# 为 UnionWSP 添加新硬件

`hardware` 目录负责描述两件事：

1. 一个执行单元在该硬件上的调度属性，例如发射优先级和同步/异步性。
2. 如何根据 extractor 提供的 IR 信息识别该硬件上的执行单元。

IR 遍历、buffer 分析和依赖构图仍由 `parseIR/extractor.py` 负责，硬件模块不应修改图结构。

## 当前文件

- `spec.py`：只保存与具体硬件无关的 `InstructionKind`、`OperationInfo`
  和 `HardwareSpec`。
- `hopper.py`：Hopper 的 operation 分类规则和 `HOPPER` 描述。
- `resolve.py`：根据 TVM `Target` 自动选择硬件描述。
- `__init__.py`：对外导出硬件描述。

## OperationInfo

Extractor 会为每个 operation 构造一个 `OperationInfo`：

```python
@dataclass(frozen=True)
class OperationInfo:
    name: str
    call_names: tuple[str, ...]
    read_scopes: tuple[str, ...]
    write_scopes: tuple[str, ...]
```

- `name`：operation 的外层名称，例如 `tl.tileop.copy`、`parallel`。
- `call_names`：operation 内部出现的调用，例如 `tl.tileop.gemm`、`tirx.exp2`。
- `read_scopes`：读取 buffer 的 scope。
- `write_scopes`：写入 buffer 的 scope。

硬件分类器只应该根据这些信息返回 `InstructionKind`，不应该创建 node 或 edge。

## 添加使用现有执行单元的新硬件

如果新硬件可以复用当前的 `InstructionKind`，只需要添加一个硬件模块。下面以 `blackwell.py` 为例：

```python
from .spec import HardwareSpec, InstructionKind, OperationInfo


def classify_blackwell_operation(operation: OperationInfo) -> InstructionKind:
    names = (operation.name, *operation.call_names)

    if any("gemm" in name or "mma" in name for name in names):
        return InstructionKind.WGMMA
    if any("tma" in name for name in names):
        return InstructionKind.TMA
    if any(name in {"tirx.exp2", "tirx.log2"} for name in names):
        return InstructionKind.FUNCTION
    return InstructionKind.GENERIC


_ISSUE_PRIORITIES = {
    InstructionKind.GENERIC: 0,
    InstructionKind.FUNCTION: 0,
    InstructionKind.RSCP: 2,
    InstructionKind.GSCP: 4,
    InstructionKind.WGMMA: 5,
    InstructionKind.TMA: 6,
}

_ROLES_CLASSIFY = {
    InstructionKind.GENERIC: 0,
    InstructionKind.FUNCTION: 0,
    InstructionKind.RSCP: 0,
    InstructionKind.GSCP: 0,
    InstructionKind.WGMMA: 1,
    InstructionKind.TMA: 2,
}

BLACKWELL = HardwareSpec(
    name="blackwell",
    supported_instructions=frozenset(_ISSUE_PRIORITIES),
    classifier=classify_blackwell_operation,
    issue_priorities=_ISSUE_PRIORITIES,
    roles_classify=_ROLES_CLASSIFY,
    async_instruction_kinds=frozenset(
        {InstructionKind.GSCP, InstructionKind.WGMMA, InstructionKind.TMA}
    ),
    multiversion_output_instruction_kinds=frozenset(
        {InstructionKind.WGMMA, InstructionKind.TMA}
    ),
    stage_splittable_pairs=frozenset(
        {
            (InstructionKind.TMA, InstructionKind.TMA),
            (InstructionKind.TMA, InstructionKind.WGMMA),
        }
    ),
    cross_group_dependency_pairs=frozenset(
        {
            (InstructionKind.TMA, InstructionKind.WGMMA),
            (InstructionKind.RSCP, InstructionKind.TMA),
        }
    ),
    standalone_group_kinds=frozenset({InstructionKind.TMA}),
    warp_size=32,
    max_threads_per_block=1024,
    warpgroup_warps=4,
    specialized_group_warp_multiple=4,
    warpgroup_collective_kinds=frozenset({InstructionKind.WGMMA}),
    warpgroup_collective_max_warps=None,
    warpgroup_collective_tile_rows=64,
    fixed_one_granule_kinds=frozenset({InstructionKind.TMA}),
    register_receiver_kinds=frozenset({InstructionKind.WGMMA}),
    setmaxnreg_required_for_specialization=True,
    setmaxnreg_min_registers=24,
    setmaxnreg_max_registers=240,
    setmaxnreg_granularity=8,
    shared_memory_capacity_bytes=232448,
    register_file_capacity=64512,
    max_registers_per_thread=255,
    mbarrier_bytes=8,
)
```

`stage_splittable_pairs` 是该硬件唯一的 stage 性能约束白名单。它保存有向
的 `(producer_kind, consumer_kind)`：只有直接依赖边对应的 pair 出现在
集合中，两端才允许分配到不同 stage；其余同类或异类 pair 都必须处于相同
stage。`(TMA, WGMMA)` 不会自动允许反方向 `(WGMMA, TMA)`。

同类拆分也使用同一种表达，例如 `(TMA, TMA)`。这个配置属于
`HardwareSpec`，添加 AMD 时只需列出 AMD 硬件真正允许拆分的 pair，无需
修改 `schedule/stage.py`。

Hopper 使用显式白名单，而不是所有不同类型 pair 的笛卡尔积。

`cross_group_dependency_pairs` 是独立的 warp-group 依赖边界白名单，不能复用
`stage_splittable_pairs`。所有实际跨 group 的直接依赖都必须出现在这个
白名单中，并且使用 group-visible buffer；其他直接依赖两端会被合并到
同一 group。实现位于 `schedule/warp_specialization.py`。

`standalone_group_kinds` 表示哪些 SERIAL operation 可以独立创建一个逻辑
group。普通 SERIAL operation 只能跟随已有角色，不能增加 group 数量。
例如 Hopper 将 `TMA` 放入该集合，并通过 `(RSCP, TMA)` 的 shared-memory
边界允许最终 TMA store 独立成为写回 group。这个属性不能绕过
`cross_group_dependency_pairs` 或 buffer 可见性检查。

## 执行角色与自动 stage/group 上限

`roles_classify` 将该硬件支持的每一种 `InstructionKind` 映射为一个整数
执行角色。整数本身没有跨硬件的固定含义，只用于判断两种指令是否使用
同一类硬件执行角色：相同整数表示同一角色，不同整数表示不同角色。

当前 Hopper 配置为：

```text
role 0: GENERIC、FUNCTION、RRCP、RSCP、GRCP、GSCP
role 1: WGMMA
role 2: TMA
```

自动 stage 上限按单个 pipeline region 中实际出现的角色数和明确允许同类
拆分的 kind 数量计算：

```text
max_stages = min(
    region node 数量,
    region role 数量 + 允许同类拆分的 kind 数量,
)
```

自动 group 上限按整个 graph 计算：

```text
physical_limit =
    max_threads_per_block / warp_size / warpgroup_warps

max_groups = min(
    不可拆分 component 数量,
    graph 中实际出现的 role 数量,
    physical_limit,
)
```

结果至少为 1。这里得到的只是搜索上限，真正的 group assignment 仍需通过
stage opportunity、`cross_group_dependency_pairs`、buffer 可见性和 SERIAL component
等约束。新增硬件时，`roles_classify` 应覆盖该硬件的全部
`supported_instructions`。

## 物理资源配置

物理 warp 分配还使用以下硬件属性：

- `warp_size`：一个 warp 的线程数。
- `max_threads_per_block`：一个 CTA 可以使用的最大线程数。
- `warpgroup_warps`：一个硬件 warpgroup 包含的 warp 数。
- `specialized_group_warp_multiple`：WSP group 的基础 warp 粒度。Hopper
  设为 4，因此不会枚举无法形成完整 warpgroup 的 warp 数。
- `warpgroup_collective_kinds`：必须以完整 warpgroup 执行的指令类型。
- `warpgroup_collective_max_warps`：当前后端 layout 实现允许一个包含
  warpgroup collective 的逻辑 group 使用的绝对最大 warp 数；`None` 表示不
  设置固定上限。
- `warpgroup_collective_tile_rows`：一个 warpgroup 负责的 collective 输出
  tile 行数。Hopper 设置为 64，所以从 WGMMA 输出 fragment 的第一维动态
  推导 `max_warps = block_M // 64 * 4`。如果一个 group 包含多个 collective，
  使用所有推导结果中的最小值。这个限制与输入 CTA 的原始 warp 数无关；
  `warpgroup_collective_max_warps` 非空时，它会作为额外的绝对上限。
- `fixed_one_granule_kinds`：只需要一个基础执行粒度的纯执行单元类型，
  例如 Hopper 的纯 TMA group 固定为一个 warpgroup。
- `register_receiver_kinds`：包含这些指令的 group 优先作为寄存器重分配的
  接收方，其余 group 优先作为释放方。如果所有 group 都属于优先接收方，
  联合求解器会在 warp 数和寄存器下限确定后，优先选择不含 warpgroup
  collective、且寄存器下限最小的 group 作为唯一后备 donor。
- `setmaxnreg_required_for_specialization`：多 group 时是否必须生成
  `setmaxnreg` 配置。Hopper 开启该约束，单 group 官方路径不设置。
- `setmaxnreg_min_registers`、`setmaxnreg_max_registers` 和
  `setmaxnreg_granularity`：每个 group 的合法寄存器范围与对齐。枚举会先按
  buffer 生命周期求出该 group 的最小需求，再向上对齐到 8 的倍数。
- `shared_memory_capacity_bytes`：一个 CTA 可使用的 shared-memory 上限。
- `register_file_capacity`：一个 CTA/SM 可分配的 32-bit register 总量上限。
  warp 分配和最终资源检查共同使用这一个字段。Hopper 设为 64512，与
  TileLang C++ pass 一致；它小于物理 65536，因为需要保留架构记账空间。
- `max_registers_per_thread`：架构允许的每线程 register 绝对上限；联合求解
  会同时取它与 `setmaxnreg_max_registers` 的较小值。
- `mbarrier_bytes`：一个 mbarrier 槽位占用的 shared-memory 字节数。

warp 数和 `setmaxnreg` 联合求解：纯访存 donor 固定使用一个执行粒度，
其 register 上限是满足自身估计需求的最小对齐值；receiver 的 warp 数按
硬件粒度递增，每一种 warp 数都会根据 order、buffer version 和 local
buffer 生命周期重新估计每线程下限。求解器先满足全部 group 的对齐下限，
再在全 CTA register 预算内把剩余容量分给 receiver。只有线程数、每线程
上下限、register 对齐和全 CTA 预算全部满足时才产生方案。开启前半区间
开关时，过滤发生在完整可行性求解之后。这些属性都属于硬件描述；添加新
硬件时按其执行粒度和寄存器限制填写即可，无需修改
`physical/warp_allocation.py`。

指令的基础调度属性同样属于 `HardwareSpec`：

- `issue_priorities`：必须覆盖该硬件的全部 `supported_instructions`，数值
  越大，在依赖已经满足的 ready nodes 中越早发射。
- `async_instruction_kinds`：该硬件上发射后异步完成的指令集合。
- `multiversion_output_instruction_kinds`：该硬件上允许输出 buffer 独立
  枚举 `{minimum, minimum + 1}` 的指令集合。

这些属性不能放回全局表。同一种 `InstructionKind` 在不同硬件上可以具有
不同属性，例如一个硬件可以将某类 copy 异步执行，另一个硬件可以将其
同步执行。

然后在 `hardware/__init__.py` 中导出：

```python
from .blackwell import BLACKWELL
```

最后，在 `hardware/resolve.py` 中注册 CUDA compute capability：

```python
_CUDA_ARCHITECTURES = {
    "90": HOPPER,
    "100": BLACKWELL,
}
```

解析器会忽略单个架构后缀，因此 `sm_90` 和 `sm_90a` 都映射到键
`"90"`，但不会把 `sm_900` 误认为 `sm_90`。

Extractor 选择硬件的优先级是：

1. 调用者显式传入的 `hardware`。
2. 调用者显式传入的 `target`。
3. PrimFunc 的 `target` attribute。
4. 无法识别时抛出异常。

显式选择硬件适合测试分类器，或者覆盖 target 的自动判断：

```python
from unionwsp.hardware import BLACKWELL
from unionwsp.parseIR import extract_dataflow_graph

graph = extract_dataflow_graph(prim_func, hardware=BLACKWELL)
```

正常编译路径只需提供 target：

```python
from tvm.target import Target

target = Target({"kind": "cuda", "arch": "sm_90a"})
graph = extract_dataflow_graph(prim_func, target=target)
```

如果 PrimFunc 已有同样的 `target` attribute，则可以直接调用：

```python
graph = extract_dataflow_graph(prim_func)
```

当前 `sm_90` 和 `sm_90a` 会自动选择 `HOPPER`。没有架构信息或遇到尚未
注册的架构时会明确报错，不会静默回退到错误的硬件分类器。

## 添加新的执行单元

如果新硬件存在当前枚举无法表达的执行单元，例如 `TCGEN05`，需要完成以下三步。

第一步，在 `spec.py` 的 `InstructionKind` 中添加成员：

```python
class InstructionKind(str, Enum):
    # Existing kinds...
    TCGEN05 = "tcgen05"
```

第二步，只在新硬件文件中设置它的调度属性：

```python
_ISSUE_PRIORITIES = {
    # This hardware's existing kinds...
    InstructionKind.TCGEN05: 7,
}

_ROLES_CLASSIFY = {
    # This hardware's existing roles...
    InstructionKind.TCGEN05: 3,
}

MY_HARDWARE = HardwareSpec(
    ...,
    supported_instructions=frozenset(_ISSUE_PRIORITIES),
    issue_priorities=_ISSUE_PRIORITIES,
    roles_classify=_ROLES_CLASSIFY,
    async_instruction_kinds=frozenset(
        {..., InstructionKind.TCGEN05}
    ),
    multiversion_output_instruction_kinds=frozenset(
        {..., InstructionKind.TCGEN05}
    ),
)
```

这里需要明确四类属性：

- `issue_priorities`：数值越大，在依赖已经满足的 ready nodes 中越早
  发射。它不能覆盖 RAW、WAR 或 WAW 依赖。
- `async_instruction_kinds`：表示这些指令在当前硬件上发射后异步完成。
  同一种 `InstructionKind` 在另一个硬件上可以具有不同的同步属性。
- `roles_classify`：表示该指令在当前硬件上属于哪个执行角色，用于自动
  推导 stage/group 搜索上限。
- `multiversion_output_instruction_kinds`：该类指令在 pipeline 中写出的
  buffer 是否允许参与独立 version 枚举。只有存在访问范围可能重叠且位于
  不同 stage 的 writer/accessor，并且该 writer 属于此集合时，才会实际
  枚举 `{minimum, minimum + 1}`；正确性最小值可以是 1。

第三步，在新硬件分类器中返回这个成员，并将它加入 `supported_instructions`。如果分类器返回了硬件不支持的成员，`HardwareSpec.classify()` 会立即抛出异常。

## 分类规则建议

分类器应遵循从具体到通用的顺序：

1. 特定 intrinsic，例如 `tcgen05`、`wgmma`、`tma`。
2. copy 的 source/destination scope 组合。
3. SFU 初等函数。
4. 最后回退到 `GENERIC`。

不要把普通 `fill`、`reduce` 或算术 operation 标记为 `FUNCTION`。`FUNCTION` 当前只表示 SFU 执行的初等函数，例如 `exp2`、`log2`、`sin` 和 `sqrt`。

## 测试

每个新硬件至少应测试：

- MMA 类 operation 的分类。
- 异步 copy 的分类。
- SFU operation 的分类。
- 普通 operation 回退到 `GENERIC`。
- 一个真实 kernel 通过该硬件描述提取后，node、region 和直接依赖边正确。
- 该硬件的 target 架构能够被 `resolve_hardware()` 正确解析。
- 未注册的相邻架构不会被误识别。

可以参考 `unionwsp/test/test_hardware.py`。运行全部 UnionWSP CPU 测试：

```bash
PYTHONPATH=3rdparty/tvm/python:. \
TVM_LIBRARY_PATH=build/lib \
conda run -n tilelang python -m pytest unionwsp/test -q
```

增加 `InstructionKind` 后，只需要在实际支持它的硬件文件中加入对应配置；
其他硬件不应把它加入 `supported_instructions`。`HardwareSpec` 会检查
`issue_priorities` 是否完整覆盖该硬件支持的指令；同时应测试
`roles_classify` 覆盖全部 supported instruction。
