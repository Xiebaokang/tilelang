# WSPipeline

`wspipeline/` 只实现自动 warp-specialized pipeline 策略路径。手工策略、枚举统计、
拒绝回调、正确性证书和独立特征对象均已移除；合法性由验证函数直接检查。

## 主路径

```text
apply_active_schedule
└─ analyze_program_dataflow                 analysis/extractor.py
└─ 默认自动 planner（可由 use_schedule_planner 覆盖）
   ├─ enumerate_realized_schedules          search/realization.py
   │  ├─ enumerate_logical_schedules        scheduling/warp_specialization.py
   │  │  ├─ enumerate_stage_assignments     scheduling/stage.py
   │  │  ├─ enumerate_logical_warp_partitions
   │  │  └─ enumerate_region_schedules_for_stages
   │  │     └─ enumerate_feasible_orders    scheduling/order.py
   │  ├─ enumerate_warp_allocations         physical/warp_allocation.py
   │  ├─ enumerate_buffer_plans             physical/buffer_planning.py
   │  └─ build_realized_schedule
   │     ├─ build_synchronization_channels  synchronization.py
   │     ├─ build_register_plan             warp_allocation.py
   │     ├─ build_register_pressure_plan    register_pressure.py
   │     └─ validate_realized_schedule      physical/validation.py
   └─ score_realized_schedule               search/scoring.py
└─ apply_schedule_to_ir                     integration/apply.py
   ├─ validate_realized_schedule
   └─ build_ir_plan
└─ C++ lowering                             src/
```

### 1. 提取

`analyze_program_dataflow` 把整个 `PrimFunc` 转成 operation、buffer、region 和
RAW/WAR/WAW 边。pipeline loop 前后的 prologue/epilogue 都属于同一 program graph，
分别形成 serial region；它们会参与 group 划分、跨 region 依赖和同步，而 stage 只在
pipeline region 内有意义。

用户通过下面的唯一开关请求自动 WSP：

```python
for k in T.Pipelined(loop_range, auto_wsp=True):
    ...
```

它生成 `tl.wsp.auto_schedule=1`。该标记优先于 `num_stages` 和手工
stage/order/group/sync；启用后这些决策全部由 WSP 重新生成。同一 `PrimFunc` 中的
pipeline loop 必须统一启用或统一关闭自动 WSP，混合模式会报错。显式请求但找不到
合法候选时也会报错，不会静默回退旧 pipeline 路径。

`auto_wsp` 是不可绕过的编译路径开关：

- `auto_wsp=False`：保留手工 `num_stages` 并使用 TileLang 官方 pipeline；即使外层安装
  `use_schedule_planner`，WSP planner 也不会被调用。
- `auto_wsp=True`：忽略手工 pipeline 决策，使用 WSP 自动搜索和 program schedule
  lowering。

### 2. 逻辑枚举

`infer_max_stages` 用 pipeline operation 数、memory/MMA/scalar 执行角色、异步角色
和可证明的静态 loop extent 推导 stage 上界。`enumerate_stage_assignments` 公平
枚举 1 到该上界的合法 assignment，选中深度不再取自原始 loop 的
`num_stages`。`num_stages=1` 表示所有 operation 都在 stage 0，即不做跨迭代
SWP。

`num_groups` 的上界由 operation 角色和 target 可容纳的 warp group 数推导，
候选从上界一直覆盖到 1。`num_groups=1` 是不做 warp specialization 的基线。
因此 `(num_groups, num_stages)=(1, 1)` 是不做两种优化的完整基线，也会和
其他候选一起评分。随后 `order.py` 为每个 region/group 枚举满足零距离
依赖的拓扑序。stage、group、order 的评分只改变候选顺序，不替代合法性检查。

Stage assignment 使用 best-first 枚举，避免为每个目标评分重复遍历整棵搜索树。
`local.fragment` 的单版本复用和 handoff import 约束在 order 枚举阶段就会剪枝，
同时仍在物理 buffer 验证中复查。

### 3. 物理实现

逻辑候选依次展开为 target 合法的 warp 分配和 buffer version。每个组合由
`build_realized_schedule` 补全同步、寄存器控制和寄存器压力，然后立即调用
`validate_realized_schedule`。失败组合在枚举器内部跳过；成功组合才会交给评分器。
符号 region overlap 结果在一次搜索生命周期内共享缓存，不会对每个 warp/group 候选
重复调用 TVM `arith.Analyzer`。

### 4. 选择与 lowering

`score_realized_schedule` 直接返回一个 `float`，综合逻辑重叠、同步、通信、shared
memory、寄存器压力和 CTA 扩张，值越小越好。选中的候选经再次验证后由
`build_ir_plan` 转为 C++ lowering 所需的紧凑整数数组，并作为
`tl.program_schedule.*` 属性附到 `PrimFunc`。

## 目录职责

- `analysis/`：`core.py`、`program.py` 和 `extractor.py` 共同定义程序图并从 TIR 提取它。
- `scheduling/`：`stage.py`、`order.py` 和 `warp_specialization.py` 负责逻辑候选。
- `physical/`：负责 warp、buffer、register、synchronization 的物理实现和硬约束验证。
- `search/`：展开完整实现候选并评分。
- `integration/`：将选中策略写入 IR 并接入 CUDA 编译流水线。

公开 API 刻意限制在 `wspipeline/__init__.py` 所列的自动主路径函数。内部枚举器和计划
对象应从其所属模块阅读，不作为稳定的调用入口。

## 测试

```bash
python -m pytest wspipeline/test/test_full_path.py -x
python -m pytest wspipeline/test/test_auto_wsp_switch.py -x
python -m pytest wspipeline/test/test_performance.py -x
```

执行 CUDA kernel 的测试需要相应 GPU；不具备设备时应跳过。
