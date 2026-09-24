# OverlapPlaner：TileLang 中的流水线与 Warp Specialization 联合调度

OverlapPlaner 为 TileLang kernel 搜索 software pipeline 与 warp specialization 的联合调度。它用**依赖与硬件资源约束**生成可执行候选，用**编译、正确性检查和 GPU 实测**比较性能。系统不把预估的单个图节点 latency 当作最优调度的求解依据；动态模型只决定下一批值得测量的候选。

完整路径可以概括为：

```text
TileLang 算子与 tile 配置
  → LayoutReducer 后的 TIR
  → 事实图：操作、buffer、依赖、循环区域
  → Hopper 节点分类
  → stage / group / order / version / sync 结构
  → warp / register / shared memory 物理实现
  → 类型化 OverlapPlan
  → TileLang lowering、正确性检查、GPU 实测
```

## 一、图的构建与系统基础设施

### 从编译器实际处理的 IR 出发

搜索在 `LayoutReducer` 之后的 TIR 上抽取事实图。此时 TileLang 已完成早期规范化和布局约简，fragment 布局、线程配置等信息更接近后续编译所见的状态。这对 GEMM 的 warp 覆盖范围、跨 group fragment 通信尤为重要。相关入口是 `contract.py` 中的 `layout_reduced_prim_func` 和 `facts/extract.py` 中的 `extract_fact_graph`。

`@T.prim_func(auto_overlap=True)` 表示 kernel 请求 OverlapPlaner 调度路径。它不是 CUDA kernel 参数，也不会自动产生默认调度；编译该路径需要提供具体的 `OverlapPlan`。group 是整个 kernel 范围内的分配，`T.Pipelined` 本身不携带这个开关。

### 抽取与目标架构无关的事实图

事实图有四类核心信息：

| 对象 | 记录的信息 | 后续用途 |
| --- | --- | --- |
| 操作节点 | copy、GEMM、reduce、elementwise、store 等类型，原始 TIR statement，GEMM 的 M/N/K、dtype、布局策略等 | 定义调度单元，供架构层分类 |
| Buffer | scope、dtype、静态大小、原始 TIR Buffer 对象 | 判断通信形式、版本、寄存器和 shared memory 用量 |
| 依赖边 | 对应 buffer、RAW/WAR/WAW、迭代距离 | 约束 stage、group、order 和同步 |
| 区域 | 串行区域或 `T.Pipelined` 区域、可知时的循环长度 | 区分一次性操作与逐迭代流水线 |

抽取器利用访问范围判断两个操作是否可能触碰同一片 buffer。能证明不相交的访问不必建立冲突；无法证明时采用保守判断。对于流水线循环，还会寻找跨迭代的 RAW 依赖。这样，搜索改变源程序顺序或让 warp group 并发执行时，有明确的依赖依据。

事实图本身不写死 Hopper ISA 名称。它描述“程序必须保持哪些关系”，硬件如何执行节点由架构层回答。

### 架构分类与分层接口

`arch/api.py` 定义通用架构接口和节点特征：内存/计算类型、执行 engine、是否异步完成、是否占用 CTA 计算分区，以及发射优先级。`arch/hopper.py` 实现 Hopper 分类与资源绑定。

例如，gmem→smem copy 会询问 TileLang 自身的 producer-copy 分类器以判断 TMA/cp.async 能力，再应用当前的小 copy 默认策略；GEMM 的 WGMMA/MMA 宽度、fragment 必须保留的线程覆盖，也由 Hopper 层处理。发射优先级帮助构造初始 order，但不作为最终性能判定。

### 类型化计划与 TileLang lowering 的合同

完整候选先成为 `PhysicalPlan`，再由 `to_overlap_plan` 转换为类型化 `OverlapPlan`：

- `GroupPlan`：各 group 的 warp 数，以及可选的寄存器再分配。
- `OperationPlacement`：操作的 group、stage 和组内 order，并持有原始 statement 身份。
- `BufferPlan`：buffer 版本数、通信方式、计划的 shared memory 偏移。
- `SyncEdge`：生产者和消费者、依赖类型、同步范围、ring slot 数、完成方式及可能的 handoff 偏移。

Python 侧把计划作为 `tl.overlap_plan` 属性附在 `PrimFunc` 上。CUDA 编译流水线在 `LayoutReducer` 后调用 `LowerOverlapPlan`；C++ 侧将计划中的 statement/buffer 与当前 TIR 匹配，生成 group 分支、buffer 版本、fragment 的 shared handoff 和同步，再验证降低结果。后续仍由 TileLang 完成 software pipeline、布局推断、TileOp lowering、shared memory 合并和 CUDA 代码生成。已规划的 shared 偏移传给后续合并过程，计划外的剩余分配由合并过程处理。

这种分层使搜索器表达“如何安排”，让 lowering 负责“如何生成代码”。`operation_id` 和 `buffer_id` 必须与各自数组下标相同；statement/Buffer 身份也必须能与编译时 IR 匹配，否则计划会被拒绝。

## 二、枚举、约束与物理可行性

调度有三个主要自由度：**stage** 决定跨迭代流水线中的相对位置，**group** 决定哪些 warp 能独立前进，**order** 决定同一 group 内的发射顺序。它们可以共同改变重叠形态。固定其中任何一个维度，都可能隐藏更快的组合。

### Stage：流水线时间上的位置

`structure/stage.py` 对每个 pipeline 区域尝试稠密 stage 编号，默认最多三级。对于从生产者到消费者的依赖边，必须满足：

```text
stage(producer) ≤ stage(consumer) + iteration_distance
```

即 `iteration_distance + stage(consumer) - stage(producer)` 非负，保证生产者不会在流水线时间上落到所依赖的消费者之后。串行区域不枚举 pipeline stage。

当前还有用于限制搜索空间的拆分条件：initializer 的相关边不拆 stage；涉及内存操作的边可以拆；两端都是计算节点时，只有不同 engine 才作为拆 stage 机会。这些条件反映当前实现支持的有价值切分，不能当作所有合法调度的数学定理。节点较少的区域逐项枚举；较大区域使用受 `stage_beam` 限制的中间状态。动态循环没有统一的“两级 stage”限制，短循环候选最终仍须接受编译和运行验证。

### Group：独立执行的 warp 分区

`structure/group.py` 先收缩不能跨 group 的依赖边，使这些节点留在同一连通分量，再枚举分量的分组。依赖所用 buffer 对 group 可见时——global、shared 或 tmem——可以作为跨组机会，串行区域中的相关边也包括在内。某些跨 engine 的 `local.fragment` RAW 边亦可跨组，由 lowering 插入 shared handoff 和显式通信。register-layout copy 等边目前仍受限制。

组标签还要枚举物理排列：当前计划合同把 group 编号映射到连续的物理 warp 区间，所以相同逻辑切分、不同 group 编号可能产生不同的物理位置。枚举只保留每个 group 都参与至少一条切分机会的分组，避免没有跨组工作可做的分区。

### Order：满足依赖的组内发射顺序

对于固定的 stage/group，`structure/order.py` 生成按可达高优先级工作组织的拓扑序，以及尽量贴近源程序的拓扑序。同一轮必须满足的依赖不能颠倒。不同 group 通过显式同步联系，因此 order 记录为每个区域、每个 group 的局部顺序。

初始枚举没有列出所有拓扑序。动态搜索随后可交换同一 group 中相邻且依赖允许的节点；每次交换后重新计算版本、同步和资源。这样 order 与 stage/group 一起参与搜索，又避免初始阶段承担全排列的组合开销。

### Buffer 版本、ring slot 与同步

固定 stage/group/order 后，`structure/version.py` 根据读写冲突和访问范围计算每个 buffer 安全复用的最少版本。跨组流水线中的 shared buffer 至少双缓冲；系统也可以尝试在最小值上增加 ring slot，使 producer 在有利的计划中跑得更远。fragment 的同组流水线复用由多个私有 register tile 实现；fragment 跨组时，私有 fragment 与 shared handoff ring 分别处理。

`structure/sync.py` 派生两类同步：

- **Forward dependency**：消费者使用某轮数据前等待生产者；同组异步 gmem→smem copy 的完成也需要等待。
- **Buffer reuse**：旧数据消费完毕后才允许 producer 覆盖对应 ring slot，为 producer 提供反压。

同步区分一次性、逐迭代和区域边界等范围。系统检查同一轮的组内顺序和同步等待是否形成环；成环候选被丢弃。因此，把依赖边标记为“可跨组”还不够，版本数、完成事件和复用反压必须一起成立。

### Warp、寄存器和 shared memory 的实现检查

`HOPPER.realize` 把结构候选映射到实际资源。单 group 保持原 kernel 的线程宽度；多 group 使用连续 warp 区间。WGMMA 及其 fragment 使用四 warp 集体粒度，MMA-only 分区可用单 warp 粒度；计算 group 还必须保留 `LayoutReducer` 建立的 fragment 线程覆盖。随后估计各 group 的寄存器活跃峰值，并在可用且合法时考虑 `setmaxnreg`。

shared memory 规划计算版本化 buffer、跨组 fragment handoff、mbarrier、对齐和可复用区间，再与设备容量比较。warp、寄存器或 shared memory 不满足约束时，候选不进入 GPU 实测。这些是搜索阶段的近似可行性模型：它们能避免明显不可用的编译，但最终资源分配、lowering 和运行正确性仍须由编译器与测试确认。

枚举器在 group 数和 stage 深度之间交错产出候选，避免有限预算被单一结构桶耗尽。`SearchBudget.max_structures` 限制最终实现出的计划数量；它、stage beam 和其他预算都不是硬件常量。`exhaustive` 模式评测的是**当前已生成的候选池**，不意味着无限制地穷举全部合法调度。TileLang native 单独编译与测量，不能将其底层调度逐项出现在 OverlapPlan 候选池中视为已有保证。

## 三、动态调优的设计与实现

### 建立可追溯的候选池

`tune/run.py` 对每个 operator 和 tile 先编译、测量 TileLang native，再生成 OverlapPlan JSON 候选。动态模式还加入异步 shared producer 与计算分组的额外初始种子。每个候选都有稳定 fingerprint；目录中保存计划、结构特征、manifest、生成的 CUDA、成功结果、失败原因和动态搜索状态，便于复现、续跑和排除过期结果。

从仓库根目录启动。省略 `--operators` 时运行 `SEARCH_OPERATORS` 中的全部算子；也可以只选择本次需要搜索的算子，例如 GQA/MHA backward：

```bash
python -m OverlapPlaner.tune.run --output results --search-mode dynamic
python -m OverlapPlaner.tune.run --output results_bwd --operators gqa_bwd mha_bwd --search-mode dynamic
python -m OverlapPlaner.tune.run --output results_research --operators gdn_chunk_o_bwd gdn_chunk_delta_bwd kda_chunk_bwd_intra dequant_gemm_fp4 kda_wy_fast_bwd fused_moe --search-mode dynamic
```

`--candidate-pool` 控制初始候选池规模，`--evaluation-budget` 控制最多尝试多少次 GPU 评测；`--search-mode exhaustive` 则评测生成的全部候选，不使用动态选择器。结果位于 `<output>/<operator>/<tile>/`，算子排名见 `top30.json`。已有计划可用 `python -m OverlapPlaner.tune.replay <schedule.json>` 重新编译测试。

### 用分桶探索取得第一批实测数据

`tune/dynamic.py` 按 group 数、stage 深度、独立异步 shared producer copy 数、粗粒度 order 类型，以及物理 group 排列分桶。物理排列保留各组的 warp 数与寄存器增减角色；同一桶内优先测试异步 copy 位于 producer 组、GEMM 位于 consumer 组的 native-like 方案。初始样本轮流覆盖这些桶，使有限的首批编译预算同时观察不同重叠机制。它并不预设更多 stage 或 group 一定更快，只是避免搜索从单一结构出发。

### 用观测延迟决定下一批测谁

搜索特征描述计划本身，而不是预估单节点 latency，包括 warp 与寄存器需求、shared memory、版本数、同步 slot、跨 stage/group 的边、GEMM/copy 工作分布、order 位置和异步完成方式。达到足够样本后，系统对**已测得的 kernel latency** 的对数训练 bootstrap 回归集合。后续四个候选通常由两个预测性能较好的候选、一个实测优良计划的联合局部变体，以及一个高不确定性或未覆盖候选组成。覆盖是有限配额的软探索，不会再阻止模型利用已经获得的性能反馈。

编译失败、正确性失败和执行超时同样作为观测使用。搜索器根据结构特征附近成功与失败候选的分布估计不可行概率，并在 acquisition score 中惩罚失败密集的区域，避免不断把预算投入相似的无效计划。

模型只影响**评测顺序**。最终性能排名始终由真实 GPU benchmark 给出，预测不能替代正确性或性能结论。它也不会预先决定某个算子必须采用 stage、group 或两者结合。

### 从实测较好的方案生成联合局部变体

初始候选池不是搜索终点。`tune/joint_search.py` 从已测较好的候选及不同桶的代表候选出发，尝试五类局部移动：

1. 将一个节点移到相邻 stage。
2. 将一个可移动的 group 连通分量移到另一 group 或新 group。
3. 交换组内相邻且依赖允许的两个操作。
4. 增减跨组 shared buffer 的额外版本。
5. 对合格的小型 gmem→smem copy 比较当前后端与 TMA 变体。

copy 变体要求 TileLang 分类器确认 TMA 可用，也不会覆盖用户显式指定的指令偏好。每个移动都重新构造受影响的版本、同步和物理计划，去重后才加入候选池。所有合法移动统一使用反馈模型排序，不再强制轮换移动维度；copy 后端选择也因此与 stage/group/order 联合，而非独立的一次性静态决定。

### 编译、校验、计时与停止

每个被选择的候选由独立 worker 编译并运行：先与 operator 参考实现比较数值结果，通过后再测量延迟。独立进程隔离失败的 CUDA context；编译、正确性检查和 benchmark 分阶段监控超时。结果、错误和计划 fingerprint 一同保存，重跑时不会把旧调度的记录误用在同编号的新计划上。

达到评测预算时停止；若连续多批提升不足也可提前停止，未覆盖结构不会强制搜索耗尽预算。`dynamic_state.json` 额外记录每批候选的选择角色、成功与失败数、失败率、batch 最小值和中位数，以及累计最优值。最终输出每个 tile 的成功和失败候选数、最优已测计划，以及相对 native 的性能。

**结果的适用范围：**当前系统搜索的是在已有图抽取、机会边、stage/group 上限、beam、候选池和局部移动定义下能够生成并通过校验的调度。动态策略提高有限测量预算的利用率，但不证明全局最优。少数候选仍可能在后续 lowering 或运行时失败；它们会作为失败记录，而不会被当作有效性能结果。
