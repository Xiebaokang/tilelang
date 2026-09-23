# UnionWSP

`unionwsp` is a new implementation developed with `wspipeline` as a read-only
reference. Its graph extraction, logical scheduling and physical-resource
checks are separated by responsibility; the existing `wspipeline` package
remains unchanged.

The current realization path is:

```text
graph -> stage -> group -> order -> buffer versions -> synchronization
      -> warp/register allocation -> shared-memory pruning
```

`physical/shared_memory.py` is the final shared-memory resource gate.
Versioned shared buffers and synchronization mbarriers receive first/last-touch
intervals and are packed into one aligned arena with the same linear-scan
policy as `MergeSharedMemoryAllocations`. Candidates exceeding the selected
hardware description's shared-memory capacity are not yielded. Per-thread and
CTA-wide register limits are handled entirely by `physical/warp_allocation.py`.

The complete entry point is `unionwsp.enumerate_wsp_schedules`:

```python
from unionwsp import enumerate_wsp_schedules

for schedule in enumerate_wsp_schedules(graph, original_threads=256):
    ...
```

By default, stage depths and logical group counts are inferred and enumerated.
Either dimension can be fixed independently:

```python
enumerate_wsp_schedules(
    graph,
    original_threads=256,
    num_stages=2,
    num_groups=2,
)
```

`original_threads` describes the thread count of the input CTA. It is retained
unchanged when the schedule has only one group. For a specialized multi-group
schedule it is not a lower bound. Each group follows the execution granularity
declared by the hardware: kinds in `fixed_one_granule_kinds` use one granule,
while other groups are enumerated in `specialized_group_warp_multiple` steps up
to the CTA thread limit. For Hopper collective groups, the output fragment's
tile rows dynamically determine the layout limit through
`collective_max_warps`; this is not derived from the input CTA width. An
optional `warpgroup_collective_max_warps` can impose an additional absolute
ceiling. On hardware that requires register redistribution,
each multi-group allocation also carries a mandatory, 8-register-aligned
`setmaxnreg` assignment. Warp counts and register counts are accepted jointly
only when every group's estimated local-buffer demand and the hardware-wide
register budget are both satisfied.
TileLang's later layout-inference pass derives each group's layout from its
selected thread count.

## IR application and measured search

An active `unionwsp.use_schedule_planner` is called immediately after
`tilelang.transform.LayoutReducer`. The callback receives the graph extracted
from that exact IR, and its selected `WSPSchedule` is encoded as a checked
`tl.program_schedule.*` contract. The existing C++ `LowerProgramSchedule` pass
then materializes group branches, pipeline stage/order annotations, versioned
buffers, fragment handoffs, mbarriers, and `setmaxnreg` operations.

`unionwsp.search_wsp_schedules` enumerates at this boundary, gives every input
PrimFunc a distinct cache identity, compiles each candidate, optionally checks
it against a reference implementation, measures latency, and maintains a
TFLOPS-ranked Top-K. The default Top-K is 20. `top20.json`, the selected CUDA
sources, and `failures.jsonl` are written to the requested output directory.

The standalone FA3 entry is:

```bash
python unionwsp/test/test_search_fa3.py
```

For a short end-to-end smoke test before an exhaustive run:

```bash
python unionwsp/test/test_search_fa3.py --max-schedules 1 --warmup 10 --rep 10
```
