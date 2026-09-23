"""Shared-memory reuse tests against backend merge semantics."""

import pytest
import tilelang
import tilelang.language as T
from tvm import IRModule, tirx
from tvm.error import TVMError
from tvm.target import Target
from tvm.tirx.stmt_functor import post_order_visit

from history.overlaper.analysis.physical import analyze_shared_memory
from history.overlaper.analysis.schedule import (
    analyze_buffer_versions,
    build_program_orders,
    build_synchronizations,
    enumerate_group_assignments,
)
from history.overlaper.headware import HOPPER
from history.overlaper.headware.hopper import GENERIC, TMA, WGMMA
from history.overlaper.parse import (
    BufferDescriptor,
    DataflowEdge,
    DataflowGraph,
    DataflowNode,
    RegionKind,
    extract_dataflow_graph,
)
from history.overlaper.test.test_extractor import make_fa3_prim_func
from history.overlaper.test.test_stage import find_fa3_stage_assignment


@T.prim_func
def _oversized_shared_kernel(A: T.Tensor((1,), T.float32)):
    with T.Kernel(1, threads=128):
        shared = T.alloc_shared((1024,), T.float32)
        shared[0] = A[0]


def _buffer(buffer_id: int, name: str, size: int) -> BufferDescriptor:
    return BufferDescriptor(buffer_id, name, "shared", size, object())


def _reuse_graph(overlap: bool) -> DataflowGraph:
    output_region = 0 if overlap else 1
    return DataflowGraph(
        buffers=(
            _buffer(0, "pipeline_shared", 96),
            _buffer(1, "output_shared", 64),
        ),
        nodes=(
            DataflowNode(0, 0, "load", GENERIC, writes=(0,)),
            DataflowNode(1, 0, "consume", GENERIC, reads=(0,)),
            DataflowNode(2, output_region, "store", GENERIC, writes=(1,)),
        ),
        edges=(DataflowEdge(0, 1, buffer_id=0),),
        region_kinds=(RegionKind.PIPELINE, RegionKind.SERIAL),
        hardware=HOPPER,
    )


def test_reuses_only_non_overlapping_lifetimes() -> None:
    groups = {0: 0, 1: 0, 2: 0}
    versions = {0: 1, 1: 1}

    graph = _reuse_graph(overlap=False)
    orders = {0: {0: {0: 0, 1: 1}}, 1: {0: {2: 0}}}
    plan = analyze_shared_memory(graph, groups, orders, versions, ())
    assert plan.shared_buffer_bytes == 96
    assert {item.byte_offset for item in plan.shared_allocations} == {0}

    graph = _reuse_graph(overlap=True)
    orders = {0: {0: {0: 0, 1: 1, 2: 2}}, 1: {0: {}}}
    plan = analyze_shared_memory(graph, groups, orders, versions, ())
    assert plan.shared_buffer_bytes == 160
    assert len({item.byte_offset for item in plan.shared_allocations}) == 2


def test_disjoint_multi_group_buffers_do_not_alias() -> None:
    """Two cross-group buffers must not alias without happens-before."""

    graph = DataflowGraph(
        buffers=(
            _buffer(0, "a_shared", 64),
            _buffer(1, "b_shared", 64),
        ),
        nodes=(
            DataflowNode(0, 0, "a0", GENERIC, writes=(0,)),
            DataflowNode(1, 0, "a1", GENERIC, writes=(0,)),
            DataflowNode(2, 0, "b2", GENERIC, writes=(1,)),
            DataflowNode(3, 0, "b3", GENERIC, writes=(1,)),
        ),
        edges=(),
        region_kinds=(RegionKind.PIPELINE,),
        hardware=HOPPER,
    )
    groups = {0: 0, 1: 1, 2: 2, 3: 3}
    orders = {0: {0: {0: 0}, 1: {1: 0}, 2: {2: 0}, 3: {3: 0}}}
    plan = analyze_shared_memory(graph, groups, orders, {0: 1, 1: 1}, ())
    offsets = {item.name: item.byte_offset for item in plan.shared_allocations}
    assert offsets["a_shared"] != offsets["b_shared"]
    assert plan.shared_buffer_bytes >= 128


def test_tma_shared_buffer_uses_backend_fallback_alignment() -> None:
    graph = DataflowGraph(
        buffers=(
            _buffer(0, "larger_generic", 128),
            _buffer(1, "tma_shared", 96),
        ),
        nodes=(
            DataflowNode(0, 0, "generic", GENERIC, writes=(0,)),
            DataflowNode(1, 0, "tma", TMA, writes=(1,)),
        ),
        edges=(),
        region_kinds=(RegionKind.PIPELINE,),
        hardware=HOPPER,
    )
    groups = {0: 0, 1: 0}
    orders = {0: {0: {0: 0, 1: 1}}}

    plan = analyze_shared_memory(graph, groups, orders, {0: 1, 1: 1}, ())
    allocations = {item.name: item for item in plan.shared_allocations}

    assert allocations["larger_generic"].byte_offset == 0
    assert allocations["tma_shared"].alignment == 1024
    assert allocations["tma_shared"].byte_offset == 1024
    assert plan.shared_buffer_bytes == 1120


def test_fa3_shared_memory_plan() -> None:
    graph = extract_dataflow_graph(make_fa3_prim_func(), hardware=HOPPER)
    stages = {1: find_fa3_stage_assignment(graph)}
    groups = tuple(enumerate_group_assignments(graph, 2))[1]
    orders = build_program_orders(graph, stages, groups)
    versions = analyze_buffer_versions(graph, stages, groups, orders)
    synchronizations = build_synchronizations(
        graph, stages, groups, orders, versions
    )

    plan = analyze_shared_memory(
        graph, groups, orders, versions, synchronizations
    )

    assert plan.fits
    assert versions[13] == 2
    assert plan.synchronization_bytes == 64
    assert plan.merged_shared_bytes >= (
        plan.shared_buffer_bytes + plan.synchronization_bytes
    )
    assert all(
        item.byte_offset % item.alignment == 0
        for item in plan.shared_allocations
    )

    print(
        f"\nFA3 shared memory: buffers={plan.shared_buffer_bytes}, "
        f"barriers={plan.synchronization_bytes}, "
        f"merged={plan.merged_shared_bytes}, "
        f"capacity={plan.shared_memory_capacity_bytes}"
    )
    for item in plan.shared_allocations:
        print(
            f"  {item.name}: lifetime=[{item.start}, {item.end}), "
            f"size={item.size_bytes}, alignment={item.alignment}, "
            f"offset={item.byte_offset}"
        )


_WS_BUFFER_BYTES = 16384 * 2


def _merge_shared(func, max_shared: int | None = 40000) -> IRModule:
    target_config = {
        "kind": "cuda",
        "arch": "sm_90a",
        "max_threads_per_block": 1024,
    }
    if max_shared is not None:
        target_config["max_shared_memory_per_block"] = max_shared
    target = Target(target_config)
    mod = IRModule.from_expr(func)
    with target, tilelang.transform.PassContext(
        config={
            tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
        }
    ):
        mod = tirx.transform.BindTarget(target)(mod)
        return tilelang.transform.MergeSharedMemoryAllocations(
            enable_aggressive_merge=True
        )(mod)


def _shared_layout(mod: IRModule) -> tuple[dict[str, int], int]:
    offsets: dict[str, int] = {}
    merged: list[int] = []

    def visit(node) -> None:
        if isinstance(node, tirx.AllocBuffer):
            buf = node.buffer
            if buf.scope() == "shared.dyn" and str(buf.name) == "buf_dyn_shmem":
                merged.append(int(buf.shape[0]))
            return
        if not isinstance(node, tirx.Bind):
            return
        value = node.value
        if not isinstance(value, tirx.Call):
            return
        op_name = getattr(value.op, "name", str(value.op))
        if "handle_add_byte_offset" not in op_name:
            return
        offsets[str(node.var.name)] = int(value.args[1])

    for func in mod.functions.values():
        post_order_visit(func.body, visit)
    assert merged, "expected a merged shared.dyn allocation"
    return offsets, merged[0]


def test_merge_does_not_reuse_across_concurrent_ws_groups() -> None:
    """Relative positions in concurrent warp groups must not imply aliasing."""

    from tvm.script import tirx as TX

    @TX.prim_func(private=True)
    def func():
        Q = TX.alloc_buffer((16384,), dtype="float16", scope="shared.dyn")
        O = TX.alloc_buffer((16384,), dtype="float16", scope="shared.dyn")
        tx = TX.launch_thread("threadIdx.x", 384)
        TX.attr([256, 128], "tl.thread_sync.warp_specialization_scope", 0)
        if tx < 256:
            Q[tx] = TX.float16(1)
            O[tx] = TX.float16(2)
        else:
            Q[tx - 256] = TX.float16(3)

    with pytest.raises(TVMError, match="requires 65536 bytes"):
        _merge_shared(func, max_shared=40000)

    offsets, size = _shared_layout(_merge_shared(func, max_shared=None))
    assert offsets["Q"] != offsets["O"]
    assert size >= _WS_BUFFER_BYTES * 2


def test_merge_splits_concurrent_groups_under_finalized_ws_scope() -> None:
    """Thread-sync then/else are concurrent groups and must not alias."""

    from tvm.script import tirx as TX

    @TX.prim_func(private=True)
    def func():
        Q = TX.alloc_buffer((16384,), dtype="float16", scope="shared.dyn")
        O = TX.alloc_buffer((16384,), dtype="float16", scope="shared.dyn")
        tx = TX.launch_thread("threadIdx.x", 256)
        TX.attr([128, 128], "tl.thread_sync.warp_specialization_scope", 0)
        if tx < 128:
            Q[tx] = TX.float16(1)
        else:
            O[tx - 128] = TX.float16(1)

    with pytest.raises(TVMError, match="requires 65536 bytes"):
        _merge_shared(func, max_shared=40000)

    offsets, size = _shared_layout(_merge_shared(func, max_shared=None))
    assert offsets["Q"] != offsets["O"]
    assert size >= _WS_BUFFER_BYTES * 2


def test_merge_reuses_sequential_buffers_in_same_ws_branch() -> None:
    """Same warp-group sequential lifetimes may still alias under WS."""

    from tvm.script import tirx as TX

    @TX.prim_func(private=True)
    def func():
        Q = TX.alloc_buffer((16384,), dtype="float16", scope="shared.dyn")
        O = TX.alloc_buffer((16384,), dtype="float16", scope="shared.dyn")
        dummy = TX.alloc_buffer((1,), dtype="float16", scope="local")
        tx = TX.launch_thread("threadIdx.x", 256)
        TX.attr([128, 128], "tl.thread_sync.warp_specialization_scope", 0)
        if tx < 128:
            Q[tx] = TX.float16(1)
            O[tx] = TX.float16(2)
        else:
            dummy[0] = TX.float16(0)

    offsets, size = _shared_layout(_merge_shared(func, max_shared=40000))
    assert offsets["Q"] == offsets["O"]
    assert size == _WS_BUFFER_BYTES


def test_merge_applies_planned_offsets_instead_of_liveness() -> None:
    """A program-schedule packing is applied verbatim, even if liveness would alias."""

    from tvm.script import tirx as TX

    @TX.prim_func(private=True)
    def func():
        Q = TX.alloc_buffer((16384,), dtype="float16", scope="shared.dyn")
        O = TX.alloc_buffer((16384,), dtype="float16", scope="shared.dyn")
        dummy = TX.alloc_buffer((1,), dtype="float16", scope="local")
        tx = TX.launch_thread("threadIdx.x", 256)
        TX.attr([128, 128], "tl.thread_sync.warp_specialization_scope", 0)
        if tx < 128:
            Q[tx] = TX.float16(1)
            O[tx] = TX.float16(2)
        else:
            dummy[0] = TX.float16(0)

    func = func.with_attr("tl.smem_offset_map", {"Q": 0, "O": _WS_BUFFER_BYTES})
    func = func.with_attr("tl.smem_planned_arena_bytes", _WS_BUFFER_BYTES * 2)
    offsets, size = _shared_layout(_merge_shared(func, max_shared=None))
    assert offsets["Q"] == 0
    assert offsets["O"] == _WS_BUFFER_BYTES
    assert size >= _WS_BUFFER_BYTES * 2


def test_merge_appends_unplanned_buffers_after_planned_arena() -> None:
    """Buffers absent from the plan are appended after the reserved arena."""

    from tvm.script import tirx as TX

    @TX.prim_func(private=True)
    def func():
        Q = TX.alloc_buffer((16384,), dtype="float16", scope="shared.dyn")
        extra = TX.alloc_buffer((16,), dtype="float16", scope="shared.dyn")
        tx = TX.launch_thread("threadIdx.x", 128)
        Q[tx] = TX.float16(1)
        extra[0] = TX.float16(2)

    func = func.with_attr("tl.smem_offset_map", {"Q": 0})
    func = func.with_attr("tl.smem_planned_arena_bytes", _WS_BUFFER_BYTES)
    offsets, size = _shared_layout(_merge_shared(func, max_shared=None))
    assert offsets["Q"] == 0
    assert offsets["extra"] >= _WS_BUFFER_BYTES
    assert size >= _WS_BUFFER_BYTES + 32


def test_merge_reuses_then_else_under_default_ws_scope() -> None:
    """Default kWarpSpecializationScope keeps sequential then/else liveness."""

    from tvm.script import tirx as TX

    @TX.prim_func(private=True)
    def func():
        Q = TX.alloc_buffer((16384,), dtype="float16", scope="shared.dyn")
        O = TX.alloc_buffer((16384,), dtype="float16", scope="shared.dyn")
        tx = TX.launch_thread("threadIdx.x", 256)
        TX.attr([128, 128], "kWarpSpecializationScope", 0)
        if tx < 128:
            Q[tx] = TX.float16(1)
        else:
            O[tx - 128] = TX.float16(1)

    offsets, size = _shared_layout(_merge_shared(func, max_shared=40000))
    assert offsets["Q"] == offsets["O"]
    assert size == _WS_BUFFER_BYTES


def test_merge_ignores_seqstmt_prefix_when_splitting_ws_groups() -> None:
    """Common fence/init prefixes are not independent warp groups."""

    from tvm.script import tirx as TX

    @TX.prim_func(private=True)
    def func():
        Q = TX.alloc_buffer((16384,), dtype="float16", scope="shared.dyn")
        O = TX.alloc_buffer((16384,), dtype="float16", scope="shared.dyn")
        tx = TX.launch_thread("threadIdx.x", 256)
        with TX.attr([128, 128], "tl.thread_sync.warp_specialization_scope", 0):
            TX.evaluate(0)
            if tx < 128:
                Q[tx] = TX.float16(1)
            else:
                O[tx - 128] = TX.float16(1)

    offsets, size = _shared_layout(_merge_shared(func, max_shared=None))
    assert offsets["Q"] != offsets["O"]
    assert size >= _WS_BUFFER_BYTES * 2


def test_merge_does_not_alias_disjoint_multi_group_buffers() -> None:
    """A in groups 0/1 and B in groups 2/3 must not share an offset."""

    from tvm.script import tirx as TX

    @TX.prim_func(private=True)
    def func():
        A = TX.alloc_buffer((16384,), dtype="float16", scope="shared.dyn")
        B = TX.alloc_buffer((16384,), dtype="float16", scope="shared.dyn")
        tx = TX.launch_thread("threadIdx.x", 256)
        TX.attr([64, 64, 64, 64], "tl.thread_sync.warp_specialization_scope", 0)
        if tx < 64:
            A[tx] = TX.float16(1)
        elif tx < 128:
            A[tx - 64] = TX.float16(2)
        elif tx < 192:
            B[tx - 128] = TX.float16(3)
        else:
            B[tx - 192] = TX.float16(4)

    with pytest.raises(TVMError, match="requires 65536 bytes"):
        _merge_shared(func, max_shared=40000)

    offsets, size = _shared_layout(_merge_shared(func, max_shared=None))
    assert offsets["A"] != offsets["B"]
    assert size >= _WS_BUFFER_BYTES * 2


def test_merge_skips_cuda_default_static_shared_memory_limit() -> None:
    """cudaDevAttrMaxSharedMemoryPerBlock is not the opt-in dynamic cap."""

    from tvm.script import tirx as TX

    @TX.prim_func(private=True)
    def func():
        Q = TX.alloc_buffer((16384,), dtype="float16", scope="shared.dyn")
        O = TX.alloc_buffer((16384,), dtype="float16", scope="shared.dyn")
        tx = TX.launch_thread("threadIdx.x", 384)
        TX.attr([256, 128], "tl.thread_sync.warp_specialization_scope", 0)
        if tx < 256:
            Q[tx] = TX.float16(1)
            O[tx] = TX.float16(2)
        else:
            Q[tx - 256] = TX.float16(3)

    offsets, size = _shared_layout(_merge_shared(func, max_shared=49152))
    assert offsets["Q"] != offsets["O"]
    assert size >= _WS_BUFFER_BYTES * 2


def test_backend_rejects_final_shared_memory_over_target_limit() -> None:
    target = Target(
        {
            "kind": "cuda",
            "arch": "sm_90a",
            "max_shared_memory_per_block": 1024,
        }
    )
    mod = IRModule({"main": _oversized_shared_kernel})
    with target, tilelang.transform.PassContext(config={}):
        mod = tirx.transform.BindTarget(target)(mod)
        mod = tilelang.transform.MaterializeKernelLaunch()(mod)
        mod = tilelang.transform.PlanAndUpdateBufferAllocationLocation()(mod)
        mod = tilelang.transform.LowerOpaqueBlock()(mod)
        mod = tilelang.transform.FlattenBuffer()(mod)
        with pytest.raises(TVMError, match="requires 4096 bytes"):
            tilelang.transform.MergeSharedMemoryAllocations()(mod)


def test_mla_no_split_dyn_smem_fits_h100() -> None:
    """Default MLA decode must stay under Hopper opt-in dynamic shared memory."""

    import importlib.util
    from pathlib import Path

    pytest.importorskip("einops")
    example_path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "deepseek_mla"
        / "example_mla_decode.py"
    )
    spec = importlib.util.spec_from_file_location("example_mla_decode", example_path)
    example = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(example)

    prim = example.flashattn.get_tir(
        1, 128, 1, 64, 512, 64, 64, 64, 1, (512 + 64) ** -0.5
    )
    target = Target("cuda")
    with target, tilelang.transform.PassContext(
        config={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
    ):
        artifact = tilelang.lower(prim, target=target)
    smem = max(
        int(func.attrs.get("dyn_shared_memory_buf", 0) or 0)
        for func in artifact.device_mod.functions.values()
    )
    from tilelang.carver.arch.driver import get_max_dynamic_shared_size_bytes

    limit = get_max_dynamic_shared_size_bytes() or (227 * 1024)
    assert smem < 296960
    assert smem <= limit, (
        f"MLA dyn shared memory {smem} exceeds opt-in limit {limit}"
    )
