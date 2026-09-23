"""Test final shared-memory arena planning and capacity pruning."""

import os
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "3rdparty" / "tvm" / "python"))
os.environ.setdefault("TVM_LIBRARY_PATH", str(PROJECT_ROOT / "build" / "lib"))

from history.unionwsp.hardware import HOPPER, InstructionKind
from history.unionwsp.parseIR import (
    BufferDescriptor,
    DataflowEdge,
    DataflowGraph,
    DataflowNode,
    RegionKind,
)
from history.unionwsp.physical import (
    analyze_shared_memory,
    enumerate_feasible_shared_memory_plans,
    enumerate_warp_allocations,
)


def _buffer(buffer_id: int, name: str, scope: str, nbytes: int):
    return BufferDescriptor(buffer_id, name, scope, nbytes, object())


def _resource_graph(hardware=HOPPER) -> DataflowGraph:
    return DataflowGraph(
        buffers=(
            _buffer(0, "shared", "shared", 96),
            _buffer(1, "fragment", "local.fragment", 512),
            _buffer(2, "private", "local", 16),
        ),
        nodes=(
            DataflowNode(
                0,
                0,
                "load",
                writes=(0,),
                instruction_kind=InstructionKind.TMA,
            ),
            DataflowNode(
                1,
                0,
                "mma",
                reads=(0,),
                writes=(1,),
                instruction_kind=InstructionKind.WGMMA,
            ),
            DataflowNode(2, 0, "consume", reads=(1,), writes=(2,)),
        ),
        edges=(
            DataflowEdge(0, 1, buffer_id=0),
            DataflowEdge(1, 2, buffer_id=1),
        ),
        region_kinds=(RegionKind.PIPELINE,),
        hardware=hardware,
    )


def _shared_reuse_graph(overlap: bool) -> DataflowGraph:
    output_region = 0 if overlap else 1
    return DataflowGraph(
        buffers=(
            _buffer(0, "pipeline_shared", "shared", 96),
            _buffer(1, "output_shared", "shared", 64),
        ),
        nodes=(
            DataflowNode(
                0,
                0,
                "load",
                writes=(0,),
                instruction_kind=InstructionKind.TMA,
            ),
            DataflowNode(
                1,
                0,
                "consume",
                reads=(0,),
                instruction_kind=InstructionKind.WGMMA,
            ),
            DataflowNode(2, output_region, "store", writes=(1,)),
        ),
        edges=(DataflowEdge(0, 1, buffer_id=0),),
        region_kinds=(RegionKind.PIPELINE, RegionKind.SERIAL),
        hardware=HOPPER,
    )


def test_counts_versioned_shared_buffers() -> None:
    graph = _resource_graph()
    groups = {0: 0, 1: 1, 2: 1}
    orders = {0: {0: {0: 0}, 1: {1: 0, 2: 1}}}
    versions = {0: 2, 1: 2, 2: 1}

    plan = analyze_shared_memory(graph, groups, orders, versions, ())

    assert plan.shared_buffer_bytes == 192
    assert plan.synchronization_bytes == 0
    assert plan.fits


def test_excess_shared_memory_is_pruned() -> None:
    groups = {0: 0, 1: 1, 2: 1}
    orders = {0: {0: {0: 0}, 1: {1: 0, 2: 1}}}
    versions = {0: 2, 1: 2, 2: 1}
    graph = _resource_graph(
        replace(HOPPER, shared_memory_capacity_bytes=191)
    )
    allocations = tuple(
        enumerate_warp_allocations(
            graph, groups, orders, versions, 128
        )
    )
    assert not tuple(
        enumerate_feasible_shared_memory_plans(
            graph,
            groups,
            orders,
            versions,
            (),
            allocations,
        )
    )


def test_shared_arena_reuses_only_non_overlapping_lifetimes() -> None:
    groups = {0: 0, 1: 0, 2: 0}
    versions = {0: 1, 1: 1}

    graph = _shared_reuse_graph(overlap=False)
    orders = {0: {0: {0: 0, 1: 1}}, 1: {0: {2: 0}}}
    plan = analyze_shared_memory(graph, groups, orders, versions, ())
    offsets = {
        item.name: item.byte_offset for item in plan.shared_allocations
    }
    assert plan.shared_buffer_bytes == 96
    assert offsets == {"pipeline_shared": 0, "output_shared": 0}

    graph = _shared_reuse_graph(overlap=True)
    orders = {0: {0: {0: 0, 1: 1, 2: 2}}, 1: {0: {}}}
    plan = analyze_shared_memory(graph, groups, orders, versions, ())
    offsets = {
        item.name: item.byte_offset for item in plan.shared_allocations
    }
    assert plan.shared_buffer_bytes == 160
    assert offsets["pipeline_shared"] != offsets["output_shared"]


def test_serial_groups_are_treated_as_concurrent() -> None:
    graph = replace(
        _shared_reuse_graph(overlap=True),
        region_kinds=(RegionKind.SERIAL,),
    )
    groups = {0: 0, 1: 1, 2: 1}
    orders = {0: {0: {0: 0}, 1: {1: 0, 2: 1}}}

    plan = analyze_shared_memory(graph, groups, orders, {0: 1, 1: 1}, ())

    assert plan.shared_buffer_bytes == 160


def test_versions_expand_the_reused_arena_not_every_lifetime() -> None:
    graph = _shared_reuse_graph(overlap=False)
    groups = {0: 0, 1: 0, 2: 0}
    orders = {0: {0: {0: 0, 1: 1}}, 1: {0: {2: 0}}}
    versions = {0: 2, 1: 1}

    plan = analyze_shared_memory(graph, groups, orders, versions, ())

    assert plan.shared_buffer_bytes == 192
    assert {item.byte_offset for item in plan.shared_allocations} == {0}


def enumerate_fa3_shared_memory_plans():
    from history.unionwsp.test.test_synchronization import (
        enumerate_fa3_synchronization_plans,
    )

    for synchronization_result in enumerate_fa3_synchronization_plans():
        (
            graph,
            _,
            _,
            result_index,
            schedule_index,
            stage_index,
            group_index,
            version_plan_index,
            _,
            groups,
            orders,
            versions,
            channels,
        ) = synchronization_result
        allocations = tuple(
            enumerate_warp_allocations(
                graph, groups, orders, versions, 128
            )
        )
        feasible = tuple(
            enumerate_feasible_shared_memory_plans(
                graph,
                groups,
                orders,
                versions,
                channels,
                allocations,
            )
        )
        yield (
            graph,
            result_index,
            schedule_index,
            stage_index,
            group_index,
            version_plan_index,
            versions,
            channels,
            allocations,
            feasible,
        )


def test_fa3_prunes_every_shared_memory_overflow() -> None:
    distribution = Counter()
    total_before = 0
    total_after = 0
    found_qkv_output_reuse = False
    found_barrier_reuse = False
    for result in enumerate_fa3_shared_memory_plans():
        allocations = result[8]
        feasible = result[9]
        total_before += len(allocations)
        total_after += len(feasible)
        distribution[(len(allocations), len(feasible))] += 1
        for _, plan in feasible:
            assert plan.fits
            assert plan.total_shared_bytes <= plan.shared_memory_capacity_bytes
            if (
                plan.synchronization_bytes
                and plan.total_shared_bytes
                < plan.shared_buffer_bytes + plan.synchronization_bytes
            ):
                found_barrier_reuse = True
            allocations_by_name = {
                item.name: item for item in plan.shared_allocations
            }
            if {
                "Q_shared",
                "K_shared",
                "V_shared",
                "O_shared",
            } <= set(allocations_by_name):
                output = allocations_by_name["O_shared"]
                qkv = tuple(
                    allocations_by_name[name]
                    for name in ("Q_shared", "K_shared", "V_shared")
                )
                assert all(item.end <= output.start for item in qkv)
                assert any(
                    output.byte_offset < item.byte_offset + item.size_bytes
                    and item.byte_offset
                    < output.byte_offset + output.size_bytes
                    for item in qkv
                )
                found_qkv_output_reuse = True

    print(
        f"FA3 shared-memory pruning: before={total_before}, "
        f"after={total_after}, "
        f"distribution={dict(distribution)}"
    )
    assert total_before > 0
    assert 0 < total_after <= total_before
    assert found_qkv_output_reuse
    assert found_barrier_reuse


if __name__ == "__main__":
    test_fa3_prunes_every_shared_memory_overflow()
