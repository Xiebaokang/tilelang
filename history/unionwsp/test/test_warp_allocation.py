"""Test physical warp allocation, including every feasible FA3 schedule."""

import os
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace


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
    estimate_group_registers_per_thread,
    enumerate_warp_allocations,
    register_receiver_groups,
    validate_warp_allocation,
)
from history.unionwsp.schedule import analyze_buffer_versions


_UNBOUNDED_COLLECTIVE_HOPPER = replace(
    HOPPER,
    warpgroup_collective_max_warps=None,
    warpgroup_collective_tile_rows=None,
)


def _two_role_graph() -> DataflowGraph:
    return DataflowGraph(
        buffers=(),
        nodes=(
            DataflowNode(
                0,
                0,
                "load",
                instruction_kind=InstructionKind.TMA,
            ),
            DataflowNode(
                1,
                0,
                "mma",
                instruction_kind=InstructionKind.WGMMA,
            ),
        ),
        edges=(),
        region_kinds=(RegionKind.PIPELINE,),
        hardware=_UNBOUNDED_COLLECTIVE_HOPPER,
    )


def _buffer(
    buffer_id: int,
    name: str,
    scope: str,
    nbytes: int,
    shape=(),
):
    return BufferDescriptor(
        buffer_id, name, scope, nbytes, SimpleNamespace(shape=shape)
    )


def _load_mma_consumer_graph(
    private_registers_per_thread: int = 0,
    hardware=_UNBOUNDED_COLLECTIVE_HOPPER,
) -> DataflowGraph:
    buffers = [
        _buffer(0, "shared", "shared", 128),
        _buffer(
            1,
            "accumulator",
            "local.fragment",
            131 * 128 * 4,
            shape=(128, 131),
        ),
    ]
    mma_writes = [1]
    consumer_reads = [1]
    if private_registers_per_thread:
        buffers.append(
            _buffer(
                2,
                "private",
                "local",
                private_registers_per_thread * 4,
            )
        )
        mma_writes.append(2)
        consumer_reads.append(2)
    return DataflowGraph(
        buffers=tuple(buffers),
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
                writes=tuple(mma_writes),
                instruction_kind=InstructionKind.WGMMA,
            ),
            DataflowNode(
                2,
                0,
                "consumer",
                reads=tuple(consumer_reads),
                instruction_kind=InstructionKind.GENERIC,
            ),
        ),
        edges=(
            DataflowEdge(0, 1, buffer_id=0),
            DataflowEdge(1, 2, buffer_id=1),
        ),
        region_kinds=(RegionKind.PIPELINE,),
        hardware=hardware,
    )


def _load_mma_consumer_inputs(graph: DataflowGraph):
    groups = {0: 0, 1: 1, 2: 1}
    orders = {0: {0: {0: 0}, 1: {1: 0, 2: 1}}}
    versions = {buffer.buffer_id: 1 for buffer in graph.buffers}
    return groups, orders, versions


def test_memory_only_group_uses_exactly_one_warpgroup() -> None:
    graph = _two_role_graph()
    groups = {0: 0, 1: 1}
    orders = {0: {0: {0: 0}, 1: {1: 0}}}
    allocations = tuple(
        enumerate_warp_allocations(graph, groups, orders, {}, 128)
    )

    assert len(allocations) == 7
    assert register_receiver_groups(graph, groups) == frozenset({1})
    assert {allocation.groups[0].warp_count for allocation in allocations} == {
        HOPPER.warpgroup_warps
    }
    assert {
        allocation.groups[1].warp_count for allocation in allocations
    } == {4, 8, 12, 16, 20, 24, 28}
    assert all(allocation.setmaxnreg_enabled for allocation in allocations)
    assert all(
        allocation.register_is_increase == (False, True)
        for allocation in allocations
    )


def test_load_mma_consumer_jointly_solves_warps_and_register_limits() -> None:
    graph = _load_mma_consumer_graph()
    groups, orders, versions = _load_mma_consumer_inputs(graph)

    allocations = tuple(
        enumerate_warp_allocations(
            graph, groups, orders, versions, original_threads=128
        )
    )

    assert [
        (
            allocation.groups[1].warp_count,
            estimate_group_registers_per_thread(
                graph,
                1,
                groups,
                orders,
                versions,
                allocation.groups[1].warp_count,
            ),
            allocation.register_counts,
        )
        for allocation in allocations
    ] == [
        (4, 131, (24, 240)),
        (8, 66, (24, 240)),
        (12, 44, (24, 160)),
        (16, 33, (24, 120)),
        (20, 27, (24, 96)),
        (24, 22, (24, 80)),
        (28, 19, (24, 64)),
    ]


def test_private_register_floor_prunes_wide_consumer_groups() -> None:
    graph = _load_mma_consumer_graph(private_registers_per_thread=50)
    groups, orders, versions = _load_mma_consumer_inputs(graph)

    allocations = tuple(
        enumerate_warp_allocations(
            graph, groups, orders, versions, original_threads=128
        )
    )

    assert [
        allocation.groups[1].warp_count for allocation in allocations
    ] == [4, 8, 12, 16, 20, 24]
    assert [allocation.register_counts for allocation in allocations] == [
        (24, 240),
        (24, 240),
        (24, 160),
        (24, 120),
        (24, 96),
        (24, 80),
    ]
    assert all(
        all(count % 8 == 0 for count in allocation.register_counts)
        for allocation in allocations
    )
    assert all(
        sum(
            group.warp_count * HOPPER.warp_size * register_count
            for group, register_count in zip(
                allocation.groups, allocation.register_counts
            )
        )
        <= HOPPER.register_file_capacity
        for allocation in allocations
    )


def test_fixed_memory_donor_uses_its_aligned_register_requirement() -> None:
    graph = DataflowGraph(
        buffers=(_buffer(0, "producer_private", "local", 25 * 4),),
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
                instruction_kind=InstructionKind.WGMMA,
            ),
        ),
        edges=(),
        region_kinds=(RegionKind.PIPELINE,),
        hardware=HOPPER,
    )
    groups = {0: 0, 1: 1}
    orders = {0: {0: {0: 0}, 1: {1: 0}}}

    allocations = tuple(
        enumerate_warp_allocations(
            graph, groups, orders, {0: 1}, original_threads=128
        )
    )

    assert allocations
    assert all(allocation.groups[0].warp_count == 4 for allocation in allocations)
    assert all(allocation.register_counts[0] == 32 for allocation in allocations)
    assert all(
        allocation.register_is_increase == (False, True)
        for allocation in allocations
    )


def test_low_pressure_compute_group_becomes_fallback_donor() -> None:
    hardware = replace(
        HOPPER,
        register_receiver_kinds=frozenset(
            {InstructionKind.WGMMA, InstructionKind.GENERIC}
        ),
    )
    graph = DataflowGraph(
        buffers=(
            _buffer(
                0,
                "mma_accumulator",
                "local.fragment",
                100 * 128 * 4,
            ),
            _buffer(1, "generic_private", "local", 40 * 4),
        ),
        nodes=(
            DataflowNode(
                0,
                0,
                "mma",
                writes=(0,),
                instruction_kind=InstructionKind.WGMMA,
            ),
            DataflowNode(
                1,
                0,
                "consumer",
                writes=(1,),
                instruction_kind=InstructionKind.GENERIC,
            ),
        ),
        edges=(),
        region_kinds=(RegionKind.PIPELINE,),
        hardware=hardware,
    )
    groups = {0: 0, 1: 1}
    orders = {0: {0: {0: 0}, 1: {1: 0}}}
    versions = {0: 1, 1: 1}

    assert register_receiver_groups(graph, groups) == frozenset({0, 1})
    allocations = tuple(
        enumerate_warp_allocations(
            graph, groups, orders, versions, original_threads=128
        )
    )

    assert allocations
    assert all(
        allocation.register_is_increase == (True, False)
        for allocation in allocations
    )
    assert all(
        allocation.register_counts[1] == 40 for allocation in allocations
    )


def test_single_group_preserves_original_cta() -> None:
    graph = _two_role_graph()
    groups = {0: 0, 1: 0}
    orders = {0: {0: {0: 0, 1: 1}}}

    allocations = tuple(
        enumerate_warp_allocations(graph, groups, orders, {}, 128)
    )

    assert len(allocations) == 1
    assert allocations[0].effective_threads == 128
    assert allocations[0].register_counts is None


def test_single_group_prunes_threads_above_collective_tile_limit() -> None:
    graph = DataflowGraph(
        buffers=(
            _buffer(
                0,
                "accumulator",
                "local.fragment",
                64 * 64 * 4,
                shape=(64, 64),
            ),
        ),
        nodes=(
            DataflowNode(
                0,
                0,
                "mma",
                writes=(0,),
                instruction_kind=InstructionKind.WGMMA,
            ),
        ),
        edges=(),
        region_kinds=(RegionKind.PIPELINE,),
        hardware=HOPPER,
    )
    groups = {0: 0}
    orders = {0: {0: {0: 0}}}
    versions = {0: 1}

    assert tuple(
        enumerate_warp_allocations(
            graph, groups, orders, versions, original_threads=128
        )
    )
    assert not tuple(
        enumerate_warp_allocations(
            graph, groups, orders, versions, original_threads=256
        )
    )


def test_single_group_register_limits_are_pruned_during_warp_allocation() -> None:
    def make_graph(registers_per_thread: int) -> DataflowGraph:
        return DataflowGraph(
            buffers=(
                _buffer(
                    0,
                    "private",
                    "local",
                    registers_per_thread * 4,
                ),
            ),
            nodes=(DataflowNode(0, 0, "compute", writes=(0,)),),
            edges=(),
            region_kinds=(RegionKind.PIPELINE,),
            hardware=HOPPER,
        )

    groups = {0: 0}
    orders = {0: {0: {0: 0}}}

    assert not tuple(
        enumerate_warp_allocations(
            make_graph(256), groups, orders, {0: 1}, 128
        )
    )
    assert not tuple(
        enumerate_warp_allocations(
            make_graph(64), groups, orders, {0: 1}, 1024
        )
    )


def test_hardware_without_setmaxnreg_still_checks_register_budget() -> None:
    hardware = replace(
        HOPPER,
        setmaxnreg_required_for_specialization=False,
        register_file_capacity=16000,
    )
    graph = _load_mma_consumer_graph(hardware=hardware)
    groups, orders, versions = _load_mma_consumer_inputs(graph)

    assert not tuple(
        enumerate_warp_allocations(
            graph, groups, orders, versions, original_threads=128
        )
    )


def test_specialized_compute_width_is_independent_of_original_cta() -> None:
    graph = _two_role_graph()
    groups = {0: 0, 1: 1}
    orders = {0: {0: {0: 0}, 1: {1: 0}}}

    allocations = tuple(
        enumerate_warp_allocations(graph, groups, orders, {}, 256)
    )

    assert len(allocations) == 7
    assert {
        allocation.groups[1].warp_count for allocation in allocations
    } == {4, 8, 12, 16, 20, 24, 28}
    assert any(
        allocation.groups[1].warp_count * HOPPER.warp_size < 256
        for allocation in allocations
    )


def test_collective_layout_limit_is_independent_of_original_cta() -> None:
    graph = _load_mma_consumer_graph(hardware=HOPPER)
    groups, orders, versions = _load_mma_consumer_inputs(graph)

    allocations = tuple(
        enumerate_warp_allocations(
            graph,
            groups,
            orders,
            versions,
            original_threads=128,
        )
    )

    collective_warps = {
        allocation.groups[1].warp_count for allocation in allocations
    }
    assert max(collective_warps) == 128 // 64 * HOPPER.warpgroup_warps
    assert all(
        warp_count % HOPPER.warpgroup_warps == 0
        for warp_count in collective_warps
    )
    assert any(
        allocation.groups[1].warp_count * HOPPER.warp_size > 128
        for allocation in allocations
    )


def test_collective_layout_limit_tracks_block_m() -> None:
    for block_m, expected_warps in ((64, 4), (128, 8), (192, 12)):
        graph = _load_mma_consumer_graph(hardware=HOPPER)
        accumulator = graph.buffers[1]
        graph = DataflowGraph(
            buffers=(
                graph.buffers[0],
                BufferDescriptor(
                    accumulator.buffer_id,
                    accumulator.name,
                    accumulator.scope,
                    accumulator.nbytes,
                    SimpleNamespace(shape=(block_m, 131)),
                ),
            ),
            nodes=graph.nodes,
            edges=graph.edges,
            region_kinds=graph.region_kinds,
            hardware=graph.hardware,
        )
        groups, orders, versions = _load_mma_consumer_inputs(graph)

        allocations = tuple(
            enumerate_warp_allocations(
                graph, groups, orders, versions, original_threads=128
            )
        )

        assert max(
            allocation.groups[1].warp_count for allocation in allocations
        ) == expected_warps


def enumerate_fa3_warp_allocations():
    from history.unionwsp.test.test_order import enumerate_fa3_ordered_schedules

    graph, stage_assignments, schedules = enumerate_fa3_ordered_schedules()
    results = tuple(
        (
            stage_index,
            group_index,
            stages_by_region,
            groups,
            orders,
            analyze_buffer_versions(
                graph, stages_by_region, groups, orders
            ),
        )
        for (
            stage_index,
            group_index,
            stages_by_region,
            groups,
            orders,
        ) in schedules
    )
    realized = tuple(
        (*result, tuple(enumerate_warp_allocations(
            graph, result[3], result[4], result[5], 128
        )))
        for result in results
    )
    return graph, stage_assignments, realized


def print_fa3_warp_allocations(graph, stage_assignments, results) -> None:
    print(
        f"\nFA3 warp allocations: stage_assignments={len(stage_assignments)}, "
        f"logical_schedules={len(results)}, "
        f"physical_schedules={sum(len(result[-1]) for result in results)}"
    )
    for result_index, result in enumerate(results, start=1):
        stage_index, group_index, _, groups, _, _, allocations = result
        receivers = sorted(register_receiver_groups(graph, groups))
        print(
            f"result {result_index:03d}: stage={stage_index:03d}, "
            f"group={group_index:02d}, receivers={receivers}, "
            f"allocations={len(allocations)}"
        )
        for allocation in allocations:
            warp_counts = tuple(group.warp_count for group in allocation.groups)
            print(
                f"  warps={warp_counts}, threads={allocation.effective_threads}, "
                f"setmaxnreg={allocation.register_counts}"
            )


def test_every_realized_fa3_warp_allocation_is_valid() -> None:
    graph, stage_assignments, results = enumerate_fa3_warp_allocations()

    assert stage_assignments
    assert results
    distribution = Counter()
    pruned = 0
    for _, _, _, groups, orders, versions, allocations in results:
        if not allocations:
            pruned += 1
            continue
        receivers = register_receiver_groups(graph, groups)
        distribution[(len(receivers), len(allocations))] += 1
        for allocation in allocations:
            validate_warp_allocation(
                graph, groups, orders, versions, 128, allocation
            )
            assert allocation.setmaxnreg_enabled
            for group in allocation.groups:
                assert group.warp_count % HOPPER.warpgroup_warps == 0
                if group.group_id not in receivers:
                    assert group.warp_count == HOPPER.warpgroup_warps

    assert distribution
    assert pruned
    print(f"FA3 allocation distribution: {dict(distribution)}")


if __name__ == "__main__":
    graph, stage_assignments, results = enumerate_fa3_warp_allocations()
    print_fa3_warp_allocations(graph, stage_assignments, results)
