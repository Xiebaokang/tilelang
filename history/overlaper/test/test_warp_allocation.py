"""Physical warp-allocation tests, including FA3."""

from history.overlaper.analysis.physical import (
    estimate_group_registers_per_thread,
    enumerate_warp_allocations,
    register_receiver_groups,
    warp_requirements_are_feasible,
)
from history.overlaper.analysis.schedule import (
    analyze_buffer_versions,
    build_program_orders,
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


def _two_group_graph() -> DataflowGraph:
    return DataflowGraph(
        buffers=(),
        nodes=(
            DataflowNode(0, 0, "load", TMA),
            DataflowNode(1, 0, "mma", WGMMA),
        ),
        edges=(),
        region_kinds=(RegionKind.PIPELINE,),
        hardware=HOPPER,
        kernel_threads=128,
    )


def test_single_group_preserves_original_cta() -> None:
    graph = _two_group_graph()
    groups = {0: 0, 1: 0}
    orders = {0: {0: {0: 0, 1: 1}}}

    allocations = tuple(
        enumerate_warp_allocations(graph, groups, orders, versions={})
    )

    assert len(allocations) == 1
    assert allocations[0].effective_threads == 128
    assert allocations[0].total_warps == 4
    assert not allocations[0].setmaxnreg_enabled


def test_memory_group_is_fixed_and_wgmma_group_receives_registers() -> None:
    graph = _two_group_graph()
    groups = {0: 0, 1: 1}
    orders = {0: {0: {0: 0}, 1: {1: 0}}}

    allocations = tuple(
        enumerate_warp_allocations(graph, groups, orders, versions={})
    )

    assert register_receiver_groups(graph, groups) == frozenset({1})
    assert len(allocations) == 7
    assert {item.groups[0].warp_count for item in allocations} == {4}
    assert {item.groups[1].warp_count for item in allocations} == {
        4,
        8,
        12,
        16,
        20,
        24,
        28,
    }
    assert all(item.register_is_increase == (False, True) for item in allocations)


def test_fragment_pressure_jointly_changes_warps_and_registers() -> None:
    graph = DataflowGraph(
        buffers=(
            BufferDescriptor(0, "shared", "shared", 128, object()),
            BufferDescriptor(
                1,
                "accumulator",
                "local.fragment",
                131 * 128 * 4,
                object(),
            ),
        ),
        nodes=(
            DataflowNode(0, 0, "load", TMA, writes=(0,)),
            DataflowNode(1, 0, "mma", WGMMA, reads=(0,), writes=(1,)),
            DataflowNode(2, 0, "consumer", GENERIC, reads=(1,)),
        ),
        edges=(
            DataflowEdge(0, 1, buffer_id=0),
            DataflowEdge(1, 2, buffer_id=1),
        ),
        region_kinds=(RegionKind.PIPELINE,),
        hardware=HOPPER,
        kernel_threads=128,
    )
    groups = {0: 0, 1: 1, 2: 1}
    orders = {0: {0: {0: 0}, 1: {1: 0, 2: 1}}}
    versions = {0: 1, 1: 1}

    allocations = tuple(
        enumerate_warp_allocations(graph, groups, orders, versions)
    )

    assert [
        (
            item.groups[1].warp_count,
            estimate_group_registers_per_thread(
                graph,
                1,
                groups,
                orders,
                versions,
                item.groups[1].warp_count,
            ),
            item.register_counts,
        )
        for item in allocations
    ] == [
        (4, 131, (24, 240)),
        (8, 66, (24, 232)),
        (12, 44, (24, 152)),
        (16, 33, (24, 112)),
        (20, 27, (24, 88)),
        (24, 22, (24, 72)),
        (28, 19, (24, 64)),
    ]

    capacity = HOPPER.device_resource.register_file_capacity
    warp_size = HOPPER.device_resource.warp_size
    assert all(
        sum(
            group.warp_count * warp_size * allocation.register_counts[group.group_id]
            for group in allocation.groups
        )
        < capacity
        for allocation in allocations
    )


def test_fa3_warp_allocation() -> None:
    graph = extract_dataflow_graph(make_fa3_prim_func(), hardware=HOPPER)
    stages = {1: find_fa3_stage_assignment(graph)}
    groups = tuple(enumerate_group_assignments(graph, 2))[1]
    orders = build_program_orders(graph, stages, groups)
    versions = analyze_buffer_versions(graph, stages, groups, orders)

    allocations = tuple(
        enumerate_warp_allocations(graph, groups, orders, versions)
    )

    assert allocations
    assert register_receiver_groups(graph, groups) == frozenset({0})
    assert all(item.total_warps <= 32 for item in allocations)
    assert all(item.register_is_increase == (True, False) for item in allocations)

    print(f"\nFA3 warp allocations: {len(allocations)}")
    for item in allocations:
        print(
            "  warps="
            f"{tuple(group.warp_count for group in item.groups)}, "
            f"threads={item.effective_threads}, registers={item.register_counts}, "
            f"increase={item.register_is_increase}"
        )


def test_fa3_block_m192_requires_twelve_compute_warps() -> None:
    graph = extract_dataflow_graph(
        make_fa3_prim_func(block_m=192, block_n=64, threads=384),
        hardware=HOPPER,
    )
    stages = {1: find_fa3_stage_assignment(graph)}
    groups = tuple(enumerate_group_assignments(graph, 2))[1]
    orders = build_program_orders(graph, stages, groups)
    versions = analyze_buffer_versions(graph, stages, groups, orders)

    allocations = tuple(
        enumerate_warp_allocations(graph, groups, orders, versions)
    )

    compute_group = next(iter(register_receiver_groups(graph, groups)))
    assert allocations
    assert {
        item.groups[compute_group].warp_count for item in allocations
    } == {12}


def test_fa3_block_m192_three_groups_leave_register_headroom() -> None:
    graph = extract_dataflow_graph(
        make_fa3_prim_func(block_m=192, block_n=64, threads=384),
        hardware=HOPPER,
    )
    groups = {node.node_id: 0 for node in graph.nodes}
    groups[
        next(
            node.node_id
            for node in graph.nodes
            if node.name == "copy_V_to_V_shared"
        )
    ] = 1
    groups[
        next(
            node.node_id
            for node in graph.nodes
            if node.name == "copy_O_shared_to_Output"
        )
    ] = 2
    stages = {
        1: {node.node_id: 0 for node in graph.nodes_for_region(1)}
    }
    orders = build_program_orders(graph, stages, groups)
    versions = analyze_buffer_versions(graph, stages, groups, orders)

    allocations = tuple(
        enumerate_warp_allocations(graph, groups, orders, versions)
    )

    assert len(allocations) == 1
    allocation = allocations[0]
    assert tuple(group.warp_count for group in allocation.groups) == (
        12,
        4,
        4,
    )
    assert allocation.register_counts == (144, 24, 24)
    assert (
        sum(
            group.warp_count
            * HOPPER.device_resource.warp_size
            * allocation.register_counts[group.group_id]
            for group in allocation.groups
        )
        < HOPPER.device_resource.register_file_capacity
    )


def test_conflicting_wgmma_rows_are_infeasible() -> None:
    class _Buffer:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape

    graph = DataflowGraph(
        buffers=(
            BufferDescriptor(
                0, "acc64", "local.fragment", 64 * 64 * 4, _Buffer((64, 64))
            ),
            BufferDescriptor(
                1,
                "acc128",
                "local.fragment",
                128 * 128 * 4,
                _Buffer((128, 128)),
            ),
        ),
        nodes=(
            DataflowNode(0, 0, "mma64", WGMMA, writes=(0,)),
            DataflowNode(1, 0, "mma128", WGMMA, writes=(1,)),
        ),
        edges=(),
        region_kinds=(RegionKind.PIPELINE,),
        hardware=HOPPER,
        kernel_threads=128,
    )

    assert warp_requirements_are_feasible(graph, {0: 0, 1: 0}) is False
    assert warp_requirements_are_feasible(graph, {0: 0, 1: 1}) is True
