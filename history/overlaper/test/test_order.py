"""Priority-order regression tests, including the full FA3 combination."""

from history.overlaper.analysis.schedule import (
    MAX_NUM_GROUPS,
    build_program_orders,
    effective_stage_distance,
    enumerate_group_assignments,
    enumerate_stage_assignments,
)
from history.overlaper.headware import HOPPER
from history.overlaper.headware.hopper import FUNCTION, GENERIC, RRCP, TMA, WGMMA
from history.overlaper.parse import (
    BufferDescriptor,
    DataflowEdge,
    DataflowGraph,
    DataflowNode,
    RegionKind,
)
from history.overlaper.parse import extract_dataflow_graph
from history.overlaper.test.test_extractor import make_fa3_prim_func
from history.overlaper.test.test_stage import FA3_QK_PV_STAGE_ASSIGNMENT, find_fa3_stage_assignment


def test_parallel_ready_nodes_keep_only_priority_order() -> None:
    graph = DataflowGraph(
        buffers=(),
        nodes=(
            DataflowNode(0, 0, "generic", GENERIC),
            DataflowNode(1, 0, "tma", TMA),
            DataflowNode(2, 0, "wgmma", WGMMA),
        ),
        edges=(),
        region_kinds=(RegionKind.PIPELINE,),
        hardware=HOPPER,
    )

    orders = build_program_orders(
        graph,
        stages_by_region={0: {0: 0, 1: 0, 2: 0}},
        groups={0: 0, 1: 1, 2: 0},
    )

    # The one retained global choice is TMA, WGMMA, GENERIC: reachable
    # priority still prefers TMA, then WGMMA, and all three share stage 0.
    assert orders == {0: {0: {2: 0, 0: 1}, 1: {1: 0}}}


def test_serial_ready_nodes_also_use_priority_without_breaking_dependencies() -> None:
    graph = DataflowGraph(
        buffers=(BufferDescriptor(0, "value", "shared", 4, object()),),
        nodes=(
            DataflowNode(0, 0, "producer", GENERIC, writes=(0,)),
            DataflowNode(1, 0, "independent_wgmma", WGMMA),
            DataflowNode(2, 0, "dependent_tma", TMA, reads=(0,)),
        ),
        edges=(DataflowEdge(0, 2, buffer_id=0),),
        region_kinds=(RegionKind.SERIAL,),
        hardware=HOPPER,
    )

    orders = build_program_orders(
        graph,
        stages_by_region={},
        groups={0: 0, 1: 0, 2: 0},
    )

    # Serial regions stay priority-first. WGMMA is ready and wins first. TMA
    # has the highest priority but remains blocked until its GENERIC producer
    # has executed.
    assert orders == {0: {0: {1: 0, 0: 1, 2: 2}}}


def test_ready_lower_stage_wgmma_is_issued_before_softmax() -> None:
    """QK (stage 0) then PV (stage 1) then leftover softmax (stage 0)."""

    graph = DataflowGraph(
        buffers=(
            BufferDescriptor(0, "acc_s", "local.fragment", 8, object()),
            BufferDescriptor(1, "acc_o", "local.fragment", 8, object()),
        ),
        nodes=(
            DataflowNode(0, 0, "init_acc_s", GENERIC, writes=(0,)),
            DataflowNode(1, 0, "qk", WGMMA, reads=(0,), writes=(0,)),
            DataflowNode(2, 0, "softmax", FUNCTION, reads=(0,), writes=(0, 1)),
            DataflowNode(3, 0, "pv", WGMMA, reads=(1,), writes=(1,)),
        ),
        edges=(
            DataflowEdge(0, 1, buffer_id=0),
            DataflowEdge(1, 2, buffer_id=0),
            DataflowEdge(2, 3, buffer_id=1),
            DataflowEdge(3, 2, iteration_distance=1, buffer_id=1),
        ),
        region_kinds=(RegionKind.PIPELINE,),
        hardware=HOPPER,
    )
    orders = build_program_orders(
        graph,
        stages_by_region={0: {0: 0, 1: 0, 2: 0, 3: 1}},
        groups={0: 0, 1: 0, 2: 0, 3: 0},
    )

    assert orders == {0: {0: {0: 0, 1: 1, 3: 2, 2: 3}}}


def _group_pipeline_order(orders, region_id, group_id) -> tuple[int, ...]:
    local_order = orders[region_id][group_id]
    return tuple(sorted(local_order, key=local_order.__getitem__))


def test_fa3_qk_pv_stages_issue_qk_then_pv_then_softmax() -> None:
    graph = extract_dataflow_graph(make_fa3_prim_func(), hardware=HOPPER)
    stages = {1: find_fa3_stage_assignment(graph, FA3_QK_PV_STAGE_ASSIGNMENT)}
    names = {node.name: node.node_id for node in graph.nodes}
    qk = names["gemm_acc_s"]
    pv = names["gemm_acc_o"]
    load_q = names["copy_Q_to_Q_shared"]
    load_k = names["copy_K_to_K_shared"]
    load_v = names["copy_V_to_V_shared"]
    groups = next(
        assignment
        for assignment in enumerate_group_assignments(graph, 2)
        if assignment[load_k] == assignment[load_v] == assignment[load_q]
        and assignment[qk] == assignment[pv]
        and assignment[load_k] != assignment[qk]
    )
    orders = build_program_orders(graph, stages, groups)

    compute_group = groups[qk]
    load_group = groups[load_k]
    compute = _group_pipeline_order(orders, 1, compute_group)
    load = _group_pipeline_order(orders, 1, load_group)
    softmax = tuple(
        node_id
        for node_id in compute
        if graph.node_for_id(node_id).instruction in {GENERIC, FUNCTION, RRCP}
        and node_id != names["parallel_acc_s"]
    )

    assert load == (load_k, load_v)
    assert compute.index(names["parallel_acc_s"]) < compute.index(qk)
    assert compute.index(qk) < compute.index(pv)
    assert compute.index(pv) < compute.index(softmax[0])
    assert all(compute.index(pv) < compute.index(node_id) for node_id in softmax)


def _combined_region_order_is_acyclic(
    graph,
    region_id,
    stages_by_region,
    group_orders,
) -> bool:
    nodes = graph.nodes_for_region(region_id)
    node_ids = {node.node_id for node in nodes}
    predecessors = {node_id: set() for node_id in node_ids}
    for edge in graph.edges:
        if edge.producer_id not in node_ids or edge.consumer_id not in node_ids:
            continue
        if graph.region_kinds[region_id] == RegionKind.PIPELINE:
            distance = effective_stage_distance(
                edge, stages_by_region[region_id]
            )
            if distance != 0:
                continue
        if edge.producer_id != edge.consumer_id:
            predecessors[edge.consumer_id].add(edge.producer_id)

    for local_order in group_orders.values():
        ordered = sorted(local_order, key=local_order.__getitem__)
        for earlier, later in zip(ordered, ordered[1:]):
            predecessors[later].add(earlier)

    pending = dict(predecessors)
    while pending:
        ready = [
            node_id for node_id, required in pending.items() if not required
        ]
        if not ready:
            return False
        for node_id in ready:
            del pending[node_id]
        for required in pending.values():
            required.difference_update(ready)
    return True


def _format_order(graph, stages, groups, orders) -> str:
    lines = []
    for region_id, group_orders in orders.items():
        region_stages = stages.get(region_id, {})
        for group_id, local_order in group_orders.items():
            operations = ", ".join(
                f"{position}:{node_id}:{graph.node_for_id(node_id).name}"
                f"[{graph.node_for_id(node_id).instruction.name},"
                f"priority={graph.issue_priority(node_id)},"
                f"stage={region_stages.get(node_id, '-')}]"
                for node_id, position in sorted(
                    local_order.items(), key=lambda item: item[1]
                )
            )
            lines.append(f"  region {region_id} group {group_id}=[{operations}]")
    return "\n".join(lines)


def test_fa3_stage_group_then_priority_order() -> None:
    graph = extract_dataflow_graph(make_fa3_prim_func(), hardware=HOPPER)
    stage_assignments = tuple(enumerate_stage_assignments(graph, 1))
    groups_by_count = {
        num_groups: tuple(enumerate_group_assignments(graph, num_groups))
        for num_groups in range(1, MAX_NUM_GROUPS + 1)
    }

    schedule_count = 0
    samples = {}
    for stages in stage_assignments:
        stages_by_region = {1: stages}
        for num_groups, group_assignments in groups_by_count.items():
            for groups in group_assignments:
                orders = build_program_orders(graph, stages_by_region, groups)
                schedule_count += 1
                samples.setdefault(
                    num_groups,
                    (stages_by_region, groups, orders),
                )

                assert set(orders) == set(range(len(graph.region_kinds)))
                for region_id, group_orders in orders.items():
                    assert set(group_orders) == set(range(num_groups))
                    region_node_ids = {
                        node.node_id
                        for node in graph.nodes_for_region(region_id)
                    }
                    covered = set()
                    for group_id, local_order in group_orders.items():
                        expected = {
                            node_id
                            for node_id in region_node_ids
                            if groups[node_id] == group_id
                        }
                        assert set(local_order) == expected
                        assert set(local_order.values()) == set(
                            range(len(expected))
                        )
                        covered.update(local_order)
                    assert covered == region_node_ids
                    assert _combined_region_order_is_acyclic(
                        graph,
                        region_id,
                        stages_by_region,
                        group_orders,
                    )

    expected_group_assignments = sum(map(len, groups_by_count.values()))
    assert len(stage_assignments) == 447
    assert expected_group_assignments == 41
    # Parallel ready-node permutations are pruned to one reachable-priority,
    # lower-stage order for every stage/group combination.
    assert schedule_count == len(stage_assignments) * expected_group_assignments
    assert schedule_count == 18327

    print(
        f"\nFA3 ordered schedules: stages={len(stage_assignments)}, "
        f"groups={expected_group_assignments}, schedules={schedule_count}"
    )
    for num_groups, (stages, groups, orders) in samples.items():
        print(f"\npriority-pruned sample for num_groups={num_groups}")
        print(_format_order(graph, stages, groups, orders))
