"""Tests for L1 structure enumeration."""

from OverlapPlaner.arch import FAKE, HOPPER, is_group_opportunity
from OverlapPlaner.facts import OpKind, RegionKind, extract_fact_graph
from OverlapPlaner.structure import (
    SearchBudget,
    enumerate_group_assignments,
    enumerate_structures,
    group_local_stages,
)
from OverlapPlaner.structure.order import build_program_orders
from OverlapPlaner.structure.enumerate import _diverse_stage_group_pairs, _group_maps
from OverlapPlaner.structure.group import connected_components
from OverlapPlaner.structure.stage import enumerate_program_stages
from OverlapPlaner.structure.version import analyze_buffer_versions
from OverlapPlaner.tune.operators.fa3 import build as build_fa3
from OverlapPlaner.tune.operators.gemm import build as build_gemm
from OverlapPlaner.tune.operators.mamba_chunk_scan import build as build_mamba_scan


def _gemm_graph():
    return extract_fact_graph(
        build_gemm(
            {"gemm_m": 256, "gemm_n": 256, "gemm_k": 256},
            {"block_m": 128, "block_n": 128, "block_k": 64},
        ).prim_func
    )


def _fa3_graph():
    return extract_fact_graph(
        build_fa3(
            {
                "fa3_batch": 1,
                "fa3_heads": 1,
                "fa3_seq_q": 256,
                "fa3_seq_kv": 256,
                "fa3_dim": 128,
                "fa3_causal": False,
            },
            {"block_m": 128, "block_n": 128},
        ).prim_func
    )


def _mamba_scan_graph():
    return extract_fact_graph(
        build_mamba_scan(
            {
                "mamba_scan_batch": 1,
                "mamba_scan_heads": 4,
                "mamba_scan_groups": 1,
                "mamba_scan_seq": 256,
                "mamba_scan_chunk": 256,
                "mamba_scan_dim": 64,
                "mamba_scan_dstate": 128,
            },
            {
                "block_m": 64,
                "block_n": 32,
                "block_k": 64,
                "block_dstate": 128,
            },
        ).prim_func
    )


def test_mamba_global_shared_priority_matches_target_order() -> None:
    from OverlapPlaner.contract import layout_reduced_prim_func

    options = {
        "mamba_scan_batch": 8, "mamba_scan_heads": 80,
        "mamba_scan_groups": 1, "mamba_scan_seq": 4096,
        "mamba_scan_chunk": 256, "mamba_scan_dim": 64,
        "mamba_scan_dstate": 128,
    }
    tile = {"block_m": 64, "block_n": 64, "block_k": 64, "block_dstate": 128}
    classified = HOPPER.classify(
        extract_fact_graph(layout_reduced_prim_func(build_mamba_scan(options, tile).prim_func))
    )
    producer_group = {0, 4, 5, 8, 10, 13, 17, 20, 24}
    groups = {
        node.node_id: int(node.node_id in producer_group)
        for node in classified.graph.nodes
    }
    stages = {1: dict(zip(range(8, 19), (0, 1, 0, 1, 2, 0, 1, 2, 2, 1, 2)))}
    orders = build_program_orders(classified, stages, groups)
    assert tuple(sorted(orders[0][1], key=orders[0][1].get)) == (0, 4, 5)
    assert tuple(sorted(orders[1][1], key=orders[1][1].get)) == (8, 10, 13, 17)
    assert tuple(sorted(orders[1][0], key=orders[1][0].get)) == (
        12, 15, 16, 18, 9, 11, 14
    )


def _structures(classified, arch=None, **budget):
    return list(
        enumerate_structures(classified, SearchBudget(**budget), arch=arch)
    )


def _buffer_named(graph, name: str):
    return next(
        buffer
        for buffer in graph.buffers
        if buffer.name == name and buffer.scope == "local.fragment"
    )


def _splits_buffer(classified, groups, buffer_id: int) -> bool:
    return any(
        groups[edge.producer_id] != groups[edge.consumer_id]
        and edge.buffer_id == buffer_id
        and is_group_opportunity(classified, edge)
        for edge in classified.graph.edges
    )


def test_gemm_enumerates_one_group_all_stage_zero() -> None:
    classified = HOPPER.classify(_gemm_graph())
    structures = _structures(classified, max_structures=32)
    assert structures
    assert any(
        item.num_groups == 1
        and all(
            stage == 0
            for stages in item.stages_by_region.values()
            for stage in stages.values()
        )
        for item in structures
    )


def test_dynamic_trip_count_enumerates_three_stages() -> None:
    # This Mamba loop has a one-iteration path, so a min-trip >= stage-depth
    # rule would incorrectly remove the three-stage assignments.
    classified = HOPPER.classify(_mamba_scan_graph())
    stages = list(
        enumerate_program_stages(
            classified,
            SearchBudget(max_stages=3, stage_beam=64),
        )
    )
    assert stages
    assert any(
        max(region.values(), default=0) == 2
        for program in stages
        for region in program.values()
    )


def test_wide_stage_beam_retains_mamba_three_stage_regression() -> None:
    classified = HOPPER.classify(_mamba_scan_graph())
    expected = {
        8: 0, 9: 1, 10: 0, 11: 1, 12: 2, 13: 0,
        14: 1, 15: 2, 16: 2, 17: 1, 18: 2,
    }
    assert expected in [
        program[1]
        for program in enumerate_program_stages(
            classified,
            SearchBudget(
                max_stages=3,
                stage_beam=1024,
            ),
        )
    ]


def test_mamba_serial_shared_handoffs_can_split_groups() -> None:
    classified = HOPPER.classify(_mamba_scan_graph())
    graph = classified.graph
    components = {
        node_id: component_id
        for component_id, component in enumerate(connected_components(classified))
        for node_id in component
    }
    for producer_id, consumer_id in ((0, 1), (4, 6), (5, 6), (20, 21)):
        edges = [
            edge for edge in graph.edges
            if edge.producer_id == producer_id
            and edge.consumer_id == consumer_id
            and graph.buffer_for_id(edge.buffer_id).scope.startswith("shared")
        ]
        assert edges
        assert all(is_group_opportunity(classified, edge) for edge in edges)
        assert components[producer_id] != components[consumer_id]
    # This partition is a regression case from the earlier Mamba search: its
    # prologue and epilogue shared handoffs must remain representable.
    producer_group = {0, 4, 5, 8, 10, 13, 17, 20, 24}
    assert all(
        is_group_opportunity(classified, edge)
        for edge in graph.edges
        if (edge.producer_id in producer_group)
        != (edge.consumer_id in producer_group)
    )


def test_group_stage_pairs_do_not_exhaust_first_group_before_second() -> None:
    groups = [{0: 0, 1: 1}, {0: 1, 1: 0}]
    stages = [{0: {0: 0, 1: 1}}, {0: {0: 1, 1: 0}}]
    pairs = list(_diverse_stage_group_pairs(groups, stages))
    assert pairs.index((groups[1], stages[0])) < pairs.index(
        (groups[0], stages[1])
    )


def test_group_stage_pairs_stream_same_order_as_materialized_maps() -> None:
    classified = HOPPER.classify(_gemm_graph())
    budget = SearchBudget(max_groups=2, max_stages=2, stage_beam=8)
    stages = list(enumerate_program_stages(classified, budget))
    factories = _group_maps(classified, budget)
    materialized = [groups for factory in factories.values() for groups in factory()]
    assert list(_diverse_stage_group_pairs(factories, stages)) == list(
        _diverse_stage_group_pairs(materialized, stages)
    )


def test_gemm_copy_may_split_stage_and_group() -> None:
    classified = HOPPER.classify(_gemm_graph())
    graph = classified.graph
    gemm = next(node for node in graph.nodes if node.kind == OpKind.GEMM)
    copy = next(
        node
        for node in graph.nodes
        if node.kind == OpKind.COPY and node.region_id == gemm.region_id
    )
    structures = _structures(classified, max_structures=64)
    pipeline = gemm.region_id
    assert any(
        item.stages_by_region[pipeline][copy.node_id]
        != item.stages_by_region[pipeline][gemm.node_id]
        for item in structures
    )
    assert any(
        item.groups[copy.node_id] != item.groups[gemm.node_id] for item in structures
    )


def test_non_opportunity_edges_cannot_split_group() -> None:
    classified = HOPPER.classify(_gemm_graph())
    graph = classified.graph
    structures = _structures(classified, max_structures=64)
    assert structures
    for item in structures:
        for edge in graph.edges:
            if not is_group_opportunity(classified, edge):
                assert item.groups[edge.producer_id] == item.groups[edge.consumer_id]


def test_fa3_engine_crossing_fragment_may_split_group() -> None:
    classified = HOPPER.classify(_fa3_graph())
    graph = classified.graph
    qk = next(
        node
        for node in graph.nodes
        if node.kind == OpKind.GEMM and node.gemm is not None and node.gemm.transpose_b
    )
    softmax = next(
        node
        for node in graph.nodes
        if classified.traits_for(node.node_id).engine == "sfu"
    )
    assignments = list(enumerate_group_assignments(classified, 2))
    assert assignments
    assert any(
        item[qk.node_id] != item[softmax.node_id] for item in assignments
    )
    for groups in assignments:
        for edge in graph.edges:
            if not is_group_opportunity(classified, edge):
                assert groups[edge.producer_id] == groups[edge.consumer_id]


def test_fragment_register_copy_stays_with_its_consumer() -> None:
    classified = HOPPER.classify(_fa3_graph())
    graph = classified.graph
    register_copy = next(
        node
        for node in graph.nodes
        if classified.traits_for(node.node_id).engine == "register_copy"
        and any(
            graph.buffer_for_id(buffer_id).name == "acc_s_cast"
            for buffer_id in node.writes
        )
    )
    producer_edges = [
        edge for edge in graph.edges if edge.consumer_id == register_copy.node_id
        and graph.buffer_for_id(edge.buffer_id).scope == "local.fragment"
        and classified.traits_for(edge.producer_id).engine == "sfu"
    ]
    assert producer_edges
    assert any(is_group_opportunity(classified, edge) for edge in producer_edges)
    consumer_edges = [
        edge for edge in graph.edges if edge.producer_id == register_copy.node_id
    ]
    assert consumer_edges
    for groups in enumerate_group_assignments(classified, 2):
        for edge in consumer_edges:
            assert groups[edge.producer_id] == groups[edge.consumer_id]


def test_structure_enumeration_has_no_architecture_score() -> None:
    assert not hasattr(HOPPER, "score")
    assert not hasattr(HOPPER, "score_groups")
    assert not hasattr(FAKE, "score")


def test_structure_cutoff_preserves_group_stage_diversity() -> None:
    classified = HOPPER.classify(_fa3_graph())
    structures = _structures(
        classified, arch=HOPPER, max_groups=2, max_structures=32
    )
    assert structures
    buckets = {
        (
            item.num_groups,
            max(
                (
                    max(stages.values(), default=0) + 1
                    for stages in item.stages_by_region.values()
                ),
                default=1,
            ),
        )
        for item in structures
    }
    assert (1, 1) in buckets
    assert any(groups > 1 for groups, _ in buckets)
    assert any(stages > 1 for _, stages in buckets)


def test_same_engine_compute_cannot_split_stage() -> None:
    classified = HOPPER.classify(_fa3_graph())
    graph = classified.graph
    gemms = [node for node in graph.nodes if node.kind == OpKind.GEMM]
    assert len(gemms) == 2
    direct = [
        edge
        for edge in graph.edges
        if {edge.producer_id, edge.consumer_id} == {gemms[0].node_id, gemms[1].node_id}
        and not edge.is_loop_carried
        and graph.node_for_id(edge.producer_id).region_id
        == graph.node_for_id(edge.consumer_id).region_id
        and graph.region_kinds[graph.node_for_id(edge.producer_id).region_id]
        == RegionKind.PIPELINE
    ]
    structures = _structures(classified, max_structures=32)
    assert structures
    for item in structures:
        for edge in graph.edges:
            producer = graph.node_for_id(edge.producer_id)
            consumer = graph.node_for_id(edge.consumer_id)
            if producer.region_id != consumer.region_id:
                continue
            if graph.region_kinds[producer.region_id] != RegionKind.PIPELINE:
                continue
            if graph.is_initializer(edge.producer_id):
                stages = item.stages_by_region[producer.region_id]
                assert stages[edge.producer_id] == stages[edge.consumer_id]
        for edge in direct:
            region_id = graph.node_for_id(edge.producer_id).region_id
            stages = item.stages_by_region[region_id]
            assert stages[edge.producer_id] == stages[edge.consumer_id]


def test_fa3_can_use_multiple_groups() -> None:
    classified = HOPPER.classify(_fa3_graph())
    structures = _structures(classified, max_structures=64)
    assert structures
    assert any(item.num_groups >= 2 for item in structures)


def test_fa3_epilogue_shared_copy_can_split_group() -> None:
    classified = HOPPER.classify(_fa3_graph())
    graph = classified.graph
    serial_shared = [
        item
        for item in graph.edges
        if item.producer_id != item.consumer_id
        and graph.region_kinds[graph.node_for_id(item.producer_id).region_id]
        != RegionKind.PIPELINE
        and graph.region_kinds[graph.node_for_id(item.consumer_id).region_id]
        != RegionKind.PIPELINE
        and graph.buffer_for_id(item.buffer_id).scope.startswith("shared")
    ]
    assert serial_shared
    assert any(
        structure.groups[edge.producer_id]
        != structure.groups[edge.consumer_id]
        for structure in _structures(classified, max_structures=64)
        for edge in serial_shared
    )


def test_structure_has_no_isa_or_physical_resources() -> None:
    classified = HOPPER.classify(_gemm_graph())
    item = _structures(classified, max_structures=1)[0]
    assert not hasattr(item, "warp_allocation")
    assert not hasattr(item, "shared_memory")
    dumped = repr(item)
    assert "wgmma" not in dumped
    assert "tma" not in dumped
    assert "warp" not in dumped


def test_group_local_idle_stages_are_equivalent() -> None:
    classified = HOPPER.classify(_gemm_graph())
    item = _structures(classified, max_structures=1)[0]
    shifted = {
        region_id: {node_id: stage + 1 for node_id, stage in stages.items()}
        for region_id, stages in item.stages_by_region.items()
    }
    assert group_local_stages(
        classified.graph, item.stages_by_region, item.groups
    ) == group_local_stages(classified.graph, shifted, item.groups)


def test_one_group_async_copy_still_gets_a_sync_edge() -> None:
    classified = HOPPER.classify(_gemm_graph())
    structures = _structures(classified, max_structures=16)
    one_group = next(item for item in structures if item.num_groups == 1)
    assert any(
        edge.producer_group == edge.consumer_group for edge in one_group.sync_edges
    )


def test_single_stage_keeps_fragment_version_one() -> None:
    classified = HOPPER.classify(_fa3_graph())
    structures = _structures(
        classified, arch=HOPPER, max_groups=1, max_stages=1, max_structures=8
    )
    assert structures
    fragments = [
        buffer.buffer_id
        for buffer in classified.graph.buffers
        if buffer.scope == "local.fragment"
    ]
    assert fragments
    for item in structures:
        for buffer_id in fragments:
            assert item.buffer_versions[buffer_id] == 1


def test_same_group_fragment_stage_cut_can_need_two_versions() -> None:
    classified = HOPPER.classify(_fa3_graph())
    graph = classified.graph
    acc_s = _buffer_named(graph, "acc_s")
    qk = next(
        node
        for node in graph.nodes
        if node.kind == OpKind.GEMM and node.gemm is not None and node.gemm.transpose_b
    )
    softmax = next(
        node
        for node in graph.nodes
        if classified.traits_for(node.node_id).engine == "sfu"
    )
    groups = next(enumerate_group_assignments(classified, 1))
    found = False
    for stages_by_region in enumerate_program_stages(
        classified, SearchBudget(max_stages=2)
    ):
        stages = stages_by_region[qk.region_id]
        if stages[qk.node_id] >= stages[softmax.node_id]:
            continue
        try:
            orders = build_program_orders(classified, stages_by_region, groups)
        except ValueError:
            continue
        versions = analyze_buffer_versions(
            graph, stages_by_region, groups, orders
        )
        if versions[acc_s.buffer_id] >= 2:
            found = True
            break
    assert found


def test_fake_arch_enumerates_the_same_shape() -> None:
    classified = FAKE.classify(_gemm_graph())
    structures = _structures(classified, max_structures=8)
    assert structures
    assert any(item.num_groups == 1 for item in structures)
    assert all(not hasattr(item, "warp_allocation") for item in structures)
