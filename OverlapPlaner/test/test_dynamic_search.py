"""Tests for measurement-driven candidate selection."""

from __future__ import annotations

import json
from pathlib import Path

from OverlapPlaner.tune.dynamic import (
    Candidate,
    DynamicSearchPolicy,
    file_fingerprint,
    should_stop,
)


def _candidate(index: int, groups: int, stages: int, signal: float) -> Candidate:
    return Candidate(
        index=index,
        path=Path(f"schedule_{index:05d}.json"),
        fingerprint=str(index),
        bucket=(groups, stages),
        features=(float(groups), float(stages), signal, signal * signal),
    )


def test_initial_selection_round_robins_structure_buckets() -> None:
    candidates = (
        _candidate(0, 1, 1, 0.0),
        _candidate(1, 1, 1, 1.0),
        _candidate(2, 1, 2, 2.0),
        _candidate(3, 2, 1, 3.0),
        _candidate(4, 2, 2, 4.0),
    )
    selected = DynamicSearchPolicy(candidates).initial(set(), 4)
    assert selected == [0, 2, 3, 4]


def test_initial_selection_covers_independent_async_producer() -> None:
    candidates = (
        _candidate(0, 2, 1, 0.0),
        Candidate(
            index=1,
            path=Path("schedule_00001.json"),
            fingerprint="1",
            bucket=(2, 1),
            features=(2.0, 1.0, 1.0),
            producer_copies=2,
        ),
        _candidate(2, 2, 1, 2.0),
    )
    policy = DynamicSearchPolicy(candidates)
    assert policy.initial(set(), 1) == [1]
    assert policy.has_uncovered_bucket({0})
    assert policy.next_batch({0: 1.0}, set(), 1) == [1]
    assert not policy.has_uncovered_bucket({0, 1})


def test_initial_selection_covers_physical_group_permutations() -> None:
    common = {
        "bucket": (2, 1),
        "features": (2.0, 1.0, 2.0),
        "producer_copies": 2,
    }
    candidates = (
        Candidate(
            index=0,
            path=Path("schedule_00000.json"),
            fingerprint="0",
            physical_bucket=((8, 1), (4, 0)),
            **common,
        ),
        Candidate(
            index=1,
            path=Path("schedule_00001.json"),
            fingerprint="1",
            physical_bucket=((4, 0), (8, 1)),
            **common,
        ),
    )
    policy = DynamicSearchPolicy(candidates)
    assert set(policy.initial(set(), 2)) == {0, 1}
    assert policy.has_uncovered_bucket({0})


def test_failed_candidate_does_not_cover_structure_bucket() -> None:
    candidates = (
        _candidate(0, 1, 1, 0.0),
        _candidate(1, 2, 3, 1.0),
        _candidate(2, 2, 3, 2.0),
    )
    policy = DynamicSearchPolicy(candidates)
    assert policy.has_uncovered_bucket({0}, {1})
    assert policy.next_batch({0: 1.0}, {1}, 1) == [2]
    assert not policy.has_uncovered_bucket({0}, {1, 2})


def test_feedback_selection_is_deterministic_and_skips_completed() -> None:
    candidates = tuple(
        _candidate(index, 1 + index % 2, 1 + index % 3, float(index))
        for index in range(10)
    )
    measured = {0: 10.0, 1: 8.0, 2: 6.0, 3: 4.0}
    policy = DynamicSearchPolicy(candidates, seed=7)
    first = policy.next_batch(measured, {4}, 3)
    second = policy.next_batch(measured, {4}, 3)
    assert first == second
    assert len(first) == 3
    assert not (set(first) & ({0, 1, 2, 3, 4}))


def test_dynamic_stopping_uses_measured_improvement() -> None:
    assert should_stop(
        [10.0, 9.0, 8.99, 8.98, 8.98],
        patience=3,
        minimum_improvement=0.01,
    )
    assert not should_stop(
        [10.0, 9.0, 8.0, 7.0],
        patience=2,
        minimum_improvement=0.01,
    )


def test_order_expansion_recomputes_plan_and_replays_on_regeneration(tmp_path):
    import tilelang

    from OverlapPlaner.apply import apply_plan_to_ir
    from OverlapPlaner.arch import HOPPER
    from OverlapPlaner.contract import layout_reduced_module, layout_reduced_prim_func
    from OverlapPlaner.facts import extract_fact_graph
    from OverlapPlaner.serialization import load_plan_json, plan_to_dict
    from OverlapPlaner.structure import SearchBudget
    from OverlapPlaner.tune.operators.mla import OPERATOR as MLA
    from OverlapPlaner.tune.order_search import adjacent_order_swaps
    from OverlapPlaner.tune.run import _expand_order_candidates, generate_plans

    options = {
        "mla_batch": 1, "mla_heads": 64, "mla_kv_heads": 1,
        "mla_seq": 256, "mla_dim": 512, "mla_pe_dim": 64,
    }
    tile = {"block_h": 64, "block_n": 64}
    plan_dir = tmp_path / "candidates"
    budget = SearchBudget(max_structures=128)
    generate_plans(
        MLA, plan_dir, tile, options, budget=budget,
        replay_order_mutations=True,
    )
    base_count = json.loads((plan_dir / "manifest.json").read_text())["schedule_count"]
    classified = HOPPER.classify(
        extract_fact_graph(layout_reduced_prim_func(MLA.build(options, tile).prim_func))
    )
    parent_index = next(
        index
        for index, path in enumerate(sorted(plan_dir.glob("schedule_*.json")))
        if any(adjacent_order_swaps(classified, plan_to_dict(load_plan_json(path))))
    )
    added = _expand_order_candidates(
        plan_dir, classified, {parent_index: 1.0}, 2, max_extra=8
    )
    assert added
    mod, target = layout_reduced_module(MLA.build(options, tile).prim_func)
    mutated = load_plan_json(plan_dir / f"schedule_{added[0]:05d}.json")
    with target, tilelang.transform.PassContext(config={}):
        lowered = tilelang.transform.LowerOverlapPlan()(
            apply_plan_to_ir(mod, mutated)
        )
    assert lowered[lowered.get_global_var("main")]
    fingerprints = [
        file_fingerprint(plan_dir / f"schedule_{index:05d}.json")
        for index in added
    ]
    generate_plans(MLA, plan_dir, tile, options, budget=budget)
    assert json.loads((plan_dir / "manifest.json").read_text())["schedule_count"] == base_count
    generate_plans(
        MLA, plan_dir, tile, options, budget=budget,
        replay_order_mutations=True,
    )
    assert fingerprints == [
        file_fingerprint(plan_dir / f"schedule_{index:05d}.json")
        for index in added
    ]


def test_joint_expansion_samples_stage_group_and_order_and_replays(tmp_path):
    from OverlapPlaner.arch import HOPPER
    from OverlapPlaner.arch.hopper import optional_tma_copy_ids
    from OverlapPlaner.contract import layout_reduced_prim_func
    from OverlapPlaner.facts import extract_fact_graph
    from OverlapPlaner.structure import SearchBudget
    from OverlapPlaner.tune.joint_search import adjacent_joint_moves, realize_joint_move
    from OverlapPlaner.tune.operators.mamba_chunk_scan import OPERATOR
    from OverlapPlaner.tune.run import _expand_joint_candidates, generate_plans

    options = {
        "mamba_scan_batch": 8, "mamba_scan_heads": 80,
        "mamba_scan_groups": 1, "mamba_scan_seq": 4096,
        "mamba_scan_chunk": 256, "mamba_scan_dim": 64,
        "mamba_scan_dstate": 128,
    }
    tile = {"block_m": 64, "block_n": 64, "block_k": 64, "block_dstate": 128}
    plan_dir = tmp_path / "candidates"
    budget = SearchBudget(max_structures=1)
    generate_plans(OPERATOR, plan_dir, tile, options, budget=budget)
    classified = HOPPER.classify(
        extract_fact_graph(layout_reduced_prim_func(OPERATOR.build(options, tile).prim_func))
    )
    assert optional_tma_copy_ids(classified) == (0,)
    parent = json.loads((plan_dir / "schedule_00000.json").read_text())
    dimensions = {
        move[0]
        for move in adjacent_joint_moves(classified, parent)
        if realize_joint_move(classified, parent, move) is not None
    }
    assert dimensions == {"stage", "group", "order", "copy"}
    copy_move = next(
        move for move in adjacent_joint_moves(classified, parent)
        if move[0] == "copy"
    )
    copy_plan = realize_joint_move(classified, parent, copy_move)
    assert copy_plan is not None
    assert copy_plan["operations"] == parent["operations"]
    assert copy_plan["groups"] == parent["groups"]
    assert copy_plan["buffers"] == parent["buffers"]
    assert any(
        edge["producer_id"] == copy_move[1]
        and edge["completion_mode"] == 1
        for edge in copy_plan["sync_edges"]
    )
    assert realize_joint_move(
        classified, copy_plan, ("copy", copy_move[1], 0, 0, 0)
    ) is not None
    grouped = next(
        proposal
        for move in adjacent_joint_moves(classified, parent)
        if move[0] == "group"
        if (proposal := realize_joint_move(classified, parent, move)) is not None
        if any(item[0] == "version" for item in adjacent_joint_moves(classified, proposal))
    )
    version_moves = [
        move for move in adjacent_joint_moves(classified, grouped) if move[0] == "version"
    ]
    assert version_moves
    assert any(
        realize_joint_move(classified, grouped, move) is not None
        for move in version_moves
    )

    added = _expand_joint_candidates(
        plan_dir, classified, {0: 1.0}, 4, max_extra=12
    )
    assert len(added) == 4
    mutations = [json.loads(row) for row in (plan_dir / "joint_mutations.jsonl").read_text().splitlines()]
    assert {row["move"][0] for row in mutations} == dimensions
    fingerprints = [
        file_fingerprint(plan_dir / f"schedule_{index:05d}.json") for index in added
    ]
    generate_plans(
        OPERATOR, plan_dir, tile, options, budget=budget,
        replay_order_mutations=True,
    )
    assert fingerprints == [
        file_fingerprint(plan_dir / f"schedule_{index:05d}.json") for index in added
    ]


def test_mamba_joint_pool_covers_target_structure_and_extra_version(tmp_path):
    from OverlapPlaner.arch import HOPPER
    from OverlapPlaner.contract import layout_reduced_prim_func
    from OverlapPlaner.facts import extract_fact_graph
    from OverlapPlaner.structure import SearchBudget
    from OverlapPlaner.tune.joint_search import adjacent_joint_moves, realize_joint_move
    from OverlapPlaner.tune.operators.mamba_chunk_scan import OPERATOR
    from OverlapPlaner.tune.run import generate_plans

    options = {
        "mamba_scan_batch": 8, "mamba_scan_heads": 80,
        "mamba_scan_groups": 1, "mamba_scan_seq": 4096,
        "mamba_scan_chunk": 256, "mamba_scan_dim": 64,
        "mamba_scan_dstate": 128,
    }
    tile = {"block_m": 64, "block_n": 64, "block_k": 64, "block_dstate": 128}
    plan_dir = tmp_path / "candidates"
    generate_plans(
        OPERATOR, plan_dir, tile, options,
        budget=SearchBudget(max_structures=1024, stage_beam=64),
        joint_seed_budget=1024,
    )
    producer_group = {0, 4, 5, 8, 10, 13, 17, 20, 24}
    target_stages = dict(zip(
        range(8, 19), (0, 1, 0, 1, 2, 0, 1, 2, 2, 1, 2)
    ))
    matched = None
    for path in sorted(plan_dir.glob("schedule_*.json")):
        payload = json.loads(path.read_text())
        operations = {item["operation_id"]: item for item in payload["operations"]}
        if (
            {node_id for node_id, item in operations.items() if item["group_id"] == 1}
            == producer_group
            and all(operations[node_id]["stage"] == stage for node_id, stage in target_stages.items())
            and tuple(
                sorted(producer_group & {0, 4, 5}, key=lambda node_id: operations[node_id]["order"])
            ) == (0, 4, 5)
        ):
            matched = payload
            break
    assert matched is not None
    classified = HOPPER.classify(
        extract_fact_graph(layout_reduced_prim_func(OPERATOR.build(options, tile).prim_func))
    )
    version_move = next(
        move for move in adjacent_joint_moves(classified, matched)
        if move[:3] == ("version", 10, 3)
    )
    completed = realize_joint_move(classified, matched, version_move)
    assert completed is not None
    assert completed["buffers"][10]["version_count"] == 3
    assert [item["warp_count"] for item in completed["groups"]] == [4, 4]
