"""Feed FA3 IR into Overlaper and print every candidate as JSON."""

import json

import history.overlaper.candidates as candidates_module
from history.overlaper import (
    GuidedSearchConfig,
    enumerate_guided_schedules,
    enumerate_schedules,
    schedules_to_json,
)
from history.overlaper.candidates import _group_local_stages
from history.overlaper.headware import HOPPER
from history.overlaper.headware.hopper import TMA, WGMMA
from history.overlaper.parse import (
    DataflowGraph,
    DataflowNode,
    RegionKind,
    extract_dataflow_graph,
)
from history.overlaper.test.test_extractor import make_fa3_prim_func


def fa3_candidates_json() -> tuple[int, str]:
    """Extract the input IR, enumerate all candidates, and encode JSON."""

    prim_func = make_fa3_prim_func()
    graph = extract_dataflow_graph(prim_func, hardware=HOPPER)
    candidates = tuple(enumerate_schedules(graph))
    return len(candidates), schedules_to_json(graph, candidates)


def test_fa3_ir_prints_all_candidates_as_json() -> None:
    candidate_count, result = fa3_candidates_json()
    payload = json.loads(result)

    assert candidate_count == 1197
    assert payload["candidate_count"] == candidate_count
    assert len(payload["candidates"]) == candidate_count
    assert len(payload["graph"]["operations"]) == 22
    assert all(
        candidate["shared_memory"]["merged_shared_bytes"]
        <= candidate["shared_memory"]["capacity_bytes"]
        for candidate in payload["candidates"]
    )

    print(result)


def test_group_local_stage_offsets_have_one_canonical_form() -> None:
    graph = DataflowGraph(
        buffers=(),
        nodes=(
            DataflowNode(0, 0, "load", TMA),
            DataflowNode(1, 0, "compute", WGMMA),
            DataflowNode(2, 0, "consume", WGMMA),
        ),
        edges=(),
        region_kinds=(RegionKind.PIPELINE,),
    )
    groups = {0: 0, 1: 1, 2: 1}

    first = _group_local_stages(
        graph, {0: {0: 0, 1: 1, 2: 2}}, groups
    )
    shifted = _group_local_stages(
        graph, {0: {0: 0, 1: 0, 2: 1}}, groups
    )

    assert first == shifted == ((0, 0, 0), (0, 1, 0), (0, 2, 1))


def test_group_local_stage_equivalence_is_pruned_before_version(monkeypatch) -> None:
    graph = DataflowGraph(
        buffers=(),
        nodes=(
            DataflowNode(0, 0, "load", TMA),
            DataflowNode(1, 0, "compute", WGMMA),
        ),
        edges=(),
        region_kinds=(RegionKind.PIPELINE,),
        hardware=HOPPER,
        kernel_threads=128,
    )
    stage_assignments = (
        {0: {0: 0, 1: 1}},
        {0: {0: 1, 1: 0}},
    )
    groups = {0: 0, 1: 1}
    calls = {"order": 0, "version": 0}
    original_build_orders = candidates_module.build_program_orders

    monkeypatch.setattr(
        candidates_module,
        "_program_stage_assignments",
        lambda _graph: iter(stage_assignments),
    )
    monkeypatch.setattr(
        candidates_module,
        "enumerate_group_assignments",
        lambda _graph, num_groups: iter((groups,))
        if num_groups == 2
        else iter(()),
    )

    def build_orders(*args, **kwargs):
        calls["order"] += 1
        return original_build_orders(*args, **kwargs)

    def no_version_plans(*_args, **_kwargs):
        calls["version"] += 1
        return iter(())

    monkeypatch.setattr(candidates_module, "build_program_orders", build_orders)
    monkeypatch.setattr(
        candidates_module,
        "enumerate_buffer_version_plans",
        no_version_plans,
    )

    assert list(enumerate_schedules(graph)) == []
    assert calls == {"order": 2, "version": 1}


def test_guided_enumeration_is_budgeted_and_deterministic() -> None:
    graph = DataflowGraph(
        buffers=(),
        nodes=(
            DataflowNode(0, 0, "load", TMA),
            DataflowNode(1, 0, "compute", WGMMA),
        ),
        edges=(),
        region_kinds=(RegionKind.PIPELINE,),
        hardware=HOPPER,
        kernel_threads=128,
    )
    policy = GuidedSearchConfig(
        groups_per_count=1,
        structures=2,
        schedules=1,
    )

    first = tuple(enumerate_guided_schedules(graph, config=policy))
    second = tuple(enumerate_guided_schedules(graph, config=policy))

    assert first == second
    assert len(first) == 1


if __name__ == "__main__":
    _, json_text = fa3_candidates_json()
    print(json_text)
