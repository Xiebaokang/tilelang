"""Global generation must escape the initial cache and resume without replay drift."""
import json
import random

import pytest

from OverlapPlaner.structure import SearchBudget
from OverlapPlaner.tune.dynamic import load_candidates
from OverlapPlaner.tune.global_search import sample_global_plan, sample_structure
from OverlapPlaner.tune.run import (
    _expand_global_candidates,
    _fresh_quotas,
    _proposal_limits,
    generate_plans,
)
from OverlapPlaner.test.test_copy_backend import OPERATOR, candidates


def test_global_stream_is_reproducible_and_samples_both_backends():
    classified, _ = candidates()
    budget = SearchBudget(max_groups=1, max_structures=1, stage_beam=1)
    first = [sample_global_plan(classified, budget, seed) for seed in range(24)]
    assert first == [sample_global_plan(classified, budget, seed) for seed in range(24)]
    choices = {tuple(op['copy_backend'] for op in p['operations']) for p in first if p}
    assert choices == {('simt', 'simt'), ('simt', 'tma'), ('tma', 'simt'), ('tma', 'tma')}


def test_global_cache_escape_dedup_and_resume(tmp_path):
    classified, _ = candidates()
    budget = SearchBudget(max_groups=1, max_structures=1)
    generate_plans(OPERATOR, tmp_path, {}, {}, budget=budget)
    original = {c.fingerprint for c in load_candidates(tmp_path)}
    added = _expand_global_candidates(tmp_path, classified, 3, seconds=30)
    assert len(added) == 3
    expanded = load_candidates(tmp_path)
    assert len({c.fingerprint for c in expanded} - original) == 3
    before = [(c.index, c.fingerprint) for c in expanded]
    cursor = json.loads((tmp_path / 'manifest.json').read_text())['global_cursor']
    generate_plans(OPERATOR, tmp_path, {}, {}, budget=budget, replay_order_mutations=True)
    assert [(c.index, c.fingerprint) for c in load_candidates(tmp_path)] == before
    assert json.loads((tmp_path / 'manifest.json').read_text())['global_cursor'] == cursor
    _expand_global_candidates(tmp_path, classified, 1, max_attempts=2, seconds=30)
    after = json.loads((tmp_path / 'manifest.json').read_text())
    assert cursor < after['global_cursor'] <= cursor + 2


def test_global_generation_attempt_limit_even_without_legal_proposals(tmp_path, monkeypatch):
    import OverlapPlaner.tune.run as run
    classified, _ = candidates()
    generate_plans(OPERATOR, tmp_path, {}, {}, budget=SearchBudget(max_structures=1))
    monkeypatch.setattr(run, 'sample_global_plan', lambda *args: None)
    assert _expand_global_candidates(tmp_path, classified, 1, max_attempts=5, seconds=30) == []
    assert json.loads((tmp_path / 'manifest.json').read_text())['global_cursor'] == 5


def test_small_batches_do_not_starve_sources():
    assert {_fresh_quotas(1, i) for i in range(3)} == {(1, 0), (0, 1), (0, 0)}
    assert {_fresh_quotas(2, i) for i in range(2)} == {(1, 1), (1, 0)}
    assert _fresh_quotas(8, 0) == (1, 1)


def test_feedback_slots_get_a_rankable_proposal_reservoir():
    assert _proposal_limits(4, 1, 1) == (5, 5)
    assert _proposal_limits(3, 1, 1) == (3, 3)
    assert _proposal_limits(2, 1, 1) == (1, 1)
    assert _proposal_limits(1, 1, 0) == (1, 0)


def test_pipeline_sampling_varies_stage_and_group():
    from OverlapPlaner.arch import HOPPER
    from OverlapPlaner.contract import layout_reduced_prim_func
    from OverlapPlaner.facts import extract_fact_graph
    from OverlapPlaner.test.test_copy_backend import build_loop
    classified = HOPPER.classify(extract_fact_graph(layout_reduced_prim_func(build_loop({}, {}).prim_func)))
    budget = SearchBudget(max_groups=2, stage_beam=1)
    structures = [sample_structure(classified, budget, random.Random(i)) for i in range(100)]
    structures = [s for s in structures if s]
    assert structures
    assert {s.num_groups for s in structures} == {1, 2}
    assert max(max(v.values()) for s in structures for v in s.stages_by_region.values()) >= 1


def test_large_region_sampling_escapes_beam_and_fixed_orders():
    from OverlapPlaner.arch import HOPPER
    from OverlapPlaner.test.test_structure import _mamba_scan_graph
    from OverlapPlaner.structure.stage import enumerate_program_stages
    from OverlapPlaner.structure.order import enumerate_program_orders
    from OverlapPlaner.structure.model import SynchronizationCycleError
    classified = HOPPER.classify(_mamba_scan_graph())
    budget = SearchBudget(max_groups=1, stage_beam=1)
    beam_stages = list(enumerate_program_stages(classified, budget))
    escaped_stage = escaped_order = False
    for seed in range(300):
        try:
            structure = sample_structure(classified, budget, random.Random(seed))
        except SynchronizationCycleError:
            continue
        if structure is None:
            continue
        escaped_stage |= structure.stages_by_region not in beam_stages
        fixed = list(enumerate_program_orders(classified, structure.stages_by_region, structure.groups))
        escaped_order |= structure.orders not in fixed
        if escaped_stage and escaped_order:
            break
    assert escaped_stage
    assert escaped_order


@pytest.mark.parametrize("batch_size", [1, 3])
def test_dynamic_refills_exhausted_initial_cache_and_honors_budget(tmp_path, monkeypatch, batch_size):
    import OverlapPlaner.tune.run as run
    budget = SearchBudget(max_groups=1, max_structures=1)
    generate_plans(OPERATOR, tmp_path / 'candidates', {}, {}, budget=budget)
    evaluated = []

    def measure(*args, schedule_indices):
        candidates_now = load_candidates(
            tmp_path / 'candidates', fingerprint_salt=run._workload_fingerprint(OPERATOR, {}, {})
        )
        by_id = {c.index: c for c in candidates_now}
        for index in schedule_indices:
            assert index not in evaluated
            evaluated.append(index)
            run._append_jsonl(tmp_path / 'results.jsonl', {
                'schedule_index': index, 'candidate_fingerprint': by_id[index].fingerprint,
                'latency_ms': 1.0,
            })
        rows = run._read_jsonl(tmp_path / 'results.jsonl')
        return {'successful': rows, 'failures': [], 'examined': len(rows)}

    monkeypatch.setattr(run, '_supervise_config', measure)
    def no_local_during_warmup(*args, **kwargs):
        raise AssertionError("warmup should sample independently")
    monkeypatch.setattr(run, '_expand_joint_candidates', no_local_during_warmup)
    result = run._supervise_dynamic_config(
        OPERATOR, {}, {}, tmp_path, 0, 1, 1, 1, 1,
        evaluation_budget=4, initial_samples=8, batch_size=batch_size,
        patience=20, minimum_improvement=0.01,
    )
    assert result['examined'] == len(evaluated) == 4
    state = json.loads((tmp_path / 'dynamic_state.json').read_text())
    assert any('global_exploration' in row['selection_roles'].values() for row in state['batch_diagnostics'])
    generate_plans(OPERATOR, tmp_path / 'candidates', {}, {}, budget=budget, replay_order_mutations=True)
    result = run._supervise_dynamic_config(
        OPERATOR, {}, {}, tmp_path, 0, 1, 1, 1, 1,
        evaluation_budget=4, initial_samples=8, batch_size=batch_size,
        patience=20, minimum_improvement=0.01,
    )
    assert result['examined'] == len(evaluated) == 4


def test_dynamic_seed_count_is_small_and_independent_of_enumeration_limit():
    from OverlapPlaner.tune.run import _generation_options
    small = _generation_options('dynamic', 1, 24, 64, 64)
    large = _generation_options('dynamic', 100000, 24, 64, 64)
    assert small == large
    assert small['budget'].max_structures + small['joint_seed_budget'] == 8
    tiny = _generation_options('dynamic', 100000, 24, 1, 64)
    assert tiny['budget'].max_structures + tiny['joint_seed_budget'] == 1
    exhaustive = _generation_options('exhaustive', 19, 24, 64, 64)
    assert exhaustive['budget'].max_structures == 19
    assert 'joint_seed_budget' not in exhaustive


def test_cli_exhaustive_limit_and_dynamic_default(monkeypatch):
    import OverlapPlaner.tune.run as run
    import sys
    calls = []
    monkeypatch.setattr(run, 'run', lambda *args: calls.append(args))
    monkeypatch.setattr(sys, 'argv', ['run', '--operators', 'gemm'])
    run.main()
    assert calls[-1][7] == 'dynamic'
    monkeypatch.setattr(sys, 'argv', ['run', '--search-mode', 'exhaustive', '--max-candidates', '19'])
    run.main()
    assert calls[-1][8] == 19


def test_cli_rejects_obsolete_or_misapplied_pool_option(monkeypatch):
    import OverlapPlaner.tune.run as run
    import pytest
    import sys
    for options in (['--candidate-pool', '8'], ['--max-candidates', '8'],
                    ['--search-mode', 'exhaustive', '--max-candidates', '0']):
        monkeypatch.setattr(sys, 'argv', ['run', *options])
        with pytest.raises(SystemExit) as error:
            run.main()
        assert error.value.code == 2
