"""Representative added tiles must be generated, realizable and lower natively."""
import pytest
import tilelang
from tilelang.engine import lower
from tvm.target import Target

from OverlapPlaner.tune.operators import get_operator
from OverlapPlaner.tune.run import _default_options
from OverlapPlaner.contract import enumerate_overlap_plans
from OverlapPlaner.structure import SearchBudget


CASES = (
    ('fa3', {'block_m': 256, 'block_n': 128}),
    ('fa3', {'block_m': 64, 'block_n': 32}),
    ('gqa', {'block_m': 256, 'block_n': 64}),
    ('gqa_bwd', {'block_m': 64, 'block_n': 32}),
    ('gqa_bwd', {'block_m': 128, 'block_n': 64}),
    ('mha_bwd', {'block_m': 64, 'block_n': 64}),
    ('gemm', {'block_m': 256, 'block_n': 256, 'block_k': 128}),
    ('gemm', {'block_m': 192, 'block_n': 128, 'block_k': 128}),
    ('gemm_fp8', {'block_m': 128, 'block_n': 256, 'block_k': 256}),
    ('mla', {'block_h': 64, 'block_n': 32}),
    ('linear_attn_fwd', {'block_k': 32, 'block_v': 128}),
    ('mamba_chunk_state', {'block_m': 32, 'block_n': 128, 'block_k': 128}),
    ('mamba_chunk_scan', {'block_m': 256, 'block_n': 32, 'block_k': 128, 'block_dstate': 128}),
    ('fused_moe', {'block_token': 128, 'block_hidden': 64, 'block_expert': 64}),
)


@pytest.mark.parametrize('name,tile', CASES)
def test_added_tile_is_reachable_and_lowers(name, tile):
    op = get_operator(name)
    options = _default_options(op)
    assert tile in op.configurations(options)
    workload = op.build(options, tile)
    assert next(enumerate_overlap_plans(
        workload.prim_func, budget=SearchBudget(max_structures=1)
    )).operations
    native = op.build_native(options, tile)
    target = Target({'kind': 'cuda', 'arch': 'sm_90a'})
    with target, tilelang.transform.PassContext(config=dict(native.pass_configs)):
        assert lower(native.prim_func, target=target).kernel_source


def test_gemm_new_tiles_avoid_known_native_layout_failures():
    op = get_operator('gemm')
    configs = op.configurations(_default_options(op))
    assert {'block_m': 256, 'block_n': 80, 'block_k': 64} not in configs
    assert {'block_m': 192, 'block_n': 80, 'block_k': 64} not in configs
    assert {'block_m': 192, 'block_n': 96, 'block_k': 64} not in configs
    assert {'block_m': 192, 'block_n': 128, 'block_k': 128} in configs
    assert {'block_m': 128, 'block_n': 96, 'block_k': 64} in configs
