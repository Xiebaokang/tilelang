"""Mixed GEMM fact extraction and Hopper classification regressions."""
import importlib.util
from pathlib import Path

import pytest
import tilelang.language as T

from OverlapPlaner.arch import HOPPER
from OverlapPlaner.contract import layout_reduced_prim_func
from OverlapPlaner.facts import DependencyKind, OpKind, extract_fact_graph


def mixed_kernel(transpose_b, clear_accum):
    b_shape = (64, 128) if not transpose_b else (128, 64)

    @T.prim_func(auto_overlap=True)
    def main(A: T.Tensor((128, 64), T.float16),
             B: T.Tensor(b_shape, T.float16),
             Out: T.Tensor((128, 128), T.float16)):
        with T.Kernel(1, threads=256):
            SA = T.alloc_shared((128, 32), T.float16)
            RA = T.alloc_fragment((128, 32), T.float16)
            SB = T.alloc_shared(b_shape, T.float16)
            C = T.alloc_fragment((128, 128), T.float32)
            T.clear(C)
            for _ in T.Pipelined(2):
                T.copy(A[:, :32], SA)
                T.copy(A[:, 32:], RA)
                T.copy(B, SB)
                T.gemm_mix(SA, RA, SB, C, transpose_B=transpose_b,
                           clear_accum=clear_accum, policy=T.GemmWarpPolicy.FullRow)
            T.copy(C, Out)
    return main


@pytest.mark.parametrize('transpose_b', [False, True])
@pytest.mark.parametrize('clear_accum', [False, True])
def test_extract_mixed_gemm_inputs_and_dependencies(transpose_b, clear_accum):
    graph = extract_fact_graph(mixed_kernel(transpose_b, clear_accum))
    gemms = [node for node in graph.nodes if node.kind == OpKind.GEMM]
    assert len(gemms) == 1
    node = gemms[0]
    fact = node.gemm
    assert node.tileop == 'gemm_mix'
    assert (fact.m, fact.n, fact.k) == (128, 128, 64)
    assert (fact.k_shared, fact.k_register) == (32, 32)
    assert fact.k == fact.k_shared + fact.k_register
    assert fact.ra_scope == 'local.fragment'
    assert fact.ra_dtype == fact.a_dtype == fact.b_dtype == 'float16'
    assert fact.transpose_b == transpose_b
    assert fact.clear_accum == clear_accum
    assert fact.policy == 'full_row'
    buffers = {buffer.name: buffer.buffer_id for buffer in graph.buffers}
    for name in ('SA', 'RA', 'SB'):
        buffer_id = buffers[name]
        assert buffer_id in node.reads
        assert any(edge.consumer_id == node.node_id and edge.buffer_id == buffer_id
                   and edge.iteration_distance == 0 and DependencyKind.RAW in edge.dependency_kinds
                   for edge in graph.edges), name
    assert buffers['C'] in node.writes
    if not clear_accum:
        assert buffers['C'] in node.reads
    traits = HOPPER.classify(graph).traits[node.node_id]
    assert traits.engine == 'warpgroup_tensorcore'
    assert traits.async_completion
    assert traits.issue_priority == 5


@pytest.mark.parametrize('reduce_layout', [False, True])
def test_mixed_attention_is_two_gemm_nodes(reduce_layout):
    path = Path(__file__).resolve().parents[2] / 'examples/flash_attention/example_mha_fwd_bhsd_mix.py'
    spec = importlib.util.spec_from_file_location('overlap_mixed_attention', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prim = module.flashattn.jit_impl.get_tir(
        1, 1, 256, 256, 128, False,
        block_M=128, block_N=128, num_stages=1, threads=256)
    if reduce_layout:
        prim = layout_reduced_prim_func(prim)
    graph = extract_fact_graph(prim)
    nodes = [node for node in graph.nodes if node.kind == OpKind.GEMM]
    assert len(nodes) == 2
    assert all(node.tileop == 'gemm_mix' for node in nodes)
    qk, pv = nodes
    assert (qk.gemm.k_shared, qk.gemm.k_register) == (96, 32)
    assert (pv.gemm.k_shared, pv.gemm.k_register) == (32, 96)
    for node, ra_name in ((qk, 'Q_reg'), (pv, 'P_reg')):
        ra = next(buffer for buffer in graph.buffers if buffer.name == ra_name)
        assert ra.buffer_id in node.reads
        assert any(edge.consumer_id == node.node_id and edge.buffer_id == ra.buffer_id
                   and DependencyKind.RAW in edge.dependency_kinds for edge in graph.edges)
        assert HOPPER.classify(graph).traits[node.node_id].engine == 'warpgroup_tensorcore'
