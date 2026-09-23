"""Operators with native TileLang and searched OverlapPlan lowering paths."""

from __future__ import annotations

from .block_causal_bwd import OPERATOR as BLOCK_CAUSAL_BWD
from .convolution import OPERATOR as CONVOLUTION
from .dequant_gemm_fp4 import OPERATOR as DEQUANT_GEMM_FP4
from .fa3 import OPERATOR as FA3
from .flash_decode import OPERATOR as FLASH_DECODE
from .fused_moe import OPERATOR as FUSED_MOE
from .gemm import OPERATOR as GEMM
from .gemm_fp8 import OPERATOR as GEMM_FP8
from .gqa import OPERATOR as GQA
from .gqa_bwd import OPERATOR as GQA_BWD
from .gdn_chunk_delta_bwd import OPERATOR as GDN_CHUNK_DELTA_BWD
from .gdn_chunk_o_bwd import OPERATOR as GDN_CHUNK_O_BWD
from .kda_chunk_bwd_intra import OPERATOR as KDA_CHUNK_BWD_INTRA
from .kda_wy_fast_bwd import OPERATOR as KDA_WY_FAST_BWD
from .linear_attn_fwd import OPERATOR as LINEAR_ATTN_FWD
from .mamba_chunk_scan import OPERATOR as MAMBA_CHUNK_SCAN
from .mamba_chunk_state import OPERATOR as MAMBA_CHUNK_STATE
from .mla import OPERATOR as MLA
from .mha_bwd import OPERATOR as MHA_BWD
from .workloads import OperatorSpec, Options, SearchWorkload, TileConfig


OPERATORS = (
    FA3,
    GEMM,
    CONVOLUTION,
    GQA,
    GQA_BWD,
    MHA_BWD,
    BLOCK_CAUSAL_BWD,
    DEQUANT_GEMM_FP4,
    GDN_CHUNK_O_BWD,
    GDN_CHUNK_DELTA_BWD,
    KDA_WY_FAST_BWD,
    KDA_CHUNK_BWD_INTRA,
    FLASH_DECODE,
    FUSED_MOE,
    GEMM_FP8,
    MLA,
    LINEAR_ATTN_FWD,
    MAMBA_CHUNK_SCAN,
    MAMBA_CHUNK_STATE,
)
_OPERATORS_BY_NAME = {operator.name: operator for operator in OPERATORS}
if len(_OPERATORS_BY_NAME) != len(OPERATORS):
    raise RuntimeError("OverlapPlan operator names must be unique")

OPERATOR_NAMES = tuple(operator.name for operator in OPERATORS)


def get_operator(name: str) -> OperatorSpec:
    try:
        return _OPERATORS_BY_NAME[name]
    except KeyError as error:
        choices = ", ".join(OPERATOR_NAMES)
        raise ValueError(
            f"unknown operator {name!r}; choose from {choices}"
        ) from error


__all__ = [
    "BLOCK_CAUSAL_BWD",
    "CONVOLUTION",
    "DEQUANT_GEMM_FP4",
    "FA3",
    "FLASH_DECODE",
    "FUSED_MOE",
    "GEMM",
    "GEMM_FP8",
    "GQA",
    "GQA_BWD",
    "GDN_CHUNK_DELTA_BWD",
    "GDN_CHUNK_O_BWD",
    "KDA_CHUNK_BWD_INTRA",
    "KDA_WY_FAST_BWD",
    "LINEAR_ATTN_FWD",
    "MAMBA_CHUNK_SCAN",
    "MAMBA_CHUNK_STATE",
    "MLA",
    "MHA_BWD",
    "OPERATORS",
    "OPERATOR_NAMES",
    "OperatorSpec",
    "Options",
    "SearchWorkload",
    "TileConfig",
    "get_operator",
]
