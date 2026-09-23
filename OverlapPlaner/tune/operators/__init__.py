"""Operators with native TileLang and searched OverlapPlan lowering paths."""

from __future__ import annotations

from .convolution import OPERATOR as CONVOLUTION
from .fa3 import OPERATOR as FA3
from .gemm import OPERATOR as GEMM
from .gemm_fp8 import OPERATOR as GEMM_FP8
from .gqa import OPERATOR as GQA
from .gqa_bwd import OPERATOR as GQA_BWD
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
    "CONVOLUTION",
    "FA3",
    "GEMM",
    "GEMM_FP8",
    "GQA",
    "GQA_BWD",
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
