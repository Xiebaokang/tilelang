"""Registry for kernels searchable by the Overlaper tuner."""

from __future__ import annotations

from .convolution import OPERATOR as CONVOLUTION
from .fa3 import OPERATOR as FA3
from .gemm import OPERATOR as GEMM
from .gemm_fp8 import OPERATOR as GEMM_FP8
from .gqa import OPERATOR as GQA
from .gqa_bwd import OPERATOR as GQA_BWD
from .mamba_scan import OPERATOR as MAMBA_SCAN
from .mha_bwd import OPERATOR as MHA_BWD
from .mla import OPERATOR as MLA
from .workloads import OperatorSpec, Options, SearchWorkload, TileConfig


OPERATORS = (
    FA3,
    GEMM,
    CONVOLUTION,
    GQA,
    GEMM_FP8,
    MHA_BWD,
    GQA_BWD,
    MAMBA_SCAN,
    MLA,
)
_OPERATORS_BY_NAME = {operator.name: operator for operator in OPERATORS}
if len(_OPERATORS_BY_NAME) != len(OPERATORS):
    raise RuntimeError("Overlaper tuning operator names must be unique")

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
    "MAMBA_SCAN",
    "MHA_BWD",
    "MLA",
    "OPERATORS",
    "OPERATOR_NAMES",
    "OperatorSpec",
    "Options",
    "SearchWorkload",
    "TileConfig",
    "get_operator",
]
