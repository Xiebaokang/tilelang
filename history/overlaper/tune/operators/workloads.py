"""Stable interface shared by every Overlaper tuning operator."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from tvm import tirx


Options = Mapping[str, Any]
TileConfig = dict[str, int]


@dataclass(frozen=True, slots=True)
class SearchWorkload:
    """Everything the schedule search needs for one concrete tile."""

    prim_func: tirx.PrimFunc
    out_idx: tuple[int, ...]
    total_flops: float
    reference_program: Callable[..., Any]
    input_tensors: Sequence[Any] | None = None
    pass_configs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.prim_func, tirx.PrimFunc):
            raise TypeError("prim_func must be a TIRX PrimFunc")
        if not self.out_idx or any(
            not isinstance(index, int) for index in self.out_idx
        ):
            raise ValueError("out_idx must contain at least one integer")
        if not math.isfinite(self.total_flops) or self.total_flops <= 0:
            raise ValueError("total_flops must be finite and positive")
        if not callable(self.reference_program):
            raise TypeError("reference_program must be callable")


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    """One registered operator and its three uniform extension points."""

    name: str
    description: str
    add_cli_arguments: Callable[[Any], None]
    configuration_factory: Callable[[Options], Sequence[TileConfig]]
    workload_factory: Callable[[Options, TileConfig], SearchWorkload]

    def __post_init__(self) -> None:
        if not self.name.isidentifier() or self.name.lower() != self.name:
            raise ValueError("operator name must be a lowercase Python identifier")
        if not self.description:
            raise ValueError("operator description cannot be empty")

    def add_arguments(self, parser: Any) -> None:
        self.add_cli_arguments(parser)

    def configurations(self, options: Options) -> tuple[TileConfig, ...]:
        configurations = tuple(
            dict(config) for config in self.configuration_factory(options)
        )
        seen = set()
        for config in configurations:
            if not config:
                raise ValueError(f"{self.name} produced an empty tile configuration")
            if any(
                not isinstance(key, str)
                or not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
                for key, value in config.items()
            ):
                raise ValueError(
                    f"{self.name} tile configuration keys must be strings and "
                    "values must be positive integers"
                )
            identity = tuple(sorted(config.items()))
            if identity in seen:
                raise ValueError(
                    f"{self.name} produced a duplicate tile configuration"
                )
            seen.add(identity)
        return configurations

    def build(self, options: Options, config: TileConfig) -> SearchWorkload:
        workload = self.workload_factory(options, dict(config))
        if not isinstance(workload, SearchWorkload):
            raise TypeError(
                f"{self.name} workload_factory must return SearchWorkload"
            )
        return workload
