"""Resolve a UnionWSP hardware description from a TVM target."""

from __future__ import annotations

import re

from tvm.target import Target

from .hopper import HOPPER
from .spec import HardwareSpec


_CUDA_ARCHITECTURES: dict[str, HardwareSpec] = {"90": HOPPER}


def resolve_hardware(target: Target) -> HardwareSpec:
    """Return the registered hardware for ``target`` or reject it."""

    if not isinstance(target, Target):
        raise TypeError(f"expected Target, got {type(target).__name__}")
    kind = target.kind.name
    arch = str(target.attrs.get("arch", ""))
    if not arch:
        raise ValueError(f"target kind {kind} does not specify an architecture")
    if kind == "cuda":
        match = re.fullmatch(r"sm_(\d+)[a-z]?", arch)
        if match is not None:
            hardware = _CUDA_ARCHITECTURES.get(match.group(1))
            if hardware is not None:
                return hardware
    raise ValueError(f"unsupported UnionWSP target: kind={kind}, arch={arch}")
