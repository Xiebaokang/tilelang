"""Resolve an Overlaper hardware description from a TVM-like target."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from .hopper import HOPPER
from .spec import HardwareSpec, Resource


_CUDA_ARCHITECTURES: dict[str, HardwareSpec] = {"90": HOPPER}


def resolve_hardware(target: Any) -> HardwareSpec:
    """Return the registered hardware for a target with ``kind`` and ``attrs``."""

    try:
        kind = target.kind.name
        arch = str(target.attrs.get("arch", ""))
    except AttributeError as error:
        raise TypeError("target must provide kind.name and attrs") from error
    if not arch:
        raise ValueError(f"target kind {kind} does not specify an architecture")
    if kind == "cuda":
        match = re.fullmatch(r"sm_(\d+)[a-z]?", arch)
        if match is not None:
            hardware = _CUDA_ARCHITECTURES.get(match.group(1))
            if hardware is not None:
                capacity = target.attrs.get("max_shared_memory_per_block")
                if capacity is None:
                    return hardware
                base = hardware.device_resource
                resource = Resource(
                    name=base.name,
                    warp_size=base.warp_size,
                    max_threads_per_block=base.max_threads_per_block,
                    register_file_capacity=base.register_file_capacity,
                    shared_memory_capacity_bytes=int(capacity),
                    max_registers_per_thread=base.max_registers_per_thread,
                    specialized_group_warp_multiple=(
                        base.specialized_group_warp_multiple
                    ),
                )
                return replace(hardware, device_resource=resource)
    raise ValueError(f"unsupported Overlaper target: kind={kind}, arch={arch}")
