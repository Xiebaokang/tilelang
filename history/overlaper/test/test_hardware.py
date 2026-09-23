"""Target-to-hardware resource resolution tests."""

from tvm.target import Target

from history.overlaper.headware import HOPPER, resolve_hardware


def test_target_shared_memory_limit_overrides_architecture_default() -> None:
    target = Target(
        {
            "kind": "cuda",
            "arch": "sm_90a",
            "max_shared_memory_per_block": 232448,
        }
    )

    hardware = resolve_hardware(target)

    assert hardware.device_resource.shared_memory_capacity_bytes == 232448
    assert HOPPER.device_resource.shared_memory_capacity_bytes == 253952
