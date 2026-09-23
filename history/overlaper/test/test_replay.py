"""CPU-only tests for the generic single-schedule replay CLI."""

import json
from pathlib import Path

import pytest

from history.overlaper.tune.replay import _infer_replay_context, _parse_override


@pytest.mark.parametrize(
    ("operator_name", "leaf"),
    [
        ("fa3", "candidates/schedule_00001.json"),
        ("gqa", "failures/timeout_schedule_00001.json"),
    ],
)
def test_infer_context_from_result_manifest(
    tmp_path: Path, operator_name: str, leaf: str
) -> None:
    config_dir = tmp_path / operator_name / "m128_n96"
    manifest = config_dir / "candidates" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "operator": operator_name,
                "tile": {"block_m": 128, "block_n": 96},
                "schedule_count": 2,
                "options": {f"{operator_name}_batch": 3},
            }
        ),
        encoding="utf-8",
    )

    operator, tile, options = _infer_replay_context(config_dir / leaf)

    assert operator.name == operator_name
    assert tile == {"block_m": 128, "block_n": 96}
    assert options == {f"{operator_name}_batch": 3}


def test_infer_context_requires_manifest(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot infer replay context"):
        _infer_replay_context(tmp_path / "schedule.json")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("gemm_m=2048", ("gemm_m", 2048)),
        ("fa3_causal=true", ("fa3_causal", True)),
        ("arch_name=custom", ("arch_name", "custom")),
    ],
)
def test_parse_operator_option_override(text: str, expected: tuple) -> None:
    assert _parse_override(text) == expected
