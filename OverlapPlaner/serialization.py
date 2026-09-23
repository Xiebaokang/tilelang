"""JSON serialization for handle-free, typed OverlapPlan candidates."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .ir import BufferPlan, GroupPlan, OperationPlacement, OverlapPlan, SyncEdge


def _integer(value: Any, field: str, *, optional: bool = False) -> int | None:
    if optional and value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    return value


def _objects(value: Any, field: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    if any(not isinstance(item, dict) for item in value):
        raise ValueError(f"every {field} entry must be an object")
    return value


def plan_from_dict(payload: Mapping[str, Any]) -> OverlapPlan:
    """Build a handle-free OverlapPlan from its native JSON representation."""

    groups = [
        GroupPlan(
            _integer(item.get("warp_count"), f"groups[{index}].warp_count"),
            _integer(
                item.get("register_count"),
                f"groups[{index}].register_count",
                optional=True,
            ),
            _integer(
                item.get("register_increase"),
                f"groups[{index}].register_increase",
                optional=True,
            ),
        )
        for index, item in enumerate(_objects(payload.get("groups"), "groups"))
    ]

    operations = [
        OperationPlacement(
            _integer(
                item.get("operation_id"),
                f"operations[{index}].operation_id",
            ),
            None,
            _integer(item.get("group_id"), f"operations[{index}].group_id"),
            _integer(
                item.get("stage"),
                f"operations[{index}].stage",
                optional=True,
            ),
            _integer(item.get("order"), f"operations[{index}].order"),
        )
        for index, item in enumerate(
            _objects(payload.get("operations"), "operations")
        )
    ]

    buffers = [
        BufferPlan(
            _integer(item.get("buffer_id"), f"buffers[{index}].buffer_id"),
            None,
            _integer(
                item.get("version_count"),
                f"buffers[{index}].version_count",
            ),
            _integer(
                item.get("communication"),
                f"buffers[{index}].communication",
            ),
            _integer(
                item.get("byte_offset"),
                f"buffers[{index}].byte_offset",
                optional=True,
            ),
        )
        for index, item in enumerate(
            _objects(payload.get("buffers"), "buffers")
        )
    ]

    sync_edges = [
        SyncEdge(
            _integer(item.get("producer_id"), f"sync_edges[{index}].producer_id"),
            _integer(item.get("consumer_id"), f"sync_edges[{index}].consumer_id"),
            _integer(
                item.get("buffer_id"),
                f"sync_edges[{index}].buffer_id",
                optional=True,
            ),
            _integer(item.get("kind"), f"sync_edges[{index}].kind"),
            _integer(item.get("scope"), f"sync_edges[{index}].scope"),
            _integer(
                item.get("iteration_distance"),
                f"sync_edges[{index}].iteration_distance",
            ),
            _integer(item.get("slot_count"), f"sync_edges[{index}].slot_count"),
            _integer(
                item.get("dependency_kind"),
                f"sync_edges[{index}].dependency_kind",
            ),
            _integer(
                item.get("completion_mode"),
                f"sync_edges[{index}].completion_mode",
            ),
            _integer(
                item.get("byte_offset"),
                f"sync_edges[{index}].byte_offset",
                optional=True,
            ),
        )
        for index, item in enumerate(
            _objects(payload.get("sync_edges"), "sync_edges")
        )
    ]

    return OverlapPlan(
        groups,
        operations,
        buffers,
        sync_edges,
        shared_arena_bytes=_integer(
            payload.get("shared_arena_bytes"),
            "shared_arena_bytes",
            optional=True,
        ),
    )


def plan_to_dict(plan: OverlapPlan) -> dict[str, Any]:
    """Return the native, handle-free JSON representation of ``plan``."""

    return {
        "groups": [
            {
                "warp_count": int(group.warp_count),
                "register_count": (
                    None
                    if group.register_count is None
                    else int(group.register_count)
                ),
                "register_increase": (
                    None
                    if group.register_increase is None
                    else int(group.register_increase)
                ),
            }
            for group in plan.groups
        ],
        "operations": [
            {
                "operation_id": int(operation.operation_id),
                "group_id": int(operation.group_id),
                "stage": (
                    None if operation.stage is None else int(operation.stage)
                ),
                "order": int(operation.order),
            }
            for operation in plan.operations
        ],
        "buffers": [
            {
                "buffer_id": int(buffer.buffer_id),
                "version_count": int(buffer.version_count),
                "communication": int(buffer.communication),
                "byte_offset": (
                    None
                    if buffer.byte_offset is None
                    else int(buffer.byte_offset)
                ),
            }
            for buffer in plan.buffers
        ],
        "sync_edges": [
            {
                "producer_id": int(edge.producer_id),
                "consumer_id": int(edge.consumer_id),
                "buffer_id": (
                    None if edge.buffer_id is None else int(edge.buffer_id)
                ),
                "kind": int(edge.kind),
                "scope": int(edge.scope),
                "iteration_distance": int(edge.iteration_distance),
                "slot_count": int(edge.slot_count),
                "dependency_kind": int(edge.dependency_kind),
                "completion_mode": int(edge.completion_mode),
                "byte_offset": (
                    None if edge.byte_offset is None else int(edge.byte_offset)
                ),
            }
            for edge in plan.sync_edges
        ],
        "shared_arena_bytes": (
            None
            if plan.shared_arena_bytes is None
            else int(plan.shared_arena_bytes)
        ),
    }


def load_plan_json(path: str | Path) -> OverlapPlan:
    """Load one native OverlapPlan candidate JSON file."""

    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("OverlapPlan JSON root must be an object")
    return plan_from_dict(payload)


def save_plan_json(path: str | Path, plan: OverlapPlan) -> None:
    """Atomically save one native OverlapPlan candidate JSON file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(plan_to_dict(plan), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
