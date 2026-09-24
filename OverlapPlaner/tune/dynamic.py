"""Measurement-driven selection from a resource-feasible candidate pool."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from OverlapPlaner.serialization import load_plan_json, plan_to_dict


@dataclass(frozen=True, slots=True)
class Candidate:
    """A serialized plan and the schedule-only features used for selection."""

    index: int
    path: Path
    fingerprint: str
    bucket: tuple[int, int]
    features: tuple[float, ...]
    producer_copies: int = 0
    order_bucket: int = 0
    physical_bucket: tuple[tuple[int, int], ...] = ()
    seed_priority: float = 0.0


def _plan_payload(path: Path) -> dict[str, Any]:
    return plan_to_dict(load_plan_json(path))


def file_fingerprint(path: str | Path) -> str:
    """Fingerprint a plan semantically, with a raw fallback for diagnostics."""

    path = Path(path)
    try:
        return plan_fingerprint(_plan_payload(path))
    except (KeyError, TypeError, ValueError):
        return hashlib.sha256(path.read_bytes()).hexdigest()


def plan_fingerprint(payload: dict[str, Any]) -> str:
    """Return a stable identity independent of schedule file numbering."""

    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def workload_plan_fingerprint(
    plan_identity: str, workload_fingerprint: str
) -> str:
    """Combine a semantic plan identity with its concrete workload."""

    return plan_fingerprint(
        {"plan": plan_identity, "workload": workload_fingerprint}
    )


def plan_features(payload: dict[str, Any]) -> tuple[float, ...]:
    """Extract latency-agnostic structural and resource features."""

    groups = payload["groups"]
    operations = payload["operations"]
    buffers = payload["buffers"]
    sync_edges = payload["sync_edges"]
    stages = [item["stage"] or 0 for item in operations]
    warps = [item["warp_count"] for item in groups]
    registers = [item["register_count"] or 0 for item in groups]
    versions = [item["version_count"] for item in buffers]
    communications = [item["communication"] for item in buffers]
    slots = [item["slot_count"] for item in sync_edges]
    group_count = len(groups)
    stage_depth = max(stages, default=0) + 1
    return (
        float(group_count),
        float(stage_depth),
        float(sum(stage > 0 for stage in stages)),
        float(len(operations)),
        float(sum(warps)),
        float(min(warps, default=0)),
        float(max(warps, default=0)),
        float(sum(registers)),
        float(max(registers, default=0)),
        float(payload.get("shared_arena_bytes") or 0),
        float(len(sync_edges)),
        float(sum(slots)),
        float(max(slots, default=0)),
        float(sum(versions)),
        float(max(versions, default=0)),
        float(sum(value == 4 for value in communications)),
    )


def classified_plan_features(payload: dict[str, Any], classified) -> tuple[float, ...]:
    """Describe which graph edges and engine work a plan separates."""

    placements = {
        int(item["operation_id"]): item for item in payload["operations"]
    }
    graph = classified.graph
    cross_group = 0
    cross_stage = 0
    memory_compute = 0
    cross_engine = 0
    async_cuts = 0
    fragment_handoff_bytes = 0
    for edge in graph.edges:
        producer = placements[edge.producer_id]
        consumer = placements[edge.consumer_id]
        group_cut = producer["group_id"] != consumer["group_id"]
        producer_stage = producer["stage"]
        consumer_stage = consumer["stage"]
        stage_cut = (
            producer_stage is not None
            and consumer_stage is not None
            and producer_stage != consumer_stage
        )
        if not group_cut and not stage_cut:
            continue
        producer_traits = classified.traits_for(edge.producer_id)
        consumer_traits = classified.traits_for(edge.consumer_id)
        cross_group += int(group_cut)
        cross_stage += int(stage_cut)
        memory_compute += int(
            producer_traits.kind != consumer_traits.kind
        )
        cross_engine += int(producer_traits.engine != consumer_traits.engine)
        async_cuts += int(producer_traits.async_completion)
        buffer = graph.buffer_for_id(edge.buffer_id)
        if group_cut and buffer.scope == "local.fragment":
            fragment_handoff_bytes += buffer.nbytes or 0

    group_count = len(payload["groups"])
    gemm_work = [0.0] * group_count
    copy_bytes = [0.0] * group_count
    for node in graph.nodes:
        group_id = int(placements[node.node_id]["group_id"])
        if node.gemm is not None and None not in (
            node.gemm.m,
            node.gemm.n,
            node.gemm.k,
        ):
            gemm_work[group_id] += float(
                2 * node.gemm.m * node.gemm.n * node.gemm.k
            )
        if node.kind.value == "copy":
            sizes = [
                graph.buffer_for_id(buffer_id).nbytes
                for buffer_id in (*node.reads, *node.writes)
            ]
            static_sizes = [size for size in sizes if size is not None]
            copy_bytes[group_id] += float(min(static_sizes, default=0))

    def imbalance(values: list[float]) -> float:
        total = sum(values)
        return 0.0 if total == 0 else max(values, default=0.0) / total

    # Order is a search dimension: two placements with identical stages,
    # groups, buffers, and synchronization can still execute very differently.
    local_orders: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for node in graph.nodes:
        placement = placements[node.node_id]
        local_orders.setdefault(
            (node.region_id, int(placement["group_id"])), []
        ).append((int(placement["order"]), node.node_id))
    inversions = 0
    weighted_order = 0
    weighted_order_square = 0
    async_position = 0.0
    tensorcore_position = 0.0
    for sequence in local_orders.values():
        sequence.sort()
        node_ids = [node_id for _, node_id in sequence]
        inversions += sum(
            left > right
            for offset, left in enumerate(node_ids)
            for right in node_ids[offset + 1 :]
        )
        for position, node_id in enumerate(node_ids):
            weighted_order += (node_id + 1) * position
            weighted_order_square += (node_id + 1) ** 2 * position
            normalized = position / max(1, len(node_ids) - 1)
            traits = classified.traits_for(node_id)
            async_position += normalized * int(traits.async_completion)
            tensorcore_position += normalized * int(traits.engine == "tensorcore")

    return (
        float(cross_group),
        float(cross_stage),
        float(memory_compute),
        float(cross_engine),
        float(async_cuts),
        float(fragment_handoff_bytes),
        float(sum(gemm_work)),
        float(max(gemm_work, default=0.0)),
        imbalance(gemm_work),
        float(sum(copy_bytes)),
        float(max(copy_bytes, default=0.0)),
        imbalance(copy_bytes),
        float(inversions),
        float(weighted_order),
        float(weighted_order_square),
        async_position,
        tensorcore_position,
        float(sum(edge["completion_mode"] == 1 for edge in payload["sync_edges"])),
    )


def producer_copy_count(payload: dict[str, Any], classified) -> int:
    """Count in-loop async shared producers separated from their consumers."""

    placements = {
        int(item["operation_id"]): int(item["group_id"])
        for item in payload["operations"]
    }
    graph = classified.graph
    compute_groups = {
        placements[node.node_id]
        for node in graph.nodes
        if node.gemm is not None
    }
    counted = set()
    for edge in graph.edges:
        producer = graph.node_for_id(edge.producer_id)
        if producer.node_id in counted or producer.kind.value != "copy":
            continue
        if graph.region_for_id(producer.region_id).kind.value != "pipeline":
            continue
        if not classified.traits_for(producer.node_id).async_completion:
            continue
        if not graph.buffer_for_id(edge.buffer_id).scope.startswith("shared"):
            continue
        producer_group = placements[producer.node_id]
        if (
            producer_group not in compute_groups
            and producer_group != placements[edge.consumer_id]
        ):
            counted.add(producer.node_id)
    return len(counted)


def order_bucket(features: tuple[float, ...]) -> int:
    """Coarsely distinguish source-like and reordered issue sequences."""

    # plan_features has 16 fields; classified_plan_features' inversion count
    # is its thirteenth field. Old feature files may contain only the prefix.
    inversion_index = 16 + 12
    return int(len(features) > inversion_index and features[inversion_index] > 0)


def _physical_group_signature(
    payload: dict[str, Any],
) -> tuple[tuple[int, int], ...]:
    """Preserve physical group order and its register-allocation role."""

    return tuple(
        (
            int(group["warp_count"]),
            -1
            if group.get("register_increase") is None
            else int(group["register_increase"]),
        )
        for group in payload["groups"]
    )


def physical_plan_features(payload: dict[str, Any]) -> tuple[float, ...]:
    """Describe the physical order of the first and last warp groups."""

    signature = _physical_group_signature(payload)
    first_warps, first_role = signature[0]
    last_warps, last_role = signature[-1]
    return tuple(
        float(value)
        for value in (first_warps, first_role, last_warps, last_role)
    )


def _native_role_seed_priority(
    payload: dict[str, Any], classified: Any | None
) -> float:
    """Prefer native-like producer/consumer roles within one coverage bucket."""

    if classified is None:
        return 0.0
    placements = {
        int(operation["operation_id"]): int(operation["group_id"])
        for operation in payload["operations"]
    }
    roles = [group.get("register_increase") for group in payload["groups"]]
    priority = 0.0
    for node in classified.graph.nodes:
        role = roles[placements[node.node_id]]
        traits = classified.traits_for(node.node_id)
        if traits.async_completion and node.kind.value == "copy":
            priority += 2.0 if role == 0 else -2.0
        if node.gemm is not None:
            priority += 1.0 if role == 1 else -1.0
    return priority


def load_candidates(
    plan_dir: str | Path,
    *,
    fingerprint_salt: str | None = None,
    classified: Any | None = None,
) -> tuple[Candidate, ...]:
    """Load a deterministic candidate pool and verify unique identities."""

    plan_dir = Path(plan_dir)
    feature_rows = {}
    feature_path = plan_dir / "features.jsonl"
    if feature_path.is_file():
        feature_rows = {
            int(row["schedule_index"]): row
            for row in (
                json.loads(line)
                for line in feature_path.read_text(encoding="utf-8").splitlines()
                if line
            )
        }
    candidates = []
    fingerprints = set()
    for fallback, path in enumerate(sorted(plan_dir.glob("schedule_*.json"))):
        suffix = path.stem.removeprefix("schedule_")
        index = int(suffix) if suffix.isdigit() else fallback
        payload = _plan_payload(path)
        plan_identity = plan_fingerprint(payload)
        if plan_identity in fingerprints:
            continue
        fingerprints.add(plan_identity)
        fingerprint = (
            plan_identity
            if fingerprint_salt is None
            else workload_plan_fingerprint(plan_identity, fingerprint_salt)
        )
        row = feature_rows.get(index, {})
        stored_features = (
            row.get("features")
            if row.get("fingerprint") == plan_identity
            else None
        )
        features = tuple(
            float(value)
            for value in (
                stored_features
                if stored_features is not None
                else plan_features(payload)
            )
        )
        # Previous feature files lacked the completion-mode dimension.  Keep
        # resumable candidate pools compatible with the new copy search.
        if len(features) == 33:
            features += (
                float(sum(edge["completion_mode"] == 1 for edge in payload["sync_edges"])),
            )
        physical_bucket = _physical_group_signature(payload)
        # Aggregate resource features previously made physical permutations
        # such as [producer, consumer] and [consumer, producer] identical to
        # the learned model.  Add a fixed-size description of both ends; the
        # complete signature remains part of coverage bucketing below.
        features += physical_plan_features(payload)
        candidates.append(
            Candidate(
                index=index,
                path=path,
                fingerprint=fingerprint,
                bucket=(int(features[0]), int(features[1])),
                features=features,
                producer_copies=(
                    int(row.get("producer_copies", 0))
                    if row.get("fingerprint") == plan_identity
                    else 0
                ),
                order_bucket=order_bucket(features),
                physical_bucket=physical_bucket,
                seed_priority=_native_role_seed_priority(payload, classified),
            )
        )
    return tuple(sorted(candidates, key=lambda item: item.index))


class DynamicSearchPolicy:
    """Select diverse seeds, then learn priorities from measured latency."""

    def __init__(
        self,
        candidates: tuple[Candidate, ...],
        *,
        seed: int = 0,
        exploration: float = 0.35,
        ensemble_size: int = 16,
    ) -> None:
        if not candidates:
            raise ValueError("dynamic search requires at least one candidate")
        if exploration < 0 or ensemble_size < 2:
            raise ValueError("invalid dynamic search policy parameters")
        self.candidates = candidates
        self.seed = seed
        self.exploration = exploration
        self.ensemble_size = ensemble_size
        self.last_selection_roles: dict[int, str] = {}

    def initial(self, measured: set[int], limit: int) -> list[int]:
        """Round-robin over group, stage, producer, and order buckets."""

        buckets: dict[
            tuple[int, int, int, int, tuple[tuple[int, int], ...]], list[int]
        ] = {}
        for candidate in self.candidates:
            if candidate.index not in measured:
                buckets.setdefault(self._coverage_bucket(candidate), []).append(
                    candidate.index
                )
        by_index = {candidate.index: candidate for candidate in self.candidates}
        for bucket in buckets.values():
            bucket.sort(
                key=lambda index: (-by_index[index].seed_priority, index)
            )
        selected = []
        keys = sorted(buckets, key=lambda key: (key[0], key[1], -key[2], key[3]))
        while keys and len(selected) < limit:
            remaining = []
            for key in keys:
                if buckets[key] and len(selected) < limit:
                    selected.append(buckets[key].pop(0))
                if buckets[key]:
                    remaining.append(key)
            keys = remaining
        return selected

    @staticmethod
    def _coverage_bucket(
        candidate: Candidate,
    ) -> tuple[int, int, int, int, tuple[tuple[int, int], ...]]:
        return (
            *candidate.bucket,
            min(candidate.producer_copies, 2),
            candidate.order_bucket,
            candidate.physical_bucket,
        )

    def has_uncovered_bucket(
        self, successful: set[int], unavailable: set[int] | None = None
    ) -> bool:
        """Return whether an untried candidate can cover an unmeasured bucket."""

        unavailable = unavailable or set()
        covered = {
            self._coverage_bucket(candidate)
            for candidate in self.candidates
            if candidate.index in successful
        }
        return any(
            candidate.index not in unavailable
            and self._coverage_bucket(candidate) not in covered
            for candidate in self.candidates
        )

    def next_batch(
        self,
        measured_latency: dict[int, float],
        unavailable: set[int],
        limit: int,
    ) -> list[int]:
        """Mix predicted winners with uncertainty and soft structural coverage."""

        self.last_selection_roles = {}
        excluded = set(measured_latency) | unavailable
        remaining = [
            item for item in self.candidates if item.index not in excluded
        ]
        if not remaining or limit < 1:
            return []
        if len(measured_latency) < 4:
            selected = self.initial(excluded, min(limit, len(remaining)))
            self.last_selection_roles = {
                index: "coverage" for index in selected
            }
            return selected

        by_index = {item.index: item for item in self.candidates}
        train_ids = sorted(measured_latency)
        x_train = np.asarray(
            [by_index[index].features for index in train_ids], dtype=np.float64
        )
        y_train = np.log(
            np.asarray(
                [measured_latency[index] for index in train_ids],
                dtype=np.float64,
            )
        )
        x_test = np.asarray([item.features for item in remaining], dtype=np.float64)
        known_ids = [
            item.index
            for item in self.candidates
            if item.index in measured_latency or item.index in unavailable
        ]
        x_known = np.asarray(
            [by_index[index].features for index in known_ids], dtype=np.float64
        )
        mean = x_known.mean(axis=0)
        scale = x_known.std(axis=0)
        scale[scale < 1e-9] = 1.0
        x_train = (x_train - mean) / scale
        x_test = (x_test - mean) / scale
        x_known = (x_known - mean) / scale
        x_test_features = x_test
        x_train = np.column_stack((np.ones(len(x_train)), x_train))
        x_test = np.column_stack((np.ones(len(x_test)), x_test))

        rng = np.random.default_rng(self.seed + len(train_ids))
        predictions = []
        regularizer = np.eye(x_train.shape[1], dtype=np.float64) * 1e-3
        regularizer[0, 0] = 0.0
        for _ in range(self.ensemble_size):
            sample = rng.integers(0, len(x_train), size=len(x_train))
            sampled_x = x_train[sample]
            sampled_y = y_train[sample]
            weights = np.linalg.pinv(
                sampled_x.T @ sampled_x + regularizer
            ) @ sampled_x.T @ sampled_y
            predictions.append(x_test @ weights)
        prediction = np.asarray(predictions)
        predicted_mean = prediction.mean(axis=0)
        predicted_std = prediction.std(axis=0)

        # Failed schedules are useful measurements too. Estimate local
        # infeasibility from nearby successful and failed structures, then
        # penalize regions dominated by compilation, correctness, or timeout
        # failures. The beta prior prevents one isolated failure from banning
        # an otherwise unexplored neighborhood.
        known_failed = np.asarray(
            [float(index in unavailable) for index in known_ids],
            dtype=np.float64,
        )
        invalid_probability = np.zeros(len(remaining), dtype=np.float64)
        neighbor_count = min(8, len(known_ids))
        distance_scale = math.sqrt(max(1, x_known.shape[1]))
        for offset, sample in enumerate(x_test_features):
            distances = np.linalg.norm(x_known - sample, axis=1)
            nearest = np.argsort(distances)[:neighbor_count]
            weights = np.exp(-distances[nearest] / distance_scale)
            invalid_probability[offset] = (
                0.5 + float(weights @ known_failed[nearest])
            ) / (2.0 + float(weights.sum()))

        covered = {
            self._coverage_bucket(item)
            for item in self.candidates
            if item.index in measured_latency
        }
        uncovered = np.asarray(
            [self._coverage_bucket(item) not in covered for item in remaining],
            dtype=np.float64,
        )
        acquisition = (
            predicted_mean
            - self.exploration * predicted_std
            + 0.5 * invalid_probability
        )

        def ranked(offsets, key):
            return sorted(
                offsets,
                key=lambda offset: (
                    key(offset),
                    remaining[offset].bucket,
                    remaining[offset].index,
                ),
            )

        available_offsets = set(range(len(remaining)))
        selected_offsets: list[int] = []
        exploit_count = min(limit, max(1, (limit + 1) // 2))
        for offset in ranked(available_offsets, lambda item: acquisition[item]):
            selected_offsets.append(offset)
            available_offsets.remove(offset)
            self.last_selection_roles[remaining[offset].index] = "exploit"
            if len(selected_offsets) >= exploit_count:
                break

        exploration_slots = min(limit - len(selected_offsets), len(available_offsets))
        for slot in range(exploration_slots):
            uncovered_offsets = [
                offset for offset in available_offsets if uncovered[offset]
            ]
            if slot == exploration_slots - 1 and uncovered_offsets:
                offset = ranked(
                    uncovered_offsets, lambda item: acquisition[item]
                )[0]
                role = "coverage"
            else:
                offset = ranked(
                    available_offsets,
                    lambda item: (
                        -predicted_std[item] + 0.5 * invalid_probability[item]
                    ),
                )[0]
                role = "uncertainty"
            selected_offsets.append(offset)
            available_offsets.remove(offset)
            self.last_selection_roles[remaining[offset].index] = role

        return [remaining[offset].index for offset in selected_offsets]


def should_stop(
    best_history: list[float],
    *,
    patience: int,
    minimum_improvement: float,
) -> bool:
    """Stop when recent batches do not improve the prior best enough."""

    if patience < 1 or len(best_history) <= patience:
        return False
    previous = min(best_history[: -patience])
    recent = min(best_history[-patience:])
    improvement = (previous - recent) / previous
    return math.isfinite(improvement) and improvement < minimum_improvement
