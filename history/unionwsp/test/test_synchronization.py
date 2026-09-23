"""Build and print logical synchronization channels for FA3 schedules."""

import os
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "3rdparty" / "tvm" / "python"))
os.environ.setdefault("TVM_LIBRARY_PATH", str(PROJECT_ROOT / "build" / "lib"))

from history.unionwsp.schedule import (
    SynchronizationChannel,
    SynchronizationChannelKind,
    SynchronizationCycleError,
    SynchronizationScope,
    build_synchronization_channels,
    enumerate_buffer_version_plans,
    validate_synchronization_channels,
)


def enumerate_fa3_synchronization_plans():
    from history.unionwsp.test.test_order import enumerate_fa3_ordered_schedules

    graph, stage_assignments, schedules = enumerate_fa3_ordered_schedules()
    result_index = 0
    for schedule_index, (
        stage_index,
        group_index,
        stages_by_region,
        groups,
        orders,
    ) in enumerate(schedules, start=1):
        for version_plan_index, versions in enumerate(
            enumerate_buffer_version_plans(
                graph, stages_by_region, groups, orders
            ),
            start=1,
        ):
            result_index += 1
            try:
                channels = build_synchronization_channels(
                    graph,
                    stages_by_region,
                    groups,
                    orders,
                    versions,
                )
            except SynchronizationCycleError:
                # The public enumerate_wsp_schedules path performs the same
                # synchronization-feasibility pruning.  Keep this lower-level
                # test helper aligned with that behavior.
                continue
            yield (
                graph,
                stage_assignments,
                schedules,
                result_index,
                schedule_index,
                stage_index,
                group_index,
                version_plan_index,
                stages_by_region,
                groups,
                orders,
                versions,
                channels,
            )


def _format_channel(graph, channel: SynchronizationChannel) -> str:
    buffer_name = (
        "-"
        if channel.buffer_id is None
        else graph.buffer_for_id(channel.buffer_id).name
    )
    distance = (
        "-"
        if channel.effective_stage_distance is None
        else str(channel.effective_stage_distance)
    )
    return (
        f"{channel.kind.value}:"
        f"{channel.producer_id}(g{channel.producer_group})->"
        f"{channel.consumer_id}(g{channel.consumer_group})/"
        f"{buffer_name}/scope={channel.scope.value}/"
        f"iter={channel.iteration_distance}/eff={distance}/"
        f"slots={channel.slot_count}"
    )


def print_fa3_synchronization_plans() -> int:
    results = tuple(enumerate_fa3_synchronization_plans())
    if not results:
        raise AssertionError("FA3 must produce synchronization plans")
    graph, stage_assignments, schedules = results[0][:3]
    print(
        f"\nFA3 logical synchronization: "
        f"stage_assignments={len(stage_assignments)}, "
        f"schedules={len(schedules)}, plans={len(results)}"
    )
    for result in results:
        (
            _,
            _,
            _,
            result_index,
            schedule_index,
            stage_index,
            group_index,
            version_plan_index,
            _,
            _,
            _,
            versions,
            channels,
        ) = result
        multiversion = ", ".join(
            f"{graph.buffer_for_id(buffer_id).name}x{count}"
            for buffer_id, count in versions.items()
            if count > 1
        )
        channel_text = "; ".join(
            _format_channel(graph, channel) for channel in channels
        )
        print(
            f"result {result_index:04d}: schedule={schedule_index:03d}, "
            f"stage_assignment={stage_index:03d}, "
            f"group_assignment={group_index:02d}, "
            f"version_plan={version_plan_index:02d}, "
            f"multiversion=[{multiversion}], channels=[{channel_text}]"
        )
    return len(results)


def test_every_fa3_version_plan_has_complete_synchronization() -> None:
    distributions = Counter()
    total = 0
    for result in enumerate_fa3_synchronization_plans():
        (
            graph,
            stage_assignments,
            schedules,
            _,
            _,
            _,
            _,
            _,
            stages_by_region,
            groups,
            orders,
            versions,
            channels,
        ) = result
        validate_synchronization_channels(
            graph,
            stages_by_region,
            groups,
            orders,
            versions,
            channels,
        )
        forward_count = sum(
            channel.kind
            == SynchronizationChannelKind.FORWARD_DEPENDENCY
            for channel in channels
        )
        reuse_count = len(channels) - forward_count
        distributions[(forward_count, reuse_count)] += 1
        total += 1

    assert stage_assignments
    assert schedules
    assert total > 0
    assert sum(distributions.values()) == total


def test_fa3_k_v_handoff_has_forward_and_reuse_channels() -> None:
    expected = {
        (
            SynchronizationChannelKind.FORWARD_DEPENDENCY,
            SynchronizationScope.PER_ITERATION,
            4,
            6,
            6,
            1,
            1,
        ),
        (
            SynchronizationChannelKind.BUFFER_REUSE,
            SynchronizationScope.PER_ITERATION,
            6,
            4,
            6,
            0,
            1,
        ),
        (
            SynchronizationChannelKind.FORWARD_DEPENDENCY,
            SynchronizationScope.PER_ITERATION,
            17,
            18,
            13,
            1,
            1,
        ),
        (
            SynchronizationChannelKind.BUFFER_REUSE,
            SynchronizationScope.PER_ITERATION,
            18,
            17,
            13,
            0,
            1,
        ),
    }

    def selected_channels(result):
        return {
            (
                channel.kind,
                channel.scope,
                channel.producer_id,
                channel.consumer_id,
                channel.buffer_id,
                channel.effective_stage_distance,
                channel.slot_count,
            )
            for channel in result[12]
            if channel.buffer_id in {6, 13}
        }

    target = next(
        (
            result
            for result in enumerate_fa3_synchronization_plans()
            if all(version == 1 for version in result[11].values())
            and selected_channels(result) == expected
        ),
        None,
    )
    assert target is not None


def test_fa3_once_only_o_handoff_has_full_without_empty_channel() -> None:
    """O_shared is filled and stored once, so it needs no reuse notification."""
    found_once_handoff = False
    for result in enumerate_fa3_synchronization_plans():
        graph = result[0]
        channels = result[12]
        o_shared_id = next(
            buffer.buffer_id
            for buffer in graph.buffers
            if buffer.name == "O_shared"
        )
        o_channels = tuple(
            channel
            for channel in channels
            if channel.buffer_id == o_shared_id
        )
        if any(
            channel.kind
            == SynchronizationChannelKind.FORWARD_DEPENDENCY
            and channel.scope == SynchronizationScope.ONCE
            for channel in o_channels
        ):
            found_once_handoff = True
            assert all(
                channel.kind != SynchronizationChannelKind.BUFFER_REUSE
                for channel in o_channels
            )
            break

    assert found_once_handoff


if __name__ == "__main__":
    print_fa3_synchronization_plans()
