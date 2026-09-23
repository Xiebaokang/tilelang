"""Python handles for the typed OverlapPlan IR objects."""

from __future__ import annotations

import tvm_ffi
from tvm.ir.base import Node

from . import _ffi_api


@tvm_ffi.register_object("tl.overlap_plan.GroupPlan")
class GroupPlan(Node):
    def __init__(self, warp_count, register_count=None, register_increase=None):
        self.__init_handle_by_constructor__(
            _ffi_api.GroupPlan, int(warp_count), register_count, register_increase
        )


@tvm_ffi.register_object("tl.overlap_plan.OperationPlacement")
class OperationPlacement(Node):
    def __init__(self, operation_id, statement, group_id, stage, order):
        self.__init_handle_by_constructor__(
            _ffi_api.OperationPlacement,
            int(operation_id),
            statement,
            int(group_id),
            stage,
            int(order),
        )


@tvm_ffi.register_object("tl.overlap_plan.BufferPlan")
class BufferPlan(Node):
    def __init__(self, buffer_id, buffer, version_count, communication, byte_offset=None):
        self.__init_handle_by_constructor__(
            _ffi_api.BufferPlan,
            int(buffer_id),
            buffer,
            int(version_count),
            int(communication),
            byte_offset,
        )


@tvm_ffi.register_object("tl.overlap_plan.SyncEdge")
class SyncEdge(Node):
    def __init__(
        self,
        producer_id,
        consumer_id,
        buffer_id,
        kind,
        scope,
        iteration_distance,
        slot_count,
        dependency_kind,
        completion_mode,
        byte_offset=None,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.SyncEdge,
            int(producer_id),
            int(consumer_id),
            buffer_id,
            int(kind),
            int(scope),
            int(iteration_distance),
            int(slot_count),
            int(dependency_kind),
            int(completion_mode),
            byte_offset,
        )


@tvm_ffi.register_object("tl.overlap_plan.OverlapPlan")
class OverlapPlan(Node):
    def __init__(
        self,
        groups,
        operations,
        buffers,
        sync_edges,
        shared_arena_bytes=None,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.OverlapPlan,
            list(groups),
            list(operations),
            list(buffers),
            list(sync_edges),
            shared_arena_bytes,
        )
