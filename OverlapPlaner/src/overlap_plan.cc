/*!
 * \file overlap_plan.cc
 * \brief Typed OverlapPlan objects, derived lowering view, and validation.
 */

#include "overlap_plan.h"

#include <algorithm>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>

#include <tvm/ir/expr.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/logging.h>

#include "op/builtin.h"
#include "support/check.h"

namespace tvm {
namespace tl {
namespace overlap_plan {

namespace {

constexpr int64_t kRegisterFileBudget = 64512;
constexpr int64_t kMinRegisterCount = 24;
constexpr int64_t kMaxRegisterCount = 240;

int64_t RequiredInt(const ffi::Optional<Integer> &value, const char *field) {
  ICHECK(value.defined()) << field << " is required";
  return value.value()->value;
}

void CheckIndex(int64_t value, size_t upper_bound, const char *field) {
  ICHECK_GE(value, 0) << field << " cannot be negative";
  ICHECK_LT(value, static_cast<int64_t>(upper_bound))
      << field << " " << value << " is outside [0, " << upper_bound << ")";
}

void CheckPermutation(std::vector<int64_t> values, const char *field) {
  std::sort(values.begin(), values.end());
  for (size_t index = 0; index < values.size(); ++index) {
    ICHECK_EQ(values[index], static_cast<int64_t>(index))
        << field << " must be a dense permutation beginning at zero";
  }
}

} // namespace

GroupPlan::GroupPlan(int64_t warp_count, ffi::Optional<Integer> register_count,
                     ffi::Optional<Integer> register_increase) {
  auto node = ffi::make_object<GroupPlanNode>();
  node->warp_count = warp_count;
  node->register_count = std::move(register_count);
  node->register_increase = std::move(register_increase);
  data_ = std::move(node);
}

void GroupPlanNode::RegisterReflection() {
  namespace refl = ffi::reflection;
  refl::ObjectDef<GroupPlanNode>()
      .def_ro("warp_count", &GroupPlanNode::warp_count)
      .def_ro("register_count", &GroupPlanNode::register_count)
      .def_ro("register_increase", &GroupPlanNode::register_increase);
}

OperationPlacement::OperationPlacement(int64_t operation_id,
                                       ffi::Optional<tirx::Stmt> statement,
                                       int64_t group_id,
                                       ffi::Optional<Integer> stage,
                                       int64_t order) {
  auto node = ffi::make_object<OperationPlacementNode>();
  node->operation_id = operation_id;
  node->statement = std::move(statement);
  node->group_id = group_id;
  node->stage = std::move(stage);
  node->order = order;
  data_ = std::move(node);
}

void OperationPlacementNode::RegisterReflection() {
  namespace refl = ffi::reflection;
  refl::ObjectDef<OperationPlacementNode>()
      .def_ro("operation_id", &OperationPlacementNode::operation_id)
      .def_ro("statement", &OperationPlacementNode::statement)
      .def_ro("group_id", &OperationPlacementNode::group_id)
      .def_ro("stage", &OperationPlacementNode::stage)
      .def_ro("order", &OperationPlacementNode::order);
}

BufferPlan::BufferPlan(int64_t buffer_id, ffi::Optional<tirx::Buffer> buffer,
                       int64_t version_count, int64_t communication,
                       ffi::Optional<Integer> byte_offset) {
  auto node = ffi::make_object<BufferPlanNode>();
  node->buffer_id = buffer_id;
  node->buffer = std::move(buffer);
  node->version_count = version_count;
  node->communication = communication;
  node->byte_offset = std::move(byte_offset);
  data_ = std::move(node);
}

void BufferPlanNode::RegisterReflection() {
  namespace refl = ffi::reflection;
  refl::ObjectDef<BufferPlanNode>()
      .def_ro("buffer_id", &BufferPlanNode::buffer_id)
      .def_ro("buffer", &BufferPlanNode::buffer)
      .def_ro("version_count", &BufferPlanNode::version_count)
      .def_ro("communication", &BufferPlanNode::communication)
      .def_ro("byte_offset", &BufferPlanNode::byte_offset);
}

SyncEdge::SyncEdge(int64_t producer_id, int64_t consumer_id,
                   ffi::Optional<Integer> buffer_id, int64_t kind,
                   int64_t scope, int64_t iteration_distance,
                   int64_t slot_count, int64_t dependency_kind,
                   int64_t completion_mode,
                   ffi::Optional<Integer> byte_offset) {
  auto node = ffi::make_object<SyncEdgeNode>();
  node->producer_id = producer_id;
  node->consumer_id = consumer_id;
  node->buffer_id = std::move(buffer_id);
  node->kind = kind;
  node->scope = scope;
  node->iteration_distance = iteration_distance;
  node->slot_count = slot_count;
  node->dependency_kind = dependency_kind;
  node->completion_mode = completion_mode;
  node->byte_offset = std::move(byte_offset);
  data_ = std::move(node);
}

void SyncEdgeNode::RegisterReflection() {
  namespace refl = ffi::reflection;
  refl::ObjectDef<SyncEdgeNode>()
      .def_ro("producer_id", &SyncEdgeNode::producer_id)
      .def_ro("consumer_id", &SyncEdgeNode::consumer_id)
      .def_ro("buffer_id", &SyncEdgeNode::buffer_id)
      .def_ro("kind", &SyncEdgeNode::kind)
      .def_ro("scope", &SyncEdgeNode::scope)
      .def_ro("iteration_distance", &SyncEdgeNode::iteration_distance)
      .def_ro("slot_count", &SyncEdgeNode::slot_count)
      .def_ro("dependency_kind", &SyncEdgeNode::dependency_kind)
      .def_ro("completion_mode", &SyncEdgeNode::completion_mode)
      .def_ro("byte_offset", &SyncEdgeNode::byte_offset);
}

OverlapPlan::OverlapPlan(ffi::Array<GroupPlan> groups,
                         ffi::Array<OperationPlacement> operations,
                         ffi::Array<BufferPlan> buffers,
                         ffi::Array<SyncEdge> sync_edges,
                         ffi::Optional<Integer> shared_arena_bytes) {
  auto node = ffi::make_object<OverlapPlanNode>();
  node->groups = std::move(groups);
  node->operations = std::move(operations);
  node->buffers = std::move(buffers);
  node->sync_edges = std::move(sync_edges);
  node->shared_arena_bytes = std::move(shared_arena_bytes);
  data_ = std::move(node);
}

void OverlapPlanNode::RegisterReflection() {
  namespace refl = ffi::reflection;
  refl::ObjectDef<OverlapPlanNode>()
      .def_ro("groups", &OverlapPlanNode::groups)
      .def_ro("operations", &OverlapPlanNode::operations)
      .def_ro("buffers", &OverlapPlanNode::buffers)
      .def_ro("sync_edges", &OverlapPlanNode::sync_edges)
      .def_ro("shared_arena_bytes", &OverlapPlanNode::shared_arena_bytes);
}

ffi::Optional<OverlapPlan> GetOverlapPlan(const tirx::PrimFunc &func) {
  return func->GetAttr<OverlapPlan>(::tvm::tl::attr::kOverlapPlan);
}

bool HasAutoOverlap(const tirx::PrimFunc &func) {
  if (auto value = func->GetAttr<Integer>(::tvm::tl::attr::kAutoOverlap)) {
    return value.value()->value != 0;
  }
  return false;
}

void CheckOverlapPlanDenseIds(const OverlapPlan &plan) {
  for (size_t operation_id = 0; operation_id < plan->operations.size();
       ++operation_id) {
    ICHECK_EQ(plan->operations[operation_id]->operation_id,
              static_cast<int64_t>(operation_id))
        << "OverlapPlan operation_id must equal the operations array index; "
        << "got operation_id="
        << plan->operations[operation_id]->operation_id << " at index "
        << operation_id;
  }
  for (size_t buffer_id = 0; buffer_id < plan->buffers.size(); ++buffer_id) {
    ICHECK_EQ(plan->buffers[buffer_id]->buffer_id,
              static_cast<int64_t>(buffer_id))
        << "OverlapPlan buffer_id must equal the buffers array index; "
        << "got buffer_id=" << plan->buffers[buffer_id]->buffer_id
        << " at index " << buffer_id;
  }
}

LoweringView MakeLoweringView(const OverlapPlan &plan,
                              const OverlapIR &program_ir) {
  ICHECK(!plan->groups.empty()) << "OverlapPlan must contain at least one group";
  ICHECK_EQ(plan->operations.size(), program_ir.operations.size())
      << "OverlapPlan operation count does not match the input TIR";
  ICHECK_EQ(plan->buffers.size(), program_ir.buffers.size())
      << "OverlapPlan buffer count does not match the input TIR";
  CheckOverlapPlanDenseIds(plan);

  LoweringView view;
  view.group_warp_counts.reserve(plan->groups.size());
  view.group_first_warps.reserve(plan->groups.size());
  view.group_register_counts.assign(plan->groups.size(), -1);
  view.group_register_increase.assign(plan->groups.size(), 0);
  int64_t next_warp = 0;
  bool has_increase = false;
  bool has_decrease = false;
  int64_t allocated_registers = 0;
  for (size_t group_id = 0; group_id < plan->groups.size(); ++group_id) {
    const GroupPlanNode *group = plan->groups[group_id].get();
    ICHECK_GT(group->warp_count, 0) << "each group must receive at least one warp";
    view.group_first_warps.push_back(next_warp);
    view.group_warp_counts.push_back(group->warp_count);
    next_warp += group->warp_count;
    if (group->register_count.defined()) {
      int64_t register_count = RequiredInt(group->register_count, "register_count");
      int64_t is_increase =
          RequiredInt(group->register_increase, "register_increase");
      ICHECK_GE(register_count, kMinRegisterCount);
      ICHECK_LE(register_count, kMaxRegisterCount);
      ICHECK_EQ(register_count % 8, 0)
          << "setmaxnreg count must be a multiple of eight";
      ICHECK(is_increase == 0 || is_increase == 1)
          << "register_increase must be 0 or 1";
      view.group_register_counts[group_id] = register_count;
      view.group_register_increase[group_id] = is_increase;
      view.setmaxnreg_enabled = true;
      has_increase = has_increase || is_increase != 0;
      has_decrease = has_decrease || is_increase == 0;
      allocated_registers += register_count * group->warp_count * kWarpSize;
    } else {
      ICHECK(!group->register_increase.defined())
          << "register_increase requires register_count";
    }
  }
  view.effective_threads = next_warp * kWarpSize;
  if (view.setmaxnreg_enabled) {
    ICHECK(has_increase && has_decrease)
        << "setmaxnreg redistribution requires donor and receiver groups";
    ICHECK_LE(allocated_registers, kRegisterFileBudget)
        << "setmaxnreg plan exceeds the per-SM register-file budget";
    for (size_t group_id = 0; group_id < plan->groups.size(); ++group_id) {
      ICHECK_GE(view.group_register_counts[group_id], 0)
          << "setmaxnreg requires a register policy for every group";
    }
  }

  view.operation_groups.resize(plan->operations.size());
  view.operation_regions.resize(plan->operations.size());
  view.operation_stages.resize(plan->operations.size());
  view.operation_local_orders.resize(plan->operations.size());
  view.region_num_stages.assign(program_ir.regions.size(), 0);

  std::map<std::pair<int64_t, int64_t>, std::vector<int64_t>> local_orders;
  std::vector<bool> used_groups(plan->groups.size(), false);
  std::vector<int64_t> max_stage(program_ir.regions.size(), -1);
  for (size_t operation_id = 0; operation_id < plan->operations.size();
       ++operation_id) {
    const OperationPlacementNode *placement = plan->operations[operation_id].get();
    CheckIndex(placement->group_id, plan->groups.size(), "operation group");
    int64_t region_id = program_ir.operations[operation_id].region_id;
    CheckIndex(region_id, program_ir.regions.size(), "operation region");
    used_groups[placement->group_id] = true;
    view.operation_groups[operation_id] = placement->group_id;
    view.operation_regions[operation_id] = region_id;
    int64_t stage = -1;
    if (placement->stage.defined()) {
      stage = placement->stage.value()->value;
      ICHECK_GE(stage, 0) << "pipeline-region stages cannot be negative";
      max_stage[region_id] = std::max(max_stage[region_id], stage);
    }
    view.operation_stages[operation_id] = stage;
    ICHECK_GE(placement->order, 0) << "operation local order cannot be negative";
    view.operation_local_orders[operation_id] = placement->order;
    local_orders[{region_id, placement->group_id}].push_back(placement->order);
  }
  ICHECK(std::all_of(used_groups.begin(), used_groups.end(),
                     [](bool used) { return used; }))
      << "every physical group must own at least one operation";
  for (const auto &[region_group, orders] : local_orders) {
    CheckPermutation(orders, "operation order within one region/group");
  }
  for (size_t region_id = 0; region_id < program_ir.regions.size();
       ++region_id) {
    const bool is_pipeline =
        program_ir.regions[region_id].pipeline_loop.has_value();
    if (is_pipeline) {
      ICHECK_GE(max_stage[region_id], 0)
          << "T.Pipelined region " << region_id
          << " must receive staged operations in the OverlapPlan";
      view.region_num_stages[region_id] = max_stage[region_id] + 1;
      if (!program_ir.auto_overlap &&
          program_ir.regions[region_id].num_stages > 0) {
        ICHECK_EQ(program_ir.regions[region_id].num_stages,
                  view.region_num_stages[region_id])
            << "OverlapPlan stage count does not match TIR region "
            << region_id;
      }
    } else {
      ICHECK_LT(max_stage[region_id], 0)
          << "serial region " << region_id
          << " cannot receive pipeline stages";
      view.region_num_stages[region_id] = 0;
    }
  }

  view.buffer_versions.resize(plan->buffers.size());
  view.buffer_communications.resize(plan->buffers.size());
  view.buffer_requires_new_allocation.resize(plan->buffers.size());
  for (size_t buffer_id = 0; buffer_id < plan->buffers.size(); ++buffer_id) {
    const BufferPlanNode *buffer = plan->buffers[buffer_id].get();
    ICHECK_GE(buffer->version_count, 1)
        << "buffer version count must be positive";
    ICHECK_GE(buffer->communication, 0);
    ICHECK_LE(buffer->communication, 4) << "unknown buffer communication code";
    view.buffer_versions[buffer_id] = buffer->version_count;
    view.buffer_communications[buffer_id] = buffer->communication;
    view.buffer_requires_new_allocation[buffer_id] =
        buffer->version_count > 1 ||
        buffer->communication ==
            static_cast<int64_t>(BufferCommunication::kMaterializedShared);
  }

  const size_t channel_count = plan->sync_edges.size();
  view.sync_producers.resize(channel_count);
  view.sync_consumers.resize(channel_count);
  view.sync_producer_groups.resize(channel_count);
  view.sync_consumer_groups.resize(channel_count);
  view.sync_producer_regions.resize(channel_count);
  view.sync_consumer_regions.resize(channel_count);
  view.sync_iteration_distances.resize(channel_count);
  view.sync_slot_counts.resize(channel_count);
  view.sync_scopes.resize(channel_count);
  view.sync_kinds.resize(channel_count);
  view.sync_buffer_ids.resize(channel_count);
  view.sync_dependency_masks.resize(channel_count);
  view.sync_completion_modes.resize(channel_count);
  view.sync_event_owners.resize(channel_count);

  for (size_t channel = 0; channel < channel_count; ++channel) {
    const SyncEdgeNode *edge = plan->sync_edges[channel].get();
    CheckIndex(edge->producer_id, plan->operations.size(), "sync producer");
    CheckIndex(edge->consumer_id, plan->operations.size(), "sync consumer");
    view.sync_producers[channel] = edge->producer_id;
    view.sync_consumers[channel] = edge->consumer_id;
    view.sync_producer_groups[channel] =
        view.operation_groups[edge->producer_id];
    view.sync_consumer_groups[channel] =
        view.operation_groups[edge->consumer_id];
    view.sync_producer_regions[channel] =
        view.operation_regions[edge->producer_id];
    view.sync_consumer_regions[channel] =
        view.operation_regions[edge->consumer_id];
    ICHECK_GE(edge->iteration_distance, 0)
        << "sync iteration distance cannot be negative";
    view.sync_iteration_distances[channel] = edge->iteration_distance;
    ICHECK_GT(edge->slot_count, 0) << "sync slot count must be positive";
    view.sync_slot_counts[channel] = edge->slot_count;
    ICHECK_GE(edge->scope, 0);
    ICHECK_LE(edge->scope, 2) << "unknown sync scope code";
    view.sync_scopes[channel] = edge->scope;
    ICHECK_GE(edge->kind, 0);
    ICHECK_LE(edge->kind, 1) << "unknown sync kind code";
    view.sync_kinds[channel] = edge->kind;
    int64_t buffer_id = -1;
    if (edge->buffer_id.defined()) {
      buffer_id = edge->buffer_id.value()->value;
      CheckIndex(buffer_id, plan->buffers.size(), "sync buffer ID");
    }
    view.sync_buffer_ids[channel] = buffer_id;
    ICHECK((edge->dependency_kind >= 1 && edge->dependency_kind <= 7) ||
           edge->dependency_kind == 8)
        << "unknown synchronization dependency mask " << edge->dependency_kind;
    view.sync_dependency_masks[channel] = edge->dependency_kind;
    ICHECK(edge->completion_mode == 0 || edge->completion_mode == 1)
        << "unknown synchronization completion mode " << edge->completion_mode;
    view.sync_completion_modes[channel] = edge->completion_mode;
    if (edge->completion_mode == 1) {
      ICHECK_EQ(edge->kind, 0)
          << "only a forward dependency may use transaction completion";
      ICHECK_GE(buffer_id, 0)
          << "transaction completion requires a buffer-backed channel";
    }
  }

  std::map<int64_t, size_t> transaction_producers;
  for (size_t channel = 0; channel < channel_count; ++channel) {
    view.sync_event_owners[channel] = static_cast<int64_t>(channel);
    if (view.sync_completion_modes[channel] != 1) {
      continue;
    }
    auto [it, inserted] =
        transaction_producers.emplace(view.sync_producers[channel], channel);
    if (inserted) {
      continue;
    }
    size_t owner = it->second;
    ICHECK_EQ(view.sync_buffer_ids[channel], view.sync_buffer_ids[owner])
        << "one transaction producer cannot complete barriers for different "
           "buffers";
    ICHECK_EQ(view.sync_scopes[channel], view.sync_scopes[owner])
        << "one transaction producer cannot use different synchronization "
           "scopes";
    ICHECK_EQ(view.sync_iteration_distances[channel],
              view.sync_iteration_distances[owner])
        << "one transaction producer cannot use different iteration "
           "distances";
    ICHECK_EQ(view.sync_slot_counts[channel], view.sync_slot_counts[owner])
        << "one transaction producer cannot use different barrier slot "
           "counts";
    view.sync_event_owners[channel] = static_cast<int64_t>(owner);
  }
  return view;
}

void ValidateLoweringView(const LoweringView &plan, const OverlapIR &program_ir) {
  ICHECK_EQ(plan.operation_groups.size(), program_ir.operations.size());
  ICHECK_EQ(plan.region_num_stages.size(), program_ir.regions.size());
  ICHECK_EQ(plan.buffer_versions.size(), program_ir.buffers.size());
  ICHECK_EQ(program_ir.operation_buffer_ids.size(),
            program_ir.operations.size());
  for (size_t operation_id = 0; operation_id < program_ir.operations.size();
       ++operation_id) {
    ICHECK_EQ(plan.operation_regions[operation_id],
              program_ir.operations[operation_id].region_id)
        << "OverlapPlan region ID does not match TIR operation "
        << operation_id;
  }
  for (size_t region_id = 0; region_id < program_ir.regions.size();
       ++region_id) {
    const bool is_pipeline =
        program_ir.regions[region_id].pipeline_loop.has_value();
    if (is_pipeline) {
      ICHECK_GT(plan.region_num_stages[region_id], 0)
          << "T.Pipelined region " << region_id
          << " must receive staged operations";
      if (!program_ir.auto_overlap &&
          program_ir.regions[region_id].num_stages > 0) {
        ICHECK_EQ(plan.region_num_stages[region_id],
                  program_ir.regions[region_id].num_stages)
            << "OverlapPlan stage count does not match TIR region "
            << region_id;
      }
    } else {
      ICHECK_EQ(plan.region_num_stages[region_id], 0)
          << "serial region " << region_id << " cannot receive pipeline stages";
    }
  }
}

tirx::PrimFunc AttachSharedMemoryAttrs(tirx::PrimFunc func,
                                       const OverlapPlan &plan,
                                       const OverlapIR &program_ir) {
  ICHECK_EQ(plan->buffers.size(), program_ir.buffers.size());
  ffi::Map<ffi::String, IntImm> offset_map;
  int64_t arena_bytes = 0;
  bool has_offset = false;
  for (size_t buffer_id = 0; buffer_id < plan->buffers.size(); ++buffer_id) {
    const BufferPlan &buffer_plan = plan->buffers[buffer_id];
    if (!buffer_plan->byte_offset.defined()) {
      continue;
    }
    has_offset = true;
    int64_t offset = buffer_plan->byte_offset.value()->value;
    ICHECK_GE(offset, 0);
    const tirx::Buffer &buffer = program_ir.buffers[buffer_id];
    offset_map.Set(ffi::String(buffer->name),
                   IntImm(DataType::Int(64), offset));
    offset_map.Set(ffi::String(buffer->data->name_hint),
                   IntImm(DataType::Int(64), offset));
    arena_bytes = std::max(arena_bytes, offset);
  }
  for (size_t channel = 0; channel < plan->sync_edges.size(); ++channel) {
    const SyncEdge &edge = plan->sync_edges[channel];
    if (!edge->byte_offset.defined() || !edge->buffer_id.defined()) {
      continue;
    }
    has_offset = true;
    int64_t offset = edge->byte_offset.value()->value;
    ICHECK_GE(offset, 0);
    int64_t buffer_id = edge->buffer_id.value()->value;
    ICHECK_GE(buffer_id, 0);
    ICHECK_LT(buffer_id, static_cast<int64_t>(program_ir.buffers.size()));
    const tirx::Buffer &buffer = program_ir.buffers[buffer_id];
    offset_map.Set(ffi::String(FragmentHandoffName(std::string(buffer->name),
                                                   channel)),
                   IntImm(DataType::Int(64), offset));
    offset_map.Set(
        ffi::String(FragmentHandoffName(
            std::string(buffer->data->name_hint), channel)),
        IntImm(DataType::Int(64), offset));
    arena_bytes = std::max(arena_bytes, offset);
  }
  if (!has_offset) {
    return func;
  }
  int64_t planned_arena = arena_bytes;
  if (plan->shared_arena_bytes.defined()) {
    planned_arena = plan->shared_arena_bytes.value()->value;
    ICHECK_GE(planned_arena, arena_bytes)
        << "shared_arena_bytes is smaller than the largest planned offset";
  }
  func = WithAttr(std::move(func), ffi::String(::tvm::tl::kSmemOffsetMap),
                  offset_map);
  func = WithAttr(std::move(func), ffi::String(::tvm::tl::kSmemPlannedArenaBytes),
                  Integer(planned_arena));
  return func;
}

} // namespace overlap_plan
} // namespace tl
} // namespace tvm

namespace tvm {
namespace tl {
namespace overlap_plan {

TVM_FFI_STATIC_INIT_BLOCK() {
  GroupPlanNode::RegisterReflection();
  OperationPlacementNode::RegisterReflection();
  BufferPlanNode::RegisterReflection();
  SyncEdgeNode::RegisterReflection();
  OverlapPlanNode::RegisterReflection();

  namespace refl = ffi::reflection;
  refl::GlobalDef()
      .def("tl.overlap_plan.GroupPlan",
           [](int64_t warp_count, ffi::Optional<Integer> register_count,
              ffi::Optional<Integer> register_increase) {
             return GroupPlan(warp_count, std::move(register_count),
                              std::move(register_increase));
           })
      .def("tl.overlap_plan.OperationPlacement",
           [](int64_t operation_id, ffi::Optional<tirx::Stmt> statement,
              int64_t group_id, ffi::Optional<Integer> stage, int64_t order) {
             return OperationPlacement(operation_id, std::move(statement),
                                       group_id, std::move(stage), order);
           })
      .def("tl.overlap_plan.BufferPlan",
           [](int64_t buffer_id, ffi::Optional<tirx::Buffer> buffer,
              int64_t version_count, int64_t communication,
              ffi::Optional<Integer> byte_offset) {
             return BufferPlan(buffer_id, std::move(buffer), version_count,
                               communication, std::move(byte_offset));
           })
      .def("tl.overlap_plan.SyncEdge",
           [](int64_t producer_id, int64_t consumer_id,
              ffi::Optional<Integer> buffer_id, int64_t kind, int64_t scope,
              int64_t iteration_distance, int64_t slot_count,
              int64_t dependency_kind, int64_t completion_mode,
              ffi::Optional<Integer> byte_offset) {
             return SyncEdge(producer_id, consumer_id, std::move(buffer_id),
                             kind, scope, iteration_distance, slot_count,
                             dependency_kind, completion_mode,
                             std::move(byte_offset));
           })
      .def("tl.overlap_plan.OverlapPlan",
           [](ffi::Array<GroupPlan> groups,
              ffi::Array<OperationPlacement> operations,
              ffi::Array<BufferPlan> buffers, ffi::Array<SyncEdge> sync_edges,
              ffi::Optional<Integer> shared_arena_bytes) {
             return OverlapPlan(std::move(groups), std::move(operations),
                                std::move(buffers),
                                std::move(sync_edges),
                                std::move(shared_arena_bytes));
           });
}

} // namespace overlap_plan
} // namespace tl
} // namespace tvm
