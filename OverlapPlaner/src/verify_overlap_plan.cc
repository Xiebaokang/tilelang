/*!
 * \file verify_overlap_plan.cc
 * \brief Structural and semantic verifier for a materialized OverlapPlan.
 *
 * Checks group structure plus the buffer, sync, pipeline, and shared-offset
 * artifacts that LowerOverlapPlan emits, including planned values rather than
 * mere presence of annotations.
 */

#include "overlap_plan.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include <tvm/ir/expr.h>
#include <tvm/ir/op.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include "op/builtin.h"
#include "support/check.h"
#include "transform/common/pipeline_utils.h"

namespace tvm {
namespace tl {
namespace overlap_plan {

namespace {

using VarSet =
    std::unordered_set<tirx::Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

ffi::Array<tirx::Stmt> FlattenSequence(const tirx::Stmt &body) {
  ffi::Array<tirx::Stmt> result;
  if (const auto *sequence = body.as<tirx::SeqStmtNode>()) {
    for (const tirx::Stmt &child : sequence->seq) {
      ffi::Array<tirx::Stmt> nested = FlattenSequence(child);
      result.insert(result.end(), nested.begin(), nested.end());
    }
  } else {
    result.push_back(body);
  }
  return result;
}

class OperationMarkerCollector : public tirx::StmtExprVisitor {
public:
  static std::vector<int64_t> Collect(const tirx::Stmt &statement,
                                      int64_t expected_group) {
    OperationMarkerCollector collector(expected_group);
    collector(statement);
    return std::move(collector.operation_ids_);
  }

private:
  explicit OperationMarkerCollector(int64_t expected_group)
      : expected_group_(expected_group) {}

  void VisitStmt_(const tirx::AttrStmtNode *op) final {
    if (op->attr_key == kOperationScope) {
      const auto *operation_id = op->node.as<IntImmNode>();
      const auto *group_id = op->value.as<IntImmNode>();
      ICHECK(operation_id != nullptr && group_id != nullptr);
      ICHECK_EQ(group_id->value, expected_group_)
          << "operation marker is nested in the wrong group branch";
      operation_ids_.push_back(operation_id->value);
      return;
    }
    tirx::StmtExprVisitor::VisitStmt_(op);
  }

  int64_t expected_group_;
  std::vector<int64_t> operation_ids_;
};

class LoweredScheduleVerifier : public tirx::StmtExprVisitor {
public:
  explicit LoweredScheduleVerifier(const LoweringView &view,
                                   const OverlapIR &program_ir)
      : view_(view), program_ir_(program_ir),
        group_counts_(view.group_warp_counts.size(), 0),
        operation_counts_(view.operation_groups.size(), 0),
        pipeline_loop_counts_(program_ir.regions.size(), 0),
        channel_arrives_(view.sync_producers.size(), 0),
        channel_waits_(view.sync_producers.size(), 0),
        channel_async_arrives_(view.sync_producers.size(), 0) {}

  static void Verify(const tirx::PrimFunc &func, const OverlapPlan &plan,
                     const LoweringView &view, const OverlapIR &program_ir) {
    LoweredScheduleVerifier verifier(view, program_ir);
    verifier(func->body);
    verifier.CheckCollected(func, plan);
  }

private:
  struct PipelineLoopSite {
    tirx::For loop;
    int64_t region_id{-1};
    int64_t group_id{-1};
  };

  int64_t ExpectedOperations(int64_t group_id, int64_t region_id) const {
    int64_t count = 0;
    for (size_t operation_id = 0; operation_id < view_.operation_groups.size();
         ++operation_id) {
      if (view_.operation_groups[operation_id] == group_id &&
          view_.operation_regions[operation_id] == region_id) {
        ++count;
      }
    }
    return count;
  }

  std::string RealizedBufferName(size_t buffer_id) const {
    std::string name = program_ir_.buffers[buffer_id]->name;
    if (view_.buffer_communications[buffer_id] ==
        static_cast<int64_t>(BufferCommunication::kMaterializedShared)) {
      name += "_wsp_shared";
    }
    return name;
  }

  std::string RealizedDataName(size_t buffer_id) const {
    std::string name = program_ir_.buffers[buffer_id]->data->name_hint;
    if (view_.buffer_communications[buffer_id] ==
        static_cast<int64_t>(BufferCommunication::kMaterializedShared)) {
      name += "_wsp_shared";
    }
    return name;
  }

  std::vector<std::string> PlannedDataNames(size_t buffer_id) const {
    if (IsFragmentPingPong(buffer_id)) {
      std::vector<std::string> names;
      int64_t versions = view_.buffer_versions[buffer_id];
      names.reserve(versions);
      for (int64_t slot = 0; slot < versions; ++slot) {
        names.push_back(FragmentSlotName(
            std::string(program_ir_.buffers[buffer_id]->data->name_hint),
            slot));
      }
      return names;
    }
    return {RealizedDataName(buffer_id)};
  }

  bool IsFragmentPingPong(size_t buffer_id) const {
    return view_.buffer_versions[buffer_id] > 1 &&
           program_ir_.buffers[buffer_id].scope() == "local.fragment";
  }

  bool BufferUsedInRegion(size_t buffer_id, int64_t region_id) const {
    for (size_t operation_id = 0; operation_id < program_ir_.operations.size();
         ++operation_id) {
      if (view_.operation_regions[operation_id] != region_id) {
        continue;
      }
      const std::vector<int64_t> &buffers =
          program_ir_.operation_buffer_ids[operation_id];
      if (std::find(buffers.begin(), buffers.end(),
                    static_cast<int64_t>(buffer_id)) != buffers.end()) {
        return true;
      }
    }
    return false;
  }

  void CheckPipelineLoop(const PipelineLoopSite &site) const {
    const tirx::For &loop = site.loop;
    int64_t expected = ExpectedOperations(site.group_id, site.region_id);
    if (expected > 0) {
      ICHECK(loop->annotations.count("tl_pipeline_stage") != 0 &&
             loop->annotations.count("tl_pipeline_order") != 0)
          << "T.Pipelined region " << site.region_id << " group "
          << site.group_id << " is missing planned stage/order annotations";
      auto stages = Downcast<ffi::Array<Integer>>(
          loop->annotations.Get("tl_pipeline_stage").value());
      auto orders = Downcast<ffi::Array<Integer>>(
          loop->annotations.Get("tl_pipeline_order").value());
      ffi::Array<tirx::Stmt> statements = FlattenSequence(loop->body);
      ICHECK_EQ(stages.size(), orders.size());
      ICHECK_EQ(stages.size(), statements.size())
          << "pipeline stage/order length must match the flattened loop body";

      int64_t minimum_stage = -1;
      for (size_t operation_id = 0;
           operation_id < view_.operation_groups.size(); ++operation_id) {
        if (view_.operation_groups[operation_id] != site.group_id ||
            view_.operation_regions[operation_id] != site.region_id) {
          continue;
        }
        int64_t stage = view_.operation_stages[operation_id];
        ICHECK_GE(stage, 0);
        minimum_stage =
            minimum_stage < 0 ? stage : std::min(minimum_stage, stage);
      }

      std::unordered_set<int64_t> seen_operations;
      int64_t auxiliary_order = expected;
      ICHECK_EQ(statements.size(), stages.size());
      for (size_t index = 0; index < statements.size(); ++index) {
        std::vector<int64_t> operation_ids =
            OperationMarkerCollector::Collect(statements[index], site.group_id);
        ICHECK_LE(operation_ids.size(), 1U);
        int64_t expected_stage = 0;
        int64_t expected_order = auxiliary_order;
        if (operation_ids.empty()) {
          ++auxiliary_order;
        } else {
          int64_t operation_id = operation_ids.front();
          ICHECK_EQ(view_.operation_groups[operation_id], site.group_id);
          ICHECK_EQ(view_.operation_regions[operation_id], site.region_id);
          ICHECK(seen_operations.insert(operation_id).second)
              << "operation " << operation_id
              << " appears more than once in one group pipeline loop";
          expected_stage = view_.operation_stages[operation_id] - minimum_stage;
          expected_order = view_.operation_local_orders[operation_id];
        }
        ICHECK_EQ(stages[index]->value, expected_stage)
            << "pipeline stage for region " << site.region_id << " group "
            << site.group_id << " statement " << index
            << " does not match the OverlapPlan";
        ICHECK_EQ(orders[index]->value, expected_order)
            << "pipeline order for region " << site.region_id << " group "
            << site.group_id << " statement " << index
            << " does not match the OverlapPlan";
      }
      ICHECK_EQ(seen_operations.size(), static_cast<size_t>(expected))
          << "pipeline region " << site.region_id << " group " << site.group_id
          << " did not retain exactly the operations described by the schedule";
    }

    if (view_.region_num_stages[site.region_id] <= 0) {
      return;
    }
    ICHECK(loop->annotations.count(kPipelinePlannedVersionBuffers) != 0)
        << "pipeline region " << site.region_id
        << " is missing planned version-buffer annotations";
    auto planned_vars = Downcast<ffi::Array<tirx::Var>>(
        loop->annotations.Get(kPipelinePlannedVersionBuffers).value());
    std::unordered_set<std::string> planned_names;
    for (const tirx::Var &var : planned_vars) {
      planned_names.insert(std::string(var->name_hint));
    }
    for (size_t buffer_id = 0; buffer_id < program_ir_.buffers.size();
         ++buffer_id) {
      if (view_.buffer_communications[buffer_id] == 0 ||
          !BufferUsedInRegion(buffer_id, site.region_id)) {
        continue;
      }
      for (const std::string &expected : PlannedDataNames(buffer_id)) {
        ICHECK(planned_names.count(expected) != 0)
            << "pipeline region " << site.region_id
            << " is missing planned version buffer " << expected;
      }
    }
    for (size_t channel = 0; channel < view_.sync_producers.size(); ++channel) {
      if (!IsFragmentHandoff(channel)) {
        continue;
      }
      int64_t producer_region = view_.sync_producer_regions[channel];
      int64_t consumer_region = view_.sync_consumer_regions[channel];
      if (producer_region != site.region_id &&
          consumer_region != site.region_id) {
        continue;
      }
      int64_t buffer_id = view_.sync_buffer_ids[channel];
      std::string expected = FragmentHandoffName(
          std::string(program_ir_.buffers[buffer_id]->data->name_hint),
          channel);
      ICHECK(planned_names.count(expected) != 0)
          << "pipeline region " << site.region_id
          << " is missing fragment-handoff buffer " << expected;
    }
    if (!view_.sync_producers.empty()) {
      ICHECK(loop->annotations.count(kPipelinePlannedBarrierBuffers) != 0)
          << "pipeline region " << site.region_id
          << " is missing planned barrier-buffer annotations";
    }
  }

  bool IsFragmentHandoff(size_t channel) const {
    int64_t buffer_id = view_.sync_buffer_ids[channel];
    if (view_.sync_kinds[channel] != 0 || buffer_id < 0 ||
        (view_.sync_dependency_masks[channel] & 1) == 0) {
      return false;
    }
    return program_ir_.buffers[buffer_id].scope() == "local.fragment";
  }

  void CheckVersionedBuffers() const {
    for (size_t buffer_id = 0; buffer_id < program_ir_.buffers.size();
         ++buffer_id) {
      int64_t versions = view_.buffer_versions[buffer_id];
      if (versions <= 1 || !view_.buffer_requires_new_allocation[buffer_id]) {
        continue;
      }
      if (IsFragmentPingPong(buffer_id)) {
        for (int64_t slot = 0; slot < versions; ++slot) {
          std::string expected = FragmentSlotName(
              std::string(program_ir_.buffers[buffer_id]->name), slot);
          int64_t found = 0;
          for (const tirx::Buffer &buffer : allocated_buffers_) {
            if (std::string(buffer->name) != expected) {
              continue;
            }
            ICHECK_EQ(buffer->shape.size(),
                      program_ir_.buffers[buffer_id]->shape.size())
                << "fragment ping-pong buffer " << expected
                << " must keep the original fragment rank";
            ++found;
          }
          ICHECK_GE(found, 1) << "fragment ping-pong buffer " << expected
                              << " was not allocated";
        }
        continue;
      }
      std::string expected = RealizedBufferName(buffer_id);
      int64_t found = 0;
      for (const tirx::Buffer &buffer : allocated_buffers_) {
        if (std::string(buffer->name) != expected) {
          continue;
        }
        ICHECK_EQ(buffer->shape.size(),
                  program_ir_.buffers[buffer_id]->shape.size() + 1)
            << "versioned buffer " << expected
            << " must grow by one leading dimension";
        const auto *leading = buffer->shape[0].as<IntImmNode>();
        ICHECK(leading != nullptr && leading->value == versions)
            << "versioned buffer " << expected
            << " leading dimension must equal the planned version count";
        ++found;
      }
      ICHECK_GE(found, 1)
          << "versioned buffer " << expected
          << " was not allocated with the planned leading version dimension";
    }
  }

  void CheckBarriers() const {
    if (view_.sync_producers.empty()) {
      return;
    }
    ICHECK(!planned_barrier_vars_.empty())
        << "lowered schedule is missing planned barrier-buffer annotations";
    int64_t expected_slots = 0;
    for (size_t channel = 0; channel < view_.sync_slot_counts.size(); ++channel) {
      if (view_.sync_event_owners[channel] == static_cast<int64_t>(channel)) {
        expected_slots += view_.sync_slot_counts[channel];
      }
    }
    int64_t planned_slots = 0;
    for (const auto &[var, slots] : barrier_inits_) {
      if (planned_barrier_vars_.count(var) != 0) {
        planned_slots += slots;
      }
    }
    ICHECK_EQ(planned_slots, expected_slots)
        << "lowered mbarrier slot count does not match the OverlapPlan";
  }

  void CheckSyncEvents() const {
    if (view_.sync_producers.empty()) {
      return;
    }
    for (size_t channel = 0; channel < view_.sync_producers.size(); ++channel) {
      if (view_.sync_completion_modes[channel] == 1) {
        int64_t expected_arrives =
            view_.sync_event_owners[channel] == static_cast<int64_t>(channel)
                ? 1
                : 0;
        ICHECK_EQ(channel_async_arrives_[channel], expected_arrives)
            << "sync channel " << channel
            << " has the wrong number of asynchronous arrive events";
        ICHECK_EQ(channel_arrives_[channel], 0)
            << "sync channel " << channel
            << " cannot contain both asynchronous and thread arrive events";
      } else {
        ICHECK_EQ(channel_arrives_[channel], 1)
            << "sync channel " << channel
            << " must contain exactly one thread arrive event";
        ICHECK_EQ(channel_async_arrives_[channel], 0)
            << "sync channel " << channel
            << " cannot contain an asynchronous arrive event";
      }
      ICHECK_EQ(channel_waits_[channel], 1)
          << "sync channel " << channel
          << " must contain exactly one wait event";
    }
  }

  void CheckSharedOffsets(const tirx::PrimFunc &func,
                          const OverlapPlan &plan) const {
    bool has_offset = false;
    for (const BufferPlan &buffer_plan : plan->buffers) {
      has_offset = has_offset || buffer_plan->byte_offset.defined();
    }
    if (!has_offset) {
      return;
    }
    auto offset_map =
        func->GetAttr<ffi::Map<ffi::String, IntImm>>(::tvm::tl::kSmemOffsetMap);
    ICHECK(offset_map.defined())
        << "planned shared-memory offsets were not attached to the PrimFunc";
    const ffi::Map<ffi::String, IntImm> &offsets = offset_map.value();
    for (size_t buffer_id = 0; buffer_id < plan->buffers.size(); ++buffer_id) {
      const BufferPlan &buffer_plan = plan->buffers[buffer_id];
      if (!buffer_plan->byte_offset.defined() ||
          buffer_id >= program_ir_.buffers.size()) {
        continue;
      }
      int64_t expected = buffer_plan->byte_offset.value()->value;
      const tirx::Buffer &buffer = program_ir_.buffers[buffer_id];
      auto by_buffer_name = offsets.Get(ffi::String(buffer->name));
      auto by_data_name = offsets.Get(ffi::String(buffer->data->name_hint));
      ICHECK(by_buffer_name.has_value())
          << "planned offset for buffer " << buffer->name
          << " is missing from tl.smem_offset_map";
      ICHECK(by_data_name.has_value())
          << "planned offset for buffer data " << buffer->data->name_hint
          << " is missing from tl.smem_offset_map";
      ICHECK_EQ(by_buffer_name.value()->value, expected)
          << "tl.smem_offset_map offset for " << buffer->name
          << " does not match the OverlapPlan";
      ICHECK_EQ(by_data_name.value()->value, expected)
          << "tl.smem_offset_map offset for " << buffer->data->name_hint
          << " does not match the OverlapPlan";
    }
    for (size_t channel = 0; channel < plan->sync_edges.size(); ++channel) {
      const SyncEdge &edge = plan->sync_edges[channel];
      if (!edge->byte_offset.defined() || !edge->buffer_id.defined()) {
        continue;
      }
      int64_t buffer_id = edge->buffer_id.value()->value;
      if (buffer_id < 0 ||
          buffer_id >= static_cast<int64_t>(program_ir_.buffers.size())) {
        continue;
      }
      int64_t expected = edge->byte_offset.value()->value;
      const tirx::Buffer &buffer = program_ir_.buffers[buffer_id];
      std::string by_name =
          FragmentHandoffName(std::string(buffer->name), channel);
      std::string by_data = FragmentHandoffName(
          std::string(buffer->data->name_hint), channel);
      auto named = offsets.Get(ffi::String(by_name));
      auto data_named = offsets.Get(ffi::String(by_data));
      ICHECK(named.has_value())
          << "planned offset for fragment handoff " << by_name
          << " is missing from tl.smem_offset_map";
      ICHECK(data_named.has_value())
          << "planned offset for fragment handoff " << by_data
          << " is missing from tl.smem_offset_map";
      ICHECK_EQ(named.value()->value, expected)
          << "tl.smem_offset_map offset for " << by_name
          << " does not match the OverlapPlan";
      ICHECK_EQ(data_named.value()->value, expected)
          << "tl.smem_offset_map offset for " << by_data
          << " does not match the OverlapPlan";
    }
    if (plan->shared_arena_bytes.defined()) {
      auto arena = func->GetAttr<Integer>(::tvm::tl::kSmemPlannedArenaBytes);
      ICHECK(arena.defined())
          << "planned shared-memory arena size was not attached to the "
             "PrimFunc";
      ICHECK_EQ(arena.value()->value, plan->shared_arena_bytes.value()->value)
          << "tl.smem_planned_arena_bytes does not match the OverlapPlan";
    }
  }

  void CheckCollected(const tirx::PrimFunc &func, const OverlapPlan &plan) {
    ICHECK_EQ(thread_x_extent_count_, 1)
        << "lowered schedule must contain exactly one threadIdx.x extent";
    ICHECK_EQ(warp_specialization_scope_count_, 1)
        << "lowered schedule must contain exactly one warp-specialization "
           "scope";
    for (size_t group_id = 0; group_id < group_counts_.size(); ++group_id) {
      ICHECK_EQ(group_counts_[group_id], 1)
          << "lowered schedule must materialize group " << group_id
          << " exactly once";
    }
    for (size_t operation_id = 0; operation_id < operation_counts_.size();
         ++operation_id) {
      ICHECK_EQ(operation_counts_[operation_id], 1)
          << "lowered schedule must retain operation " << operation_id
          << " exactly once";
    }

    const int64_t group_count =
        static_cast<int64_t>(view_.group_warp_counts.size());
    for (size_t region_id = 0; region_id < program_ir_.regions.size();
         ++region_id) {
      if (!program_ir_.regions[region_id].pipeline_loop.has_value()) {
        ICHECK_EQ(pipeline_loop_counts_[region_id], 0);
        continue;
      }
      ICHECK_EQ(pipeline_loop_counts_[region_id], group_count)
          << "each group must retain T.Pipelined region " << region_id;
    }
    for (const PipelineLoopSite &site : pipeline_loops_) {
      CheckPipelineLoop(site);
    }
    CheckVersionedBuffers();
    CheckBarriers();
    CheckSyncEvents();
    CheckSharedOffsets(func, plan);
  }

  void VisitStmt_(const tirx::AttrStmtNode *op) final {
    if (op->attr_key == kSyncEventScope) {
      const auto *channel_imm = op->node.as<IntImmNode>();
      const auto *role_imm = op->value.as<IntImmNode>();
      ICHECK(channel_imm != nullptr && role_imm != nullptr)
          << kSyncEventScope << " must carry static channel/role IDs";
      ICHECK_GE(channel_imm->value, 0);
      ICHECK_LT(channel_imm->value,
                static_cast<int64_t>(view_.sync_producers.size()));
      ICHECK_GE(role_imm->value,
                static_cast<int64_t>(SyncEventRole::kThreadArrive));
      ICHECK_LE(role_imm->value,
                static_cast<int64_t>(SyncEventRole::kAsyncArrive));
      ICHECK_EQ(active_sync_channel_, -1)
          << "overlap-plan synchronization event scopes cannot be nested";
      active_sync_channel_ = channel_imm->value;
      active_sync_role_ = static_cast<SyncEventRole>(role_imm->value);
      VisitStmt(op->body);
      active_sync_channel_ = -1;
      return;
    }
    if (op->attr_key == tirx::attr::thread_extent) {
      ffi::Optional<tirx::IterVar> iter_var = op->node.as<tirx::IterVar>();
      if (iter_var.has_value() &&
          iter_var.value()->thread_tag == "threadIdx.x") {
        const auto *extent = op->value.as<IntImmNode>();
        ICHECK(extent != nullptr);
        ICHECK_EQ(extent->value, view_.effective_threads)
            << "lowered threadIdx.x extent does not match the schedule";
        ++thread_x_extent_count_;
      }
    }
    if (op->attr_key == ::tvm::tl::attr::kWarpSpecializationScope) {
      ++warp_specialization_scope_count_;
    }
    if (op->attr_key == kProgramScheduleGroupScope) {
      const auto *group_id_imm = op->value.as<IntImmNode>();
      ICHECK(group_id_imm != nullptr);
      int64_t group_id = group_id_imm->value;
      ICHECK_GE(group_id, 0);
      ICHECK_LT(group_id, static_cast<int64_t>(group_counts_.size()));
      ICHECK_EQ(active_group_, -1)
          << "overlap-plan group scopes cannot be nested";
      auto [first_thread, thread_count] =
          GetProgramScheduleGroupThreadRange(op);
      ICHECK_EQ(first_thread.as<IntImmNode>()->value,
                view_.group_first_warps[group_id] * 32);
      ICHECK_EQ(thread_count.as<IntImmNode>()->value,
                view_.group_warp_counts[group_id] * 32);
      ++group_counts_[group_id];
      active_group_ = group_id;
      VisitStmt(op->body);
      active_group_ = -1;
      return;
    }
    if (op->attr_key == kOperationScope) {
      const auto *operation_id_imm = op->node.as<IntImmNode>();
      const auto *group_id_imm = op->value.as<IntImmNode>();
      ICHECK(operation_id_imm != nullptr && group_id_imm != nullptr);
      int64_t operation_id = operation_id_imm->value;
      ICHECK_GE(operation_id, 0);
      ICHECK_LT(operation_id, static_cast<int64_t>(operation_counts_.size()));
      ICHECK_EQ(active_group_, view_.operation_groups[operation_id])
          << "operation marker is materialized in the wrong group";
      ICHECK_EQ(group_id_imm->value, active_group_)
          << "operation marker carries the wrong group ID";
      ++operation_counts_[operation_id];
      VisitStmt(op->body);
      return;
    }
    tirx::StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const tirx::ForNode *op) final {
    tirx::For loop = ffi::GetRef<tirx::For>(op);
    auto region_annotation = loop->annotations.Get(kRegionIdAnnotation);
    if (region_annotation.has_value()) {
      const auto *region_id_imm = region_annotation.value().as<IntImmNode>();
      ICHECK(region_id_imm != nullptr);
      int64_t region_id = region_id_imm->value;
      ICHECK_GE(region_id, 0);
      ICHECK_LT(region_id, static_cast<int64_t>(pipeline_loop_counts_.size()));
      ++pipeline_loop_counts_[region_id];
      ICHECK_GE(active_group_, 0)
          << "pipeline region annotation must be inside a group scope";
      pipeline_loops_.push_back(
          PipelineLoopSite{loop, region_id, active_group_});
      if (auto planned_barriers =
              loop->annotations.Get(kPipelinePlannedBarrierBuffers)) {
        auto vars = Downcast<ffi::Array<tirx::Var>>(planned_barriers.value());
        for (const tirx::Var &var : vars) {
          planned_barrier_vars_.insert(var);
        }
      }
    }
    tirx::StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const tirx::SBlockNode *op) final {
    for (const tirx::Buffer &buffer : op->alloc_buffers) {
      allocated_buffers_.push_back(buffer);
    }
    if (op->annotations.count("barrier_init")) {
      auto barrier_init = Downcast<ffi::Map<tirx::Var, ffi::Array<PrimExpr>>>(
          op->annotations.Get("barrier_init").value());
      for (const auto &kv : barrier_init) {
        barrier_inits_.emplace_back(kv.first,
                                    static_cast<int64_t>(kv.second.size()));
      }
    }
    tirx::StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const tirx::AllocBufferNode *op) final {
    allocated_buffers_.push_back(op->buffer);
    tirx::StmtExprVisitor::VisitStmt_(op);
  }

  void RecordSyncEvent(SyncEventRole expected_role,
                       std::vector<int64_t> *counts) {
    if (active_sync_channel_ < 0) {
      return;
    }
    ICHECK(active_sync_role_ == expected_role)
        << "synchronization channel " << active_sync_channel_
        << " carries the wrong event role";
    ++(*counts)[active_sync_channel_];
  }

  void VisitExpr_(const tirx::CallNode *op) final {
    tirx::Call call = ffi::GetRef<tirx::Call>(op);
    static const Op &tma_copy_op = Op::Get("tl.tileop.tma_copy");
    if (call->op.same_as(tirx::builtin::ptx_arrive_barrier())) {
      RecordSyncEvent(SyncEventRole::kThreadArrive, &channel_arrives_);
    } else if (call->op.same_as(mbarrier_wait_parity())) {
      RecordSyncEvent(SyncEventRole::kWait, &channel_waits_);
    } else if (call->op.same_as(tma_copy_op)) {
      if (auto emit = call->annotations.Get("emit_arrive")) {
        const auto *imm = emit.value().as<IntImmNode>();
        if (imm != nullptr && imm->value != 0) {
          RecordSyncEvent(SyncEventRole::kAsyncArrive, &channel_async_arrives_);
        }
      }
    }
    tirx::StmtExprVisitor::VisitExpr_(op);
  }

  const LoweringView &view_;
  const OverlapIR &program_ir_;
  std::vector<int64_t> group_counts_;
  std::vector<int64_t> operation_counts_;
  std::vector<int64_t> pipeline_loop_counts_;
  std::vector<int64_t> channel_arrives_;
  std::vector<int64_t> channel_waits_;
  std::vector<int64_t> channel_async_arrives_;
  std::vector<PipelineLoopSite> pipeline_loops_;
  std::vector<tirx::Buffer> allocated_buffers_;
  std::vector<std::pair<tirx::Var, int64_t>> barrier_inits_;
  VarSet planned_barrier_vars_;
  int64_t active_group_{-1};
  int64_t active_sync_channel_{-1};
  SyncEventRole active_sync_role_{SyncEventRole::kThreadArrive};
  int64_t thread_x_extent_count_{0};
  int64_t warp_specialization_scope_count_{0};
};

} // namespace

void VerifyLoweredOverlapPlan(const tirx::PrimFunc &func,
                              const OverlapPlan &plan, const LoweringView &view,
                              const OverlapIR &program_ir) {
  LoweredScheduleVerifier::Verify(func, plan, view, program_ir);
}

} // namespace overlap_plan
} // namespace tl
} // namespace tvm
