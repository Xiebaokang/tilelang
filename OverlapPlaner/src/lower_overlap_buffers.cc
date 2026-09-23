/*!
 * \file lower_overlap_buffers.cc
 * \brief Materialize OverlapPlan buffer storage and version slots.
 */

#include "overlap_plan.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <tvm/ir/type.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include "layout/layout.h"
#include "op/builtin.h"
#include "op/copy.h"
#include "op/region.h"
#include "support/check.h"
#include "transform/common/pipeline_utils.h"

namespace tvm {
namespace tl {
namespace overlap_plan {

namespace {

using BufferIdMap = std::unordered_map<tirx::Buffer, int64_t,
                                       ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;
using BufferMap = std::unordered_map<tirx::Buffer, tirx::Buffer,
                                     ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;
using VarIdMap = std::unordered_map<tirx::Var, int64_t, ffi::ObjectPtrHash,
                                    ffi::ObjectPtrEqual>;
using VarMap = std::unordered_map<tirx::Var, tirx::Var, ffi::ObjectPtrHash,
                                  ffi::ObjectPtrEqual>;

struct FragmentHandoff {
  size_t channel_id{0};
  int64_t buffer_id{-1};
  int64_t producer_operation_id{-1};
  int64_t consumer_operation_id{-1};
  int64_t producer_region_id{-1};
  int64_t consumer_region_id{-1};
  int64_t iteration_distance{0};
  int64_t slots{1};
  tirx::Buffer shared_buffer;
};

tirx::Stmt MakeSequence(std::vector<tirx::Stmt> statements) {
  ICHECK(!statements.empty());
  if (statements.size() == 1) {
    return statements.front();
  }
  ffi::Array<tirx::Stmt> sequence;
  sequence.reserve(statements.size());
  for (tirx::Stmt &statement : statements) {
    sequence.push_back(std::move(statement));
  }
  return tirx::SeqStmt(sequence);
}

class ProgramBufferLowerer : public tirx::StmtExprMutator {
public:
  static tirx::PrimFunc Lower(tirx::PrimFunc func, const LoweringView &plan,
                              const OverlapIR &program_ir) {
    ProgramBufferLowerer lowerer(plan, program_ir);
    tirx::PrimFuncNode *node = func.CopyOnWrite();
    node->body = lowerer(node->body);
    lowerer.ValidateAllocations();
    return func;
  }

private:
  ProgramBufferLowerer(const LoweringView &plan, const OverlapIR &program_ir)
      : plan_(plan), program_ir_(program_ir),
        versioned_region_ids_(program_ir.buffers.size()),
        allocation_counts_(program_ir.buffers.size(), 0),
        planned_buffers_by_region_(program_ir.regions.size()),
        planned_handoffs_by_region_(program_ir.regions.size()),
        fragment_slots_(program_ir.buffers.size()) {
    ICHECK_EQ(program_ir_.operation_buffer_ids.size(),
              program_ir_.operations.size());
    for (size_t buffer_id = 0; buffer_id < program_ir_.buffers.size();
         ++buffer_id) {
      const tirx::Buffer &buffer = program_ir_.buffers[buffer_id];
      ICHECK(buffer_ids_.emplace(buffer, buffer_id).second)
          << "one Buffer ObjectRef cannot have two program buffer IDs";
      var_buffer_ids_.emplace(buffer->data, buffer_id);
    }

    BuildFragmentHandoffs();

    for (size_t operation_id = 0; operation_id < program_ir_.operations.size();
         ++operation_id) {
      int64_t region_id = program_ir_.operations[operation_id].region_id;
      for (int64_t buffer_id : program_ir_.operation_buffer_ids[operation_id]) {
        ICHECK_GE(buffer_id, 0);
        ICHECK_LT(buffer_id, static_cast<int64_t>(program_ir_.buffers.size()));
        if (plan_.region_num_stages[region_id] > 0 &&
            plan_.buffer_communications[buffer_id] != 0) {
          planned_buffers_by_region_[region_id].insert(buffer_id);
        }
      }
    }

    for (size_t buffer_id = 0; buffer_id < program_ir_.buffers.size();
         ++buffer_id) {
      ValidateCommunication(buffer_id);
      if (plan_.buffer_versions[buffer_id] > 1) {
        versioned_region_ids_[buffer_id] = FindVersionedRegion(buffer_id);
      }
      if (!plan_.buffer_requires_new_allocation[buffer_id]) {
        continue;
      }
      const tirx::Buffer &old_buffer = program_ir_.buffers[buffer_id];
      if (IsFragmentPingPong(buffer_id)) {
        fragment_slots_[buffer_id] =
            MakeFragmentSlots(old_buffer, plan_.buffer_versions[buffer_id]);
        for (const tirx::Buffer &slot : fragment_slots_[buffer_id]) {
          var_buffer_ids_.emplace(slot->data, buffer_id);
        }
        continue;
      }
      tirx::Buffer new_buffer = MakeRealizedBuffer(old_buffer, buffer_id);
      buffer_remap_.emplace(old_buffer, new_buffer);
      var_remap_.emplace(old_buffer->data, new_buffer->data);
      var_buffer_ids_.emplace(new_buffer->data, buffer_id);
    }
  }

  void BuildFragmentHandoffs() {
    for (size_t channel = 0; channel < plan_.sync_producers.size(); ++channel) {
      int64_t buffer_id = plan_.sync_buffer_ids[channel];
      if (plan_.sync_kinds[channel] != 0 || buffer_id < 0 ||
          (plan_.sync_dependency_masks[channel] & 1) == 0) {
        continue;
      }
      const tirx::Buffer &buffer = program_ir_.buffers[buffer_id];
      if (buffer.scope() != "local.fragment") {
        continue;
      }
      ICHECK_EQ(plan_.buffer_communications[buffer_id], 1)
          << "cross-group fragment handoff must preserve the original "
             "PRIVATE buffer: "
          << buffer->name;
      int64_t slots = 1;
      if (plan_.sync_scopes[channel] == 1) {
        int64_t producer_region = plan_.sync_producer_regions[channel];
        ICHECK_GE(producer_region, 0);
        ICHECK_EQ(producer_region, plan_.sync_consumer_regions[channel]);
        ICHECK_LT(channel, plan_.sync_slot_counts.size())
            << "per-iteration fragment handoff needs a planned slot count: "
            << buffer->name;
        slots = plan_.sync_slot_counts[channel];
        ICHECK_GT(slots, 0);
      }
      FragmentHandoff handoff{
          channel,
          buffer_id,
          plan_.sync_producers[channel],
          plan_.sync_consumers[channel],
          plan_.sync_producer_regions[channel],
          plan_.sync_consumer_regions[channel],
          plan_.sync_iteration_distances[channel],
          slots,
          MakeFragmentHandoffBuffer(buffer, channel, slots),
      };
      size_t handoff_id = fragment_handoffs_.size();
      fragment_handoffs_.push_back(std::move(handoff));
      handoffs_by_buffer_[buffer].push_back(handoff_id);
      const FragmentHandoff &stored = fragment_handoffs_.back();
      if (stored.producer_region_id >= 0 &&
          plan_.region_num_stages[stored.producer_region_id] > 0) {
        planned_handoffs_by_region_[stored.producer_region_id].insert(
            handoff_id);
      }
      if (stored.consumer_region_id >= 0 &&
          plan_.region_num_stages[stored.consumer_region_id] > 0) {
        planned_handoffs_by_region_[stored.consumer_region_id].insert(
            handoff_id);
      }
    }
    handoff_allocation_counts_.assign(fragment_handoffs_.size(), 0);
  }

  tirx::Buffer MakeFragmentHandoffBuffer(const tirx::Buffer &fragment,
                                         size_t channel, int64_t slots) const {
    const auto *pointer_type =
        fragment->data->type_annotation.as<PointerTypeNode>();
    ICHECK(pointer_type != nullptr)
        << "fragment handoff buffer must have pointer type: " << fragment->name;
    ffi::ObjectPtr<tirx::BufferNode> node =
        ffi::make_object<tirx::BufferNode>(*fragment.get());
    std::string suffix = FragmentHandoffSuffix(channel);
    node->data =
        tirx::Var(std::string(fragment->data->name_hint) + suffix,
                  PointerType(pointer_type->element_type, "shared.dyn"));
    node->name = std::string(fragment->name) + suffix;
    if (slots > 1) {
      DataType shape_dtype = fragment->shape.empty()
                                 ? DataType::Int(32)
                                 : fragment->shape[0].dtype();
      node->shape.insert(node->shape.begin(), IntImm(shape_dtype, slots));
    }
    node->strides.clear();
    return tirx::Buffer(node);
  }

  void ValidateCommunication(size_t buffer_id) const {
    const tirx::Buffer &buffer = program_ir_.buffers[buffer_id];
    std::string scope = buffer.scope();
    int64_t communication = plan_.buffer_communications[buffer_id];
    int64_t versions = plan_.buffer_versions[buffer_id];
    if (communication == 0) {
      ICHECK_EQ(versions, 1)
          << "external buffer " << buffer->name
          << " cannot be physically versioned by LowerOverlapPlan";
      ICHECK(!plan_.buffer_requires_new_allocation[buffer_id])
          << "external buffer " << buffer->name
          << " cannot request a new kernel-local allocation";
    } else if (communication == 2) {
      ICHECK(scope.rfind("shared", 0) == 0)
          << "SHARED communication requires shared storage for buffer "
          << buffer->name;
    } else if (communication == 3) {
      ICHECK(scope.find("tmem") != std::string::npos)
          << "TMEM communication requires a TMEM buffer: " << buffer->name;
    } else if (communication == 4) {
      ICHECK(scope.rfind("shared", 0) != 0)
          << "MATERIALIZED_SHARED expects a non-shared source buffer: "
          << buffer->name;
    }
  }

  int64_t FindVersionedRegion(size_t buffer_id) const {
    std::optional<int64_t> versioned_region;
    for (size_t operation_id = 0; operation_id < program_ir_.operations.size();
         ++operation_id) {
      const std::vector<int64_t> &buffers =
          program_ir_.operation_buffer_ids[operation_id];
      if (std::find(buffers.begin(), buffers.end(), buffer_id) ==
          buffers.end()) {
        continue;
      }
      int64_t region_id = program_ir_.operations[operation_id].region_id;
      if (plan_.region_num_stages[region_id] == 0) {
        continue;
      }
      if (versioned_region.has_value()) {
        ICHECK_EQ(versioned_region.value(), region_id)
            << "versioned buffer " << program_ir_.buffers[buffer_id]->name
            << " is used by more than one pipeline region";
      } else {
        versioned_region = region_id;
      }
    }
    ICHECK(versioned_region.has_value())
        << "multiversion buffer " << program_ir_.buffers[buffer_id]->name
        << " is not used by a pipeline region";
    return versioned_region.value();
  }

  tirx::Buffer MakeRealizedBuffer(const tirx::Buffer &old_buffer,
                                  size_t buffer_id) const {
    ffi::ObjectPtr<tirx::BufferNode> node =
        ffi::make_object<tirx::BufferNode>(*old_buffer.get());
    if (plan_.buffer_communications[buffer_id] == 4) {
      const auto *pointer_type =
          old_buffer->data->type_annotation.as<PointerTypeNode>();
      ICHECK(pointer_type != nullptr)
          << "materialized-shared buffer data must have pointer type: "
          << old_buffer->name;
      std::string data_name =
          std::string(old_buffer->data->name_hint) + "_wsp_shared";
      node->data = tirx::Var(
          data_name, PointerType(pointer_type->element_type, "shared.dyn"));
      node->name = std::string(old_buffer->name) + "_wsp_shared";
    }

    int64_t versions = plan_.buffer_versions[buffer_id];
    if (versions > 1 && old_buffer.scope() != "local.fragment") {
      DataType shape_dtype = old_buffer->shape.empty()
                                 ? DataType::Int(32)
                                 : old_buffer->shape[0].dtype();
      node->shape.insert(node->shape.begin(), IntImm(shape_dtype, versions));
      if (!node->strides.empty()) {
        ICHECK_EQ(node->strides.size() + 1, node->shape.size());
        DataType stride_dtype = node->strides[0].dtype();
        PrimExpr leading_stride = tvm::cast(stride_dtype, node->strides[0]) *
                                  tvm::cast(stride_dtype, node->shape[1]);
        node->strides.insert(node->strides.begin(), leading_stride);
      }
    }
    return tirx::Buffer(node);
  }

  bool IsFragmentPingPong(int64_t buffer_id) const {
    return plan_.buffer_versions[buffer_id] > 1 &&
           program_ir_.buffers[buffer_id].scope() == "local.fragment";
  }

  std::vector<tirx::Buffer> MakeFragmentSlots(const tirx::Buffer &old_buffer,
                                              int64_t versions) const {
    std::vector<tirx::Buffer> slots;
    slots.reserve(versions);
    for (int64_t slot = 0; slot < versions; ++slot) {
      ffi::ObjectPtr<tirx::BufferNode> node =
          ffi::make_object<tirx::BufferNode>(*old_buffer.get());
      std::string suffix = FragmentSlotSuffix(slot);
      node->data = tirx::Var(std::string(old_buffer->data->name_hint) + suffix,
                             old_buffer->data->type_annotation);
      node->name = std::string(old_buffer->name) + suffix;
      slots.emplace_back(node);
    }
    return slots;
  }

  std::vector<int64_t> PingPongBuffersForOperation(int64_t operation_id) const {
    std::vector<int64_t> result;
    auto add = [&](int64_t buffer_id) {
      if (IsFragmentPingPong(buffer_id) &&
          std::find(result.begin(), result.end(), buffer_id) == result.end()) {
        result.push_back(buffer_id);
      }
    };
    for (int64_t buffer_id : program_ir_.operation_buffer_ids[operation_id]) {
      add(buffer_id);
    }
    for (const FragmentHandoff &handoff : fragment_handoffs_) {
      if (handoff.producer_operation_id == operation_id ||
          handoff.consumer_operation_id == operation_id) {
        add(handoff.buffer_id);
      }
    }
    return result;
  }

  int64_t
  CombinedFragmentModulus(const std::vector<int64_t> &buffer_ids) const {
    int64_t combined = 1;
    for (int64_t buffer_id : buffer_ids) {
      combined = std::lcm(combined, plan_.buffer_versions[buffer_id]);
    }
    return combined;
  }

  PrimExpr PipelineModulusIndex(int64_t buffer_id, int64_t modulus) const {
    ICHECK_GT(modulus, 1);
    ICHECK(current_operation_id_.has_value())
        << "versioned buffer " << program_ir_.buffers[buffer_id]->name
        << " is accessed outside a scheduled operation";
    int64_t operation_region =
        plan_.operation_regions[current_operation_id_.value()];
    ICHECK(versioned_region_ids_[buffer_id].has_value())
        << "multiversion buffer " << program_ir_.buffers[buffer_id]->name
        << " is not bound to a pipeline region";
    int64_t versioned_region = versioned_region_ids_[buffer_id].value();
    const tirx::For &loop =
        program_ir_.regions[versioned_region].pipeline_loop.value();
    PrimExpr modulus_expr = IntImm(loop->loop_var.dtype(), modulus);
    if (operation_region == versioned_region) {
      return tirx::FloorMod(loop->loop_var, modulus_expr);
    }
    if (operation_region < versioned_region) {
      return tirx::FloorMod(loop->min, modulus_expr);
    }
    PrimExpr final_iteration =
        loop->min + loop->extent - IntImm(loop->loop_var.dtype(), 1);
    return tirx::FloorMod(final_iteration, modulus_expr);
  }

  tirx::Buffer RemapBuffer(const tirx::Buffer &buffer) const {
    auto id_iterator = buffer_ids_.find(buffer);
    if (id_iterator != buffer_ids_.end() &&
        IsFragmentPingPong(id_iterator->second)) {
      ICHECK_GE(current_combined_slot_, 0)
          << "fragment ping-pong access to " << buffer->name
          << " requires a selected version slot";
      const std::vector<tirx::Buffer> &slots =
          fragment_slots_[id_iterator->second];
      int64_t versions = plan_.buffer_versions[id_iterator->second];
      return slots[current_combined_slot_ % versions];
    }
    auto iterator = buffer_remap_.find(buffer);
    return iterator == buffer_remap_.end() ? buffer : iterator->second;
  }

  PrimExpr MakeHandoffRegion(const tirx::Buffer &buffer,
                             ffi::Optional<PrimExpr> version_index,
                             int access_mask) const {
    ffi::Array<PrimExpr> indices;
    ffi::Array<PrimExpr> arguments;
    size_t first_shape_axis = 0;
    if (version_index.has_value()) {
      indices.push_back(version_index.value());
      first_shape_axis = 1;
    }
    for (size_t axis = first_shape_axis; axis < buffer->shape.size(); ++axis) {
      indices.push_back(IntImm(buffer->shape[axis].dtype(), 0));
    }
    arguments.push_back(tirx::BufferLoad(buffer, indices));
    arguments.push_back(IntImm(DataType::Int(32), access_mask));
    if (version_index.has_value()) {
      arguments.push_back(IntImm(buffer->shape[0].dtype(), 1));
    }
    for (size_t axis = first_shape_axis; axis < buffer->shape.size(); ++axis) {
      arguments.push_back(buffer->shape[axis]);
    }
    return tirx::Call(DataType::Handle(), RegionOp::Get(), arguments);
  }

  ffi::Optional<PrimExpr> HandoffVersionIndex(const FragmentHandoff &handoff,
                                              bool consumer) const {
    if (handoff.slots == 1) {
      return ffi::Optional<PrimExpr>();
    }
    ICHECK(current_pipeline_loop_.has_value())
        << "versioned fragment handoff must be inside a pipeline loop";
    const tirx::For &loop = current_pipeline_loop_.value();
    PrimExpr epoch = loop->loop_var - loop->min;
    if (consumer && handoff.iteration_distance != 0) {
      epoch =
          epoch - IntImm(loop->loop_var.dtype(), handoff.iteration_distance);
    }
    return tirx::FloorMod(epoch, IntImm(loop->loop_var.dtype(), handoff.slots));
  }

  tirx::Stmt MakeFragmentHandoffCopy(const FragmentHandoff &handoff,
                                     bool import) const {
    tirx::Buffer fragment = RemapBuffer(program_ir_.buffers[handoff.buffer_id]);
    PrimExpr fragment_region =
        MakeHandoffRegion(fragment, ffi::Optional<PrimExpr>(), import ? 2 : 1);
    PrimExpr shared_region =
        MakeHandoffRegion(handoff.shared_buffer,
                          HandoffVersionIndex(handoff, import), import ? 1 : 2);
    ffi::Array<PrimExpr> arguments =
        import ? ffi::Array<PrimExpr>{shared_region, fragment_region}
               : ffi::Array<PrimExpr>{fragment_region, shared_region};
    tirx::Stmt copy = tirx::Evaluate(
        tirx::Call(DataType::Handle(), Copy::Get(), arguments));

    // A region-boundary handoff attached to an operation inside a pipeline
    // must execute once at the boundary, rather than once per logical
    // iteration.  Re-importing a serial-region accumulator before every GEMM
    // iteration resets the partial sum; exporting every iteration similarly
    // exposes an intermediate fragment before the pipeline has completed.
    if (handoff.producer_region_id != handoff.consumer_region_id &&
        current_pipeline_loop_.has_value()) {
      const tirx::For &loop = current_pipeline_loop_.value();
      PrimExpr boundary_iteration;
      if (import) {
        boundary_iteration = loop->min;
      } else {
        boundary_iteration =
            loop->min + loop->extent - IntImm(loop->loop_var.dtype(), 1);
      }
      copy = tirx::IfThenElse(loop->loop_var == boundary_iteration,
                              std::move(copy));
    }
    return copy;
  }

  int64_t BufferId(const tirx::Buffer &buffer) const {
    auto iterator = buffer_ids_.find(buffer);
    ICHECK(iterator != buffer_ids_.end())
        << "rewritten access references an unknown program buffer "
        << buffer->name;
    return iterator->second;
  }

  PrimExpr VersionIndex(int64_t buffer_id) const {
    return PipelineModulusIndex(buffer_id, plan_.buffer_versions[buffer_id]);
  }

  ffi::Array<PrimExpr>
  RewriteIndices(const tirx::Buffer &old_buffer,
                 const ffi::Array<PrimExpr> &indices) const {
    int64_t buffer_id = BufferId(old_buffer);
    if (plan_.buffer_versions[buffer_id] == 1 ||
        IsFragmentPingPong(buffer_id)) {
      return indices;
    }
    ffi::Array<PrimExpr> rewritten;
    rewritten.reserve(indices.size() + 1);
    rewritten.push_back(VersionIndex(buffer_id));
    rewritten.insert(rewritten.end(), indices.begin(), indices.end());
    return rewritten;
  }

  tirx::BufferRegion
  RewriteBufferRegion(const tirx::BufferRegion &buffer_region) const {
    auto iterator = buffer_ids_.find(buffer_region->buffer);
    if (iterator == buffer_ids_.end()) {
      return buffer_region;
    }
    int64_t buffer_id = iterator->second;
    if (IsFragmentPingPong(buffer_id)) {
      ICHECK_GE(current_combined_slot_, 0)
          << "fragment ping-pong region for " << buffer_region->buffer->name
          << " requires a selected version slot";
      return tirx::BufferRegion(RemapBuffer(buffer_region->buffer),
                                buffer_region->region);
    }
    tirx::Buffer buffer = RemapBuffer(buffer_region->buffer);
    ffi::Array<Range> region = buffer_region->region;
    if (plan_.buffer_versions[buffer_id] > 1) {
      ffi::Array<Range> rewritten;
      rewritten.reserve(region.size() + 1);
      if (current_operation_id_.has_value()) {
        rewritten.push_back(Range::FromMinExtent(VersionIndex(buffer_id),
                                                 IntImm(DataType::Int(32), 1)));
      } else {
        rewritten.push_back(Range::FromMinExtent(
            IntImm(DataType::Int(32), 0),
            IntImm(DataType::Int(32), plan_.buffer_versions[buffer_id])));
      }
      rewritten.insert(rewritten.end(), region.begin(), region.end());
      region = std::move(rewritten);
    }
    return tirx::BufferRegion(buffer, region);
  }

  void AppendAccessRegions(const tirx::BufferRegion &region,
                           ffi::Array<tirx::BufferRegion> *out) const {
    auto iterator = buffer_ids_.find(region->buffer);
    if (iterator != buffer_ids_.end() && IsFragmentPingPong(iterator->second) &&
        current_combined_slot_ < 0) {
      for (const tirx::Buffer &slot : fragment_slots_[iterator->second]) {
        out->push_back(tirx::BufferRegion(slot, region->region));
      }
      return;
    }
    out->push_back(RewriteBufferRegion(region));
  }

  tirx::Stmt VisitStmt_(const tirx::AttrStmtNode *op) final {
    if (op->attr_key != kOperationScope) {
      return tirx::StmtExprMutator::VisitStmt_(op);
    }
    const auto *operation_id = op->node.as<IntImmNode>();
    ICHECK(operation_id != nullptr);
    ICHECK_GE(operation_id->value, 0);
    ICHECK_LT(operation_id->value,
              static_cast<int64_t>(program_ir_.operations.size()));
    std::optional<int64_t> previous_operation = current_operation_id_;
    int64_t previous_slot = current_combined_slot_;
    current_operation_id_ = operation_id->value;
    std::vector<int64_t> pingpong =
        PingPongBuffersForOperation(operation_id->value);
    int64_t combined = CombinedFragmentModulus(pingpong);
    if (combined > 1) {
      int64_t region = versioned_region_ids_[pingpong.front()].value();
      for (int64_t buffer_id : pingpong) {
        ICHECK_EQ(versioned_region_ids_[buffer_id].value(), region)
            << "fragment ping-pong buffers in one operation must share a "
               "pipeline region";
      }
    }

    auto build_body = [&](int64_t slot) {
      current_combined_slot_ = slot;
      std::vector<tirx::Stmt> statements;
      for (const FragmentHandoff &handoff : fragment_handoffs_) {
        if (handoff.consumer_operation_id == operation_id->value) {
          statements.push_back(
              MakeFragmentHandoffCopy(handoff, /*import=*/true));
        }
      }
      statements.push_back(VisitStmt(op->body));
      for (const FragmentHandoff &handoff : fragment_handoffs_) {
        if (handoff.producer_operation_id == operation_id->value) {
          statements.push_back(
              MakeFragmentHandoffCopy(handoff, /*import=*/false));
        }
      }
      return MakeSequence(std::move(statements));
    };

    tirx::Stmt body;
    if (combined <= 1) {
      body = build_body(-1);
    } else {
      PrimExpr selector = PipelineModulusIndex(pingpong.front(), combined);
      body = build_body(combined - 1);
      for (int64_t slot = combined - 2; slot >= 0; --slot) {
        body = tirx::IfThenElse(selector == IntImm(selector.dtype(), slot),
                                build_body(slot), body);
      }
    }
    current_operation_id_ = previous_operation;
    current_combined_slot_ = previous_slot;
    return tirx::AttrStmt(op->node, op->attr_key, op->value, std::move(body),
                          op->span);
  }

  tirx::Stmt VisitStmt_(const tirx::ForNode *op) final {
    auto region_annotation = op->annotations.Get(kRegionIdAnnotation);
    if (!region_annotation.has_value()) {
      return tirx::StmtExprMutator::VisitStmt_(op);
    }
    const auto *region_id = region_annotation.value().as<IntImmNode>();
    ICHECK(region_id != nullptr);
    ICHECK_GE(region_id->value, 0);
    ICHECK_LT(region_id->value,
              static_cast<int64_t>(planned_buffers_by_region_.size()));
    std::optional<tirx::For> previous_loop = current_pipeline_loop_;
    current_pipeline_loop_ = ffi::GetRef<tirx::For>(op);
    tirx::For loop = Downcast<tirx::For>(tirx::StmtExprMutator::VisitStmt_(op));
    current_pipeline_loop_ = previous_loop;
    ffi::Array<tirx::Var> planned_buffers;
    std::vector<int64_t> ordered_buffer_ids(
        planned_buffers_by_region_[region_id->value].begin(),
        planned_buffers_by_region_[region_id->value].end());
    std::sort(ordered_buffer_ids.begin(), ordered_buffer_ids.end());
    for (int64_t buffer_id : ordered_buffer_ids) {
      if (IsFragmentPingPong(buffer_id)) {
        for (const tirx::Buffer &slot : fragment_slots_[buffer_id]) {
          planned_buffers.push_back(slot->data);
        }
        continue;
      }
      planned_buffers.push_back(
          RemapBuffer(program_ir_.buffers[buffer_id])->data);
    }
    std::vector<size_t> ordered_handoff_ids(
        planned_handoffs_by_region_[region_id->value].begin(),
        planned_handoffs_by_region_[region_id->value].end());
    std::sort(ordered_handoff_ids.begin(), ordered_handoff_ids.end());
    for (size_t handoff_id : ordered_handoff_ids) {
      planned_buffers.push_back(
          fragment_handoffs_[handoff_id].shared_buffer->data);
    }
    ffi::Map<ffi::String, ffi::Any> annotations = loop->annotations;
    annotations.Set(kPipelinePlannedVersionBuffers, planned_buffers);
    loop.CopyOnWrite()->annotations = std::move(annotations);
    return loop;
  }

  tirx::Stmt VisitStmt_(const tirx::SBlockNode *op) final {
    tirx::SBlock block =
        Downcast<tirx::SBlock>(tirx::StmtExprMutator::VisitStmt_(op));
    ffi::Array<tirx::Buffer> alloc_buffers;
    std::vector<std::pair<tirx::Buffer, tirx::Buffer>> remapped_allocations;
    for (const tirx::Buffer &old_buffer : op->alloc_buffers) {
      auto id_iterator = buffer_ids_.find(old_buffer);
      if (id_iterator != buffer_ids_.end() &&
          IsFragmentPingPong(id_iterator->second)) {
        ++allocation_counts_[id_iterator->second];
        for (const tirx::Buffer &slot : fragment_slots_[id_iterator->second]) {
          alloc_buffers.push_back(slot);
          remapped_allocations.emplace_back(old_buffer, slot);
        }
      } else {
        tirx::Buffer new_buffer = RemapBuffer(old_buffer);
        alloc_buffers.push_back(new_buffer);
        if (!new_buffer.same_as(old_buffer)) {
          ++allocation_counts_[BufferId(old_buffer)];
          remapped_allocations.emplace_back(old_buffer, new_buffer);
        }
      }
      auto handoffs = handoffs_by_buffer_.find(old_buffer);
      if (handoffs != handoffs_by_buffer_.end()) {
        for (size_t handoff_id : handoffs->second) {
          alloc_buffers.push_back(fragment_handoffs_[handoff_id].shared_buffer);
          ++handoff_allocation_counts_[handoff_id];
        }
      }
    }
    ffi::Array<tirx::BufferRegion> reads;
    for (const tirx::BufferRegion &region : block->reads) {
      AppendAccessRegions(region, &reads);
    }
    ffi::Array<tirx::BufferRegion> writes;
    for (const tirx::BufferRegion &region : block->writes) {
      AppendAccessRegions(region, &writes);
    }
    ffi::Array<tirx::MatchBufferRegion> match_buffers;
    for (const tirx::MatchBufferRegion &match : block->match_buffers) {
      ffi::Array<tirx::BufferRegion> sources;
      AppendAccessRegions(match->source, &sources);
      auto source_id = buffer_ids_.find(match->source->buffer);
      if (source_id != buffer_ids_.end() &&
          IsFragmentPingPong(source_id->second) && current_combined_slot_ < 0) {
        const std::vector<tirx::Buffer> &slots =
            fragment_slots_[source_id->second];
        ICHECK_EQ(sources.size(), slots.size());
        for (size_t index = 0; index < slots.size(); ++index) {
          match_buffers.push_back(
              tirx::MatchBufferRegion(slots[index], sources[index]));
        }
      } else {
        ICHECK_EQ(sources.size(), 1)
            << "match_buffer source cannot expand to multiple fragment slots";
        match_buffers.push_back(
            tirx::MatchBufferRegion(RemapBuffer(match->buffer), sources[0]));
      }
    }
    tirx::SBlockNode *node = block.CopyOnWrite();
    node->alloc_buffers = std::move(alloc_buffers);
    node->reads = std::move(reads);
    node->writes = std::move(writes);
    node->match_buffers = std::move(match_buffers);
    UpdateLayoutMap(remapped_allocations, &node->annotations);
    return block;
  }

  tirx::Stmt VisitStmt_(const tirx::AllocBufferNode *op) final {
    auto id_iterator = buffer_ids_.find(op->buffer);
    if (id_iterator != buffer_ids_.end() &&
        IsFragmentPingPong(id_iterator->second)) {
      ++allocation_counts_[id_iterator->second];
      ffi::Array<tirx::Stmt> allocations;
      for (const tirx::Buffer &slot : fragment_slots_[id_iterator->second]) {
        allocations.push_back(
            tirx::AllocBuffer(slot, op->annotations, op->span));
      }
      return allocations.size() == 1 ? allocations[0]
                                     : tirx::SeqStmt(allocations);
    }
    tirx::AllocBuffer allocation =
        Downcast<tirx::AllocBuffer>(tirx::StmtExprMutator::VisitStmt_(op));
    tirx::Buffer new_buffer = RemapBuffer(op->buffer);
    if (!new_buffer.same_as(op->buffer)) {
      allocation.CopyOnWrite()->buffer = new_buffer;
      ++allocation_counts_[BufferId(op->buffer)];
    }
    return allocation;
  }

  tirx::Stmt VisitStmt_(const tirx::DeclBufferNode *op) final {
    auto id_iterator = buffer_ids_.find(op->buffer);
    if (id_iterator != buffer_ids_.end() &&
        IsFragmentPingPong(id_iterator->second)) {
      ffi::Array<tirx::Stmt> declarations;
      for (const tirx::Buffer &slot : fragment_slots_[id_iterator->second]) {
        declarations.push_back(tirx::DeclBuffer(slot, op->span));
      }
      return declarations.size() == 1 ? declarations[0]
                                      : tirx::SeqStmt(declarations);
    }
    tirx::DeclBuffer declaration =
        Downcast<tirx::DeclBuffer>(tirx::StmtExprMutator::VisitStmt_(op));
    declaration.CopyOnWrite()->buffer = RemapBuffer(op->buffer);
    return declaration;
  }

  PrimExpr VisitExpr_(const tirx::VarNode *op) final {
    tirx::Var variable = ffi::GetRef<tirx::Var>(op);
    auto iterator = var_remap_.find(variable);
    if (iterator != var_remap_.end()) {
      return iterator->second;
    }
    return tirx::StmtExprMutator::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const tirx::BufferLoadNode *op) final {
    tirx::BufferLoad load =
        Downcast<tirx::BufferLoad>(tirx::StmtExprMutator::VisitExpr_(op));
    auto iterator = buffer_ids_.find(op->buffer);
    if (iterator == buffer_ids_.end()) {
      return load;
    }
    tirx::BufferLoadNode *node = load.CopyOnWrite();
    node->buffer = RemapBuffer(op->buffer);
    node->indices = RewriteIndices(op->buffer, load->indices);
    return load;
  }

  tirx::Stmt VisitStmt_(const tirx::BufferStoreNode *op) final {
    tirx::BufferStore store =
        Downcast<tirx::BufferStore>(tirx::StmtExprMutator::VisitStmt_(op));
    auto iterator = buffer_ids_.find(op->buffer);
    if (iterator == buffer_ids_.end()) {
      return store;
    }
    tirx::BufferStoreNode *node = store.CopyOnWrite();
    node->buffer = RemapBuffer(op->buffer);
    node->indices = RewriteIndices(op->buffer, store->indices);
    return store;
  }

  PrimExpr VisitExpr_(const tirx::CallNode *op) final {
    tirx::Call call =
        Downcast<tirx::Call>(tirx::StmtExprMutator::VisitExpr_(op));
    if (call->op.same_as(tirx::builtin::tvm_access_ptr()) &&
        call->args.size() >= 3) {
      return RewriteTVMAccessPtr(std::move(call));
    }
    if (call->op.same_as(RegionOp::Get()) && call->args.size() >= 2) {
      const auto *load = call->args[0].as<tirx::BufferLoadNode>();
      if (load != nullptr) {
        auto iterator = var_buffer_ids_.find(load->buffer->data);
        if (iterator != var_buffer_ids_.end() &&
            plan_.buffer_versions[iterator->second] > 1 &&
            !IsFragmentPingPong(iterator->second)) {
          size_t num_extents = call->args.size() - 2;
          ICHECK_EQ(load->indices.size(), num_extents + 1)
              << "versioned tile region rank does not match its BufferLoad";
          ffi::Array<PrimExpr> arguments;
          arguments.push_back(call->args[0]);
          arguments.push_back(call->args[1]);
          arguments.push_back(IntImm(DataType::Int(32), 1));
          for (size_t index = 2; index < call->args.size(); ++index) {
            arguments.push_back(call->args[index]);
          }
          return tirx::Call(call->dtype, call->op, arguments, call->annotations,
                            call->span);
        }
      }
    }
    return call;
  }

  PrimExpr RewriteTVMAccessPtr(tirx::Call call) const {
    const auto *data_var = call->args[1].as<tirx::VarNode>();
    if (data_var == nullptr) {
      return call;
    }
    auto iterator = var_buffer_ids_.find(ffi::GetRef<tirx::Var>(data_var));
    if (iterator == var_buffer_ids_.end() ||
        plan_.buffer_versions[iterator->second] == 1) {
      return call;
    }
    int64_t buffer_id = iterator->second;
    if (IsFragmentPingPong(buffer_id)) {
      ICHECK_GE(current_combined_slot_, 0)
          << "fragment ping-pong tvm_access_ptr requires a selected version "
             "slot";
      const tirx::Buffer &slot =
          fragment_slots_[buffer_id][current_combined_slot_ %
                                     plan_.buffer_versions[buffer_id]];
      ffi::Array<PrimExpr> arguments = call->args;
      arguments.Set(1, slot->data);
      return tirx::Call(call->dtype, call->op, arguments, call->annotations,
                        call->span);
    }
    const tirx::Buffer &old_buffer = program_ir_.buffers[buffer_id];
    const tirx::Buffer new_buffer = RemapBuffer(old_buffer);
    PrimExpr offset = call->args[2];
    PrimExpr version_stride = make_const(offset.dtype(), 1);
    if (!new_buffer->strides.empty()) {
      version_stride = tvm::cast(offset.dtype(), new_buffer->strides[0]);
    } else {
      for (const PrimExpr &extent : old_buffer->shape) {
        version_stride = version_stride * tvm::cast(offset.dtype(), extent);
      }
    }
    ffi::Array<PrimExpr> arguments = call->args;
    arguments.Set(2,
                  offset + tvm::cast(offset.dtype(), VersionIndex(buffer_id)) *
                               version_stride);
    return tirx::Call(call->dtype, call->op, arguments, call->annotations,
                      call->span);
  }

  void UpdateLayoutMap(
      const std::vector<std::pair<tirx::Buffer, tirx::Buffer>> &remapped,
      ffi::Map<ffi::String, ffi::Any> *annotations) const {
    if (remapped.empty() || !annotations->count(attr::kLayoutMap)) {
      return;
    }
    auto layout_map = annotations->Get(attr::kLayoutMap)
                          .value()
                          .as<ffi::Map<tirx::Var, Layout>>();
    if (!layout_map.has_value()) {
      return;
    }
    ffi::Map<tirx::Var, Layout> rewritten = layout_map.value();
    std::unordered_map<int64_t, Layout> source_layouts;
    for (const auto &[old_buffer, new_buffer] : remapped) {
      int64_t buffer_id = BufferId(old_buffer);
      if (rewritten.count(old_buffer->data)) {
        source_layouts[buffer_id] = rewritten[old_buffer->data];
        rewritten.erase(old_buffer->data);
      }
    }
    for (const auto &[old_buffer, new_buffer] : remapped) {
      int64_t buffer_id = BufferId(old_buffer);
      auto layout_iterator = source_layouts.find(buffer_id);
      if (layout_iterator == source_layouts.end()) {
        continue;
      }
      if (plan_.buffer_communications[buffer_id] == 4) {
        continue;
      }
      Layout layout = layout_iterator->second;
      if (plan_.buffer_versions[buffer_id] > 1 &&
          !IsFragmentPingPong(buffer_id)) {
        ffi::Array<PrimExpr> leading_shape{
            IntImm(DataType::Int(32), plan_.buffer_versions[buffer_id])};
        layout = layout->Expand(leading_shape);
      }
      rewritten.Set(new_buffer->data, layout);
    }
    annotations->Set(attr::kLayoutMap, rewritten);
  }

  void ValidateAllocations() const {
    for (const auto &[old_buffer, _] : buffer_remap_) {
      int64_t buffer_id = BufferId(old_buffer);
      ICHECK_EQ(allocation_counts_[buffer_id], 1)
          << "planned buffer " << old_buffer->name
          << " must have exactly one physical allocation, found "
          << allocation_counts_[buffer_id];
    }
    for (size_t buffer_id = 0; buffer_id < fragment_slots_.size();
         ++buffer_id) {
      if (fragment_slots_[buffer_id].empty()) {
        continue;
      }
      ICHECK_EQ(allocation_counts_[buffer_id], 1)
          << "planned fragment " << program_ir_.buffers[buffer_id]->name
          << " must have exactly one physical allocation site, found "
          << allocation_counts_[buffer_id];
    }
    for (size_t handoff_id = 0; handoff_id < fragment_handoffs_.size();
         ++handoff_id) {
      ICHECK_EQ(handoff_allocation_counts_[handoff_id], 1)
          << "fragment handoff buffer "
          << fragment_handoffs_[handoff_id].shared_buffer->name
          << " must have exactly one physical allocation";
    }
  }

  const LoweringView &plan_;
  const OverlapIR &program_ir_;
  BufferIdMap buffer_ids_;
  BufferMap buffer_remap_;
  VarMap var_remap_;
  VarIdMap var_buffer_ids_;
  std::vector<std::optional<int64_t>> versioned_region_ids_;
  std::vector<int64_t> allocation_counts_;
  std::vector<std::unordered_set<int64_t>> planned_buffers_by_region_;
  std::vector<std::unordered_set<size_t>> planned_handoffs_by_region_;
  std::vector<FragmentHandoff> fragment_handoffs_;
  std::unordered_map<tirx::Buffer, std::vector<size_t>, ffi::ObjectPtrHash,
                     ffi::ObjectPtrEqual>
      handoffs_by_buffer_;
  std::vector<int64_t> handoff_allocation_counts_;
  std::optional<int64_t> current_operation_id_;
  std::optional<tirx::For> current_pipeline_loop_;
  std::vector<std::vector<tirx::Buffer>> fragment_slots_;
  int64_t current_combined_slot_{-1};
};

} // namespace

tirx::PrimFunc LowerOverlapPlanBuffers(tirx::PrimFunc func,
                                       const LoweringView &plan,
                                       const OverlapIR &program_ir) {
  return ProgramBufferLowerer::Lower(std::move(func), plan, program_ir);
}

} // namespace overlap_plan
} // namespace tl
} // namespace tvm
