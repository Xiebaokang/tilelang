/*!
 * \file lower_program_synchronization.cc
 * \brief Materialize program-schedule synchronization channels.
 */

#include "program_schedule.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <tvm/ir/op.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>

#include "op/builtin.h"
#include "support/check.h"
#include "transform/common/mbarrier.h"
#include "transform/common/pipeline_utils.h"

namespace tvm {
namespace tl {
namespace wspipeline {

namespace {

enum class SynchronizationScope : int64_t {
  kOnce = 0,
  kPerIteration = 1,
  kRegionBoundary = 2,
};

enum class ProducerCompletionMode : int64_t {
  kThreadArrive = 0,
  kAsyncTransaction = 1,
};

struct ChannelLayout {
  int64_t base{0};
  int64_t slots{1};
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

class AsyncTransactionCopyRewriter : public tirx::StmtExprMutator {
public:
  static tirx::Stmt Rewrite(tirx::Stmt statement, PrimExpr barrier_ref,
                            int64_t leader_scope_threads) {
    AsyncTransactionCopyRewriter rewriter(std::move(barrier_ref),
                                          leader_scope_threads);
    tirx::Stmt rewritten = rewriter(std::move(statement));
    ICHECK_EQ(rewriter.rewritten_copies_, 1)
        << "an asynchronous transaction producer must contain exactly one "
           "T.copy operation";
    return rewritten;
  }

private:
  AsyncTransactionCopyRewriter(PrimExpr barrier_ref,
                               int64_t leader_scope_threads)
      : barrier_ref_(std::move(barrier_ref)),
        leader_scope_threads_(leader_scope_threads) {}

  PrimExpr VisitExpr_(const tirx::CallNode *op) final {
    static const Op &copy_op = Op::Get("tl.tileop.copy");
    static const Op &tma_copy_op = Op::Get("tl.tileop.tma_copy");
    tirx::Call call =
        Downcast<tirx::Call>(tirx::StmtExprMutator::VisitExpr_(op));
    if (!call->op.same_as(copy_op)) {
      return call;
    }
    ++rewritten_copies_;
    ffi::Map<ffi::String, ffi::ObjectRef> annotations = call->annotations;
    annotations.Set("barrier", barrier_ref_);
    annotations.Set("is_tma_copy", IntImm(DataType::Int(32), 1));
    annotations.Set("emit_arrive", IntImm(DataType::Int(32), 1));
    annotations.Set("leader_scope_threads",
                    IntImm(DataType::Int(32), leader_scope_threads_));
    // tl_shuffle_elect uses the block-global warp index.  Elect exactly the
    // first warp of this group so a later group whose first_warp is a multiple
    // of this group's warp count cannot also match.
    annotations.Set("leader_first_warp", IntImm(DataType::Int(32), 1));
    return tirx::Call(call->dtype, tma_copy_op, call->args,
                      std::move(annotations), call->span);
  }

  PrimExpr barrier_ref_;
  int64_t leader_scope_threads_;
  int64_t rewritten_copies_{0};
};

class ProgramSynchronizationLowerer : public tirx::StmtMutator {
public:
  static tirx::PrimFunc Lower(tirx::PrimFunc func,
                              const ProgramSchedulePlan &plan,
                              const ProgramScheduleIR &program_ir) {
    if (plan.sync_producers.empty()) {
      return func;
    }
    ProgramSynchronizationLowerer lowerer(plan, program_ir);
    tirx::PrimFuncNode *node = func.CopyOnWrite();
    node->body = lowerer(node->body);
    ICHECK(lowerer.barrier_allocated_)
        << "program synchronization barrier was not allocated";
    return func;
  }

private:
  ProgramSynchronizationLowerer(const ProgramSchedulePlan &plan,
                                const ProgramScheduleIR &program_ir)
      : plan_(plan), program_ir_(program_ir) {
    layouts_.reserve(plan_.sync_producers.size());
    int64_t total_slots = 0;
    for (size_t channel = 0; channel < plan_.sync_producers.size();
         ++channel) {
      int64_t slots = 1;
      if (Scope(channel) == SynchronizationScope::kPerIteration) {
        int64_t region_id = plan_.sync_producer_regions[channel];
        ICHECK_GE(region_id, 0);
        ICHECK_EQ(region_id, plan_.sync_consumer_regions[channel]);
        if (!plan_.sync_slot_counts.empty()) {
          slots = plan_.sync_slot_counts[channel];
        } else {
          // Preserve the original wspipeline contract when the optional exact
          // slot-count array is absent.
          slots = plan_.region_num_stages[region_id];
          int64_t buffer_id = plan_.sync_buffer_ids[channel];
          if (buffer_id >= 0) {
            slots = std::max(slots, plan_.buffer_versions[buffer_id]);
          }
        }
        ICHECK_GT(slots, 0)
            << "per-iteration synchronization requires a pipeline region";
      }
      layouts_.push_back(ChannelLayout{total_slots, slots});
      int64_t producer_group = plan_.sync_producer_groups[channel];
      int64_t arrive_count =
          CompletionMode(channel) == ProducerCompletionMode::kAsyncTransaction
              ? 1
              : plan_.group_warp_counts[producer_group] * 32;
      for (int64_t slot = 0; slot < slots; ++slot) {
        arrive_counts_.push_back(IntImm(DataType::Int(32), arrive_count));
      }
      total_slots += slots;
    }
    barrier_ = CreateMBarrierBuffer("program_schedule_mbar",
                                    static_cast<int>(total_slots));
  }

  SynchronizationScope Scope(size_t channel) const {
    int64_t code = plan_.sync_scopes[channel];
    ICHECK_GE(code, 0)
        << "program schedule synchronization requires an explicit scope";
    ICHECK_LE(code, 2);
    return static_cast<SynchronizationScope>(code);
  }

  ProducerCompletionMode CompletionMode(size_t channel) const {
    if (plan_.sync_completion_modes.empty()) {
      return ProducerCompletionMode::kThreadArrive;
    }
    int64_t code = plan_.sync_completion_modes[channel];
    ICHECK_GE(code, 0);
    ICHECK_LE(code, 1);
    return static_cast<ProducerCompletionMode>(code);
  }

  bool IsGroup(size_t channel, bool producer) const {
    ICHECK_GE(current_group_, 0);
    return current_group_ ==
           (producer ? plan_.sync_producer_groups[channel]
                     : plan_.sync_consumer_groups[channel]);
  }

  tirx::Stmt MakeArrive(size_t channel) const {
    PrimExpr barrier_ref =
        MakeBarrierRef(barrier_, BarrierIndex(channel, /*wait=*/false));
    tirx::Stmt arrive = tirx::Evaluate(tirx::Call(
        DataType::Handle(), tirx::builtin::ptx_arrive_barrier(),
        {barrier_ref}));
    if (!NeedsProducerProxyFence(channel)) {
      return arrive;
    }
    return MakeSequence({
        tirx::Evaluate(
            tirx::Call(DataType::Handle(), fence_proxy_async(), {})),
        std::move(arrive),
    });
  }

  tirx::Stmt MakeWait(size_t channel) const {
    PrimExpr barrier_ref =
        MakeBarrierRef(barrier_, BarrierIndex(channel, /*wait=*/true));
    PrimExpr parity = BarrierParity(channel, /*wait=*/true);
    tirx::Stmt completion = tirx::Evaluate(tirx::Call(
        DataType::Handle(), mbarrier_wait_parity(), {barrier_ref, parity}));
    if (Scope(channel) == SynchronizationScope::kRegionBoundary) {
      int64_t consumer_region = plan_.sync_consumer_regions[channel];
      ICHECK_GE(consumer_region, 0);
      ICHECK_LT(consumer_region,
                static_cast<int64_t>(program_ir_.regions.size()));
      if (program_ir_.regions[consumer_region].pipeline_loop.has_value()) {
        tirx::For loop = RegionPipelineLoop(consumer_region);
        return tirx::IfThenElse(loop->loop_var == loop->min, completion);
      }
      return completion;
    }
    if (Scope(channel) != SynchronizationScope::kPerIteration ||
        plan_.sync_iteration_distances[channel] == 0) {
      return completion;
    }
    tirx::For loop = PipelineLoop(channel);
    PrimExpr completed_iterations = loop->loop_var - loop->min;
    PrimExpr distance = IntImm(loop->loop_var.dtype(),
                               plan_.sync_iteration_distances[channel]);
    return tirx::IfThenElse(completed_iterations >= distance, completion);
  }

  bool NeedsProducerProxyFence(size_t channel) const {
    if (plan_.sync_kinds[channel] != 0) {
      return false;
    }
    int64_t buffer_id = plan_.sync_buffer_ids[channel];
    if (buffer_id < 0) {
      return false;
    }
    const tirx::Buffer &buffer = program_ir_.buffers[buffer_id];
    if (buffer.scope() == "local.fragment" &&
        (plan_.sync_dependency_masks[channel] & 1) != 0) {
      return true;
    }
    int64_t communication = plan_.buffer_communications[buffer_id];
    return communication == 2 || communication == 4;
  }

  tirx::For PipelineLoop(size_t channel) const {
    int64_t region_id = plan_.sync_producer_regions[channel];
    return RegionPipelineLoop(region_id);
  }

  tirx::For RegionPipelineLoop(int64_t region_id) const {
    ICHECK_GE(region_id, 0);
    ICHECK_LT(region_id, static_cast<int64_t>(program_ir_.regions.size()));
    ICHECK(program_ir_.regions[region_id].pipeline_loop.has_value());
    return program_ir_.regions[region_id].pipeline_loop.value();
  }

  PrimExpr EventEpoch(size_t channel, bool wait) const {
    tirx::For loop = PipelineLoop(channel);
    PrimExpr epoch = loop->loop_var;
    if (wait && plan_.sync_iteration_distances[channel] != 0) {
      epoch = epoch - IntImm(loop->loop_var.dtype(),
                             plan_.sync_iteration_distances[channel]);
    }
    return epoch - loop->min;
  }

  PrimExpr BarrierIndex(size_t channel, bool wait) const {
    const ChannelLayout &layout = layouts_[channel];
    PrimExpr base = IntImm(DataType::Int(32), layout.base);
    if (Scope(channel) != SynchronizationScope::kPerIteration) {
      return base;
    }
    PrimExpr slots = IntImm(DataType::Int(32), layout.slots);
    return base + tirx::FloorMod(EventEpoch(channel, wait), slots);
  }

  PrimExpr BarrierParity(size_t channel, bool wait) const {
    if (Scope(channel) != SynchronizationScope::kPerIteration) {
      return IntImm(DataType::Int(32), 0);
    }
    PrimExpr slots = IntImm(DataType::Int(32), layouts_[channel].slots);
    return tirx::FloorMod(tirx::FloorDiv(EventEpoch(channel, wait), slots),
                          IntImm(DataType::Int(32), 2));
  }

  tirx::Stmt VisitStmt_(const tirx::AttrStmtNode *op) final {
    if (op->attr_key == kProgramScheduleGroupScope) {
      const auto *group_id = op->value.as<IntImmNode>();
      ICHECK(group_id != nullptr);
      int64_t previous_group = current_group_;
      current_group_ = group_id->value;
      tirx::Stmt body = VisitStmt(op->body);
      current_group_ = previous_group;
      return tirx::AttrStmt(op->node, op->attr_key, op->value,
                            std::move(body), op->span);
    }
    if (op->attr_key != kOperationScope) {
      return tirx::StmtMutator::VisitStmt_(op);
    }

    const auto *operation_id = op->node.as<IntImmNode>();
    ICHECK(operation_id != nullptr);
    std::vector<tirx::Stmt> statements;
    for (size_t channel = 0; channel < plan_.sync_producers.size();
         ++channel) {
      if (plan_.sync_consumers[channel] == operation_id->value &&
          IsGroup(channel, /*producer=*/false) &&
          OperationAnchorsEvent(channel, /*producer=*/false)) {
        statements.push_back(MakeWait(channel));
      }
    }
    tirx::Stmt body = VisitStmt(op->body);
    std::optional<size_t> transaction_channel;
    for (size_t channel = 0; channel < plan_.sync_producers.size();
         ++channel) {
      if (plan_.sync_producers[channel] == operation_id->value &&
          IsGroup(channel, /*producer=*/true) &&
          CompletionMode(channel) ==
              ProducerCompletionMode::kAsyncTransaction) {
        ICHECK(!transaction_channel.has_value())
            << "one operation cannot own multiple transaction barriers";
        transaction_channel = channel;
      }
    }
    if (transaction_channel.has_value()) {
      size_t channel = transaction_channel.value();
      PrimExpr barrier_ref =
          MakeBarrierRef(barrier_, BarrierIndex(channel, /*wait=*/false));
      int64_t producer_group = plan_.sync_producer_groups[channel];
      int64_t leader_scope_threads =
          plan_.group_warp_counts[producer_group] * 32;
      body = AsyncTransactionCopyRewriter::Rewrite(
          std::move(body), std::move(barrier_ref), leader_scope_threads);
    }
    statements.push_back(std::move(body));
    for (size_t channel = 0; channel < plan_.sync_producers.size();
         ++channel) {
      if (plan_.sync_producers[channel] == operation_id->value &&
          IsGroup(channel, /*producer=*/true) &&
          CompletionMode(channel) == ProducerCompletionMode::kThreadArrive &&
          OperationAnchorsEvent(channel, /*producer=*/true)) {
        statements.push_back(MakeArrive(channel));
      }
    }
    return tirx::AttrStmt(op->node, op->attr_key, op->value,
                          MakeSequence(std::move(statements)), op->span);
  }

  bool OperationAnchorsEvent(size_t channel, bool producer) const {
    if (Scope(channel) != SynchronizationScope::kRegionBoundary) {
      return true;
    }
    // A region-boundary wait belongs immediately before the actual consumer.
    // If that consumer is in a pipeline loop, MakeWait guards it with the
    // first logical iteration so the one-shot event is not waited repeatedly.
    if (!producer) {
      return true;
    }
    int64_t region_id = plan_.sync_producer_regions[channel];
    ICHECK_GE(region_id, 0);
    ICHECK_LT(region_id, static_cast<int64_t>(program_ir_.regions.size()));
    return !program_ir_.regions[region_id].pipeline_loop.has_value();
  }

  tirx::Stmt VisitStmt_(const tirx::ForNode *op) final {
    tirx::For loop =
        Downcast<tirx::For>(tirx::StmtMutator::VisitStmt_(op));
    auto region_annotation = loop->annotations.Get(kRegionIdAnnotation);
    if (!region_annotation.has_value()) {
      return loop;
    }
    const auto *region_id = region_annotation.value().as<IntImmNode>();
    ICHECK(region_id != nullptr);
    ffi::Map<ffi::String, ffi::Any> annotations = loop->annotations;
    annotations.Set(kPipelinePlannedBarrierBuffers,
                    ffi::Array<tirx::Var>{barrier_->data});
    loop.CopyOnWrite()->annotations = std::move(annotations);

    std::vector<tirx::Stmt> statements;
    statements.push_back(loop);
    for (size_t channel = 0; channel < plan_.sync_producers.size();
         ++channel) {
      if (Scope(channel) == SynchronizationScope::kRegionBoundary &&
          plan_.sync_producer_regions[channel] == region_id->value &&
          IsGroup(channel, /*producer=*/true) &&
          CompletionMode(channel) == ProducerCompletionMode::kThreadArrive) {
        statements.push_back(MakeArrive(channel));
      }
    }
    return MakeSequence(std::move(statements));
  }

  tirx::Stmt VisitStmt_(const tirx::SBlockNode *op) final {
    if (barrier_allocated_) {
      return tirx::StmtMutator::VisitStmt_(op);
    }
    if (op->name_hint == "root") {
      return tirx::StmtMutator::VisitStmt_(op);
    }
    barrier_allocated_ = true;
    tirx::SBlock block = ffi::GetRef<tirx::SBlock>(op);
    tirx::SBlockNode *node = block.CopyOnWrite();
    node->body = VisitStmt(node->body);
    node->alloc_buffers.push_back(barrier_);
    ffi::Map<tirx::Var, ffi::Array<PrimExpr>> barrier_init;
    if (node->annotations.count("barrier_init")) {
      barrier_init = Downcast<ffi::Map<tirx::Var, ffi::Array<PrimExpr>>>(
          node->annotations.Get("barrier_init").value());
    }
    barrier_init.Set(barrier_->data, arrive_counts_);
    node->annotations.Set("barrier_init", barrier_init);
    return block;
  }

  const ProgramSchedulePlan &plan_;
  const ProgramScheduleIR &program_ir_;
  std::vector<ChannelLayout> layouts_;
  tirx::Buffer barrier_;
  ffi::Array<PrimExpr> arrive_counts_;
  int64_t current_group_{-1};
  bool barrier_allocated_{false};
};

} // namespace

tirx::PrimFunc LowerProgramScheduleSynchronization(
    tirx::PrimFunc func, const ProgramSchedulePlan &plan,
    const ProgramScheduleIR &program_ir) {
  func = ProgramSynchronizationLowerer::Lower(std::move(func), plan,
                                              program_ir);
  return WithAttr(std::move(func),
                  "tl.program_schedule.synchronization_lowered", Integer(1));
}

} // namespace wspipeline
} // namespace tl
} // namespace tvm
