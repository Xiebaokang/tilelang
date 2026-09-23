/*!
 * \file lower_program_schedule.cc
 * \brief Materialize program schedule groups in TIRX.
 */

#include "program_schedule.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "op/builtin.h"
#include "transform/common/pipeline_utils.h"
#include "support/check.h"

namespace tvm {
namespace tl {
namespace wspipeline {

namespace {

using OperationIdMap =
    std::unordered_map<tirx::Stmt, int64_t, ffi::ObjectPtrHash,
                       ffi::ObjectPtrEqual>;
using PipelineRegionIdMap =
    std::unordered_map<tirx::For, int64_t, ffi::ObjectPtrHash,
                       ffi::ObjectPtrEqual>;

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
      ICHECK(operation_id != nullptr && group_id != nullptr)
          << kOperationScope << " must carry static operation/group IDs";
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

class ThreadEnvironmentCollector : public tirx::StmtExprVisitor {
public:
  static tirx::IterVar Collect(const tirx::Stmt &body) {
    ThreadEnvironmentCollector collector;
    collector(body);
    ICHECK(collector.thread_x_.defined())
        << "LowerProgramSchedule requires a materialized threadIdx.x "
           "thread_extent; run MaterializeKernelLaunch first";
    return collector.thread_x_;
  }

private:
  void VisitStmt_(const tirx::AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::thread_extent) {
      ffi::Optional<tirx::IterVar> iter_var = op->node.as<tirx::IterVar>();
      ICHECK(iter_var.has_value())
          << "thread_extent node must be an IterVar";
      std::string thread_tag = iter_var.value()->thread_tag;
      if (thread_tag == "threadIdx.x") {
        ICHECK(!thread_x_.defined())
            << "LowerProgramSchedule requires exactly one threadIdx.x extent";
        thread_x_ = iter_var.value();
      } else if (thread_tag.rfind("threadIdx.", 0) == 0) {
        const auto *extent = op->value.as<IntImmNode>();
        ICHECK(extent != nullptr && extent->value == 1)
            << "LowerProgramSchedule currently supports only a one-dimensional "
               "threadIdx.x launch; non-unit "
            << thread_tag << " is not supported";
      }
    }
    tirx::StmtExprVisitor::VisitStmt_(op);
  }

  tirx::IterVar thread_x_;
};

class OperationGroupFilter : public tirx::StmtMutator {
public:
  OperationGroupFilter(const OperationIdMap &operation_ids,
                       const PipelineRegionIdMap &pipeline_region_ids,
                       const ProgramSchedulePlan &plan, int64_t group_id,
                       std::vector<int64_t> *visit_counts,
                       std::vector<int64_t> *keep_counts)
      : operation_ids_(operation_ids),
        pipeline_region_ids_(pipeline_region_ids), plan_(plan),
        group_id_(group_id), visit_counts_(visit_counts),
        keep_counts_(keep_counts) {}

private:
  ffi::Optional<tirx::Stmt> FilterOperation(const tirx::Stmt &statement) {
    auto iterator = operation_ids_.find(statement);
    if (iterator == operation_ids_.end()) {
      return std::nullopt;
    }
    int64_t operation_id = iterator->second;
    ++(*visit_counts_)[operation_id];
    if (plan_.operation_groups[operation_id] == group_id_) {
      ++(*keep_counts_)[operation_id];
      return tirx::AttrStmt(Integer(operation_id), kOperationScope,
                            IntImm(DataType::Int(32), group_id_), statement);
    }
    return tirx::Evaluate(0);
  }

  tirx::Stmt VisitStmt_(const tirx::ForNode *op) final {
    tirx::For loop = ffi::GetRef<tirx::For>(op);
    if (ffi::Optional<tirx::Stmt> filtered = FilterOperation(loop)) {
      return filtered.value();
    }
    tirx::Stmt rewritten_statement = tirx::StmtMutator::VisitStmt_(op);
    auto region_iterator = pipeline_region_ids_.find(loop);
    if (region_iterator == pipeline_region_ids_.end()) {
      return rewritten_statement;
    }

    tirx::For rewritten = Downcast<tirx::For>(rewritten_statement);
    return AnnotatePipelineLoop(std::move(rewritten), region_iterator->second);
  }

  tirx::Stmt VisitStmt_(const tirx::SeqStmtNode *op) final {
    ffi::Array<tirx::Stmt> statements;
    for (const tirx::Stmt &child : op->seq) {
      auto operation_iterator = operation_ids_.find(child);
      bool remove =
          operation_iterator != operation_ids_.end() &&
          plan_.operation_groups[operation_iterator->second] != group_id_;
      tirx::Stmt rewritten = VisitStmt(child);
      if (!remove) {
        statements.push_back(std::move(rewritten));
      }
    }
    if (statements.empty()) {
      return tirx::Evaluate(0);
    }
    return MakePipelineBody(statements);
  }

  tirx::Stmt VisitStmt_(const tirx::EvaluateNode *op) final {
    tirx::Evaluate evaluate = ffi::GetRef<tirx::Evaluate>(op);
    if (ffi::Optional<tirx::Stmt> filtered = FilterOperation(evaluate)) {
      return filtered.value();
    }
    return tirx::StmtMutator::VisitStmt_(op);
  }

  tirx::Stmt VisitStmt_(const tirx::BufferStoreNode *op) final {
    tirx::BufferStore store = ffi::GetRef<tirx::BufferStore>(op);
    if (ffi::Optional<tirx::Stmt> filtered = FilterOperation(store)) {
      return filtered.value();
    }
    return tirx::StmtMutator::VisitStmt_(op);
  }

  tirx::Stmt AnnotatePipelineLoop(tirx::For loop, int64_t region_id) const {
    int64_t expected_operation_count = 0;
    int64_t minimum_stage = -1;
    for (size_t operation_id = 0;
         operation_id < plan_.operation_groups.size(); ++operation_id) {
      if (plan_.operation_groups[operation_id] == group_id_ &&
          plan_.operation_regions[operation_id] == region_id) {
        ++expected_operation_count;
        int64_t stage = plan_.operation_stages[operation_id];
        ICHECK_GE(stage, 0)
            << "pipeline operation " << operation_id
            << " must have a non-negative stage";
        minimum_stage =
            minimum_stage < 0 ? stage : std::min(minimum_stage, stage);
      }
    }

    ffi::Map<ffi::String, ffi::Any> annotations;
    for (const auto &[key, value] : loop->annotations) {
      if (key != "num_stages" && key != "tl_pipeline_stage" &&
          key != "tl_pipeline_order" && key != "tl_pipeline_group" &&
          key != "tl_pipeline_sync" && key != kAutoScheduleAnnotation &&
          key != kRegionIdAnnotation) {
        annotations.Set(key, value);
      }
    }
    annotations.Set(kRegionIdAnnotation, Integer(region_id));

    if (expected_operation_count == 0) {
      tirx::ForNode *node = loop.CopyOnWrite();
      node->annotations = std::move(annotations);
      return loop;
    }

    ffi::Array<tirx::Stmt> statements = FlattenSequence(loop->body);
    ffi::Array<Integer> stages;
    ffi::Array<Integer> orders;
    std::unordered_set<int64_t> seen_operations;
    int64_t auxiliary_order = expected_operation_count;
    for (const tirx::Stmt &statement : statements) {
      std::vector<int64_t> operation_ids =
          OperationMarkerCollector::Collect(statement, group_id_);
      ICHECK_LE(operation_ids.size(), 1U)
          << "pipeline region " << region_id << " group " << group_id_
          << " contains a top-level statement representing multiple schedule "
             "operations; operation-level stage/order cannot be lowered "
             "without changing its control-flow granularity";
      if (operation_ids.empty()) {
        ICHECK(IsPipelineDeclarationStmt(statement) ||
               statement.as<tirx::BindNode>() != nullptr)
            << "pipeline region " << region_id << " group " << group_id_
            << " contains an executable statement without a program schedule "
               "operation ID";
        stages.push_back(Integer(0));
        orders.push_back(Integer(auxiliary_order++));
        continue;
      }

      int64_t operation_id = operation_ids.front();
      ICHECK_GE(operation_id, 0);
      ICHECK_LT(operation_id,
                static_cast<int64_t>(plan_.operation_groups.size()));
      ICHECK_EQ(plan_.operation_groups[operation_id], group_id_);
      ICHECK_EQ(plan_.operation_regions[operation_id], region_id);
      ICHECK(seen_operations.insert(operation_id).second)
          << "operation " << operation_id
          << " appears more than once in one group pipeline loop";
      // Groups execute independent copies of the pipeline loop.  A common
      // leading stage offset within one group is therefore only idle
      // prologue: subtract it so InjectSoftwarePipeline emits the shortest
      // equivalent prologue/epilogue for this group.
      stages.push_back(
          Integer(plan_.operation_stages[operation_id] - minimum_stage));
      orders.push_back(Integer(plan_.operation_local_orders[operation_id]));
    }
    ICHECK_EQ(seen_operations.size(),
              static_cast<size_t>(expected_operation_count))
        << "pipeline region " << region_id << " group " << group_id_
        << " did not retain exactly the operations described by the schedule";

    annotations.Set("tl_pipeline_stage", stages);
    annotations.Set("tl_pipeline_order", orders);
    tirx::ForNode *node = loop.CopyOnWrite();
    node->annotations = std::move(annotations);
    return loop;
  }

  const OperationIdMap &operation_ids_;
  const PipelineRegionIdMap &pipeline_region_ids_;
  const ProgramSchedulePlan &plan_;
  int64_t group_id_;
  std::vector<int64_t> *visit_counts_;
  std::vector<int64_t> *keep_counts_;
};

class GroupBranchBuilder {
public:
  GroupBranchBuilder(const ProgramSchedulePlan &plan,
                     const ProgramScheduleIR &program_ir,
                     tirx::Var physical_thread_x)
      : plan_(plan), physical_thread_x_(std::move(physical_thread_x)),
        visit_counts_(program_ir.operations.size(), 0),
        keep_counts_(program_ir.operations.size(), 0) {
    for (size_t operation_id = 0;
         operation_id < program_ir.operations.size(); ++operation_id) {
      bool inserted =
          operation_ids_
              .emplace(program_ir.operations[operation_id].statement,
                       static_cast<int64_t>(operation_id))
              .second;
      ICHECK(inserted) << "one TIRX statement cannot represent two operations";
    }
    for (size_t region_id = 0; region_id < program_ir.regions.size();
         ++region_id) {
      if (!program_ir.regions[region_id].pipeline_loop.has_value()) {
        continue;
      }
      bool inserted =
          pipeline_region_ids_
              .emplace(program_ir.regions[region_id].pipeline_loop.value(),
                       static_cast<int64_t>(region_id))
              .second;
      ICHECK(inserted) << "one pipeline loop cannot represent two regions";
    }
  }

  tirx::Stmt Build(const tirx::Stmt &body) {
    std::vector<tirx::Stmt> group_bodies;
    group_bodies.reserve(plan_.group_warp_counts.size());
    for (size_t group_id = 0; group_id < plan_.group_warp_counts.size();
         ++group_id) {
      OperationGroupFilter filter(operation_ids_, pipeline_region_ids_, plan_,
                                  group_id,
                                  &visit_counts_, &keep_counts_);
      tirx::Stmt group_body = filter(body);
      if (plan_.setmaxnreg_enabled) {
        auto domain = std::find(plan_.register_domain_groups.begin(),
                                plan_.register_domain_groups.end(),
                                static_cast<int64_t>(group_id));
        ICHECK(domain != plan_.register_domain_groups.end())
            << "missing register domain for group " << group_id;
        size_t domain_id =
            std::distance(plan_.register_domain_groups.begin(), domain);
        tirx::Stmt set_register_limit = tirx::Evaluate(tirx::Call(
            DataType::Handle(), set_max_nreg(),
            {IntImm(DataType::Int(32),
                    plan_.register_domain_register_counts[domain_id]),
             IntImm(DataType::Int(32),
                    plan_.register_domain_is_increase[domain_id])}));
        group_body = tirx::SeqStmt(
            ffi::Array<tirx::Stmt>{set_register_limit, group_body});
      }
      int64_t first_thread = plan_.group_first_warps[group_id] * 32;
      if (first_thread != 0) {
        PrimExpr local_thread =
            physical_thread_x_ -
            IntImm(physical_thread_x_.dtype(), first_thread);
        group_body = tirx::Substitute(
            std::move(group_body), {{physical_thread_x_, local_thread}});
      }
      int64_t thread_count = plan_.group_warp_counts[group_id] * 32;
      ffi::Array<IntImm> group_range{
          IntImm(DataType::Int(32), first_thread),
          IntImm(DataType::Int(32), thread_count)};
      group_body = tirx::AttrStmt(
          group_range, kProgramScheduleGroupScope,
          IntImm(DataType::Int(32), static_cast<int64_t>(group_id)),
          std::move(group_body));
      group_bodies.push_back(std::move(group_body));
    }

    for (size_t operation_id = 0; operation_id < visit_counts_.size();
         ++operation_id) {
      ICHECK_EQ(visit_counts_[operation_id],
                static_cast<int64_t>(plan_.group_warp_counts.size()))
          << "operation " << operation_id
          << " was not found exactly once while constructing every group";
      ICHECK_EQ(keep_counts_[operation_id], 1)
          << "operation " << operation_id
          << " must be retained by exactly one group";
    }

    tirx::Stmt guarded = tirx::Evaluate(0);
    ffi::Array<IntImm> group_thread_counts;
    for (int64_t group_id =
             static_cast<int64_t>(plan_.group_warp_counts.size()) - 1;
         group_id >= 0; --group_id) {
      int64_t first_thread = plan_.group_first_warps[group_id] * 32;
      int64_t thread_count = plan_.group_warp_counts[group_id] * 32;
      int64_t end_thread = first_thread + thread_count;
      PrimExpr lower = IntImm(physical_thread_x_.dtype(), first_thread);
      PrimExpr upper = IntImm(physical_thread_x_.dtype(), end_thread);
      PrimExpr condition = tirx::And(physical_thread_x_ >= lower,
                                     physical_thread_x_ < upper);
      guarded = tirx::IfThenElse(condition, group_bodies[group_id], guarded);
    }
    for (int64_t warp_count : plan_.group_warp_counts) {
      group_thread_counts.push_back(IntImm(DataType::Int(32), warp_count * 32));
    }
    return tirx::AttrStmt(group_thread_counts,
                          ::tvm::tl::attr::kWarpSpecializationScope,
                          0, guarded);
  }

private:
  const ProgramSchedulePlan &plan_;
  tirx::Var physical_thread_x_;
  OperationIdMap operation_ids_;
  PipelineRegionIdMap pipeline_region_ids_;
  std::vector<int64_t> visit_counts_;
  std::vector<int64_t> keep_counts_;
};

class GroupScopeInserter : public tirx::StmtMutator {
public:
  explicit GroupScopeInserter(GroupBranchBuilder *builder) : builder_(builder) {}

  static tirx::Stmt Insert(const tirx::Stmt &body,
                           GroupBranchBuilder *builder) {
    GroupScopeInserter inserter(builder);
    tirx::Stmt rewritten = inserter(body);
    if (!inserter.inserted_) {
      rewritten = builder->Build(body);
    }
    return rewritten;
  }

private:
  tirx::Stmt VisitStmt_(const tirx::SBlockNode *op) final {
    if (inserted_) {
      return tirx::StmtMutator::VisitStmt_(op);
    }
    inserted_ = true;
    tirx::SBlock block = ffi::GetRef<tirx::SBlock>(op);
    block.CopyOnWrite()->body = builder_->Build(block->body);
    return block;
  }

  GroupBranchBuilder *builder_;
  bool inserted_{false};
};

class ThreadExtentRewriter : public tirx::StmtMutator {
public:
  ThreadExtentRewriter(const ProgramSchedulePlan &plan,
                       const ProgramScheduleIR &program_ir,
                       tirx::IterVar thread_x)
      : plan_(plan), program_ir_(program_ir), thread_x_(std::move(thread_x)) {}

  static tirx::Stmt Rewrite(const tirx::Stmt &body,
                            const ProgramSchedulePlan &plan,
                            const ProgramScheduleIR &program_ir,
                            tirx::IterVar thread_x) {
    ThreadExtentRewriter rewriter(plan, program_ir, std::move(thread_x));
    tirx::Stmt result = rewriter(body);
    ICHECK(rewriter.rewritten_)
        << "LowerProgramSchedule did not find the validated threadIdx.x extent";
    return result;
  }

private:
  tirx::Stmt VisitStmt_(const tirx::AttrStmtNode *op) final {
    if (op->attr_key != tirx::attr::thread_extent) {
      return tirx::StmtMutator::VisitStmt_(op);
    }
    ffi::Optional<tirx::IterVar> iter_var = op->node.as<tirx::IterVar>();
    if (!iter_var.has_value() ||
        !iter_var.value().same_as(thread_x_)) {
      return tirx::StmtMutator::VisitStmt_(op);
    }
    ICHECK(!rewritten_) << "threadIdx.x extent cannot be rewritten twice";
    rewritten_ = true;

    GroupBranchBuilder builder(plan_, program_ir_, thread_x_->var);
    tirx::Stmt grouped_body = GroupScopeInserter::Insert(op->body, &builder);
    PrimExpr new_extent =
        IntImm(op->value.dtype(), plan_.effective_threads);
    tirx::IterVar new_thread_x(
        Range::FromMinExtent(thread_x_->dom->min, new_extent), thread_x_->var,
        thread_x_->iter_type, thread_x_->thread_tag, thread_x_->span);
    return tirx::AttrStmt(new_thread_x, op->attr_key, new_extent,
                          std::move(grouped_body), op->span);
  }

  const ProgramSchedulePlan &plan_;
  const ProgramScheduleIR &program_ir_;
  tirx::IterVar thread_x_;
  bool rewritten_{false};
};

class ProgramScheduleMetadataCleaner : public tirx::StmtMutator {
private:
  tirx::Stmt VisitStmt_(const tirx::AttrStmtNode *op) final {
    if (op->attr_key == kOperationScope ||
        op->attr_key == kProgramScheduleGroupScope) {
      return VisitStmt(op->body);
    }
    if (op->attr_key == ::tvm::tl::attr::kWarpSpecializationScope) {
      return tirx::AttrStmt(
          op->node, ::tvm::tl::attr::kThreadSyncWarpSpecializationScope,
          op->value, VisitStmt(op->body), op->span);
    }
    return tirx::StmtMutator::VisitStmt_(op);
  }

  tirx::Stmt VisitStmt_(const tirx::ForNode *op) final {
    tirx::For loop = Downcast<tirx::For>(tirx::StmtMutator::VisitStmt_(op));
    ffi::Map<ffi::String, ffi::Any> annotations;
    for (const auto &[key, value] : loop->annotations) {
      if (key != kRegionIdAnnotation &&
          key != kAutoScheduleAnnotation &&
          key != kPipelinePlannedVersionBuffers &&
          key != kPipelinePlannedBarrierBuffers) {
        annotations.Set(key, value);
      }
    }
    tirx::ForNode *node = loop.CopyOnWrite();
    node->annotations = std::move(annotations);
    return loop;
  }
};

tirx::PrimFunc FinalizeProgramScheduleFunction(tirx::PrimFunc func) {
  ffi::Optional<Integer> version =
      func->GetAttr<Integer>("tl.program_schedule.version");
  if (!version.has_value()) {
    return func;
  }
  for (const char *completion : {
           "tl.program_schedule.groups_lowered",
           "tl.program_schedule.pipeline_schedules_lowered",
           "tl.program_schedule.buffers_lowered",
           "tl.program_schedule.synchronization_lowered",
           "tl.program_schedule.tile_ops_lowered",
       }) {
    ffi::Optional<Integer> value = func->GetAttr<Integer>(completion);
    ICHECK(value.has_value() && value.value()->value == 1)
        << "FinalizeProgramSchedule requires completed lowering attribute "
        << completion;
  }

  ProgramScheduleMetadataCleaner cleaner;
  tirx::PrimFuncNode *node = func.CopyOnWrite();
  node->body = cleaner(node->body);

  static constexpr const char *kScheduleAttributes[] = {
      "version",
      "effective_threads",
      "setmaxnreg_enabled",
      "operation_groups",
      "operation_regions",
      "operation_stages",
      "operation_local_orders",
      "region_num_stages",
      "group_first_warps",
      "group_warp_counts",
      "buffer_versions",
      "buffer_communications",
      "buffer_requires_new_allocation",
      "sync_producers",
      "sync_consumers",
      "sync_producer_groups",
      "sync_consumer_groups",
      "sync_producer_regions",
      "sync_consumer_regions",
      "sync_iteration_distances",
      "sync_effective_stage_distances",
      "sync_slot_counts",
      "sync_scopes",
      "sync_kinds",
      "sync_buffer_ids",
      "sync_dependency_masks",
      "sync_completion_modes",
      "register_domain_groups",
      "register_domain_first_warps",
      "register_domain_warp_counts",
      "register_domain_register_counts",
      "register_domain_is_increase",
      "shared_byte_offsets",
      "merged_shared_bytes",
      "groups_lowered",
      "pipeline_schedules_lowered",
      "buffers_lowered",
      "synchronization_lowered",
      "tile_ops_lowered",
  };
  for (const char *suffix : kScheduleAttributes) {
    func = tvm::WithoutAttr(std::move(func),
                            std::string("tl.program_schedule.") + suffix);
  }
  return WithAttr(std::move(func), "tl.program_schedule.lowered",
                  version.value());
}

tirx::PrimFunc LowerProgramScheduleGroups(
    tirx::PrimFunc func, const ProgramSchedulePlan &plan,
    const ProgramScheduleIR &program_ir) {
  tirx::IterVar thread_x = ThreadEnvironmentCollector::Collect(func->body);
  tirx::PrimFuncNode *node = func.CopyOnWrite();
  node->body = ThreadExtentRewriter::Rewrite(
      node->body, plan, program_ir, std::move(thread_x));
  func = WithAttr(std::move(func), "tl.program_schedule.groups_lowered",
                  Integer(1));
  func = WithAttr(std::move(func),
                  "tl.program_schedule.pipeline_schedules_lowered",
                  Integer(1));
  return func;
}

} // namespace

} // namespace wspipeline

tvm::transform::Pass LowerProgramSchedule() {
  auto pass_func = [](tirx::PrimFunc func, const IRModule &,
                      tvm::transform::PassContext) {
    std::optional<wspipeline::ProgramSchedulePlan> plan =
        wspipeline::ParseProgramSchedule(func);
    if (!plan.has_value()) {
      return func;
    }
    wspipeline::ProgramScheduleIR program_ir =
        wspipeline::CollectProgramScheduleIR(func, plan.value());
    wspipeline::ValidateProgramScheduleAgainstIR(plan.value(), program_ir);
    func = wspipeline::LowerProgramScheduleGroups(
        std::move(func), plan.value(), program_ir);
    func = wspipeline::LowerProgramScheduleBuffers(
        std::move(func), plan.value(), program_ir);
    func = wspipeline::LowerProgramScheduleSynchronization(
        std::move(func), plan.value(), program_ir);
    wspipeline::VerifyLoweredProgramSchedule(func, plan.value());
    return func;
  };
  return tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.LowerProgramSchedule", {});
}

tvm::transform::Pass FinalizeProgramSchedule() {
  auto pass_func = [](tirx::PrimFunc func, const IRModule &,
                      tvm::transform::PassContext) {
    return wspipeline::FinalizeProgramScheduleFunction(std::move(func));
  };
  return tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.FinalizeProgramSchedule", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerProgramSchedule",
                        LowerProgramSchedule);
  refl::GlobalDef().def("tl.transform.FinalizeProgramSchedule",
                        FinalizeProgramSchedule);
}

} // namespace tl
} // namespace tvm
