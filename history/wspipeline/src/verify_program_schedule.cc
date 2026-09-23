/*!
 * \file verify_program_schedule.cc
 * \brief Structural verifier for a materialized program schedule.
 */

#include "program_schedule.h"

#include <cstdint>
#include <vector>

#include <tvm/tirx/stmt_functor.h>

#include "op/builtin.h"
#include "support/check.h"
#include "transform/common/pipeline_utils.h"

namespace tvm {
namespace tl {
namespace wspipeline {

namespace {

class LoweredScheduleVerifier : public tirx::StmtExprVisitor {
public:
  explicit LoweredScheduleVerifier(const ProgramSchedulePlan &plan)
      : plan_(plan), group_counts_(plan.group_warp_counts.size(), 0),
        operation_counts_(plan.operation_groups.size(), 0) {}

  static void Verify(const tirx::PrimFunc &func,
                     const ProgramSchedulePlan &plan) {
    LoweredScheduleVerifier verifier(plan);
    verifier(func->body);
    ICHECK_EQ(verifier.thread_x_extent_count_, 1)
        << "lowered schedule must contain exactly one threadIdx.x extent";
    ICHECK_EQ(verifier.warp_specialization_scope_count_, 1)
        << "lowered schedule must contain exactly one warp-specialization scope";
    for (size_t group_id = 0; group_id < verifier.group_counts_.size();
         ++group_id) {
      ICHECK_EQ(verifier.group_counts_[group_id], 1)
          << "lowered schedule must materialize group " << group_id
          << " exactly once";
    }
    for (size_t operation_id = 0;
         operation_id < verifier.operation_counts_.size(); ++operation_id) {
      ICHECK_EQ(verifier.operation_counts_[operation_id], 1)
          << "lowered schedule must retain operation " << operation_id
          << " exactly once";
    }
  }

private:
  void VisitStmt_(const tirx::AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::thread_extent) {
      ffi::Optional<tirx::IterVar> iter_var = op->node.as<tirx::IterVar>();
      if (iter_var.has_value() &&
          iter_var.value()->thread_tag == "threadIdx.x") {
        const auto *extent = op->value.as<IntImmNode>();
        ICHECK(extent != nullptr);
        ICHECK_EQ(extent->value, plan_.effective_threads)
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
          << "program-schedule group scopes cannot be nested";
      auto [first_thread, thread_count] =
          GetProgramScheduleGroupThreadRange(op);
      ICHECK_EQ(first_thread.as<IntImmNode>()->value,
                plan_.group_first_warps[group_id] * 32);
      ICHECK_EQ(thread_count.as<IntImmNode>()->value,
                plan_.group_warp_counts[group_id] * 32);
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
      ICHECK_LT(operation_id,
                static_cast<int64_t>(operation_counts_.size()));
      ICHECK_EQ(active_group_, plan_.operation_groups[operation_id])
          << "operation marker is materialized in the wrong group";
      ICHECK_EQ(group_id_imm->value, active_group_)
          << "operation marker carries the wrong group ID";
      ++operation_counts_[operation_id];
      return;
    }
    tirx::StmtExprVisitor::VisitStmt_(op);
  }

  const ProgramSchedulePlan &plan_;
  std::vector<int64_t> group_counts_;
  std::vector<int64_t> operation_counts_;
  int64_t active_group_{-1};
  int64_t thread_x_extent_count_{0};
  int64_t warp_specialization_scope_count_{0};
};

void RequireCompletionAttribute(const tirx::PrimFunc &func,
                                const char *attribute) {
  ffi::Optional<Integer> value = func->GetAttr<Integer>(attribute);
  ICHECK(value.has_value() && value.value()->value == 1)
      << "lowered program schedule is missing completion attribute "
      << attribute;
}

} // namespace

void VerifyLoweredProgramSchedule(const tirx::PrimFunc &func,
                                  const ProgramSchedulePlan &plan) {
  RequireCompletionAttribute(func, "tl.program_schedule.groups_lowered");
  RequireCompletionAttribute(func,
                             "tl.program_schedule.pipeline_schedules_lowered");
  RequireCompletionAttribute(func, "tl.program_schedule.buffers_lowered");
  RequireCompletionAttribute(func,
                             "tl.program_schedule.synchronization_lowered");
  LoweredScheduleVerifier::Verify(func, plan);
}

} // namespace wspipeline
} // namespace tl
} // namespace tvm
