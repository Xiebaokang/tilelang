/*!
 * \file program_schedule.h
 * \brief Internal representation of a program-aware warp-specialization plan.
 */

#ifndef TVM_TL_WSPIPELINE_PROGRAM_SCHEDULE_H_
#define TVM_TL_WSPIPELINE_PROGRAM_SCHEDULE_H_

#include <cstdint>
#include <optional>
#include <vector>

#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt.h>

namespace tvm {
namespace tl {
namespace wspipeline {

inline constexpr const char *kOperationScope =
    "tl.program_schedule.operation";
inline constexpr const char *kRegionIdAnnotation =
    "tl.program_schedule.region_id";
inline constexpr const char *kAutoScheduleAnnotation =
    "tl.wsp.auto_schedule";

struct ProgramSchedulePlan {
  int64_t version{0};
  int64_t effective_threads{0};
  bool setmaxnreg_enabled{false};

  std::vector<int64_t> operation_groups;
  std::vector<int64_t> operation_regions;
  std::vector<int64_t> operation_stages;
  std::vector<int64_t> operation_local_orders;
  std::vector<int64_t> region_num_stages;

  std::vector<int64_t> group_first_warps;
  std::vector<int64_t> group_warp_counts;

  std::vector<int64_t> buffer_versions;
  std::vector<int64_t> buffer_communications;
  std::vector<int64_t> buffer_requires_new_allocation;

  std::vector<int64_t> sync_producers;
  std::vector<int64_t> sync_consumers;
  std::vector<int64_t> sync_producer_groups;
  std::vector<int64_t> sync_consumer_groups;
  std::vector<int64_t> sync_producer_regions;
  std::vector<int64_t> sync_consumer_regions;
  std::vector<int64_t> sync_iteration_distances;
  std::vector<int64_t> sync_effective_stage_distances;
  // Optional for backward compatibility with the original wspipeline path.
  // UnionWSP supplies the exact logical ring size for every channel.
  std::vector<int64_t> sync_slot_counts;
  std::vector<int64_t> sync_scopes;
  std::vector<int64_t> sync_kinds;
  std::vector<int64_t> sync_buffer_ids;
  std::vector<int64_t> sync_dependency_masks;
  // 0: every producer thread arrives; 1: an asynchronous transaction
  // completes the barrier. Optional for older schedule producers.
  std::vector<int64_t> sync_completion_modes;

  std::vector<int64_t> register_domain_groups;
  std::vector<int64_t> register_domain_first_warps;
  std::vector<int64_t> register_domain_warp_counts;
  std::vector<int64_t> register_domain_register_counts;
  std::vector<int64_t> register_domain_is_increase;
};

struct ProgramOperationSite {
  tirx::Stmt statement;
  int64_t region_id{-1};
};

struct ProgramRegionSite {
  ffi::Optional<tirx::For> pipeline_loop;
  int64_t num_stages{0};
};

struct ProgramScheduleIR {
  std::vector<ProgramOperationSite> operations;
  std::vector<std::vector<int64_t>> operation_buffer_ids;
  std::vector<ProgramRegionSite> regions;
  std::vector<tirx::Buffer> buffers;
};

/*! \brief Parse and validate a schedule contract attached to a PrimFunc. */
std::optional<ProgramSchedulePlan>
ParseProgramSchedule(const tirx::PrimFunc &func);

/*! \brief Reconstruct stable operation, region, and buffer IDs from TIRX. */
ProgramScheduleIR CollectProgramScheduleIR(const tirx::PrimFunc &func,
                                           const ProgramSchedulePlan &plan);

/*! \brief Check that a parsed contract describes the exact input TIRX. */
void ValidateProgramScheduleAgainstIR(const ProgramSchedulePlan &plan,
                                      const ProgramScheduleIR &program_ir);

/*! \brief Check that WSP lowering materialized the parsed contract exactly. */
void VerifyLoweredProgramSchedule(const tirx::PrimFunc &func,
                                  const ProgramSchedulePlan &plan);

/*! \brief Materialize planned buffer storage and version indices. */
tirx::PrimFunc LowerProgramScheduleBuffers(tirx::PrimFunc func,
                                           const ProgramSchedulePlan &plan,
                                           const ProgramScheduleIR &program_ir);

/*! \brief Materialize synchronization channels as shared mbarriers. */
tirx::PrimFunc
LowerProgramScheduleSynchronization(tirx::PrimFunc func,
                                    const ProgramSchedulePlan &plan,
                                    const ProgramScheduleIR &program_ir);

} // namespace wspipeline
} // namespace tl
} // namespace tvm

#endif // TVM_TL_WSPIPELINE_PROGRAM_SCHEDULE_H_
