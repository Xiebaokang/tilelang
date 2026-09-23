/*!
 * \file overlap_plan.h
 * \brief Typed overlap schedule contract consumed by TileLang lowering.
 */

#ifndef TVM_TL_OVERLAP_PLAN_OVERLAP_PLAN_H_
#define TVM_TL_OVERLAP_PLAN_OVERLAP_PLAN_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/optional.h>
#include <tvm/ir/expr.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt.h>

namespace tvm {
namespace tl {
namespace overlap_plan {

inline constexpr const char *kOperationScope = "tl.overlap_plan.operation";
inline constexpr const char *kRegionIdAnnotation = "tl.overlap_plan.region_id";
inline constexpr const char *kSyncEventScope = "tl.overlap_plan.sync_event";
inline constexpr int64_t kWarpSize = 32;

inline std::string FragmentSlotSuffix(int64_t slot) {
  return "_v" + std::to_string(slot);
}

inline std::string FragmentSlotName(const std::string &name, int64_t slot) {
  return name + FragmentSlotSuffix(slot);
}

inline std::string FragmentHandoffSuffix(size_t channel) {
  return "_wsp_handoff_" + std::to_string(channel);
}

inline std::string FragmentHandoffName(const std::string &name, size_t channel) {
  return name + FragmentHandoffSuffix(channel);
}

enum class SyncEventRole : int64_t {
  kThreadArrive = 0,
  kWait = 1,
  kAsyncArrive = 2,
};

enum class BufferCommunication : int64_t {
  kExternal = 0,
  kPrivate = 1,
  kShared = 2,
  kTmem = 3,
  kMaterializedShared = 4,
};

class GroupPlanNode : public ffi::Object {
public:
  int64_t warp_count{0};
  ffi::Optional<Integer> register_count;
  ffi::Optional<Integer> register_increase;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.overlap_plan.GroupPlan", GroupPlanNode,
                                    ffi::Object);
};

class GroupPlan : public ffi::ObjectRef {
public:
  TVM_DLL GroupPlan(int64_t warp_count, ffi::Optional<Integer> register_count,
                    ffi::Optional<Integer> register_increase);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(GroupPlan, ffi::ObjectRef,
                                             GroupPlanNode);
};

class OperationPlacementNode : public ffi::Object {
public:
  int64_t operation_id{-1};
  ffi::Optional<tirx::Stmt> statement;
  int64_t group_id{0};
  ffi::Optional<Integer> stage;
  int64_t order{0};

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.overlap_plan.OperationPlacement",
                                    OperationPlacementNode, ffi::Object);
};

class OperationPlacement : public ffi::ObjectRef {
public:
  TVM_DLL OperationPlacement(int64_t operation_id,
                             ffi::Optional<tirx::Stmt> statement,
                             int64_t group_id, ffi::Optional<Integer> stage,
                             int64_t order);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(OperationPlacement, ffi::ObjectRef,
                                             OperationPlacementNode);
};

class BufferPlanNode : public ffi::Object {
public:
  int64_t buffer_id{-1};
  ffi::Optional<tirx::Buffer> buffer;
  int64_t version_count{1};
  int64_t communication{0};
  ffi::Optional<Integer> byte_offset;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.overlap_plan.BufferPlan",
                                    BufferPlanNode, ffi::Object);
};

class BufferPlan : public ffi::ObjectRef {
public:
  TVM_DLL BufferPlan(int64_t buffer_id, ffi::Optional<tirx::Buffer> buffer,
                     int64_t version_count, int64_t communication,
                     ffi::Optional<Integer> byte_offset);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(BufferPlan, ffi::ObjectRef,
                                             BufferPlanNode);
};

class SyncEdgeNode : public ffi::Object {
public:
  int64_t producer_id{-1};
  int64_t consumer_id{-1};
  ffi::Optional<Integer> buffer_id;
  int64_t kind{0};
  int64_t scope{0};
  int64_t iteration_distance{0};
  int64_t slot_count{1};
  int64_t dependency_kind{1};
  int64_t completion_mode{0};
  ffi::Optional<Integer> byte_offset;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.overlap_plan.SyncEdge", SyncEdgeNode,
                                    ffi::Object);
};

class SyncEdge : public ffi::ObjectRef {
public:
  TVM_DLL SyncEdge(int64_t producer_id, int64_t consumer_id,
                   ffi::Optional<Integer> buffer_id, int64_t kind,
                   int64_t scope, int64_t iteration_distance,
                   int64_t slot_count, int64_t dependency_kind,
                   int64_t completion_mode,
                   ffi::Optional<Integer> byte_offset);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(SyncEdge, ffi::ObjectRef,
                                             SyncEdgeNode);
};

class OverlapPlanNode : public ffi::Object {
public:
  ffi::Array<GroupPlan> groups;
  ffi::Array<OperationPlacement> operations;
  ffi::Array<BufferPlan> buffers;
  ffi::Array<SyncEdge> sync_edges;
  ffi::Optional<Integer> shared_arena_bytes;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.overlap_plan.OverlapPlan",
                                    OverlapPlanNode, ffi::Object);
};

class OverlapPlan : public ffi::ObjectRef {
public:
  TVM_DLL OverlapPlan(ffi::Array<GroupPlan> groups,
                      ffi::Array<OperationPlacement> operations,
                      ffi::Array<BufferPlan> buffers,
                      ffi::Array<SyncEdge> sync_edges,
                      ffi::Optional<Integer> shared_arena_bytes);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(OverlapPlan, ffi::ObjectRef,
                                             OverlapPlanNode);
};

struct OverlapOperationSite {
  tirx::Stmt statement;
  int64_t region_id{-1};
};

struct OverlapRegionSite {
  ffi::Optional<tirx::For> pipeline_loop;
  // Original T.Pipelined annotation. Never overwritten by the plan.
  int64_t num_stages{0};
};

struct OverlapIR {
  std::vector<OverlapOperationSite> operations;
  std::vector<std::vector<int64_t>> operation_buffer_ids;
  std::vector<OverlapRegionSite> regions;
  std::vector<tirx::Buffer> buffers;
  // True when the PrimFunc requested kernel-wide OverlapPlan lowering.
  bool auto_overlap{false};
};

/*!
 * \brief Identify a T.Pipelined loop.
 *
 * Any T.Pipelined loop is a pipeline region; ``num_stages`` is not required.
 */
inline bool IsOverlapPipelineLoop(const tirx::For &loop) {
  return loop->annotations.count("tl.pipelined") != 0;
}

/*!
 * \brief Flattened, derived view used by IR rewriters.
 *
 * Fields that used to be Python-written dense arrays are computed here from
 * the typed OverlapPlan plus the collected TIR sites.
 */
struct LoweringView {
  int64_t effective_threads{0};
  bool setmaxnreg_enabled{false};

  std::vector<int64_t> operation_groups;
  std::vector<int64_t> operation_regions;
  std::vector<int64_t> operation_stages;
  std::vector<int64_t> operation_local_orders;
  std::vector<int64_t> region_num_stages;

  std::vector<int64_t> group_first_warps;
  std::vector<int64_t> group_warp_counts;
  std::vector<int64_t> group_register_counts;
  std::vector<int64_t> group_register_increase;

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
  std::vector<int64_t> sync_slot_counts;
  std::vector<int64_t> sync_scopes;
  std::vector<int64_t> sync_kinds;
  std::vector<int64_t> sync_buffer_ids;
  std::vector<int64_t> sync_dependency_masks;
  std::vector<int64_t> sync_completion_modes;
  // Canonical channel that owns the physical completion event.  Multiple
  // async forward edges may wait on one TMA transaction event.
  std::vector<int64_t> sync_event_owners;
};

ffi::Optional<OverlapPlan> GetOverlapPlan(const tirx::PrimFunc &func);

bool HasAutoOverlap(const tirx::PrimFunc &func);

OverlapIR CollectOverlapIR(const tirx::PrimFunc &func);

OverlapIR MatchOverlapPlan(const OverlapPlan &plan, OverlapIR program_ir);

LoweringView MakeLoweringView(const OverlapPlan &plan,
                              const OverlapIR &program_ir);

void ValidateLoweringView(const LoweringView &plan,
                          const OverlapIR &program_ir);

tirx::PrimFunc AttachSharedMemoryAttrs(tirx::PrimFunc func,
                                       const OverlapPlan &plan,
                                       const OverlapIR &program_ir);

tirx::PrimFunc LowerOverlapPlanGroups(tirx::PrimFunc func,
                                      const LoweringView &plan,
                                      const OverlapIR &program_ir);

tirx::PrimFunc LowerOverlapPlanBuffers(tirx::PrimFunc func,
                                       const LoweringView &plan,
                                       const OverlapIR &program_ir);

tirx::PrimFunc LowerOverlapPlanSynchronization(tirx::PrimFunc func,
                                               const LoweringView &plan,
                                               const OverlapIR &program_ir);

void CheckOverlapPlanDenseIds(const OverlapPlan &plan);

void VerifyLoweredOverlapPlan(const tirx::PrimFunc &func,
                              const OverlapPlan &plan, const LoweringView &view,
                              const OverlapIR &program_ir);

tirx::PrimFunc FinalizeOverlapPlanFunction(tirx::PrimFunc func);

} // namespace overlap_plan
} // namespace tl
} // namespace tvm

#endif // TVM_TL_OVERLAP_PLAN_OVERLAP_PLAN_H_
