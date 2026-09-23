/*!
 * \file program_schedule.cc
 * \brief Parsing and structural validation for program schedule attributes.
 */

#include "program_schedule.h"

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <tvm/ffi/container/array.h>
#include <tvm/ir/expr.h>
#include <tvm/runtime/logging.h>

#include "support/check.h"

namespace tvm {
namespace tl {
namespace wspipeline {

namespace {

constexpr const char *kPrefix = "tl.program_schedule.";
constexpr int64_t kContractVersion = 3;
constexpr int64_t kWarpSize = 32;
constexpr int64_t kWarpgroupWarps = 4;
constexpr int64_t kRegisterFileBudget = 64512;

std::string AttrKey(const char *suffix) {
  return std::string(kPrefix) + suffix;
}

int64_t RequiredInteger(const tirx::PrimFunc &func, const char *suffix) {
  std::string key = AttrKey(suffix);
  ffi::Optional<Integer> value = func->GetAttr<Integer>(key);
  ICHECK(value.defined()) << "program schedule is missing required attribute "
                          << key;
  return value.value()->value;
}

std::vector<int64_t> RequiredIntegerArray(const tirx::PrimFunc &func,
                                          const char *suffix) {
  std::string key = AttrKey(suffix);
  ffi::Optional<ffi::Array<Integer>> values =
      func->GetAttr<ffi::Array<Integer>>(key);
  ICHECK(values.defined()) << "program schedule is missing required attribute "
                           << key;
  std::vector<int64_t> result;
  result.reserve(values.value().size());
  for (const Integer &value : values.value()) {
    result.push_back(value->value);
  }
  return result;
}

void CheckSameSize(const std::vector<int64_t> &reference,
                   const std::vector<int64_t> &candidate,
                   const char *reference_name, const char *candidate_name) {
  ICHECK_EQ(reference.size(), candidate.size())
      << "program schedule arrays " << reference_name << " and "
      << candidate_name << " must have the same length";
}

void CheckIndex(int64_t value, size_t upper_bound, const char *field) {
  ICHECK_GE(value, 0) << field << " cannot be negative";
  ICHECK_LT(static_cast<size_t>(value), upper_bound)
      << field << " " << value << " is outside [0, " << upper_bound << ")";
}

void CheckOptionalIndex(int64_t value, size_t upper_bound,
                        const char *field) {
  ICHECK_GE(value, -1) << field << " must be -1 or a valid index";
  if (value >= 0) {
    CheckIndex(value, upper_bound, field);
  }
}

void CheckPermutation(std::vector<int64_t> values, const char *field) {
  std::sort(values.begin(), values.end());
  for (size_t index = 0; index < values.size(); ++index) {
    ICHECK_EQ(values[index], static_cast<int64_t>(index))
        << field << " must be a dense permutation beginning at zero";
  }
}

void ValidateWarpAllocation(const ProgramSchedulePlan &plan) {
  CheckSameSize(plan.group_first_warps, plan.group_warp_counts,
                "group_first_warps", "group_warp_counts");
  ICHECK(!plan.group_warp_counts.empty())
      << "program schedule must contain at least one group";
  int64_t next_warp = 0;
  for (size_t group = 0; group < plan.group_warp_counts.size(); ++group) {
    ICHECK_EQ(plan.group_first_warps[group], next_warp)
        << "group warp ranges must be contiguous and begin at warp zero";
    ICHECK_GT(plan.group_warp_counts[group], 0)
        << "each group must receive at least one warp";
    next_warp += plan.group_warp_counts[group];
  }
  ICHECK_EQ(plan.effective_threads, next_warp * kWarpSize)
      << "effective_threads must equal allocated warps times warp size";
}

void ValidateOperations(const ProgramSchedulePlan &plan) {
  const std::vector<int64_t> &groups = plan.operation_groups;
  ICHECK(!groups.empty())
      << "program schedule must contain at least one operation";
  CheckSameSize(groups, plan.operation_regions, "operation_groups",
                "operation_regions");
  CheckSameSize(groups, plan.operation_stages, "operation_groups",
                "operation_stages");
  CheckSameSize(groups, plan.operation_local_orders, "operation_groups",
                "operation_local_orders");
  ICHECK(!plan.region_num_stages.empty())
      << "program schedule must contain at least one region";

  std::map<std::pair<int64_t, int64_t>, std::vector<int64_t>> local_orders;
  std::vector<bool> used_groups(plan.group_warp_counts.size(), false);
  std::vector<bool> used_regions(plan.region_num_stages.size(), false);
  std::vector<int64_t> minimum_stages(plan.region_num_stages.size(), -1);
  std::vector<int64_t> maximum_stages(plan.region_num_stages.size(), -1);
  for (size_t operation_id = 0; operation_id < groups.size(); ++operation_id) {
    int64_t group = groups[operation_id];
    int64_t region = plan.operation_regions[operation_id];
    CheckIndex(group, plan.group_warp_counts.size(), "operation group");
    CheckIndex(region, plan.region_num_stages.size(), "operation region");
    used_groups[group] = true;
    used_regions[region] = true;

    int64_t num_stages = plan.region_num_stages[region];
    int64_t stage = plan.operation_stages[operation_id];
    ICHECK_GE(num_stages, 0) << "region_num_stages cannot be negative";
    if (num_stages == 0) {
      ICHECK_EQ(stage, -1)
          << "serial-region operations must use stage -1";
    } else {
      ICHECK_GE(stage, 0) << "pipeline-region stages cannot be negative";
      ICHECK_LT(stage, num_stages)
          << "operation stage exceeds its region's stage count";
      minimum_stages[region] =
          minimum_stages[region] < 0
              ? stage
              : std::min(minimum_stages[region], stage);
      maximum_stages[region] = std::max(maximum_stages[region], stage);
    }

    int64_t local_order = plan.operation_local_orders[operation_id];
    ICHECK_GE(local_order, 0) << "operation local order cannot be negative";
    local_orders[{region, group}].push_back(local_order);
  }

  ICHECK(std::all_of(used_groups.begin(), used_groups.end(),
                     [](bool used) { return used; }))
      << "every physical group must own at least one operation";
  ICHECK(std::all_of(used_regions.begin(), used_regions.end(),
                     [](bool used) { return used; }))
      << "every declared region must contain at least one operation";
  for (const auto &[region_group, orders] : local_orders) {
    CheckPermutation(orders, "operation_local_orders within one region/group");
  }
  for (size_t region = 0; region < plan.region_num_stages.size(); ++region) {
    if (plan.region_num_stages[region] > 0) {
      ICHECK_EQ(minimum_stages[region], 0)
          << "pipeline-region stages must begin at zero";
      ICHECK_EQ(maximum_stages[region], plan.region_num_stages[region] - 1)
          << "pipeline-region stages must use the declared final stage";
    }
  }
}

void ValidateBuffers(const ProgramSchedulePlan &plan) {
  CheckSameSize(plan.buffer_versions, plan.buffer_communications,
                "buffer_versions", "buffer_communications");
  CheckSameSize(plan.buffer_versions, plan.buffer_requires_new_allocation,
                "buffer_versions", "buffer_requires_new_allocation");
  for (size_t buffer_id = 0; buffer_id < plan.buffer_versions.size();
       ++buffer_id) {
    int64_t versions = plan.buffer_versions[buffer_id];
    int64_t communication = plan.buffer_communications[buffer_id];
    int64_t requires_allocation =
        plan.buffer_requires_new_allocation[buffer_id];
    ICHECK_GE(versions, 1) << "buffer version count must be positive";
    ICHECK_GE(communication, 0)
        << "buffer communication code cannot be negative";
    ICHECK_LE(communication, 4) << "unknown buffer communication code";
    ICHECK(requires_allocation == 0 || requires_allocation == 1)
        << "buffer_requires_new_allocation must contain boolean integers";
    ICHECK_EQ(requires_allocation, versions > 1 || communication == 4)
        << "new allocation is required exactly for multiversion or "
           "materialized-shared buffers";
  }
}

void ValidateSynchronization(const ProgramSchedulePlan &plan) {
  const std::vector<int64_t> &producers = plan.sync_producers;
  CheckSameSize(producers, plan.sync_consumers, "sync_producers",
                "sync_consumers");
  CheckSameSize(producers, plan.sync_producer_groups, "sync_producers",
                "sync_producer_groups");
  CheckSameSize(producers, plan.sync_consumer_groups, "sync_producers",
                "sync_consumer_groups");
  CheckSameSize(producers, plan.sync_producer_regions, "sync_producers",
                "sync_producer_regions");
  CheckSameSize(producers, plan.sync_consumer_regions, "sync_producers",
                "sync_consumer_regions");
  CheckSameSize(producers, plan.sync_iteration_distances, "sync_producers",
                "sync_iteration_distances");
  CheckSameSize(producers, plan.sync_effective_stage_distances,
                "sync_producers", "sync_effective_stage_distances");
  if (!plan.sync_slot_counts.empty()) {
    CheckSameSize(producers, plan.sync_slot_counts, "sync_producers",
                  "sync_slot_counts");
  }
  CheckSameSize(producers, plan.sync_scopes, "sync_producers", "sync_scopes");
  CheckSameSize(producers, plan.sync_kinds, "sync_producers", "sync_kinds");
  CheckSameSize(producers, plan.sync_buffer_ids, "sync_producers",
                "sync_buffer_ids");
  CheckSameSize(producers, plan.sync_dependency_masks, "sync_producers",
                "sync_dependency_masks");
  if (!plan.sync_completion_modes.empty()) {
    CheckSameSize(producers, plan.sync_completion_modes, "sync_producers",
                  "sync_completion_modes");
  }

  std::map<int64_t, size_t> transaction_producers;

  for (size_t channel = 0; channel < producers.size(); ++channel) {
    int64_t producer = producers[channel];
    int64_t consumer = plan.sync_consumers[channel];
    CheckIndex(producer, plan.operation_groups.size(), "sync producer");
    CheckIndex(consumer, plan.operation_groups.size(), "sync consumer");
    CheckIndex(plan.sync_producer_groups[channel], plan.group_warp_counts.size(),
               "sync producer group");
    CheckIndex(plan.sync_consumer_groups[channel], plan.group_warp_counts.size(),
               "sync consumer group");
    ICHECK_EQ(plan.sync_producer_groups[channel],
              plan.operation_groups[producer])
        << "sync producer group does not match its operation";
    ICHECK_EQ(plan.sync_consumer_groups[channel],
              plan.operation_groups[consumer])
        << "sync consumer group does not match its operation";
    CheckOptionalIndex(plan.sync_producer_regions[channel],
                       plan.region_num_stages.size(), "sync producer region");
    CheckOptionalIndex(plan.sync_consumer_regions[channel],
                       plan.region_num_stages.size(), "sync consumer region");
    if (plan.sync_producer_regions[channel] >= 0) {
      ICHECK_EQ(plan.sync_producer_regions[channel],
                plan.operation_regions[producer])
          << "sync producer region does not match its operation";
    }
    if (plan.sync_consumer_regions[channel] >= 0) {
      ICHECK_EQ(plan.sync_consumer_regions[channel],
                plan.operation_regions[consumer])
          << "sync consumer region does not match its operation";
    }
    ICHECK_GE(plan.sync_iteration_distances[channel], 0)
        << "sync iteration distance cannot be negative";
    ICHECK_GE(plan.sync_effective_stage_distances[channel], -1)
        << "sync effective stage distance must be -1 or nonnegative";
    if (!plan.sync_slot_counts.empty()) {
      ICHECK_GT(plan.sync_slot_counts[channel], 0)
          << "sync slot count must be positive";
    }
    ICHECK_GE(plan.sync_scopes[channel], -1) << "unknown sync scope code";
    ICHECK_LE(plan.sync_scopes[channel], 2) << "unknown sync scope code";
    ICHECK_GE(plan.sync_kinds[channel], 0) << "unknown sync kind code";
    ICHECK_LE(plan.sync_kinds[channel], 1) << "unknown sync kind code";
    CheckOptionalIndex(plan.sync_buffer_ids[channel],
                       plan.buffer_versions.size(), "sync buffer ID");
    int64_t dependency_mask = plan.sync_dependency_masks[channel];
    ICHECK((dependency_mask >= 1 && dependency_mask <= 7) ||
           dependency_mask == 8)
        << "unknown synchronization dependency mask " << dependency_mask;
    int64_t completion_mode = plan.sync_completion_modes.empty()
                                  ? 0
                                  : plan.sync_completion_modes[channel];
    ICHECK(completion_mode == 0 || completion_mode == 1)
        << "unknown synchronization completion mode " << completion_mode;
    if (completion_mode == 1) {
      ICHECK_EQ(plan.sync_kinds[channel], 0)
          << "only a forward dependency may use transaction completion";
      ICHECK_GE(plan.sync_buffer_ids[channel], 0)
          << "transaction completion requires a buffer-backed channel";
      ICHECK_GE(plan.sync_producer_regions[channel], 0)
          << "transaction completion requires a producer region";
      ICHECK(transaction_producers.emplace(producer, channel).second)
          << "one producer cannot complete multiple transaction barriers";
    }
  }
}

void ValidateRegisterDomains(const ProgramSchedulePlan &plan) {
  CheckSameSize(plan.register_domain_groups,
                plan.register_domain_first_warps, "register_domain_groups",
                "register_domain_first_warps");
  CheckSameSize(plan.register_domain_groups,
                plan.register_domain_warp_counts, "register_domain_groups",
                "register_domain_warp_counts");
  CheckSameSize(plan.register_domain_groups,
                plan.register_domain_register_counts,
                "register_domain_groups", "register_domain_register_counts");
  CheckSameSize(plan.register_domain_groups,
                plan.register_domain_is_increase, "register_domain_groups",
                "register_domain_is_increase");
  if (!plan.setmaxnreg_enabled) {
    ICHECK(plan.register_domain_groups.empty())
        << "register domains require setmaxnreg_enabled";
    return;
  }
  ICHECK(!plan.register_domain_groups.empty())
      << "setmaxnreg_enabled requires at least one register domain";
  std::vector<int64_t> next_warps = plan.group_first_warps;
  std::vector<int64_t> group_register_counts(plan.group_warp_counts.size(), -1);
  std::vector<int64_t> group_actions(plan.group_warp_counts.size(), -1);
  bool has_increase = false;
  bool has_decrease = false;
  int64_t allocated_registers = 0;
  for (size_t domain = 0; domain < plan.register_domain_groups.size();
       ++domain) {
    int64_t group = plan.register_domain_groups[domain];
    CheckIndex(group, plan.group_warp_counts.size(), "register domain group");
    ICHECK_EQ(plan.register_domain_first_warps[domain],
              next_warps[group])
        << "register domains must cover each group contiguously";
    ICHECK_EQ(plan.register_domain_warp_counts[domain],
              kWarpgroupWarps)
        << "each setmaxnreg domain must be one hardware warpgroup";
    next_warps[group] += kWarpgroupWarps;

    int64_t register_count = plan.register_domain_register_counts[domain];
    int64_t is_increase = plan.register_domain_is_increase[domain];
    ICHECK_GE(register_count, 24);
    ICHECK_LE(register_count, 240);
    ICHECK_EQ(register_count % 8, 0)
        << "setmaxnreg count must be a multiple of eight";
    ICHECK(is_increase == 0 || is_increase == 1)
        << "register_domain_is_increase must contain boolean integers";
    if (group_register_counts[group] < 0) {
      group_register_counts[group] = register_count;
      group_actions[group] = is_increase;
    } else {
      ICHECK_EQ(group_register_counts[group], register_count)
          << "all register domains in one logical group must use one count";
      ICHECK_EQ(group_actions[group], is_increase)
          << "all register domains in one logical group must use one action";
    }
    has_increase = has_increase || is_increase != 0;
    has_decrease = has_decrease || is_increase == 0;
    allocated_registers +=
        register_count * plan.register_domain_warp_counts[domain] * kWarpSize;
  }
  for (size_t group = 0; group < plan.group_warp_counts.size(); ++group) {
    ICHECK_EQ(next_warps[group],
              plan.group_first_warps[group] + plan.group_warp_counts[group])
        << "register domains must cover every warp of group " << group;
  }
  ICHECK(has_increase && has_decrease)
      << "setmaxnreg redistribution requires donor and receiver groups";
  ICHECK_LE(allocated_registers, kRegisterFileBudget)
      << "setmaxnreg plan exceeds the per-SM register-file budget";
}

void ValidateProgramSchedulePlan(const ProgramSchedulePlan &plan) {
  ICHECK_EQ(plan.version, kContractVersion)
      << "unsupported tl.program_schedule contract version " << plan.version;
  ValidateWarpAllocation(plan);
  ValidateOperations(plan);
  ValidateBuffers(plan);
  ValidateSynchronization(plan);
  ValidateRegisterDomains(plan);
}

} // namespace

std::optional<ProgramSchedulePlan>
ParseProgramSchedule(const tirx::PrimFunc &func) {
  ffi::Optional<Integer> version =
      func->GetAttr<Integer>(AttrKey("version"));
  if (!version.defined()) {
    return std::nullopt;
  }

  ProgramSchedulePlan plan;
  plan.version = version.value()->value;
  plan.effective_threads = RequiredInteger(func, "effective_threads");
  int64_t setmaxnreg_enabled = RequiredInteger(func, "setmaxnreg_enabled");
  ICHECK(setmaxnreg_enabled == 0 || setmaxnreg_enabled == 1)
      << AttrKey("setmaxnreg_enabled") << " must be 0 or 1";
  plan.setmaxnreg_enabled = setmaxnreg_enabled != 0;

#define TL_PARSE_PROGRAM_SCHEDULE_ARRAY(field)                                  \
  plan.field = RequiredIntegerArray(func, #field)
  TL_PARSE_PROGRAM_SCHEDULE_ARRAY(operation_groups);
  TL_PARSE_PROGRAM_SCHEDULE_ARRAY(operation_regions);
  TL_PARSE_PROGRAM_SCHEDULE_ARRAY(operation_stages);
  TL_PARSE_PROGRAM_SCHEDULE_ARRAY(operation_local_orders);
  TL_PARSE_PROGRAM_SCHEDULE_ARRAY(region_num_stages);
  TL_PARSE_PROGRAM_SCHEDULE_ARRAY(group_first_warps);
  TL_PARSE_PROGRAM_SCHEDULE_ARRAY(group_warp_counts);
  TL_PARSE_PROGRAM_SCHEDULE_ARRAY(buffer_versions);
  TL_PARSE_PROGRAM_SCHEDULE_ARRAY(buffer_communications);
  TL_PARSE_PROGRAM_SCHEDULE_ARRAY(buffer_requires_new_allocation);
  TL_PARSE_PROGRAM_SCHEDULE_ARRAY(sync_producers);
  TL_PARSE_PROGRAM_SCHEDULE_ARRAY(sync_consumers);
  TL_PARSE_PROGRAM_SCHEDULE_ARRAY(sync_producer_groups);
  TL_PARSE_PROGRAM_SCHEDULE_ARRAY(sync_consumer_groups);
  TL_PARSE_PROGRAM_SCHEDULE_ARRAY(sync_producer_regions);
  TL_PARSE_PROGRAM_SCHEDULE_ARRAY(sync_consumer_regions);
  TL_PARSE_PROGRAM_SCHEDULE_ARRAY(sync_iteration_distances);
  TL_PARSE_PROGRAM_SCHEDULE_ARRAY(sync_effective_stage_distances);
  ffi::Optional<ffi::Array<Integer>> sync_slot_counts =
      func->GetAttr<ffi::Array<Integer>>(AttrKey("sync_slot_counts"));
  if (sync_slot_counts.defined()) {
    TL_PARSE_PROGRAM_SCHEDULE_ARRAY(sync_slot_counts);
  }
  TL_PARSE_PROGRAM_SCHEDULE_ARRAY(sync_scopes);
  TL_PARSE_PROGRAM_SCHEDULE_ARRAY(sync_kinds);
  TL_PARSE_PROGRAM_SCHEDULE_ARRAY(sync_buffer_ids);
  TL_PARSE_PROGRAM_SCHEDULE_ARRAY(sync_dependency_masks);
  ffi::Optional<ffi::Array<Integer>> sync_completion_modes =
      func->GetAttr<ffi::Array<Integer>>(AttrKey("sync_completion_modes"));
  if (sync_completion_modes.defined()) {
    TL_PARSE_PROGRAM_SCHEDULE_ARRAY(sync_completion_modes);
  }
  TL_PARSE_PROGRAM_SCHEDULE_ARRAY(register_domain_groups);
  TL_PARSE_PROGRAM_SCHEDULE_ARRAY(register_domain_first_warps);
  TL_PARSE_PROGRAM_SCHEDULE_ARRAY(register_domain_warp_counts);
  TL_PARSE_PROGRAM_SCHEDULE_ARRAY(register_domain_register_counts);
  TL_PARSE_PROGRAM_SCHEDULE_ARRAY(register_domain_is_increase);
#undef TL_PARSE_PROGRAM_SCHEDULE_ARRAY

  ValidateProgramSchedulePlan(plan);
  return plan;
}

} // namespace wspipeline
} // namespace tl
} // namespace tvm
