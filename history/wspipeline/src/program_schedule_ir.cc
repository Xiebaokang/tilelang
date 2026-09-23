/*!
 * \file program_schedule_ir.cc
 * \brief Recover schedule IDs from the exact TIRX consumed by C++ lowering.
 */

#include "program_schedule.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <tvm/ir/op.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt_functor.h>

#include "support/check.h"

namespace tvm {
namespace tl {
namespace wspipeline {

namespace {

using BufferSet =
    std::unordered_set<tirx::Buffer, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;
using BufferIdMap =
    std::unordered_map<tirx::Buffer, int64_t, ffi::ObjectPtrHash,
                       ffi::ObjectPtrEqual>;

struct OperationAccess {
  BufferSet reads;
  BufferSet writes;
};

struct RawOperation {
  tirx::Stmt statement;
  std::optional<int64_t> pipeline_id;
  OperationAccess access;
};

struct BufferSortKey {
  std::string name;
  std::string scope;
  std::string dtype;
  std::string shape;

  bool operator<(const BufferSortKey &other) const {
    return std::tie(name, scope, dtype, shape) <
           std::tie(other.name, other.scope, other.dtype, other.shape);
  }
};

BufferSortKey MakeBufferSortKey(const tirx::Buffer &buffer) {
  std::ostringstream dtype;
  dtype << buffer->dtype;
  std::ostringstream shape;
  for (const PrimExpr &extent : buffer->shape) {
    shape << '[' << extent << ']';
  }
  return BufferSortKey{std::string(buffer->name),
                       std::string(buffer.scope()), dtype.str(), shape.str()};
}

bool IsTileOperation(const tirx::CallNode *call) {
  const auto *op = call->op.as<OpNode>();
  if (op == nullptr) {
    return false;
  }
  std::string name = op->name;
  return name.rfind("tl.tileop.", 0) == 0;
}

class ScalarAccessCollector : public tirx::StmtExprVisitor {
public:
  static OperationAccess Collect(const tirx::Stmt &statement) {
    ScalarAccessCollector collector;
    collector(statement);
    return std::move(collector.access_);
  }

private:
  void VisitExpr_(const tirx::BufferLoadNode *op) final {
    access_.reads.insert(op->buffer);
    tirx::StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const tirx::BufferStoreNode *op) final {
    access_.writes.insert(op->buffer);
    tirx::StmtExprVisitor::VisitStmt_(op);
  }

  OperationAccess access_;
};

class TileOperationAccessCollector : public tirx::StmtExprVisitor {
public:
  static OperationAccess Collect(const tirx::Call &call) {
    TileOperationAccessCollector collector;
    collector(call);
    return std::move(collector.access_);
  }

private:
  void VisitExpr_(const tirx::CallNode *op) final {
    static const Op &region_op = Op::Get("tl.tileop.region");
    if (op->op.same_as(region_op) && op->args.size() >= 2) {
      const auto *load = op->args[0].as<tirx::BufferLoadNode>();
      const auto *mask = op->args[1].as<IntImmNode>();
      if (load != nullptr && mask != nullptr) {
        if (mask->value & 1) {
          access_.reads.insert(load->buffer);
        }
        if (mask->value & 2) {
          access_.writes.insert(load->buffer);
        }
      }
    }
    tirx::StmtExprVisitor::VisitExpr_(op);
  }

  OperationAccess access_;
};

class ProgramIRCollector {
public:
  static ProgramScheduleIR Collect(const tirx::PrimFunc &func,
                                   const ProgramSchedulePlan &plan) {
    ProgramIRCollector collector(plan);
    collector.Visit(func->body, std::nullopt, false);
    return collector.Build();
  }

private:
  explicit ProgramIRCollector(const ProgramSchedulePlan &plan) : plan_(plan) {}

  void AppendOperation(const tirx::Stmt &statement,
                       std::optional<int64_t> pipeline_id,
                       OperationAccess access) {
    operations_.push_back(
        RawOperation{statement, pipeline_id, std::move(access)});
  }

  void Visit(const tirx::Stmt &statement,
             std::optional<int64_t> pipeline_id, bool under_conditional) {
    if (const auto *sequence = statement.as<tirx::SeqStmtNode>()) {
      for (const tirx::Stmt &child : sequence->seq) {
        Visit(child, pipeline_id, under_conditional);
      }
      return;
    }
    if (const auto *loop_node = statement.as<tirx::ForNode>()) {
      tirx::For loop = ffi::GetRef<tirx::For>(loop_node);
      ffi::Optional<ffi::Any> num_stages = loop->annotations.Get("num_stages");
      ffi::Optional<ffi::Any> auto_schedule =
          loop->annotations.Get(kAutoScheduleAnnotation);
      bool is_auto_schedule = false;
      if (auto_schedule.has_value()) {
        const auto *enabled = auto_schedule.value().as<IntImmNode>();
        ICHECK(enabled != nullptr)
            << kAutoScheduleAnnotation << " must be a static integer";
        is_auto_schedule = enabled->value != 0;
      }
      if (num_stages.has_value() || is_auto_schedule) {
        ICHECK(!under_conditional)
            << "pipeline loops under conditional control flow are not "
               "supported by the program schedule contract";
        ICHECK(!pipeline_id.has_value())
            << "nested pipeline loops are not supported by the program "
               "schedule contract";
        int64_t stage_count = 0;
        if (!is_auto_schedule) {
          const auto *value = num_stages.value().as<IntImmNode>();
          ICHECK(value != nullptr)
              << "pipeline num_stages must be a static integer";
          ICHECK_GE(value->value, 2)
              << "pipeline num_stages must be at least two";
          stage_count = value->value;
        }
        int64_t new_pipeline_id = pipeline_loops_.size();
        pipeline_loops_.push_back(loop);
        pipeline_stage_counts_.push_back(stage_count);
        pipeline_auto_schedule_.push_back(is_auto_schedule);
        Visit(loop->body, new_pipeline_id, false);
        return;
      }
      if (loop->kind == tirx::ForKind::kParallel) {
        AppendOperation(statement, pipeline_id,
                        ScalarAccessCollector::Collect(statement));
        return;
      }
      Visit(loop->body, pipeline_id, under_conditional);
      return;
    }
    if (const auto *realize = statement.as<tirx::SBlockRealizeNode>()) {
      Visit(realize->block->body, pipeline_id, under_conditional);
      return;
    }
    if (const auto *block = statement.as<tirx::SBlockNode>()) {
      Visit(block->body, pipeline_id, under_conditional);
      return;
    }
    if (const auto *attribute = statement.as<tirx::AttrStmtNode>()) {
      Visit(attribute->body, pipeline_id, under_conditional);
      return;
    }
    if (const auto *condition = statement.as<tirx::IfThenElseNode>()) {
      Visit(condition->then_case, pipeline_id, true);
      if (condition->else_case.defined()) {
        Visit(condition->else_case.value(), pipeline_id, true);
      }
      return;
    }
    if (const auto *evaluate = statement.as<tirx::EvaluateNode>()) {
      if (const auto *call = evaluate->value.as<tirx::CallNode>();
          call != nullptr && IsTileOperation(call)) {
        AppendOperation(
            statement, pipeline_id,
            TileOperationAccessCollector::Collect(ffi::GetRef<tirx::Call>(call)));
      }
      return;
    }
    if (statement.as<tirx::BufferStoreNode>() != nullptr) {
      AppendOperation(statement, pipeline_id,
                      ScalarAccessCollector::Collect(statement));
    }
  }

  std::vector<tirx::Buffer> OrderedBuffers(const BufferSet &buffers) const {
    std::vector<std::pair<BufferSortKey, tirx::Buffer>> keyed;
    keyed.reserve(buffers.size());
    std::map<BufferSortKey, tirx::Buffer> unique_keys;
    for (const tirx::Buffer &buffer : buffers) {
      BufferSortKey key = MakeBufferSortKey(buffer);
      auto [iterator, inserted] = unique_keys.emplace(key, buffer);
      ICHECK(inserted || iterator->second.same_as(buffer))
          << "cannot reproduce Python buffer IDs: two distinct buffers used "
             "by one operation have identical name/scope/dtype/shape metadata";
      keyed.emplace_back(std::move(key), buffer);
    }
    std::sort(keyed.begin(), keyed.end(),
              [](const auto &left, const auto &right) {
                return left.first < right.first;
              });
    std::vector<tirx::Buffer> result;
    result.reserve(keyed.size());
    for (const auto &[key, buffer] : keyed) {
      result.push_back(buffer);
    }
    return result;
  }

  ProgramScheduleIR Build() const {
    ICHECK(!operations_.empty())
        << "program schedule contract requires at least one schedulable "
           "operation in the input TIRX";

    ProgramScheduleIR result;
    std::optional<int64_t> previous_pipeline;
    bool has_previous = false;
    for (const RawOperation &operation : operations_) {
      bool starts_region =
          !has_previous || operation.pipeline_id != previous_pipeline;
      if (starts_region) {
        ProgramRegionSite region;
        if (operation.pipeline_id.has_value()) {
          int64_t pipeline = operation.pipeline_id.value();
          region.pipeline_loop = pipeline_loops_[pipeline];
          int64_t region_id = result.regions.size();
          if (pipeline_auto_schedule_[pipeline]) {
            ICHECK_LT(region_id, plan_.region_num_stages.size())
                << "automatic WSP region is missing its selected stage count";
            region.num_stages = plan_.region_num_stages[region_id];
          } else {
            region.num_stages = pipeline_stage_counts_[pipeline];
          }
        }
        result.regions.push_back(std::move(region));
      }
      result.operations.push_back(
          ProgramOperationSite{operation.statement,
                               static_cast<int64_t>(result.regions.size() - 1)});
      previous_pipeline = operation.pipeline_id;
      has_previous = true;
    }

    BufferIdMap buffer_ids;
    for (const RawOperation &operation : operations_) {
      std::vector<tirx::Buffer> reads = OrderedBuffers(operation.access.reads);
      std::vector<tirx::Buffer> writes = OrderedBuffers(operation.access.writes);
      reads.insert(reads.end(), writes.begin(), writes.end());
      for (const tirx::Buffer &buffer : reads) {
        if (buffer_ids.count(buffer) == 0) {
          int64_t buffer_id = result.buffers.size();
          buffer_ids.emplace(buffer, buffer_id);
          result.buffers.push_back(buffer);
        }
      }
    }
    for (const RawOperation &operation : operations_) {
      std::vector<int64_t> operation_buffer_ids;
      BufferSet operation_buffers = operation.access.reads;
      operation_buffers.insert(operation.access.writes.begin(),
                               operation.access.writes.end());
      for (const tirx::Buffer &buffer : operation_buffers) {
        auto iterator = buffer_ids.find(buffer);
        ICHECK(iterator != buffer_ids.end());
        operation_buffer_ids.push_back(iterator->second);
      }
      std::sort(operation_buffer_ids.begin(), operation_buffer_ids.end());
      result.operation_buffer_ids.push_back(
          std::move(operation_buffer_ids));
    }
    return result;
  }

  std::vector<RawOperation> operations_;
  std::vector<tirx::For> pipeline_loops_;
  std::vector<int64_t> pipeline_stage_counts_;
  std::vector<bool> pipeline_auto_schedule_;
  const ProgramSchedulePlan &plan_;
};

} // namespace

ProgramScheduleIR CollectProgramScheduleIR(const tirx::PrimFunc &func,
                                           const ProgramSchedulePlan &plan) {
  return ProgramIRCollector::Collect(func, plan);
}

void ValidateProgramScheduleAgainstIR(const ProgramSchedulePlan &plan,
                                      const ProgramScheduleIR &program_ir) {
  ICHECK_EQ(plan.operation_groups.size(), program_ir.operations.size())
      << "program schedule operation count does not match the input TIRX";
  ICHECK_EQ(plan.region_num_stages.size(), program_ir.regions.size())
      << "program schedule region count does not match the input TIRX";
  ICHECK_EQ(plan.buffer_versions.size(), program_ir.buffers.size())
      << "program schedule buffer count does not match the input TIRX";
  ICHECK_EQ(program_ir.operation_buffer_ids.size(),
            program_ir.operations.size())
      << "program schedule IR must record buffer users for every operation";

  for (size_t operation_id = 0; operation_id < program_ir.operations.size();
       ++operation_id) {
    ICHECK_EQ(plan.operation_regions[operation_id],
              program_ir.operations[operation_id].region_id)
        << "program schedule region ID does not match TIRX operation "
        << operation_id;
  }
  for (size_t region_id = 0; region_id < program_ir.regions.size();
       ++region_id) {
    ICHECK_EQ(plan.region_num_stages[region_id],
              program_ir.regions[region_id].num_stages)
        << "program schedule stage count does not match TIRX region "
        << region_id;
  }
}

} // namespace wspipeline
} // namespace tl
} // namespace tvm
