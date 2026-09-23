/*!
 * \file overlap_plan_ir.cc
 * \brief Collect TIR operations/buffers and match them to an OverlapPlan.
 */

#include "overlap_plan.h"

#include <algorithm>
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
namespace overlap_plan {

namespace {

using BufferSet =
    std::unordered_set<tirx::Buffer, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;
using BufferIdMap =
    std::unordered_map<tirx::Buffer, int64_t, ffi::ObjectPtrHash,
                       ffi::ObjectPtrEqual>;
using StmtIdMap =
    std::unordered_map<tirx::Stmt, int64_t, ffi::ObjectPtrHash,
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
  return BufferSortKey{std::string(buffer->name), std::string(buffer.scope()),
                       dtype.str(), shape.str()};
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
  static OverlapIR Collect(const tirx::PrimFunc &func) {
    ProgramIRCollector collector(HasAutoOverlap(func));
    collector.Visit(func->body, std::nullopt, false);
    return collector.Build();
  }

private:
  explicit ProgramIRCollector(bool auto_overlap)
      : auto_overlap_(auto_overlap) {}

  void AppendOperation(const tirx::Stmt &statement,
                       std::optional<int64_t> pipeline_id,
                       OperationAccess access) {
    operations_.push_back(
        RawOperation{statement, pipeline_id, std::move(access)});
  }

  void Visit(const tirx::Stmt &statement, std::optional<int64_t> pipeline_id,
             bool under_conditional) {
    if (const auto *sequence = statement.as<tirx::SeqStmtNode>()) {
      for (const tirx::Stmt &child : sequence->seq) {
        Visit(child, pipeline_id, under_conditional);
      }
      return;
    }
    if (const auto *loop_node = statement.as<tirx::ForNode>()) {
      tirx::For loop = ffi::GetRef<tirx::For>(loop_node);
      if (IsOverlapPipelineLoop(loop)) {
        ICHECK(!under_conditional)
            << "pipeline loops under conditional control flow are not "
               "supported by OverlapPlan";
        ICHECK(!pipeline_id.has_value())
            << "nested pipeline loops are not supported by OverlapPlan";
        int64_t stage_count = 0;
        ffi::Optional<ffi::Any> num_stages = loop->annotations.Get("num_stages");
        if (num_stages.has_value()) {
          const auto *value = num_stages.value().as<IntImmNode>();
          ICHECK(value != nullptr)
              << "pipeline num_stages must be a static integer";
          stage_count = value->value;
          if (!auto_overlap_) {
            ICHECK_GE(stage_count, 2)
                << "pipeline num_stages must be at least two";
          }
        }
        int64_t new_pipeline_id = pipeline_loops_.size();
        pipeline_loops_.push_back(loop);
        pipeline_stage_counts_.push_back(stage_count);
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
        AppendOperation(statement, pipeline_id,
                        TileOperationAccessCollector::Collect(
                            ffi::GetRef<tirx::Call>(call)));
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
          << "two distinct buffers used by one operation have identical "
             "name/scope/dtype/shape metadata";
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

  OverlapIR Build() const {
    ICHECK(!operations_.empty())
        << "OverlapPlan requires at least one schedulable operation";

    OverlapIR result;
    std::optional<int64_t> previous_pipeline;
    bool has_previous = false;
    for (const RawOperation &operation : operations_) {
      bool starts_region =
          !has_previous || operation.pipeline_id != previous_pipeline;
      if (starts_region) {
        OverlapRegionSite region;
        if (operation.pipeline_id.has_value()) {
          int64_t pipeline = operation.pipeline_id.value();
          region.pipeline_loop = pipeline_loops_[pipeline];
          region.num_stages = pipeline_stage_counts_[pipeline];
        }
        result.regions.push_back(std::move(region));
      }
      result.operations.push_back(OverlapOperationSite{
          operation.statement,
          static_cast<int64_t>(result.regions.size() - 1)});
      previous_pipeline = operation.pipeline_id;
      has_previous = true;
    }

    BufferIdMap buffer_ids;
    for (const RawOperation &operation : operations_) {
      std::vector<tirx::Buffer> reads = OrderedBuffers(operation.access.reads);
      std::vector<tirx::Buffer> writes =
          OrderedBuffers(operation.access.writes);
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
      result.operation_buffer_ids.push_back(std::move(operation_buffer_ids));
    }
    result.auto_overlap = auto_overlap_;
    return result;
  }

  std::vector<RawOperation> operations_;
  std::vector<tirx::For> pipeline_loops_;
  std::vector<int64_t> pipeline_stage_counts_;
  bool auto_overlap_{false};
};

std::vector<int64_t> MatchOperations(const OverlapPlan &plan,
                                     const OverlapIR &program_ir) {
  ICHECK_EQ(plan->operations.size(), program_ir.operations.size())
      << "OverlapPlan operation count does not match the input TIR";
  CheckOverlapPlanDenseIds(plan);
  size_t statement_count = 0;
  for (const OperationPlacement &placement : plan->operations) {
    statement_count += static_cast<size_t>(placement->statement.defined());
  }
  ICHECK(statement_count == 0 || statement_count == plan->operations.size())
      << "OverlapPlan operations must either all carry statement handles or "
         "all omit them; mixed matching is not supported";

  std::vector<int64_t> order(plan->operations.size(), -1);
  if (statement_count == 0) {
    for (size_t index = 0; index < order.size(); ++index) {
      order[index] = static_cast<int64_t>(index);
    }
    return order;
  }

  StmtIdMap collected;
  for (size_t index = 0; index < program_ir.operations.size(); ++index) {
    ICHECK(collected
               .emplace(program_ir.operations[index].statement,
                        static_cast<int64_t>(index))
               .second)
        << "one TIR statement cannot represent two collected operations";
  }
  std::vector<bool> used(program_ir.operations.size(), false);
  for (size_t plan_id = 0; plan_id < plan->operations.size(); ++plan_id) {
    const tirx::Stmt &statement = plan->operations[plan_id]->statement.value();
    auto iterator = collected.find(statement);
    ICHECK(iterator != collected.end())
        << "OverlapPlan operation " << plan_id
        << " does not match any TIR statement collected for lowering";
    ICHECK(!used[iterator->second])
        << "OverlapPlan maps two operations onto TIR statement "
        << iterator->second;
    used[iterator->second] = true;
    order[plan_id] = iterator->second;
  }
  return order;
}

std::vector<int64_t> MatchBuffers(const OverlapPlan &plan,
                                  const OverlapIR &program_ir) {
  ICHECK_EQ(plan->buffers.size(), program_ir.buffers.size())
      << "OverlapPlan buffer count does not match the input TIR";
  CheckOverlapPlanDenseIds(plan);
  size_t buffer_count = 0;
  for (const BufferPlan &buffer : plan->buffers) {
    buffer_count += static_cast<size_t>(buffer->buffer.defined());
  }
  ICHECK(buffer_count == 0 || buffer_count == plan->buffers.size())
      << "OverlapPlan buffers must either all carry Buffer handles or all "
         "omit them; mixed matching is not supported";

  std::vector<int64_t> order(plan->buffers.size(), -1);
  if (buffer_count == 0) {
    for (size_t index = 0; index < order.size(); ++index) {
      order[index] = static_cast<int64_t>(index);
    }
    return order;
  }

  BufferIdMap collected;
  for (size_t index = 0; index < program_ir.buffers.size(); ++index) {
    ICHECK(collected
               .emplace(program_ir.buffers[index], static_cast<int64_t>(index))
               .second)
        << "one Buffer ObjectRef cannot have two collected buffer IDs";
  }
  std::vector<bool> used(program_ir.buffers.size(), false);
  for (size_t plan_id = 0; plan_id < plan->buffers.size(); ++plan_id) {
    const tirx::Buffer &buffer = plan->buffers[plan_id]->buffer.value();
    auto iterator = collected.find(buffer);
    ICHECK(iterator != collected.end())
        << "OverlapPlan buffer " << plan_id << " (" << buffer->name
        << ") does not match any TIR buffer collected for lowering";
    ICHECK(!used[iterator->second])
        << "OverlapPlan maps two buffers onto TIR buffer " << iterator->second;
    used[iterator->second] = true;
    order[plan_id] = iterator->second;
  }
  return order;
}

} // namespace

OverlapIR CollectOverlapIR(const tirx::PrimFunc &func) {
  return ProgramIRCollector::Collect(func);
}

OverlapIR MatchOverlapPlan(const OverlapPlan &plan, OverlapIR program_ir) {
  std::vector<int64_t> operation_order = MatchOperations(plan, program_ir);
  std::vector<int64_t> buffer_order = MatchBuffers(plan, program_ir);

  OverlapIR remapped;
  remapped.regions = program_ir.regions;
  remapped.auto_overlap = program_ir.auto_overlap;
  remapped.operations.resize(operation_order.size());
  remapped.operation_buffer_ids.resize(operation_order.size());
  remapped.buffers.resize(buffer_order.size());

  std::vector<int64_t> old_to_new_buffer(program_ir.buffers.size(), -1);
  for (size_t new_id = 0; new_id < buffer_order.size(); ++new_id) {
    int64_t old_id = buffer_order[new_id];
    remapped.buffers[new_id] = program_ir.buffers[old_id];
    old_to_new_buffer[old_id] = static_cast<int64_t>(new_id);
  }
  for (size_t new_id = 0; new_id < operation_order.size(); ++new_id) {
    int64_t old_id = operation_order[new_id];
    remapped.operations[new_id] = program_ir.operations[old_id];
    std::vector<int64_t> buffer_ids;
    for (int64_t old_buffer_id : program_ir.operation_buffer_ids[old_id]) {
      buffer_ids.push_back(old_to_new_buffer[old_buffer_id]);
    }
    std::sort(buffer_ids.begin(), buffer_ids.end());
    remapped.operation_buffer_ids[new_id] = std::move(buffer_ids);
  }
  return remapped;
}

} // namespace overlap_plan
} // namespace tl
} // namespace tvm
