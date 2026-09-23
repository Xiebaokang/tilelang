/*!
 * \file unroll_fragment_pingpong.cc
 * \brief Unroll fragment ping-pong IfThenElse dispatch by V.
 *
 * LowerOverlapPlan emits uniform ``if k % V == slot`` across named fragment
 * slots. After InjectSoftwarePipeline this pass expands the remaining kernel
 * loop into V straight-line copies so Simplify can fold the branches. The
 * unroll factor is V (2 or 3), not the K trip count.
 */

#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "support/check.h"

namespace tvm {
namespace tl {
namespace overlap_plan {

using namespace tirx;
using namespace ffi;

namespace {

std::optional<int64_t> ModFactor(const PrimExpr &expr, const Var &loop_var) {
  auto from_binary = [&](const PrimExpr &lhs, const PrimExpr &rhs) {
    if (!lhs.same_as(loop_var)) {
      return std::optional<int64_t>();
    }
    const int64_t *value = as_const_int(rhs);
    if (value == nullptr || *value <= 1) {
      return std::optional<int64_t>();
    }
    return std::optional<int64_t>(*value);
  };
  if (const auto *node = expr.as<FloorModNode>()) {
    return from_binary(node->a, node->b);
  }
  if (const auto *node = expr.as<ModNode>()) {
    return from_binary(node->a, node->b);
  }
  return std::nullopt;
}

void CollectEqSides(const PrimExpr &condition, std::vector<PrimExpr> *sides) {
  if (const auto *eq = condition.as<EQNode>()) {
    sides->push_back(eq->a);
    sides->push_back(eq->b);
    return;
  }
  if (const auto *and_node = condition.as<AndNode>()) {
    CollectEqSides(and_node->a, sides);
    CollectEqSides(and_node->b, sides);
    return;
  }
  if (const auto *or_node = condition.as<OrNode>()) {
    CollectEqSides(or_node->a, sides);
    CollectEqSides(or_node->b, sides);
  }
}

int64_t PingPongFactor(const Stmt &body, const Var &loop_var) {
  int64_t factor = 1;
  bool found = false;
  PostOrderVisit(body, [&](const ObjectRef &node) {
    const auto *op = node.as<IfThenElseNode>();
    if (op == nullptr) {
      return;
    }
    std::vector<PrimExpr> sides;
    CollectEqSides(op->condition, &sides);
    sides.push_back(op->condition);
    for (const PrimExpr &side : sides) {
      std::optional<int64_t> value = ModFactor(side, loop_var);
      if (!value.has_value()) {
        continue;
      }
      found = true;
      factor = std::lcm(factor, *value);
    }
  });
  return found && factor > 1 ? factor : 0;
}

Stmt MakeSequence(ffi::Array<Stmt> statements) {
  ICHECK(!statements.empty());
  if (statements.size() == 1) {
    return statements[0];
  }
  return SeqStmt(std::move(statements));
}

For CopyFor(const ForNode *op, Stmt body) {
  return For(op->loop_var, op->min, op->extent, op->kind, std::move(body),
             op->thread_binding, op->annotations, op->step);
}

} // namespace

class UnrollFragmentPingPongRewriter : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc func) {
    func.CopyOnWrite()->body = UnrollFragmentPingPongRewriter()(func->body);
    return func;
  }

private:
  Stmt VisitStmt_(const ForNode *op) final {
    Stmt body = VisitStmt(op->body);
    if (op->kind == ForKind::kThreadBinding || op->kind == ForKind::kParallel ||
        op->kind == ForKind::kVectorized) {
      return CopyFor(op, std::move(body));
    }
    if (!op->HasTrivialStep()) {
      return CopyFor(op, std::move(body));
    }
    int64_t factor = PingPongFactor(body, op->loop_var);
    const int64_t *extent = as_const_int(op->extent);
    if (factor < 2 || extent == nullptr || *extent < 1) {
      return CopyFor(op, std::move(body));
    }

    DataType dtype = op->loop_var.dtype();
    int64_t nfull = *extent / factor;
    int64_t remainder = *extent % factor;
    ffi::Array<Stmt> statements;
    if (nfull > 0) {
      ffi::Array<Stmt> copies;
      copies.reserve(factor);
      for (int64_t offset = 0; offset < factor; ++offset) {
        PrimExpr index = op->loop_var * IntImm(dtype, factor) + op->min +
                         IntImm(dtype, offset);
        copies.push_back(
            tirx::Substitute(body, Map<Var, PrimExpr>{{op->loop_var, index}}));
      }
      statements.push_back(For(op->loop_var, IntImm(dtype, 0),
                               IntImm(dtype, nfull), ForKind::kSerial,
                               MakeSequence(std::move(copies)), std::nullopt,
                               op->annotations));
    }
    if (remainder > 0) {
      ffi::Array<Stmt> copies;
      copies.reserve(remainder);
      for (int64_t offset = 0; offset < remainder; ++offset) {
        PrimExpr index = op->min + IntImm(dtype, nfull * factor + offset);
        copies.push_back(
            tirx::Substitute(body, Map<Var, PrimExpr>{{op->loop_var, index}}));
      }
      statements.push_back(MakeSequence(std::move(copies)));
    }
    if (statements.empty()) {
      return body;
    }
    return MakeSequence(std::move(statements));
  }
};

} // namespace overlap_plan

tvm::transform::Pass UnrollFragmentPingPong() {
  auto pass_func = [](tirx::PrimFunc func, const IRModule &,
                      tvm::transform::PassContext) {
    return overlap_plan::UnrollFragmentPingPongRewriter::Substitute(
        std::move(func));
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0,
                                             "tl.UnrollFragmentPingPong", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.UnrollFragmentPingPong",
                        UnrollFragmentPingPong);
}

} // namespace tl
} // namespace tvm
