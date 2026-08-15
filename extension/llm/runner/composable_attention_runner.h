/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <executorch/extension/llm/runner/static_attention_planner.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/compiler.h>

namespace executorch {
namespace extension {
namespace llm {

struct ComposableAttentionRunnerTestPeer;

/// EXPERIMENTAL: Returns the method name shared with the portfolio builder.
ET_EXPERIMENTAL std::string
static_attention_method_name(bool first, size_t query_rows, size_t width);
ET_EXPERIMENTAL std::string static_attention_method_name(
    bool first,
    size_t query_rows,
    size_t width,
    bool assume_nonempty);

/// EXPERIMENTAL: One already padded, fixed-width K/V block.
///
/// The referenced tensors must remain alive for the duration of run_blocks().
/// Expected layouts are K=[B,Hkv,D,C], V=[B,Hkv,C,D], and
/// visibility=[B,Hm,R,C]. Invalid padding positions must have zero visibility.
struct ET_EXPERIMENTAL PreparedAttentionBlock {
  const executorch::aten::Tensor* key;
  const executorch::aten::Tensor* value;
  const executorch::aten::Tensor* visibility;
};

/// EXPERIMENTAL: Caller-owned output buffers for composed Attention.
///
/// Each bank contains output, running maximum, running denominator, and
/// running numerator tensors in method-output order. The runner copies only
/// the three small online-softmax states between calls and materializes the
/// final output once. Two banks avoid allocating state tensors per block.
struct ET_EXPERIMENTAL ComposableAttentionWorkspace {
  std::array<TensorPtr, 4> banks[2];
};

/// EXPERIMENTAL: Selects how caller-owned Attention outputs are validated.
///
/// The default preserves compatibility with portfolios exported before
/// caller-owned output metadata was available. Those portfolios can execute
/// through a logged copy path. RequireUnplannedOutputs is intended for a
/// zero-copy deployment contract: creation fails unless every portfolio
/// method declares all outputs as externally owned (not memory planned).
enum class ET_EXPERIMENTAL CallerOwnedOutputPolicy {
  kAllowCompatibilityCopy,
  kRequireUnplannedOutputs,
};

/// EXPERIMENTAL: Executes Attention using reusable fixed-shape methods.
///
/// The runner owns no model or KV cache. Callers retain full K/V state on the
/// host and pass the valid prefix for one invocation. The runner chooses a
/// measured-cost plan, slices that prefix into fixed-width inputs, pads only
/// the final block, and carries exact online-softmax state between methods.
class ET_EXPERIMENTAL ComposableAttentionRunner {
 public:
  /// Creates a runner for one fixed query-row shape.
  ///
  /// @param module Multi-method module containing first and merge methods.
  /// @param query_rows Static query rows represented by the method portfolio.
  /// @param graph_costs Device-measured first and merge method costs.
  /// @return A runner, or an error when the configuration is invalid.
  static runtime::Result<ComposableAttentionRunner> create(
      Module* module,
      size_t query_rows,
      std::vector<StaticAttentionGraphCost> graph_costs,
      bool assume_nonempty = false,
      CallerOwnedOutputPolicy caller_owned_output_policy =
          CallerOwnedOutputPolicy::kAllowCompatibilityCopy);

  /// Runs one composed Attention invocation.
  ///
  /// Expected contiguous layouts are Q=[B,Hq,R,D], K=[B,Hkv,D,L],
  /// V=[B,Hkv,L,D], and visibility=[B,Hm,R,L], where Hm is broadcastable to
  /// Hq. L may exceed every exported method width.
  runtime::Result<TensorPtr> run(
      const executorch::aten::Tensor& q,
      const executorch::aten::Tensor& k,
      const executorch::aten::Tensor& v,
      const executorch::aten::Tensor& visibility);

  /// Runs only the K/V prefix visible to one causal Query tile.
  runtime::Result<TensorPtr> run_causal_tile(
      const executorch::aten::Tensor& q,
      const executorch::aten::Tensor& k,
      const executorch::aten::Tensor& v,
      const executorch::aten::Tensor& visibility,
      size_t causal_query_begin,
      size_t valid_query_rows = 0);

  /// Returns the measured-cost block plan for a valid K/V prefix length.
  ///
  /// Persistent-cache callers can use this before run_blocks() so their
  /// physical layout follows the same large-block-versus-composition decision
  /// as run().
  runtime::Result<StaticAttentionPlan> plan(size_t sequence_length);

  runtime::Result<StaticAttentionPlan> plan_causal_tile(
      size_t full_sequence_length,
      size_t causal_query_begin,
      size_t valid_query_rows = 0);

  /// Returns one measured-cost persistent layout for causal Prefill tiles.
  runtime::Result<StaticAttentionPlan> plan_prefixes(
      const std::vector<size_t>& visible_prefixes);

  /// Runs Attention over caller-owned fixed-width blocks without repacking K/V.
  ///
  /// This path is intended for persistent block-oriented KV caches. Each
  /// physical block width must be present in graph_costs supplied to create().
  runtime::Result<TensorPtr> run_blocks(
      const executorch::aten::Tensor& q,
      const std::vector<PreparedAttentionBlock>& blocks);

  /// Runs caller-owned blocks and writes method outputs into reusable buffers.
  ///
  /// This is the zero-copy execution path for persistent KV caches. Callers
  /// may allocate the workspace from backend-registered shared memory.
  runtime::Result<TensorPtr> run_blocks(
      const executorch::aten::Tensor& q,
      const std::vector<PreparedAttentionBlock>& blocks,
      ComposableAttentionWorkspace& workspace);

  /// Executes one block in a caller-driven first/merge sequence.
  ///
  /// `block_index` must start at zero and increase by one. This form lets a
  /// long-context runtime refill one bounded scratch block between calls
  /// instead of materializing the complete K/V composition twice.
  runtime::Error run_prepared_block(
      const executorch::aten::Tensor& q,
      const PreparedAttentionBlock& block,
      size_t block_index,
      ComposableAttentionWorkspace& workspace);

  /// Returns the stable normalized output populated by run_prepared_block().
  static TensorPtr prepared_output(ComposableAttentionWorkspace& workspace) {
    return workspace.banks[0][0];
  }

  /// Unloads methods whose outputs were bound to caller-owned workspaces.
  ///
  /// Call this before releasing or replacing a workspace that has been passed
  /// to run_blocks(). Normal execution rebinds when an output address or its
  /// tensor contract changes, but unloading is required before
  /// backend-registered memory is destroyed so a delegate cannot retain a
  /// stale output address.
  void release_output_bindings();

  size_t planned_through() const {
    return planner_.planned_through();
  }

  const std::vector<StaticAttentionGraphCost>& graph_costs() const {
    return planner_.graph_costs();
  }

 private:
  struct OutputTensorBinding {
    std::weak_ptr<executorch::aten::Tensor> owner;
    const executorch::aten::Tensor* tensor = nullptr;
    void* data = nullptr;
    std::vector<executorch::aten::SizesType> sizes;
    executorch::aten::ScalarType dtype = executorch::aten::ScalarType::Float;
    size_t nbytes = 0;
  };

  struct OutputBinding {
    Module* module = nullptr;
    std::array<OutputTensorBinding, 4> outputs;
  };

  ComposableAttentionRunner(
      Module* module,
      size_t query_rows,
      StaticAttentionPlanner planner,
      bool assume_nonempty,
      CallerOwnedOutputPolicy caller_owned_output_policy);

  runtime::Result<TensorPtr> run_plan(
      const executorch::aten::Tensor& q,
      const executorch::aten::Tensor& k,
      const executorch::aten::Tensor& v,
      const executorch::aten::Tensor& visibility,
      const StaticAttentionPlan& plan);

  bool has_output_binding(
      Module* module,
      const std::string& method,
      const std::array<TensorPtr, 4>& outputs) const;

  void remember_output_binding(
      Module* module,
      const std::string& method,
      const std::array<TensorPtr, 4>& outputs);

  void forget_output_binding(const std::string& method);

  template <typename BindOutputs>
  runtime::Error bind_outputs_if_needed(
      Module* module,
      const std::string& method,
      const std::array<TensorPtr, 4>& outputs,
      BindOutputs&& bind_outputs,
      bool& outputs_bound) {
    outputs_bound = false;
    for (const TensorPtr& output : outputs) {
      if (output == nullptr) {
        return runtime::Error::InvalidArgument;
      }
    }
    if (has_output_binding(module, method, outputs)) {
      outputs_bound = true;
      return runtime::Error::Ok;
    }

    const runtime::Error error = std::forward<BindOutputs>(bind_outputs)();
    if (error != runtime::Error::Ok) {
      // set_outputs() can fail after changing an earlier output. The old
      // signature is no longer trustworthy until the method is reloaded.
      forget_output_binding(method);
      return error;
    }

    remember_output_binding(module, method, outputs);
    bound_methods_.insert(method);
    outputs_bound = true;
    return runtime::Error::Ok;
  }

  friend struct ComposableAttentionRunnerTestPeer;

  Module* module_;
  size_t query_rows_;
  StaticAttentionPlanner planner_;
  bool assume_nonempty_;
  CallerOwnedOutputPolicy caller_owned_output_policy_;
  std::unordered_map<std::string, OutputBinding> output_bindings_;
  std::unordered_set<std::string> bound_methods_;
  std::unordered_set<std::string> copy_fallback_methods_;
};

} // namespace llm
} // namespace extension
} // namespace executorch
