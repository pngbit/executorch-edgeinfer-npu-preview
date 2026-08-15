/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/prompt_processor.h>
#include <executorch/extension/llm/runner/composable_attention_runner.h>
#include <executorch/extension/llm/runner/static_attention_planner.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace example {

namespace edgeinfer {
namespace detail {

struct StageOutputTensorFingerprint final {
  void* data = nullptr;
  size_t nbytes = 0;
  executorch::aten::ScalarType scalar_type =
      executorch::aten::ScalarType::Undefined;
  std::vector<executorch::aten::SizesType> sizes;
  std::vector<executorch::aten::DimOrderType> dim_order;
  std::vector<executorch::aten::StridesType> strides;

  bool operator==(const StageOutputTensorFingerprint& other) const {
    return data == other.data && nbytes == other.nbytes &&
        scalar_type == other.scalar_type && sizes == other.sizes &&
        dim_order == other.dim_order && strides == other.strides;
  }
};

struct StageOutputBindingFingerprint final {
  std::vector<StageOutputTensorFingerprint> outputs;

  bool operator==(const StageOutputBindingFingerprint& other) const {
    return outputs == other.outputs;
  }
};

inline std::optional<StageOutputBindingFingerprint>
make_stage_output_binding_fingerprint(
    const std::vector<executorch::extension::TensorPtr>& outputs) {
  if (outputs.empty() ||
      std::any_of(outputs.begin(), outputs.end(), [](const auto& output) {
        return output == nullptr;
      })) {
    return std::nullopt;
  }
  StageOutputBindingFingerprint fingerprint;
  fingerprint.outputs.reserve(outputs.size());
  for (const auto& output : outputs) {
    StageOutputTensorFingerprint tensor;
    tensor.data = output->mutable_data_ptr();
    tensor.nbytes = output->nbytes();
    tensor.scalar_type = output->scalar_type();
    tensor.sizes.assign(output->sizes().begin(), output->sizes().end());
    tensor.dim_order.assign(
        output->dim_order().begin(), output->dim_order().end());
    tensor.strides.assign(output->strides().begin(), output->strides().end());
    fingerprint.outputs.emplace_back(std::move(tensor));
  }
  return fingerprint;
}

class StageOutputBindingCache final {
 public:
  using MethodBindings =
      std::unordered_map<std::string, StageOutputBindingFingerprint>;
  using ModuleBindings =
      std::unordered_map<executorch::extension::Module*, MethodBindings>;

  const StageOutputBindingFingerprint* find(
      executorch::extension::Module* module,
      const std::string& method) const {
    const auto module_it = bindings_.find(module);
    if (module_it == bindings_.end()) {
      return nullptr;
    }
    const auto method_it = module_it->second.find(method);
    return method_it == module_it->second.end() ? nullptr : &method_it->second;
  }

  void remember(
      executorch::extension::Module* module,
      std::string method,
      StageOutputBindingFingerprint fingerprint) {
    bindings_[module].insert_or_assign(
        std::move(method), std::move(fingerprint));
  }

  void forget(
      executorch::extension::Module* module,
      const std::string& method) {
    auto module_it = bindings_.find(module);
    if (module_it == bindings_.end()) {
      return;
    }
    module_it->second.erase(method);
    if (module_it->second.empty()) {
      bindings_.erase(module_it);
    }
  }

  void forget_module(executorch::extension::Module* module) {
    bindings_.erase(module);
  }

  void clear() {
    bindings_.clear();
  }

  const ModuleBindings& bindings() const {
    return bindings_;
  }

 private:
  ModuleBindings bindings_;
};

enum class StageOutputBindingDisposition {
  Bound,
  Copy,
};

template <
    typename HasMemoryPlannedOutput,
    typename BindOutputs,
    typename Unload>
StageOutputBindingDisposition prepare_stage_output_binding(
    StageOutputBindingCache& cache,
    executorch::extension::Module* module,
    const std::string& method,
    const StageOutputBindingFingerprint& fingerprint,
    HasMemoryPlannedOutput&& has_memory_planned_output,
    BindOutputs&& bind_outputs,
    Unload&& unload) {
  if (const auto* cached = cache.find(module, method)) {
    if (*cached == fingerprint) {
      return StageOutputBindingDisposition::Bound;
    }
    std::forward<Unload>(unload)();
    cache.forget(module, method);
  }
  if (std::forward<HasMemoryPlannedOutput>(has_memory_planned_output)()) {
    return StageOutputBindingDisposition::Copy;
  }
  if (std::forward<BindOutputs>(bind_outputs)() ==
      executorch::runtime::Error::Ok) {
    cache.remember(module, method, fingerprint);
    return StageOutputBindingDisposition::Bound;
  }
  // set_outputs() may have changed an earlier output before a later failure.
  std::forward<Unload>(unload)();
  cache.forget(module, method);
  return StageOutputBindingDisposition::Copy;
}

template <typename MarkCopy, typename Unload>
void reject_stage_output_binding(
    StageOutputBindingCache& cache,
    executorch::extension::Module* module,
    const std::string& method,
    MarkCopy&& mark_copy,
    Unload&& unload) {
  std::forward<Unload>(unload)();
  cache.forget(module, method);
  std::forward<MarkCopy>(mark_copy)();
}

/** Tracks the materialized visible-prefix length for every mask row. */
class VisibilityMaskCache final {
 public:
  bool reset_known_zero(size_t query_rows, size_t block_width) {
    if (query_rows == 0 || block_width == 0) {
      invalidate();
      return false;
    }
    query_rows_ = query_rows;
    block_width_ = block_width;
    visible_columns_.assign(query_rows, 0);
    initialized_ = true;
    return true;
  }

  void invalidate() noexcept {
    query_rows_ = 0;
    block_width_ = 0;
    visible_columns_.clear();
    initialized_ = false;
  }

  bool matches(size_t query_rows, size_t block_width) const noexcept {
    return initialized_ && query_rows_ == query_rows &&
        block_width_ == block_width && visible_columns_.size() == query_rows;
  }

  const std::vector<size_t>& visible_columns() const noexcept {
    return visible_columns_;
  }

  std::vector<size_t>& mutable_visible_columns() noexcept {
    return visible_columns_;
  }

 private:
  size_t query_rows_ = 0;
  size_t block_width_ = 0;
  std::vector<size_t> visible_columns_;
  bool initialized_ = false;
};

/**
 * Materializes only mask ranges whose visible-prefix length changed.
 *
 * `visible_columns(row)` returns the desired one-prefix length for that row;
 * `fill_range(offset, count, visible)` writes ones or zeros. On a shape change
 * the physical mask is cleared before the new per-row state is established.
 */
template <typename VisibleColumns, typename FillRange>
bool update_visibility_mask_cache(
    VisibilityMaskCache& cache,
    size_t query_rows,
    size_t block_width,
    VisibleColumns&& visible_columns,
    FillRange&& fill_range) {
  if (query_rows == 0 || block_width == 0 ||
      block_width > std::numeric_limits<size_t>::max() / query_rows) {
    return false;
  }
  if (!cache.matches(query_rows, block_width)) {
    if (!fill_range(0, query_rows * block_width, false) ||
        !cache.reset_known_zero(query_rows, block_width)) {
      return false;
    }
  }

  auto& materialized = cache.mutable_visible_columns();
  for (size_t row = 0; row < query_rows; ++row) {
    const size_t target = visible_columns(row);
    if (target > block_width) {
      cache.invalidate();
      return false;
    }
    const size_t previous = materialized[row];
    if (target > previous) {
      if (!fill_range(row * block_width + previous, target - previous, true)) {
        cache.invalidate();
        return false;
      }
    } else if (target < previous) {
      if (!fill_range(row * block_width + target, previous - target, false)) {
        cache.invalidate();
        return false;
      }
    }
    materialized[row] = target;
  }
  return true;
}

/** Computes the same causal row prefixes as the former full mask rewrite. */
template <typename FillRange>
bool update_causal_visibility_mask_cache(
    VisibilityMaskCache& cache,
    size_t query_rows,
    size_t block_width,
    size_t block_begin,
    size_t block_valid_width,
    size_t causal_query_begin,
    size_t valid_query_rows,
    FillRange&& fill_range) {
  if (valid_query_rows == 0 || valid_query_rows > query_rows ||
      block_valid_width > block_width ||
      causal_query_begin >
          std::numeric_limits<size_t>::max() - valid_query_rows) {
    return false;
  }
  return update_visibility_mask_cache(
      cache,
      query_rows,
      block_width,
      [&](size_t row) {
        if (row >= valid_query_rows) {
          return size_t{0};
        }
        const size_t row_visible_end = causal_query_begin + row + size_t{1};
        return block_begin >= row_visible_end
            ? size_t{0}
            : std::min(block_valid_width, row_visible_end - block_begin);
      },
      std::forward<FillRange>(fill_range));
}

/** Prepare every layer replacement before committing any layer. */
template <typename LayerCache, typename Plan, typename ReleaseBindings>
bool apply_layer_plan_atomically(
    const std::vector<std::unique_ptr<LayerCache>>& layer_caches,
    const Plan& plan,
    ReleaseBindings&& release_bindings) {
  if (layer_caches.empty()) {
    return false;
  }
  for (const auto& cache : layer_caches) {
    if (cache == nullptr) {
      return false;
    }
  }

  using PreparedLayout = typename LayerCache::PreparedLayout;
  static_assert(
      noexcept(std::declval<LayerCache&>().commit_recompose(
          std::declval<PreparedLayout&&>())),
      "Layer-plan commit must not fail");
  std::vector<PreparedLayout> prepared;
  prepared.reserve(layer_caches.size());
  bool has_replacement = false;
  for (const auto& cache : layer_caches) {
    PreparedLayout replacement;
    if (!cache->prepare_recompose(plan, &replacement)) {
      return false;
    }
    has_replacement = has_replacement || replacement.needs_commit();
    prepared.emplace_back(std::move(replacement));
  }
  if (!has_replacement) {
    return true;
  }

  std::forward<ReleaseBindings>(release_bindings)();
  for (size_t layer = 0; layer < layer_caches.size(); ++layer) {
    if (prepared[layer].needs_commit()) {
      layer_caches[layer]->commit_recompose(std::move(prepared[layer]));
    }
  }
  return true;
}

/** Selects the physical query-row shape exported for a runtime phase. */
inline size_t query_rows_for_phase(
    size_t prefill_query_rows,
    bool decode,
    bool has_separate_decode_portfolio) {
  return decode && has_separate_decode_portfolio ? size_t{1}
                                                 : prefill_query_rows;
}

} // namespace detail
} // namespace edgeinfer

/** Configuration for the opt-in split-PTE EdgeInfer Prefill path. */
struct EdgeInferPrefillOptions {
  size_t layers = 0;
  size_t dim = 0;
  size_t query_heads = 0;
  size_t kv_heads = 0;
  size_t head_dim = 0;
  size_t vocab_size = 0;
  double rope_theta = 10000.0;
  bool hf_rope = false;
  bool pre_attention_rope = false;
  std::vector<size_t> prefill_query_rows;
  std::vector<size_t> widths;
  std::vector<size_t> decode_widths;
  size_t layers_per_shard = 0;
  size_t decode_layers_per_shard = 0;
  size_t profile_warmup = 1;
  size_t profile_iterations = 3;
};

/**
 * Execute prefill through a split Llama PTE while retaining the native
 * Qualcomm decoder PTE for baseline operation and token-by-token decode.
 *
 * The split PTE owns `llama_embedding`, per-layer QKV/post methods, and the
 * fixed-width Attention methods. The original decoder PTE remains untouched
 * unless this processor completes successfully and bridges its FP32 K/V state
 * back into KVManager for the existing decode path.
 */
class EdgeInferPromptProcessor final : public PromptProcessor {
 public:
  EdgeInferPromptProcessor(
      DecoderRunner* decoder_runner,
      KVManager* kv_manager,
      const std::string& method_name,
      Metadata metadata,
      std::unique_ptr<executorch::extension::MethodMeta> method_meta,
      executorch::extension::Module* composable_module,
      std::vector<executorch::extension::Module*> composable_layer_modules,
      executorch::extension::Module* decode_composable_module,
      std::vector<executorch::extension::Module*>
          decode_composable_layer_modules,
      int32_t native_decode_ar_len,
      EdgeInferPrefillOptions options);
  ~EdgeInferPromptProcessor() override;

  executorch::runtime::Result<uint64_t> prefill(
      std::vector<uint64_t> prompt_tokens,
      int64_t start_pos,
      bool dump_logits,
      AttentionSinkRopeRunner* attention_sink_rope_runner) override;

  /** Reserve the measured-cost physical KV layout used by steady-state decode.
   */
  executorch::runtime::Error prepare_decode(size_t target_context);

  /** Materialize block-native K/V only when native decode fallback is needed.
   */
  executorch::runtime::Error bridge_to_native_decode();

  /** Execute one token through the split graph and append its K/V in place. */
  executorch::runtime::Result<uint64_t>
  decode_token(uint64_t token, int64_t position, bool dump_logits);

  bool has_edgeinfer_state() const {
    return ready_ && sequence_length_ > 0;
  }

 private:
  struct Block;
  struct LayerCache;
  struct PreparedBlocks;
  struct StageWorkspace;
  struct PrefillScratchPool;
  struct PrefillRowRuntime;

  executorch::runtime::Error ensure_ready();
  executorch::runtime::Error run_edgeinfer_prefill(
      const std::vector<uint64_t>& prompt_tokens,
      int64_t start_pos,
      bool dump_logits,
      uint64_t* next_token);
  executorch::runtime::Error run_layer_major_prefill(
      const std::vector<uint64_t>& prompt_tokens,
      int64_t start_pos,
      bool dump_logits,
      uint64_t* next_token);
  executorch::runtime::Error run_chunk(
      const uint64_t* tokens,
      size_t valid_rows,
      bool dump_logits,
      executorch::extension::TensorPtr* logits,
      bool decode);
  executorch::runtime::Error run_chunk_graphs(
      const uint64_t* tokens,
      size_t valid_rows,
      executorch::extension::TensorPtr* logits,
      bool decode);
  executorch::runtime::Result<
      executorch::extension::llm::ComposableAttentionRunner>
  profile_attention_runner(
      executorch::extension::Module* module,
      size_t query_rows,
      const std::vector<size_t>& widths);
  executorch::runtime::Error initialize_prefill_portfolio();
  executorch::runtime::Error run_layer_major_prefill_plan(
      const std::vector<uint64_t>& prompt_tokens,
      int64_t start_pos,
      bool dump_logits,
      uint64_t* next_token,
      const executorch::extension::llm::StaticAttentionPrefillPlan& plan);
  executorch::runtime::Error recompose_for_target(
      size_t target_context,
      bool decode);
  executorch::runtime::Error recompose_for_prefill(
      const std::vector<size_t>& visible_prefixes);
  executorch::runtime::Error apply_persistent_plan(
      const executorch::extension::llm::StaticAttentionPlan& plan,
      const char* phase,
      bool aggregate_cost);
  executorch::runtime::Error bridge_kv_cache();
  executorch::runtime::Error execute_into_outputs(
      executorch::extension::Module* module,
      const std::string& method,
      const std::vector<executorch::runtime::EValue>& inputs,
      const std::vector<executorch::extension::TensorPtr>& outputs);
  void release_pre_output_bindings();
  void forget_stage_output_bindings(executorch::extension::Module* module);
  executorch::extension::Module* core_module(bool decode) const;
  executorch::extension::Module* layer_module(bool decode, size_t layer) const;
  std::vector<executorch::extension::Module*> portfolio_modules(
      bool decode) const;
  bool has_separate_decode_portfolio() const;
  size_t query_rows_for_phase(bool decode) const;
  void unload_all_composable_modules();
  void unload_composable_module_methods(executorch::extension::Module* module);
  void clear_local_state();

  executorch::extension::Module* composable_module_;
  std::vector<executorch::extension::Module*> composable_layer_modules_;
  executorch::extension::Module* decode_composable_module_;
  std::vector<executorch::extension::Module*> decode_composable_layer_modules_;
  EdgeInferPrefillOptions options_;
  std::optional<executorch::extension::llm::ComposableAttentionRunner>
      attention_runner_;
  std::optional<executorch::extension::llm::ComposableAttentionRunner>
      decode_attention_runner_;
  std::unordered_map<size_t, std::unique_ptr<PrefillRowRuntime>>
      prefill_row_runtimes_;
  std::unique_ptr<PrefillScratchPool> prefill_scratch_pool_;
  std::optional<executorch::extension::llm::StaticAttentionPrefillPlanner>
      prefill_planner_;
  std::vector<std::unique_ptr<LayerCache>> layer_caches_;
  std::vector<LayerCache*> layer_cache_view_;
  std::unique_ptr<StageWorkspace> stage_workspace_;
  edgeinfer::detail::StageOutputBindingCache stage_output_bindings_;
  std::unordered_map<
      executorch::extension::Module*,
      std::unordered_set<std::string>>
      copy_stage_methods_;
  std::vector<double> rope_inverse_frequencies_;
  executorch::aten::ScalarType scalar_type_ =
      executorch::aten::ScalarType::Undefined;
  int32_t native_decode_ar_len_ = 0;
  size_t sequence_length_ = 0;
  bool ready_ = false;
};

} // namespace example
