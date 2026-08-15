/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 *
 * Validates arbitrary-context Attention composed from a finite QNN graph
 * portfolio. The runner profiles each fixed-shape method on the device, plans
 * a measured-cost composition, and compares the result with full Attention.
 */

#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/extension/llm/runner/composable_attention_runner.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>

#include <gflags/gflags.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

DEFINE_string(
    model_path,
    "composable_attention_qnn.pte",
    "Multi-method composable Attention PTE.");
DEFINE_uint32(query_heads, 8, "Number of query heads.");
DEFINE_uint32(kv_heads, 4, "Number of key/value heads.");
DEFINE_uint32(query_rows, 1, "Static query-row shape to execute.");
DEFINE_uint32(head_dim, 8, "Attention head dimension.");
DEFINE_string(widths, "16,32,64", "Comma-separated static K/V block widths.");
DEFINE_string(
    sequence_lengths,
    "15,70,160",
    "Comma-separated dynamic K/V lengths to validate.");
DEFINE_uint32(warmup, 1, "Warm-up executions per static method.");
DEFINE_uint32(iterations, 3, "Timed executions per static method.");
DEFINE_uint32(
    fixed_width,
    0,
    "Optional Fixed-Max width to benchmark against the composed planner.");
DEFINE_uint32(
    benchmark_warmup,
    5,
    "Warm-up invocations for each end-to-end Attention design.");
DEFINE_uint32(
    benchmark_iterations,
    20,
    "Timed invocations for each end-to-end Attention design.");
DEFINE_bool(
    benchmark_prepared_blocks,
    false,
    "Also benchmark caller-owned K/V blocks with packing outside timing.");
DEFINE_bool(
    prepared_only_benchmark,
    false,
    "Benchmark only the persistent caller-owned EdgeInfer path. This skips "
    "the fp32 reference and Fixed-Max execution so long-context diagnosis "
    "does not include validation or baseline work.");
DEFINE_bool(
    assume_nonempty,
    false,
    "Use Decode-only methods whose every block contains a visible key.");
DEFINE_double(
    atol,
    0.04,
    "Maximum accepted absolute error versus fp32 when strict accuracy "
    "validation is enabled.");
DEFINE_bool(
    fail_on_accuracy,
    false,
    "Strict regression-only accuracy gate. By default numerical differences "
    "are reported but never reject a performance candidate.");

namespace {

using executorch::aten::Half;
using executorch::aten::ScalarType;
using executorch::aten::SizesType;
using executorch::aten::Tensor;
using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::extension::llm::ComposableAttentionRunner;
using executorch::extension::llm::ComposableAttentionWorkspace;
using executorch::extension::llm::PreparedAttentionBlock;
using executorch::extension::llm::StaticAttentionGraphCost;
using executorch::runtime::Error;
using executorch::runtime::EValue;

std::vector<size_t> parse_positive_sizes(const std::string& values) {
  std::vector<size_t> result;
  std::stringstream stream(values);
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (token.empty()) {
      return {};
    }
    char* end = nullptr;
    const unsigned long long value = std::strtoull(token.c_str(), &end, 10);
    if (end == token.c_str() || *end != '\0' || value == 0 ||
        value > std::numeric_limits<size_t>::max()) {
      return {};
    }
    result.push_back(static_cast<size_t>(value));
  }
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

bool fits_tensor_size(size_t value) {
  return value <= static_cast<size_t>(std::numeric_limits<SizesType>::max());
}

bool is_power_of_two(size_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

TensorPtr make_half_tensor(
    std::vector<SizesType> sizes,
    std::vector<float> values) {
  return executorch::extension::make_tensor_ptr(
      std::move(sizes),
      std::move(values),
      {},
      {},
      ScalarType::Half,
      executorch::aten::TensorShapeDynamism::STATIC);
}

std::vector<float> deterministic_values(size_t count, float frequency) {
  std::vector<float> values(count);
  for (size_t index = 0; index < count; ++index) {
    values[index] = std::sin(static_cast<float>(index + 1) * frequency);
  }
  return values;
}

struct ProfileInputs {
  std::vector<TensorPtr> tensors;
  std::vector<EValue> values;
};

struct PreparedBlocks {
  class SharedArena;

  std::unique_ptr<SharedArena> arena;
  TensorPtr query;
  std::vector<std::array<TensorPtr, 3>> owned;
  std::vector<PreparedAttentionBlock> views;
  ComposableAttentionWorkspace workspace;
};

class PreparedBlocks::SharedArena {
 public:
  explicit SharedArena(size_t bytes)
      : bytes_(bytes),
        base_(QnnExecuTorchAllocCustomMem(
            bytes,
            executorch::runtime::MemoryAllocator::kDefaultAlignment)) {}

  ~SharedArena() {
    if (base_ != nullptr) {
      QnnExecuTorchFreeCustomMem(base_);
    }
  }

  bool valid() const {
    return base_ != nullptr;
  }

  TensorPtr make_half_tensor(std::vector<SizesType> sizes) {
    size_t elements = 1;
    for (const SizesType size : sizes) {
      if (size <= 0 ||
          elements >
              std::numeric_limits<size_t>::max() / static_cast<size_t>(size)) {
        return nullptr;
      }
      elements *= static_cast<size_t>(size);
    }
    if (elements > std::numeric_limits<size_t>::max() / sizeof(Half)) {
      return nullptr;
    }
    const size_t tensor_bytes = elements * sizeof(Half);
    const size_t alignment =
        executorch::runtime::MemoryAllocator::kDefaultAlignment;
    const size_t aligned_offset =
        (offset_ + alignment - 1) / alignment * alignment;
    if (aligned_offset > bytes_ || tensor_bytes > bytes_ - aligned_offset) {
      return nullptr;
    }
    auto* data = static_cast<std::byte*>(base_) + aligned_offset;
    offset_ = aligned_offset + tensor_bytes;
    std::memset(data, 0, tensor_bytes);
    QnnExecuTorchAddCustomMemTensorAddr(data, base_);
    return executorch::extension::make_tensor_ptr(
        std::move(sizes),
        data,
        ScalarType::Half,
        executorch::aten::TensorShapeDynamism::STATIC,
        [](void*) {});
  }

 private:
  size_t bytes_;
  size_t offset_ = 0;
  void* base_;
};

size_t prepared_storage_bytes(const std::vector<size_t>& widths) {
  size_t elements = FLAGS_query_heads * FLAGS_query_rows * FLAGS_head_dim;
  for (const size_t width : widths) {
    elements += FLAGS_kv_heads * FLAGS_head_dim * width * 2;
    elements += FLAGS_query_rows * width;
  }
  const size_t state_elements = 2 *
      (2 * FLAGS_query_heads * FLAGS_query_rows * FLAGS_head_dim +
       2 * FLAGS_query_heads * FLAGS_query_rows);
  const size_t tensor_count = widths.size() * 3 + 9;
  return (elements + state_elements) * sizeof(Half) +
      tensor_count * executorch::runtime::MemoryAllocator::kDefaultAlignment;
}

PreparedBlocks prepare_blocks(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& visibility,
    const std::vector<size_t>& widths,
    size_t sequence_length) {
  PreparedBlocks result;
  result.arena = std::make_unique<PreparedBlocks::SharedArena>(
      prepared_storage_bytes(widths));
  if (!result.arena->valid()) {
    return {};
  }
  result.query = result.arena->make_half_tensor(
      {1,
       static_cast<SizesType>(FLAGS_query_heads),
       static_cast<SizesType>(FLAGS_query_rows),
       static_cast<SizesType>(FLAGS_head_dim)});
  if (result.query == nullptr || result.query->nbytes() != q.nbytes()) {
    return {};
  }
  std::memcpy(result.query->mutable_data_ptr(), q.const_data_ptr(), q.nbytes());
  result.owned.reserve(widths.size());
  result.views.reserve(widths.size());
  const size_t source_width = static_cast<size_t>(k.size(3));
  const auto* k_data = k.const_data_ptr<Half>();
  const auto* v_data = v.const_data_ptr<Half>();
  const auto* visibility_data = visibility.const_data_ptr<Half>();
  size_t offset = 0;
  for (const size_t width : widths) {
    if (offset > sequence_length) {
      return {};
    }
    const size_t valid_width = std::min(width, sequence_length - offset);
    auto k_block = result.arena->make_half_tensor(
        {1,
         static_cast<SizesType>(FLAGS_kv_heads),
         static_cast<SizesType>(FLAGS_head_dim),
         static_cast<SizesType>(width)});
    auto v_block = result.arena->make_half_tensor(
        {1,
         static_cast<SizesType>(FLAGS_kv_heads),
         static_cast<SizesType>(width),
         static_cast<SizesType>(FLAGS_head_dim)});
    auto visibility_block = result.arena->make_half_tensor(
        {1,
         1,
         static_cast<SizesType>(FLAGS_query_rows),
         static_cast<SizesType>(width)});
    if (k_block == nullptr || v_block == nullptr ||
        visibility_block == nullptr) {
      return {};
    }
    auto* k_values = k_block->mutable_data_ptr<Half>();
    auto* v_values = v_block->mutable_data_ptr<Half>();
    auto* visibility_values = visibility_block->mutable_data_ptr<Half>();
    for (size_t row = 0; row < FLAGS_kv_heads * FLAGS_head_dim; ++row) {
      for (size_t token = 0; token < valid_width; ++token) {
        k_values[row * width + token] =
            k_data[row * source_width + offset + token];
      }
    }
    for (size_t head = 0; head < FLAGS_kv_heads; ++head) {
      for (size_t token = 0; token < valid_width; ++token) {
        for (size_t dim = 0; dim < FLAGS_head_dim; ++dim) {
          v_values[(head * width + token) * FLAGS_head_dim + dim] = v_data
              [(head * source_width + offset + token) * FLAGS_head_dim + dim];
        }
      }
    }
    for (size_t row = 0; row < FLAGS_query_rows; ++row) {
      for (size_t token = 0; token < valid_width; ++token) {
        visibility_values[row * width + token] =
            visibility_data[row * source_width + offset + token];
      }
    }
    result.owned.push_back(
        {std::move(k_block), std::move(v_block), std::move(visibility_block)});
    const auto& owned = result.owned.back();
    result.views.push_back({owned[0].get(), owned[1].get(), owned[2].get()});
    offset += valid_width;
  }
  if (offset != sequence_length) {
    return {};
  }
  for (auto& bank : result.workspace.banks) {
    bank[0] = result.arena->make_half_tensor(
        {1,
         static_cast<SizesType>(FLAGS_query_heads),
         static_cast<SizesType>(FLAGS_query_rows),
         static_cast<SizesType>(FLAGS_head_dim)});
    bank[1] = result.arena->make_half_tensor(
        {1,
         static_cast<SizesType>(FLAGS_query_heads),
         static_cast<SizesType>(FLAGS_query_rows),
         1});
    bank[2] = result.arena->make_half_tensor(
        {1,
         static_cast<SizesType>(FLAGS_query_heads),
         static_cast<SizesType>(FLAGS_query_rows),
         1});
    bank[3] = result.arena->make_half_tensor(
        {1,
         static_cast<SizesType>(FLAGS_query_heads),
         static_cast<SizesType>(FLAGS_query_rows),
         static_cast<SizesType>(FLAGS_head_dim)});
    if (std::any_of(bank.begin(), bank.end(), [](const TensorPtr& tensor) {
          return tensor == nullptr;
        })) {
      return {};
    }
  }
  return result;
}

ProfileInputs make_profile_inputs(
    size_t query_heads,
    size_t kv_heads,
    size_t query_rows,
    size_t head_dim,
    size_t width,
    bool merge) {
  auto q = make_half_tensor(
      {1,
       static_cast<SizesType>(query_heads),
       static_cast<SizesType>(query_rows),
       static_cast<SizesType>(head_dim)},
      deterministic_values(query_heads * query_rows * head_dim, 0.017f));
  auto k = make_half_tensor(
      {1,
       static_cast<SizesType>(kv_heads),
       static_cast<SizesType>(head_dim),
       static_cast<SizesType>(width)},
      deterministic_values(kv_heads * head_dim * width, 0.013f));
  auto v = make_half_tensor(
      {1,
       static_cast<SizesType>(kv_heads),
       static_cast<SizesType>(width),
       static_cast<SizesType>(head_dim)},
      deterministic_values(kv_heads * width * head_dim, 0.011f));
  auto visibility = make_half_tensor(
      {1, 1, static_cast<SizesType>(query_rows), static_cast<SizesType>(width)},
      std::vector<float>(query_rows * width, 1.0f));

  ProfileInputs inputs;
  inputs.tensors.reserve(merge ? 7 : 4);
  inputs.tensors.emplace_back(std::move(q));
  inputs.tensors.emplace_back(std::move(k));
  inputs.tensors.emplace_back(std::move(v));
  inputs.tensors.emplace_back(std::move(visibility));
  if (merge) {
    auto running_max = make_half_tensor(
        {1,
         static_cast<SizesType>(query_heads),
         static_cast<SizesType>(query_rows),
         1},
        std::vector<float>(query_heads * query_rows, -1.0f));
    auto running_sum = make_half_tensor(
        {1,
         static_cast<SizesType>(query_heads),
         static_cast<SizesType>(query_rows),
         1},
        std::vector<float>(query_heads * query_rows, 1.0f));
    auto running_numerator = make_half_tensor(
        {1,
         static_cast<SizesType>(query_heads),
         static_cast<SizesType>(query_rows),
         static_cast<SizesType>(head_dim)},
        deterministic_values(query_heads * query_rows * head_dim, 0.007f));
    inputs.tensors.emplace_back(std::move(running_max));
    inputs.tensors.emplace_back(std::move(running_sum));
    inputs.tensors.emplace_back(std::move(running_numerator));
  }
  inputs.values.reserve(inputs.tensors.size());
  for (const auto& tensor : inputs.tensors) {
    inputs.values.emplace_back(*tensor);
  }
  return inputs;
}

double profile_method(
    Module& module,
    const std::string& method,
    const std::vector<EValue>& inputs) {
  const size_t query_elements =
      FLAGS_query_heads * FLAGS_query_rows * FLAGS_head_dim;
  const size_t scalar_elements = FLAGS_query_heads * FLAGS_query_rows;
  std::vector<TensorPtr> owned_outputs;
  owned_outputs.reserve(4);
  owned_outputs.emplace_back(make_half_tensor(
      {1,
       static_cast<SizesType>(FLAGS_query_heads),
       static_cast<SizesType>(FLAGS_query_rows),
       static_cast<SizesType>(FLAGS_head_dim)},
      std::vector<float>(query_elements, 0.0f)));
  owned_outputs.emplace_back(make_half_tensor(
      {1,
       static_cast<SizesType>(FLAGS_query_heads),
       static_cast<SizesType>(FLAGS_query_rows),
       1},
      std::vector<float>(scalar_elements, 0.0f)));
  owned_outputs.emplace_back(make_half_tensor(
      {1,
       static_cast<SizesType>(FLAGS_query_heads),
       static_cast<SizesType>(FLAGS_query_rows),
       1},
      std::vector<float>(scalar_elements, 0.0f)));
  owned_outputs.emplace_back(make_half_tensor(
      {1,
       static_cast<SizesType>(FLAGS_query_heads),
       static_cast<SizesType>(FLAGS_query_rows),
       static_cast<SizesType>(FLAGS_head_dim)},
      std::vector<float>(query_elements, 0.0f)));
  std::vector<EValue> output_values;
  output_values.reserve(owned_outputs.size());
  for (const TensorPtr& output : owned_outputs) {
    output_values.emplace_back(*output);
  }
  if (module.set_outputs(method, output_values) != Error::Ok) {
    // Legacy PTEs memory-plan graph outputs. A multi-output bind can fail after
    // changing an earlier pointer, so reload before using that compatible path.
    module.unload_method(method);
  }

  // Prime each exported method before timing so lazy QNN setup does not bias
  // whichever graph happens to be profiled first.
  auto priming_output = module.execute(method, inputs);
  if (!priming_output.ok()) {
    ET_LOG(
        Error,
        "Priming failed for method '%s': 0x%x",
        method.c_str(),
        static_cast<unsigned int>(priming_output.error()));
    return -1.0;
  }
  for (size_t index = 0; index < FLAGS_warmup; ++index) {
    auto output = module.execute(method, inputs);
    if (!output.ok()) {
      ET_LOG(
          Error,
          "Warm-up failed for method '%s': 0x%x",
          method.c_str(),
          static_cast<unsigned int>(output.error()));
      return -1.0;
    }
  }

  const auto start = std::chrono::steady_clock::now();
  for (size_t index = 0; index < FLAGS_iterations; ++index) {
    auto output = module.execute(method, inputs);
    if (!output.ok()) {
      ET_LOG(
          Error,
          "Timed execution failed for method '%s': 0x%x",
          method.c_str(),
          static_cast<unsigned int>(output.error()));
      return -1.0;
    }
  }
  const auto end = std::chrono::steady_clock::now();
  const double elapsed_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  return std::max(elapsed_ms / static_cast<double>(FLAGS_iterations), 1.0e-9);
}

std::vector<StaticAttentionGraphCost> profile_portfolio(
    Module& module,
    const std::vector<size_t>& widths) {
  std::vector<StaticAttentionGraphCost> costs;
  costs.reserve(widths.size());
  for (const size_t width : widths) {
    const std::string first_name =
        executorch::extension::llm::static_attention_method_name(
            true, FLAGS_query_rows, width, FLAGS_assume_nonempty);
    const std::string merge_name =
        executorch::extension::llm::static_attention_method_name(
            false, FLAGS_query_rows, width, FLAGS_assume_nonempty);
    auto first_inputs = make_profile_inputs(
        FLAGS_query_heads,
        FLAGS_kv_heads,
        FLAGS_query_rows,
        FLAGS_head_dim,
        width,
        false);
    const double first_ms =
        profile_method(module, first_name, first_inputs.values);
    auto merge_inputs = make_profile_inputs(
        FLAGS_query_heads,
        FLAGS_kv_heads,
        FLAGS_query_rows,
        FLAGS_head_dim,
        width,
        true);
    const double merge_ms =
        profile_method(module, merge_name, merge_inputs.values);
    if (first_ms <= 0.0 || merge_ms <= 0.0) {
      return {};
    }
    costs.push_back({width, first_ms, merge_ms});
    ET_LOG(
        Info,
        "Profile width=%zu: first=%.4f ms, merge=%.4f ms",
        width,
        first_ms,
        merge_ms);
  }
  return costs;
}

std::vector<float> reference_attention(
    const executorch::aten::Tensor& q,
    const executorch::aten::Tensor& k,
    const executorch::aten::Tensor& v,
    const executorch::aten::Tensor& visibility,
    size_t query_heads,
    size_t kv_heads,
    size_t query_rows,
    size_t head_dim,
    size_t sequence_length) {
  const auto* q_data = q.const_data_ptr<Half>();
  const auto* k_data = k.const_data_ptr<Half>();
  const auto* v_data = v.const_data_ptr<Half>();
  const auto* visibility_data = visibility.const_data_ptr<Half>();
  const size_t groups = query_heads / kv_heads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  std::vector<float> output(query_heads * query_rows * head_dim, 0.0f);
  std::vector<float> weights(sequence_length, 0.0f);

  for (size_t head = 0; head < query_heads; ++head) {
    const size_t kv_head = head / groups;
    for (size_t row = 0; row < query_rows; ++row) {
      float maximum = -std::numeric_limits<float>::infinity();
      for (size_t token = 0; token < sequence_length; ++token) {
        const bool visible =
            static_cast<float>(visibility_data[row * sequence_length + token]) >
            0.0f;
        if (!visible) {
          weights[token] = 0.0f;
          continue;
        }
        float score = 0.0f;
        for (size_t dim = 0; dim < head_dim; ++dim) {
          score +=
              static_cast<float>(
                  q_data[(head * query_rows + row) * head_dim + dim]) *
              static_cast<float>(
                  k_data[(kv_head * head_dim + dim) * sequence_length + token]);
        }
        weights[token] = score * scale;
        maximum = std::max(maximum, weights[token]);
      }

      float denominator = 0.0f;
      for (size_t token = 0; token < sequence_length; ++token) {
        const bool visible =
            static_cast<float>(visibility_data[row * sequence_length + token]) >
            0.0f;
        weights[token] = visible ? std::exp(weights[token] - maximum) : 0.0f;
        denominator += weights[token];
      }
      for (size_t dim = 0; dim < head_dim; ++dim) {
        float numerator = 0.0f;
        for (size_t token = 0; token < sequence_length; ++token) {
          numerator +=
              weights[token] *
              static_cast<float>(
                  v_data[(kv_head * sequence_length + token) * head_dim + dim]);
        }
        output[(head * query_rows + row) * head_dim + dim] =
            numerator / denominator;
      }
    }
  }
  return output;
}

struct ErrorStats {
  double maximum{0.0};
  double root_mean_square{0.0};
  size_t finite_comparison_count{0};
  size_t reference_nan_count{0};
  size_t reference_positive_infinity_count{0};
  size_t reference_negative_infinity_count{0};
  size_t actual_nan_count{0};
  size_t actual_positive_infinity_count{0};
  size_t actual_negative_infinity_count{0};

  size_t nonfinite_count() const {
    return reference_nan_count + reference_positive_infinity_count +
        reference_negative_infinity_count + actual_nan_count +
        actual_positive_infinity_count + actual_negative_infinity_count;
  }

  bool within_tolerance(double tolerance) const {
    return std::isfinite(tolerance) && tolerance >= 0.0 &&
        finite_comparison_count != 0 && nonfinite_count() == 0 &&
        maximum <= tolerance;
  }
};

void record_nonfinite(
    double value,
    size_t& nan_count,
    size_t& positive_infinity_count,
    size_t& negative_infinity_count) {
  if (std::isnan(value)) {
    ++nan_count;
  } else if (std::isinf(value)) {
    ++(std::signbit(value) ? negative_infinity_count : positive_infinity_count);
  }
}

ErrorStats calculate_error(
    const executorch::aten::Tensor& actual,
    const std::vector<float>& expected) {
  const auto* actual_data = actual.const_data_ptr<Half>();
  ErrorStats result;
  double squared_error = 0.0;
  for (size_t index = 0; index < expected.size(); ++index) {
    const double reference = expected[index];
    const double observed = static_cast<double>(actual_data[index]);
    if (!std::isfinite(reference)) {
      record_nonfinite(
          reference,
          result.reference_nan_count,
          result.reference_positive_infinity_count,
          result.reference_negative_infinity_count);
    }
    if (!std::isfinite(observed)) {
      record_nonfinite(
          observed,
          result.actual_nan_count,
          result.actual_positive_infinity_count,
          result.actual_negative_infinity_count);
    }
    if (!std::isfinite(reference) || !std::isfinite(observed)) {
      continue;
    }
    const double error = std::abs(observed - reference);
    squared_error += error * error;
    result.maximum = std::max(result.maximum, error);
    ++result.finite_comparison_count;
  }
  if (result.nonfinite_count() != 0 || result.finite_comparison_count == 0) {
    result.maximum = std::numeric_limits<double>::infinity();
    result.root_mean_square = std::numeric_limits<double>::infinity();
  } else {
    result.root_mean_square = std::sqrt(
        squared_error / static_cast<double>(result.finite_comparison_count));
  }
  return result;
}

void log_error_stats(
    const char* label,
    size_t sequence_length,
    const ErrorStats& error) {
  ET_LOG(
      Info,
      "%s length=%zu: max_abs_error=%.6f, rmse=%.6f, finite_compared=%zu, "
      "reference_nan=%zu, reference_pos_inf=%zu, reference_neg_inf=%zu, "
      "actual_nan=%zu, actual_pos_inf=%zu, actual_neg_inf=%zu",
      label,
      sequence_length,
      error.maximum,
      error.root_mean_square,
      error.finite_comparison_count,
      error.reference_nan_count,
      error.reference_positive_infinity_count,
      error.reference_negative_infinity_count,
      error.actual_nan_count,
      error.actual_positive_infinity_count,
      error.actual_negative_infinity_count);
}

struct TimingStats {
  double median;
  double mean;
  double minimum;
  double maximum;
};

TimingStats summarize_timings(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  const size_t middle = samples.size() / 2;
  const double median = samples.size() % 2 == 0
      ? (samples[middle - 1] + samples[middle]) / 2.0
      : samples[middle];
  double sum = 0.0;
  for (const double sample : samples) {
    sum += sample;
  }
  return {
      median,
      sum / static_cast<double>(samples.size()),
      samples.front(),
      samples.back()};
}

bool time_attention(
    ComposableAttentionRunner& runner,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& visibility,
    double& elapsed_ms) {
  const auto start = std::chrono::steady_clock::now();
  auto output = runner.run(q, k, v, visibility);
  elapsed_ms = std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - start)
                   .count();
  if (!output.ok()) {
    ET_LOG(
        Error,
        "Timed Attention invocation failed: 0x%x",
        static_cast<unsigned int>(output.error()));
    return false;
  }
  return true;
}

bool time_prepared_attention(
    ComposableAttentionRunner& runner,
    PreparedBlocks& blocks,
    double& elapsed_ms) {
  const auto start = std::chrono::steady_clock::now();
  auto output =
      runner.run_blocks(*blocks.query, blocks.views, blocks.workspace);
  elapsed_ms = std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - start)
                   .count();
  if (!output.ok()) {
    ET_LOG(
        Error,
        "Timed prepared-block Attention invocation failed: 0x%x",
        static_cast<unsigned int>(output.error()));
    return false;
  }
  return true;
}

bool benchmark_prepared_designs(
    ComposableAttentionRunner& runner,
    ComposableAttentionRunner& fixed_runner,
    const Tensor& q,
    PreparedBlocks& composed_blocks,
    PreparedBlocks& fixed_blocks,
    size_t sequence_length) {
  double ignored = 0.0;
  for (size_t index = 0; index < FLAGS_benchmark_warmup; ++index) {
    if (!time_prepared_attention(runner, composed_blocks, ignored) ||
        !time_prepared_attention(fixed_runner, fixed_blocks, ignored)) {
      return false;
    }
  }

  std::vector<double> composed_samples;
  std::vector<double> fixed_samples;
  composed_samples.reserve(FLAGS_benchmark_iterations);
  fixed_samples.reserve(FLAGS_benchmark_iterations);
  for (size_t index = 0; index < FLAGS_benchmark_iterations; ++index) {
    double composed_ms = 0.0;
    double fixed_ms = 0.0;
    const bool composed_first = index % 2 == 0;
    if (composed_first) {
      if (!time_prepared_attention(runner, composed_blocks, composed_ms) ||
          !time_prepared_attention(fixed_runner, fixed_blocks, fixed_ms)) {
        return false;
      }
    } else if (
        !time_prepared_attention(fixed_runner, fixed_blocks, fixed_ms) ||
        !time_prepared_attention(runner, composed_blocks, composed_ms)) {
      return false;
    }
    composed_samples.push_back(composed_ms);
    fixed_samples.push_back(fixed_ms);
  }

  const TimingStats composed = summarize_timings(std::move(composed_samples));
  const TimingStats fixed = summarize_timings(std::move(fixed_samples));
  ET_LOG(
      Info,
      "Prepared-block benchmark length=%zu: EdgeInfer median=%.4f ms "
      "mean=%.4f ms range=[%.4f,%.4f]; Fixed-Max(C=%u) median=%.4f ms "
      "mean=%.4f ms range=[%.4f,%.4f]; speedup=%.3fx",
      sequence_length,
      composed.median,
      composed.mean,
      composed.minimum,
      composed.maximum,
      FLAGS_fixed_width,
      fixed.median,
      fixed.mean,
      fixed.minimum,
      fixed.maximum,
      fixed.median / composed.median);
  return true;
}

bool benchmark_prepared_only(
    ComposableAttentionRunner& runner,
    size_t sequence_length) {
  const size_t query_elements =
      FLAGS_query_heads * FLAGS_query_rows * FLAGS_head_dim;
  const size_t k_elements = FLAGS_kv_heads * FLAGS_head_dim * sequence_length;
  const size_t v_elements = FLAGS_kv_heads * sequence_length * FLAGS_head_dim;
  auto q = make_half_tensor(
      {1,
       static_cast<SizesType>(FLAGS_query_heads),
       static_cast<SizesType>(FLAGS_query_rows),
       static_cast<SizesType>(FLAGS_head_dim)},
      deterministic_values(query_elements, 0.017f));
  auto k = make_half_tensor(
      {1,
       static_cast<SizesType>(FLAGS_kv_heads),
       static_cast<SizesType>(FLAGS_head_dim),
       static_cast<SizesType>(sequence_length)},
      deterministic_values(k_elements, 0.013f));
  auto v = make_half_tensor(
      {1,
       static_cast<SizesType>(FLAGS_kv_heads),
       static_cast<SizesType>(sequence_length),
       static_cast<SizesType>(FLAGS_head_dim)},
      deterministic_values(v_elements, 0.011f));
  auto visibility = make_half_tensor(
      {1,
       1,
       static_cast<SizesType>(FLAGS_query_rows),
       static_cast<SizesType>(sequence_length)},
      std::vector<float>(FLAGS_query_rows * sequence_length, 1.0f));

  auto plan = runner.plan(sequence_length);
  if (!plan.ok()) {
    ET_LOG(
        Error, "Prepared-only planning failed at length %zu.", sequence_length);
    return false;
  }
  auto blocks =
      prepare_blocks(*q, *k, *v, *visibility, plan->widths, sequence_length);
  if (blocks.query == nullptr || blocks.views.empty()) {
    ET_LOG(
        Error,
        "Prepared-only block construction failed at length %zu.",
        sequence_length);
    return false;
  }

  std::string selected_widths;
  for (const size_t width : plan->widths) {
    if (!selected_widths.empty()) {
      selected_widths += "+";
    }
    selected_widths += std::to_string(width);
  }
  ET_LOG(
      Info,
      "Prepared-only plan length=%zu: widths=%s graph_calls=%zu coverage=%zu "
      "padding=%zu predicted=%.4f ms",
      sequence_length,
      selected_widths.c_str(),
      plan->graph_calls(),
      plan->coverage(),
      plan->padding(),
      plan->predicted_cost);

  double ignored = 0.0;
  for (size_t index = 0; index < FLAGS_benchmark_warmup; ++index) {
    if (!time_prepared_attention(runner, blocks, ignored)) {
      return false;
    }
  }
  std::vector<double> samples;
  samples.reserve(FLAGS_benchmark_iterations);
  for (size_t index = 0; index < FLAGS_benchmark_iterations; ++index) {
    double elapsed_ms = 0.0;
    if (!time_prepared_attention(runner, blocks, elapsed_ms)) {
      return false;
    }
    samples.push_back(elapsed_ms);
  }
  const TimingStats timing = summarize_timings(std::move(samples));
  ET_LOG(
      Info,
      "Prepared-only EdgeInfer length=%zu: median=%.4f ms mean=%.4f ms "
      "range=[%.4f,%.4f] graph_calls=%zu",
      sequence_length,
      timing.median,
      timing.mean,
      timing.minimum,
      timing.maximum,
      plan->graph_calls());
  return true;
}

bool benchmark_designs(
    ComposableAttentionRunner& runner,
    ComposableAttentionRunner& fixed_runner,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& visibility,
    size_t sequence_length) {
  double ignored = 0.0;
  for (size_t index = 0; index < FLAGS_benchmark_warmup; ++index) {
    if (!time_attention(runner, q, k, v, visibility, ignored) ||
        !time_attention(fixed_runner, q, k, v, visibility, ignored)) {
      return false;
    }
  }

  std::vector<double> composed_samples;
  std::vector<double> fixed_samples;
  composed_samples.reserve(FLAGS_benchmark_iterations);
  fixed_samples.reserve(FLAGS_benchmark_iterations);
  for (size_t index = 0; index < FLAGS_benchmark_iterations; ++index) {
    double composed_ms = 0.0;
    double fixed_ms = 0.0;
    const bool composed_first = index % 2 == 0;
    if (composed_first) {
      if (!time_attention(runner, q, k, v, visibility, composed_ms) ||
          !time_attention(fixed_runner, q, k, v, visibility, fixed_ms)) {
        return false;
      }
    } else if (
        !time_attention(fixed_runner, q, k, v, visibility, fixed_ms) ||
        !time_attention(runner, q, k, v, visibility, composed_ms)) {
      return false;
    }
    composed_samples.push_back(composed_ms);
    fixed_samples.push_back(fixed_ms);
  }

  const TimingStats composed = summarize_timings(std::move(composed_samples));
  const TimingStats fixed = summarize_timings(std::move(fixed_samples));
  ET_LOG(
      Info,
      "Benchmark length=%zu: EdgeInfer median=%.4f ms mean=%.4f ms "
      "range=[%.4f,%.4f]; Fixed-Max(C=%u) median=%.4f ms mean=%.4f ms "
      "range=[%.4f,%.4f]; speedup=%.3fx",
      sequence_length,
      composed.median,
      composed.mean,
      composed.minimum,
      composed.maximum,
      FLAGS_fixed_width,
      fixed.median,
      fixed.mean,
      fixed.minimum,
      fixed.maximum,
      fixed.median / composed.median);
  return true;
}

bool validate_length(
    ComposableAttentionRunner& runner,
    ComposableAttentionRunner* fixed_runner,
    size_t sequence_length) {
  const size_t query_elements =
      FLAGS_query_heads * FLAGS_query_rows * FLAGS_head_dim;
  const size_t k_elements = FLAGS_kv_heads * FLAGS_head_dim * sequence_length;
  const size_t v_elements = FLAGS_kv_heads * sequence_length * FLAGS_head_dim;
  auto q = make_half_tensor(
      {1,
       static_cast<SizesType>(FLAGS_query_heads),
       static_cast<SizesType>(FLAGS_query_rows),
       static_cast<SizesType>(FLAGS_head_dim)},
      deterministic_values(query_elements, 0.017f));
  auto k = make_half_tensor(
      {1,
       static_cast<SizesType>(FLAGS_kv_heads),
       static_cast<SizesType>(FLAGS_head_dim),
       static_cast<SizesType>(sequence_length)},
      deterministic_values(k_elements, 0.013f));
  auto v = make_half_tensor(
      {1,
       static_cast<SizesType>(FLAGS_kv_heads),
       static_cast<SizesType>(sequence_length),
       static_cast<SizesType>(FLAGS_head_dim)},
      deterministic_values(v_elements, 0.011f));

  std::vector<float> visibility_values(FLAGS_query_rows * sequence_length);
  for (size_t row = 0; row < FLAGS_query_rows; ++row) {
    const size_t visible_through = sequence_length - FLAGS_query_rows + row + 1;
    std::fill_n(
        visibility_values.begin() + row * sequence_length,
        visible_through,
        1.0f);
  }
  auto visibility = make_half_tensor(
      {1,
       1,
       static_cast<SizesType>(FLAGS_query_rows),
       static_cast<SizesType>(sequence_length)},
      std::move(visibility_values));
  const auto expected = reference_attention(
      *q,
      *k,
      *v,
      *visibility,
      FLAGS_query_heads,
      FLAGS_kv_heads,
      FLAGS_query_rows,
      FLAGS_head_dim,
      sequence_length);

  auto selected_plan = runner.plan(sequence_length);
  if (!selected_plan.ok()) {
    ET_LOG(Error, "Planning failed at length %zu", sequence_length);
    return false;
  }
  std::string selected_widths;
  for (const size_t width : selected_plan->widths) {
    if (!selected_widths.empty()) {
      selected_widths += "+";
    }
    selected_widths += std::to_string(width);
  }
  ET_LOG(
      Info,
      "Selected plan length=%zu: widths=%s, graph_calls=%zu, coverage=%zu, "
      "padding=%zu, predicted=%.4f ms",
      sequence_length,
      selected_widths.c_str(),
      selected_plan->graph_calls(),
      selected_plan->coverage(),
      selected_plan->padding(),
      selected_plan->predicted_cost);

  auto output = runner.run(*q, *k, *v, *visibility);
  if (!output.ok()) {
    ET_LOG(
        Error,
        "Composed Attention failed at length %zu: 0x%x",
        sequence_length,
        static_cast<unsigned int>(output.error()));
    return false;
  }
  if ((*output)->scalar_type() != ScalarType::Half ||
      static_cast<size_t>((*output)->numel()) != expected.size()) {
    ET_LOG(Error, "Unexpected output metadata at length %zu", sequence_length);
    return false;
  }

  const ErrorStats composed_error = calculate_error(**output, expected);
  log_error_stats("EdgeInfer", sequence_length, composed_error);
  if (FLAGS_fail_on_accuracy && !composed_error.within_tolerance(FLAGS_atol)) {
    return false;
  }

  if (fixed_runner == nullptr) {
    return true;
  }
  auto fixed_output = fixed_runner->run(*q, *k, *v, *visibility);
  if (!fixed_output.ok() ||
      (*fixed_output)->scalar_type() != ScalarType::Half ||
      static_cast<size_t>((*fixed_output)->numel()) != expected.size()) {
    ET_LOG(Error, "Fixed-Max Attention failed at length %zu", sequence_length);
    return false;
  }
  const ErrorStats fixed_error = calculate_error(**fixed_output, expected);
  std::vector<float> composed_values(expected.size());
  const auto* composed_data = (*output)->const_data_ptr<Half>();
  for (size_t index = 0; index < expected.size(); ++index) {
    composed_values[index] = static_cast<float>(composed_data[index]);
  }
  const ErrorStats design_delta =
      calculate_error(**fixed_output, composed_values);
  log_error_stats("Fixed-Max", sequence_length, fixed_error);
  log_error_stats("Fixed-Max-vs-EdgeInfer", sequence_length, design_delta);
  const bool benchmark_passed = benchmark_designs(
      runner, *fixed_runner, *q, *k, *v, *visibility, sequence_length);
  bool prepared_benchmark_passed = true;
  if (FLAGS_benchmark_prepared_blocks) {
    auto composed_blocks = prepare_blocks(
        *q, *k, *v, *visibility, selected_plan->widths, sequence_length);
    auto fixed_blocks = prepare_blocks(
        *q,
        *k,
        *v,
        *visibility,
        std::vector<size_t>{static_cast<size_t>(FLAGS_fixed_width)},
        sequence_length);
    prepared_benchmark_passed = !composed_blocks.views.empty() &&
        !fixed_blocks.views.empty() &&
        benchmark_prepared_designs(
            runner,
            *fixed_runner,
            *q,
            composed_blocks,
            fixed_blocks,
            sequence_length);
  }
  const bool fixed_accuracy_passed = fixed_error.within_tolerance(FLAGS_atol);
  return (!FLAGS_fail_on_accuracy || fixed_accuracy_passed) &&
      benchmark_passed && prepared_benchmark_passed;
}

} // namespace

int main(int argc, char** argv) {
  executorch::runtime::runtime_init();
  QnnExecuTorchBackendRegister(
      reinterpret_cast<void*>(executorch::runtime::register_backend));
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc != 1 || FLAGS_query_heads == 0 || FLAGS_kv_heads == 0 ||
      FLAGS_query_rows == 0 || FLAGS_head_dim == 0 ||
      FLAGS_query_heads % FLAGS_kv_heads != 0 || FLAGS_iterations == 0 ||
      (FLAGS_fixed_width > 0 && FLAGS_benchmark_iterations == 0)) {
    ET_LOG(Error, "Invalid composable Attention runner arguments.");
    return 1;
  }
  const auto widths = parse_positive_sizes(FLAGS_widths);
  const auto sequence_lengths = parse_positive_sizes(FLAGS_sequence_lengths);
  if (widths.empty() || sequence_lengths.empty() ||
      sequence_lengths.front() < FLAGS_query_rows ||
      !fits_tensor_size(FLAGS_query_heads) ||
      !fits_tensor_size(FLAGS_kv_heads) ||
      !fits_tensor_size(FLAGS_query_rows) ||
      !fits_tensor_size(FLAGS_head_dim) ||
      !std::all_of(
          widths.begin(),
          widths.end(),
          [](size_t width) {
            return fits_tensor_size(width) &&
                (is_power_of_two(width) || width == FLAGS_fixed_width);
          }) ||
      !std::all_of(
          sequence_lengths.begin(), sequence_lengths.end(), fits_tensor_size)) {
    ET_LOG(
        Error,
        "Composable widths must be powers of two; fixed_width may be an "
        "arbitrary exported capacity. All lengths and dimensions must fit "
        "SizesType.");
    return 1;
  }
  if (FLAGS_fixed_width > 0 &&
      (static_cast<size_t>(FLAGS_fixed_width) < sequence_lengths.back() ||
       !std::binary_search(
           widths.begin(),
           widths.end(),
           static_cast<size_t>(FLAGS_fixed_width)))) {
    ET_LOG(
        Error,
        "fixed_width must be an exported width no smaller than every "
        "sequence length.");
    return 1;
  }

  Module module(FLAGS_model_path, Module::LoadMode::MmapUseMlockIgnoreErrors);
  auto costs = profile_portfolio(module, widths);
  if (costs.size() != widths.size()) {
    return 2;
  }
  auto runner_result = ComposableAttentionRunner::create(
      &module, FLAGS_query_rows, costs, FLAGS_assume_nonempty);
  if (!runner_result.ok()) {
    ET_LOG(Error, "Failed to create composable Attention runner.");
    return 2;
  }
  auto runner = std::move(*runner_result);
  std::optional<ComposableAttentionRunner> fixed_runner_storage;
  if (FLAGS_fixed_width > 0) {
    const auto fixed_cost = std::find_if(
        costs.begin(), costs.end(), [](const StaticAttentionGraphCost& cost) {
          return cost.width == FLAGS_fixed_width;
        });
    if (fixed_cost == costs.end()) {
      ET_LOG(Error, "Could not find the profiled Fixed-Max graph.");
      return 2;
    }
    auto fixed_runner_result = ComposableAttentionRunner::create(
        &module,
        FLAGS_query_rows,
        std::vector<StaticAttentionGraphCost>{*fixed_cost},
        FLAGS_assume_nonempty);
    if (!fixed_runner_result.ok()) {
      ET_LOG(Error, "Failed to create Fixed-Max Attention runner.");
      return 2;
    }
    fixed_runner_storage.emplace(std::move(*fixed_runner_result));
  }
  ComposableAttentionRunner* fixed_runner =
      fixed_runner_storage ? &*fixed_runner_storage : nullptr;

  bool passed = true;
  for (const size_t sequence_length : sequence_lengths) {
    passed = (FLAGS_prepared_only_benchmark
                  ? benchmark_prepared_only(runner, sequence_length)
                  : validate_length(runner, fixed_runner, sequence_length)) &&
        passed;
  }
  if (!passed) {
    ET_LOG(Error, "Composable QNN Attention validation failed.");
    return 3;
  }
  ET_LOG(
      Info,
      "Composable QNN Attention validation passed through length %zu "
      "(largest static width: %zu).",
      sequence_lengths.back(),
      widths.back());
  return 0;
}
