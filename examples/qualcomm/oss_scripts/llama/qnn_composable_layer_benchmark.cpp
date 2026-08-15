/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Benchmarks one Qwen3 decoder layer at long-context Decode or Prefill. A
 * single exported layer may be reused for logical Decode layers while each
 * layer owns independent KV. Synthetic input initialization is outside timed
 * regions; Prefill QKV generation, KV append, causal masks, Attention, and the
 * post-Attention/FFN graph are timed end to end.
 */

#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/extension/llm/runner/composable_attention_runner.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/memory_allocator.h>
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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

DEFINE_string(
    attention_model_path,
    "composable_attention_qnn.pte",
    "PTE containing fixed-width first/merge Attention methods.");
DEFINE_string(
    pre_model_path,
    "composable_layer_pre_qnn.pte",
    "PTE containing the layer pre-Attention method.");
DEFINE_string(
    post_model_path,
    "composable_layer_post_qnn.pte",
    "PTE containing the layer post-Attention method.");
DEFINE_string(
    baseline_model_path,
    "composable_layer_full_qnn.pte",
    "PTE containing the fixed-capacity complete-layer baseline.");
DEFINE_string(design, "edge", "attention, edge, or baseline.");
DEFINE_string(phase, "decode", "decode or prefill.");
DEFINE_string(output_path, "", "Optional final FP16 hidden-state output.");
DEFINE_uint32(logical_layers, 1, "Logical layers reusing the exported layer.");
DEFINE_uint32(capacity, 40960, "Visible Decode context and baseline capacity.");
DEFINE_uint32(block_width, 1024, "Composable Attention block width.");
DEFINE_string(
    block_widths,
    "",
    "Optional comma-separated measured K/V width portfolio. Empty preserves "
    "the legacy single --block_width path.");
DEFINE_string(
    first_costs_ms,
    "",
    "Comma-separated first-graph costs aligned with --block_widths.");
DEFINE_string(
    merge_costs_ms,
    "",
    "Comma-separated merge-graph costs aligned with --block_widths.");
DEFINE_uint32(query_rows, 1, "Static Query rows in every graph invocation.");
DEFINE_uint32(
    prompt_tokens,
    40960,
    "Prompt tokens processed by one continuous Prefill invocation.");
DEFINE_uint32(dim, 1024, "Hidden dimension.");
DEFINE_uint32(hidden_dim, 3072, "Feed-forward hidden dimension.");
DEFINE_uint32(query_heads, 16, "Query heads.");
DEFINE_uint32(kv_heads, 8, "K/V heads.");
DEFINE_uint32(head_dim, 128, "Attention head dimension.");
DEFINE_uint32(warmup, 1, "Warm-up invocations.");
DEFINE_uint32(iterations, 5, "Measured invocations.");
DEFINE_bool(
    assume_nonempty,
    false,
    "Use Decode-only Attention methods where every block has a visible key.");
DEFINE_bool(
    fail_on_nonfinite,
    false,
    "Strict regression-only output gate. By default NaN and Inf are reported "
    "without failing a performance run.");

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
using executorch::extension::llm::StaticAttentionPlan;
using executorch::runtime::Error;
using executorch::runtime::EValue;

constexpr size_t kAlignment =
    executorch::runtime::MemoryAllocator::kDefaultAlignment;

std::vector<size_t> parse_positive_sizes(const std::string& text) {
  std::vector<size_t> values;
  std::stringstream stream(text);
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
    values.push_back(static_cast<size_t>(value));
  }
  return values;
}

std::vector<double> parse_positive_costs(const std::string& text) {
  std::vector<double> values;
  std::stringstream stream(text);
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (token.empty()) {
      return {};
    }
    char* end = nullptr;
    const double value = std::strtod(token.c_str(), &end);
    if (end == token.c_str() || *end != '\0' || !std::isfinite(value) ||
        value <= 0.0) {
      return {};
    }
    values.push_back(value);
  }
  return values;
}

std::vector<size_t> uniform_widths(size_t total, size_t width) {
  if (total == 0 || width == 0 || total % width != 0) {
    return {};
  }
  return std::vector<size_t>(total / width, width);
}

size_t width_coverage(const std::vector<size_t>& widths) {
  size_t coverage = 0;
  for (const size_t width : widths) {
    if (width > std::numeric_limits<size_t>::max() - coverage) {
      return 0;
    }
    coverage += width;
  }
  return coverage;
}

size_t align_up(size_t value) {
  return (value + kAlignment - 1) / kAlignment * kAlignment;
}

size_t tensor_bytes(std::initializer_list<size_t> sizes) {
  size_t elements = 1;
  for (const size_t size : sizes) {
    if (size == 0 || elements > std::numeric_limits<size_t>::max() / size) {
      return 0;
    }
    elements *= size;
  }
  return elements * sizeof(Half);
}

size_t edge_cache_storage_bytes(
    const std::vector<size_t>& widths,
    bool prefill) {
  size_t bytes = kAlignment;
  for (const size_t width : widths) {
    const size_t key = tensor_bytes({FLAGS_kv_heads, FLAGS_head_dim, width});
    const size_t value = tensor_bytes({FLAGS_kv_heads, width, FLAGS_head_dim});
    const size_t visibility =
        prefill ? tensor_bytes({1, FLAGS_query_rows, width}) : 0;
    if (key == 0 || value == 0 || (prefill && visibility == 0) ||
        key > std::numeric_limits<size_t>::max() - value ||
        key + value > std::numeric_limits<size_t>::max() - visibility ||
        key + value + visibility >
            std::numeric_limits<size_t>::max() - 3 * kAlignment) {
      return 0;
    }
    const size_t block_bytes = key + value + visibility + 3 * kAlignment;
    if (bytes > std::numeric_limits<size_t>::max() - block_bytes) {
      return 0;
    }
    bytes += block_bytes;
  }
  return bytes;
}

class SharedArena {
 public:
  explicit SharedArena(size_t bytes)
      : bytes_(align_up(bytes) + kAlignment),
        base_(QnnExecuTorchAllocCustomMem(bytes_, kAlignment)) {}

  ~SharedArena() {
    if (base_ != nullptr) {
      QnnExecuTorchFreeCustomMem(base_);
    }
  }

  SharedArena(const SharedArena&) = delete;
  SharedArena& operator=(const SharedArena&) = delete;

  bool valid() const {
    return base_ != nullptr;
  }

  TensorPtr make(std::vector<SizesType> sizes, bool zero = true) {
    size_t elements = 1;
    for (const SizesType size : sizes) {
      if (size <= 0 ||
          elements >
              std::numeric_limits<size_t>::max() / static_cast<size_t>(size)) {
        return nullptr;
      }
      elements *= static_cast<size_t>(size);
    }
    const size_t bytes = elements * sizeof(Half);
    const size_t offset = align_up(offset_);
    if (offset > bytes_ || bytes > bytes_ - offset) {
      return nullptr;
    }
    void* data = static_cast<std::byte*>(base_) + offset;
    offset_ = offset + bytes;
    if (zero) {
      std::memset(data, 0, bytes);
    }
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

float deterministic_value(size_t index, size_t layer, size_t salt) {
  const int value =
      static_cast<int>((index * 17 + layer * 29 + salt * 13) % 127);
  return static_cast<float>(value - 63) / 256.0f;
}

void fill_tensor(Tensor& tensor, size_t layer, size_t salt) {
  auto* values = tensor.mutable_data_ptr<Half>();
  const size_t count = tensor.numel();
  for (size_t index = 0; index < count; ++index) {
    values[index] = Half(deterministic_value(index, layer, salt));
  }
}

struct EdgeCache {
  EdgeCache(
      size_t layer,
      bool prefill,
      const std::vector<size_t>& layout_widths)
      : block_widths(layout_widths),
        arena(edge_cache_storage_bytes(layout_widths, prefill)) {
    if (!arena.valid()) {
      return;
    }
    const size_t count = block_widths.size();
    keys.reserve(count);
    values.reserve(count);
    visibilities.reserve(count);
    valid_widths.reserve(count);
    for (const size_t width : block_widths) {
      valid_widths.push_back(prefill ? 0 : width);
    }
    visible_columns.assign(
        count, std::vector<size_t>(FLAGS_query_rows, size_t{0}));
    for (size_t block = 0; block < count; ++block) {
      const size_t width = block_widths[block];
      auto key = arena.make(
          {1,
           static_cast<SizesType>(FLAGS_kv_heads),
           static_cast<SizesType>(FLAGS_head_dim),
           static_cast<SizesType>(width)},
          false);
      auto value = arena.make(
          {1,
           static_cast<SizesType>(FLAGS_kv_heads),
           static_cast<SizesType>(width),
           static_cast<SizesType>(FLAGS_head_dim)},
          false);
      auto visibility = prefill ? arena.make(
                                      {1,
                                       1,
                                       static_cast<SizesType>(FLAGS_query_rows),
                                       static_cast<SizesType>(width)})
                                : nullptr;
      if (key == nullptr || value == nullptr ||
          (prefill && visibility == nullptr)) {
        keys.clear();
        values.clear();
        visibilities.clear();
        return;
      }
      if (!prefill) {
        fill_tensor(*key, layer, 1);
        fill_tensor(*value, layer, 2);
      }
      keys.emplace_back(std::move(key));
      values.emplace_back(std::move(value));
      visibilities.emplace_back(std::move(visibility));
    }
  }

  void reset_prefill() {
    std::fill(valid_widths.begin(), valid_widths.end(), size_t{0});
    for (size_t block = 0; block < visibilities.size(); ++block) {
      std::fill(
          visible_columns[block].begin(),
          visible_columns[block].end(),
          size_t{0});
      if (visibilities[block] != nullptr) {
        std::memset(
            visibilities[block]->mutable_data_ptr(),
            0,
            visibilities[block]->nbytes());
      }
    }
    blocks.clear();
  }

  bool append(const Tensor& key, const Tensor& value, size_t valid_rows) {
    if (key.dim() != 4 || value.dim() != 4 || key.size(0) != 1 ||
        value.size(0) != 1 ||
        static_cast<size_t>(key.size(1)) != FLAGS_kv_heads ||
        static_cast<size_t>(value.size(1)) != FLAGS_kv_heads ||
        static_cast<size_t>(key.size(2)) != FLAGS_head_dim ||
        static_cast<size_t>(value.size(3)) != FLAGS_head_dim ||
        key.size(3) != value.size(2) || valid_rows == 0 ||
        valid_rows > static_cast<size_t>(key.size(3))) {
      return false;
    }
    const size_t rows = static_cast<size_t>(key.size(3));
    const auto* source_key = key.const_data_ptr<Half>();
    const auto* source_value = value.const_data_ptr<Half>();
    size_t source_row = 0;
    for (size_t block = 0; block < keys.size() && source_row < valid_rows;
         ++block) {
      const size_t width = block_widths[block];
      const size_t available = width - valid_widths[block];
      if (available == 0) {
        continue;
      }
      const size_t count = std::min(available, valid_rows - source_row);
      auto* destination_key = keys[block]->mutable_data_ptr<Half>();
      auto* destination_value = values[block]->mutable_data_ptr<Half>();
      for (size_t row = 0; row < FLAGS_kv_heads * FLAGS_head_dim; ++row) {
        std::memcpy(
            destination_key + row * width + valid_widths[block],
            source_key + row * rows + source_row,
            count * sizeof(Half));
      }
      for (size_t head = 0; head < FLAGS_kv_heads; ++head) {
        std::memcpy(
            destination_value +
                (head * width + valid_widths[block]) * FLAGS_head_dim,
            source_value + (head * rows + source_row) * FLAGS_head_dim,
            count * FLAGS_head_dim * sizeof(Half));
      }
      valid_widths[block] += count;
      source_row += count;
    }
    return source_row == valid_rows;
  }

  bool prepare_causal_blocks(size_t query_begin, size_t valid_rows) {
    if (valid_rows == 0 || valid_rows > FLAGS_query_rows ||
        query_begin > std::numeric_limits<size_t>::max() - valid_rows) {
      return false;
    }
    const size_t query_end = query_begin + valid_rows;
    blocks.clear();
    size_t block_begin = 0;
    for (size_t block = 0; block < keys.size(); ++block) {
      if (valid_widths[block] == 0 || block_begin >= query_end) {
        break;
      }
      const size_t width = block_widths[block];
      auto* mask = visibilities[block]->mutable_data_ptr<Half>();
      for (size_t row = 0; row < FLAGS_query_rows; ++row) {
        const size_t target =
            row >= valid_rows || block_begin > query_begin + row
            ? 0
            : std::min(
                  valid_widths[block], query_begin + row + 1 - block_begin);
        const size_t previous = visible_columns[block][row];
        if (target > previous) {
          std::fill(
              mask + row * width + previous,
              mask + row * width + target,
              Half(1.0f));
        } else if (target < previous) {
          std::fill(
              mask + row * width + target,
              mask + row * width + previous,
              Half(0.0f));
        }
        visible_columns[block][row] = target;
      }
      blocks.push_back(
          {keys[block].get(), values[block].get(), visibilities[block].get()});
      block_begin += width;
    }
    return !blocks.empty();
  }

  std::vector<size_t> block_widths;
  SharedArena arena;
  std::vector<TensorPtr> keys;
  std::vector<TensorPtr> values;
  std::vector<TensorPtr> visibilities;
  std::vector<size_t> valid_widths;
  std::vector<std::vector<size_t>> visible_columns;
  std::vector<PreparedAttentionBlock> blocks;
};

struct BaselineCache {
  explicit BaselineCache(size_t layer)
      : arena(
            tensor_bytes({FLAGS_kv_heads, FLAGS_head_dim, FLAGS_capacity - 1}) +
            tensor_bytes({FLAGS_kv_heads, FLAGS_capacity - 1, FLAGS_head_dim}) +
            3 * kAlignment) {
    if (!arena.valid()) {
      return;
    }
    key = arena.make(
        {1,
         static_cast<SizesType>(FLAGS_kv_heads),
         static_cast<SizesType>(FLAGS_head_dim),
         static_cast<SizesType>(FLAGS_capacity - 1)},
        false);
    value = arena.make(
        {1,
         static_cast<SizesType>(FLAGS_kv_heads),
         static_cast<SizesType>(FLAGS_capacity - 1),
         static_cast<SizesType>(FLAGS_head_dim)},
        false);
    if (key == nullptr || value == nullptr) {
      return;
    }

    auto* key_data = key->mutable_data_ptr<Half>();
    for (size_t row = 0; row < FLAGS_kv_heads * FLAGS_head_dim; ++row) {
      for (size_t token = 0; token < FLAGS_capacity - 1; ++token) {
        const size_t local = token % FLAGS_block_width;
        const size_t template_index = row * FLAGS_block_width + local;
        key_data[row * (FLAGS_capacity - 1) + token] =
            Half(deterministic_value(template_index, layer, 1));
      }
    }
    auto* value_data = value->mutable_data_ptr<Half>();
    for (size_t head = 0; head < FLAGS_kv_heads; ++head) {
      for (size_t token = 0; token < FLAGS_capacity - 1; ++token) {
        const size_t local = token % FLAGS_block_width;
        for (size_t dim = 0; dim < FLAGS_head_dim; ++dim) {
          const size_t template_index =
              (head * FLAGS_block_width + local) * FLAGS_head_dim + dim;
          value_data
              [(head * (FLAGS_capacity - 1) + token) * FLAGS_head_dim + dim] =
                  Half(deterministic_value(template_index, layer, 2));
        }
      }
    }
  }

  SharedArena arena;
  TensorPtr key;
  TensorPtr value;
};

size_t common_storage_bytes() {
  const size_t vector = FLAGS_query_heads * FLAGS_head_dim;
  const size_t kv_vector = FLAGS_kv_heads * FLAGS_head_dim;
  const size_t scalar = FLAGS_query_heads;
  const size_t rows = FLAGS_query_rows;
  size_t elements = 0;
  elements += 2 * rows * FLAGS_dim; // initial and working hidden
  elements += 2 * rows * FLAGS_head_dim; // cosine and sine
  elements += rows * (vector + 2 * kv_vector); // pre outputs
  elements += rows * (FLAGS_block_width + FLAGS_capacity); // visibility/mask
  elements += rows * (4 * vector + 4 * scalar); // Attention banks
  elements += rows * (FLAGS_dim + 2 * kv_vector); // layer/full outputs
  return elements * sizeof(Half) + 32 * kAlignment;
}

struct CommonBuffers {
  CommonBuffers() : arena(common_storage_bytes()) {
    if (!arena.valid()) {
      return;
    }
    initial_hidden = arena.make(
        {1,
         static_cast<SizesType>(FLAGS_query_rows),
         static_cast<SizesType>(FLAGS_dim)},
        false);
    hidden = arena.make(
        {1,
         static_cast<SizesType>(FLAGS_query_rows),
         static_cast<SizesType>(FLAGS_dim)},
        false);
    cosine = arena.make(
        {static_cast<SizesType>(FLAGS_query_rows),
         static_cast<SizesType>(FLAGS_head_dim)},
        false);
    sine = arena.make(
        {static_cast<SizesType>(FLAGS_query_rows),
         static_cast<SizesType>(FLAGS_head_dim)},
        true);
    pre_query = arena.make(
        {1,
         static_cast<SizesType>(FLAGS_query_heads),
         static_cast<SizesType>(FLAGS_query_rows),
         static_cast<SizesType>(FLAGS_head_dim)});
    pre_key = arena.make(
        {1,
         static_cast<SizesType>(FLAGS_kv_heads),
         static_cast<SizesType>(FLAGS_head_dim),
         static_cast<SizesType>(FLAGS_query_rows)});
    pre_value = arena.make(
        {1,
         static_cast<SizesType>(FLAGS_kv_heads),
         static_cast<SizesType>(FLAGS_query_rows),
         static_cast<SizesType>(FLAGS_head_dim)});
    visibility = arena.make(
        {1,
         1,
         static_cast<SizesType>(FLAGS_query_rows),
         static_cast<SizesType>(FLAGS_block_width)},
        false);
    mask = arena.make(
        {1,
         1,
         static_cast<SizesType>(FLAGS_query_rows),
         static_cast<SizesType>(FLAGS_capacity)},
        true);
    for (auto& bank : attention_workspace.banks) {
      bank[0] = arena.make(
          {1,
           static_cast<SizesType>(FLAGS_query_heads),
           static_cast<SizesType>(FLAGS_query_rows),
           static_cast<SizesType>(FLAGS_head_dim)});
      bank[1] = arena.make(
          {1,
           static_cast<SizesType>(FLAGS_query_heads),
           static_cast<SizesType>(FLAGS_query_rows),
           1});
      bank[2] = arena.make(
          {1,
           static_cast<SizesType>(FLAGS_query_heads),
           static_cast<SizesType>(FLAGS_query_rows),
           1});
      bank[3] = arena.make(
          {1,
           static_cast<SizesType>(FLAGS_query_heads),
           static_cast<SizesType>(FLAGS_query_rows),
           static_cast<SizesType>(FLAGS_head_dim)});
    }
    layer_output = arena.make(
        {1,
         static_cast<SizesType>(FLAGS_query_rows),
         static_cast<SizesType>(FLAGS_dim)});
    full_new_key = arena.make(
        {1,
         static_cast<SizesType>(FLAGS_kv_heads),
         static_cast<SizesType>(FLAGS_head_dim),
         static_cast<SizesType>(FLAGS_query_rows)});
    full_new_value = arena.make(
        {1,
         static_cast<SizesType>(FLAGS_kv_heads),
         static_cast<SizesType>(FLAGS_query_rows),
         static_cast<SizesType>(FLAGS_head_dim)});
    if (initial_hidden != nullptr) {
      fill_tensor(*initial_hidden, 0, 3);
      std::memcpy(
          hidden->mutable_data_ptr(),
          initial_hidden->const_data_ptr(),
          initial_hidden->nbytes());
    }
    if (cosine != nullptr) {
      auto* values = cosine->mutable_data_ptr<Half>();
      for (size_t index = 0; index < cosine->numel(); ++index) {
        values[index] = Half(std::cos(static_cast<float>(index) * 0.0001f));
      }
    }
    if (visibility != nullptr) {
      auto* values = visibility->mutable_data_ptr<Half>();
      std::fill(values, values + visibility->numel(), Half(1.0f));
    }
    if (pre_query != nullptr) {
      fill_tensor(*pre_query, 0, 4);
    }
  }

  bool valid() const {
    if (initial_hidden == nullptr || hidden == nullptr || cosine == nullptr ||
        sine == nullptr || pre_query == nullptr || pre_key == nullptr ||
        pre_value == nullptr || visibility == nullptr || mask == nullptr ||
        layer_output == nullptr || full_new_key == nullptr ||
        full_new_value == nullptr) {
      return false;
    }
    for (const auto& bank : attention_workspace.banks) {
      if (std::any_of(bank.begin(), bank.end(), [](const TensorPtr& value) {
            return value == nullptr;
          })) {
        return false;
      }
    }
    return true;
  }

  void reset_hidden() {
    std::memcpy(
        hidden->mutable_data_ptr(),
        initial_hidden->const_data_ptr(),
        hidden->nbytes());
  }

  SharedArena arena;
  TensorPtr initial_hidden;
  TensorPtr hidden;
  TensorPtr cosine;
  TensorPtr sine;
  TensorPtr pre_query;
  TensorPtr pre_key;
  TensorPtr pre_value;
  TensorPtr visibility;
  TensorPtr mask;
  ComposableAttentionWorkspace attention_workspace;
  TensorPtr layer_output;
  TensorPtr full_new_key;
  TensorPtr full_new_value;
};

Error bind_outputs(
    Module& module,
    const std::string& method,
    const std::vector<TensorPtr>& outputs) {
  std::vector<EValue> values;
  values.reserve(outputs.size());
  for (const TensorPtr& output : outputs) {
    values.emplace_back(*output);
  }
  return module.set_outputs(method, values);
}

bool execute(
    Module& module,
    const std::string& method,
    const std::vector<EValue>& inputs,
    size_t expected_outputs) {
  auto outputs = module.execute(method, inputs);
  return outputs.ok() && outputs->size() == expected_outputs &&
      std::all_of(outputs->begin(), outputs->end(), [](const EValue& value) {
           return value.isTensor();
         });
}

void install_current_kv(const CommonBuffers& common, EdgeCache& cache) {
  Tensor& key = *cache.keys.back();
  Tensor& value = *cache.values.back();
  const auto* source_key = common.pre_key->const_data_ptr<Half>();
  auto* destination_key = key.mutable_data_ptr<Half>();
  for (size_t row = 0; row < FLAGS_kv_heads * FLAGS_head_dim; ++row) {
    destination_key[row * FLAGS_block_width + FLAGS_block_width - 1] =
        source_key[row];
  }
  auto* destination_value = value.mutable_data_ptr<Half>();
  const auto* source_value = common.pre_value->const_data_ptr<Half>();
  for (size_t head = 0; head < FLAGS_kv_heads; ++head) {
    std::memcpy(
        destination_value +
            (head * FLAGS_block_width + FLAGS_block_width - 1) * FLAGS_head_dim,
        source_value + head * FLAGS_head_dim,
        FLAGS_head_dim * sizeof(Half));
  }
}

struct Sample {
  double total_ms = 0.0;
  double pre_ms = 0.0;
  double attention_ms = 0.0;
  double post_ms = 0.0;
  double host_ms = 0.0;
  double full_ms = 0.0;
  size_t attention_calls = 0;
};

double elapsed_ms(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

bool run_attention(
    ComposableAttentionRunner& attention,
    CommonBuffers& common,
    EdgeCache& cache,
    Sample* sample) {
  const auto start = std::chrono::steady_clock::now();
  auto output = attention.run_blocks(
      *common.pre_query, cache.blocks, common.attention_workspace);
  if (!output.ok()) {
    return false;
  }
  const double duration = elapsed_ms(start);
  if (sample != nullptr) {
    sample->total_ms = duration;
    sample->attention_ms = duration;
  }
  return true;
}

bool run_edge(
    Module& pre_module,
    Module& post_module,
    ComposableAttentionRunner& attention,
    CommonBuffers& common,
    std::vector<std::unique_ptr<EdgeCache>>& caches,
    Sample* sample) {
  common.reset_hidden();
  Sample current;
  const auto total_start = std::chrono::steady_clock::now();
  for (size_t layer = 0; layer < caches.size(); ++layer) {
    if (layer > 0) {
      const auto host_start = std::chrono::steady_clock::now();
      std::memcpy(
          common.hidden->mutable_data_ptr(),
          common.layer_output->const_data_ptr(),
          common.hidden->nbytes());
      current.host_ms += elapsed_ms(host_start);
    }
    auto start = std::chrono::steady_clock::now();
    if (!execute(
            pre_module,
            "llama_layer_0_pre",
            {*common.hidden, *common.cosine, *common.sine},
            3)) {
      return false;
    }
    current.pre_ms += elapsed_ms(start);

    start = std::chrono::steady_clock::now();
    install_current_kv(common, *caches[layer]);
    current.host_ms += elapsed_ms(start);

    start = std::chrono::steady_clock::now();
    auto output = attention.run_blocks(
        *common.pre_query, caches[layer]->blocks, common.attention_workspace);
    if (!output.ok()) {
      return false;
    }
    current.attention_ms += elapsed_ms(start);

    start = std::chrono::steady_clock::now();
    if (!execute(
            post_module, "llama_layer_0_post", {*common.hidden, **output}, 1)) {
      return false;
    }
    current.post_ms += elapsed_ms(start);
  }
  current.total_ms = elapsed_ms(total_start);
  if (sample != nullptr) {
    *sample = current;
  }
  return true;
}

void fill_hf_rope(CommonBuffers& common, size_t position_begin) {
  auto* cosine = common.cosine->mutable_data_ptr<Half>();
  auto* sine = common.sine->mutable_data_ptr<Half>();
  constexpr double kTheta = 1000000.0;
  for (size_t row = 0; row < FLAGS_query_rows; ++row) {
    for (size_t dim = 0; dim < FLAGS_head_dim / 2; ++dim) {
      const double frequency =
          std::pow(kTheta, -2.0 * static_cast<double>(dim) / FLAGS_head_dim);
      const double angle =
          static_cast<double>(position_begin + row) * frequency;
      const Half cos_value(std::cos(angle));
      const Half sin_value(std::sin(angle));
      cosine[row * FLAGS_head_dim + dim] = cos_value;
      cosine[row * FLAGS_head_dim + dim + FLAGS_head_dim / 2] = cos_value;
      sine[row * FLAGS_head_dim + dim] = sin_value;
      sine[row * FLAGS_head_dim + dim + FLAGS_head_dim / 2] = sin_value;
    }
  }
}

bool run_prefill(
    Module& pre_module,
    Module& post_module,
    ComposableAttentionRunner& attention,
    CommonBuffers& common,
    EdgeCache& cache,
    Sample* sample) {
  cache.reset_prefill();
  Sample current;
  const auto total_start = std::chrono::steady_clock::now();
  for (size_t query_begin = 0; query_begin < FLAGS_prompt_tokens;
       query_begin += FLAGS_query_rows) {
    const size_t valid_rows =
        std::min<size_t>(FLAGS_query_rows, FLAGS_prompt_tokens - query_begin);
    auto host_start = std::chrono::steady_clock::now();
    common.reset_hidden();
    fill_hf_rope(common, query_begin);
    current.host_ms += elapsed_ms(host_start);

    auto start = std::chrono::steady_clock::now();
    if (!execute(
            pre_module,
            "llama_layer_0_pre",
            {*common.hidden, *common.cosine, *common.sine},
            3)) {
      return false;
    }
    current.pre_ms += elapsed_ms(start);

    host_start = std::chrono::steady_clock::now();
    if (!cache.append(*common.pre_key, *common.pre_value, valid_rows) ||
        !cache.prepare_causal_blocks(query_begin, valid_rows)) {
      return false;
    }
    current.host_ms += elapsed_ms(host_start);
    current.attention_calls += cache.blocks.size();

    start = std::chrono::steady_clock::now();
    auto output = attention.run_blocks(
        *common.pre_query, cache.blocks, common.attention_workspace);
    if (!output.ok()) {
      return false;
    }
    current.attention_ms += elapsed_ms(start);

    start = std::chrono::steady_clock::now();
    if (!execute(
            post_module, "llama_layer_0_post", {*common.hidden, **output}, 1)) {
      return false;
    }
    current.post_ms += elapsed_ms(start);
  }
  current.total_ms = elapsed_ms(total_start);
  if (sample != nullptr) {
    *sample = current;
  }
  return true;
}

bool run_baseline(
    Module& baseline_module,
    CommonBuffers& common,
    std::vector<std::unique_ptr<BaselineCache>>& caches,
    Sample* sample) {
  common.reset_hidden();
  Sample current;
  const auto total_start = std::chrono::steady_clock::now();
  for (size_t layer = 0; layer < caches.size(); ++layer) {
    if (layer > 0) {
      const auto host_start = std::chrono::steady_clock::now();
      std::memcpy(
          common.hidden->mutable_data_ptr(),
          common.layer_output->const_data_ptr(),
          common.hidden->nbytes());
      current.host_ms += elapsed_ms(host_start);
    }
    const auto start = std::chrono::steady_clock::now();
    if (!execute(
            baseline_module,
            "llama_layer_0_full_c40960",
            {*common.hidden,
             *common.cosine,
             *common.sine,
             *common.mask,
             *caches[layer]->key,
             *caches[layer]->value},
            3)) {
      return false;
    }
    current.full_ms += elapsed_ms(start);
  }
  current.total_ms = elapsed_ms(total_start);
  if (sample != nullptr) {
    *sample = current;
  }
  return true;
}

double quantile(std::vector<double> values, double probability) {
  std::sort(values.begin(), values.end());
  const size_t rank = static_cast<size_t>(
      std::ceil(probability * static_cast<double>(values.size())));
  return values[std::max<size_t>(1, rank) - 1];
}

std::vector<double> field(
    const std::vector<Sample>& samples,
    double Sample::* member) {
  std::vector<double> values;
  values.reserve(samples.size());
  for (const Sample& sample : samples) {
    values.push_back(sample.*member);
  }
  return values;
}

void print_array(const std::vector<double>& values) {
  std::cout << '[';
  for (size_t index = 0; index < values.size(); ++index) {
    if (index > 0) {
      std::cout << ',';
    }
    std::cout << std::fixed << std::setprecision(4) << values[index];
  }
  std::cout << ']';
}

void print_size_array(const std::vector<size_t>& values) {
  std::cout << '[';
  for (size_t index = 0; index < values.size(); ++index) {
    if (index > 0) {
      std::cout << ',';
    }
    std::cout << values[index];
  }
  std::cout << ']';
}

bool finite_tensor(const Tensor& tensor) {
  const auto* values = tensor.const_data_ptr<Half>();
  for (size_t index = 0; index < tensor.numel(); ++index) {
    if (!std::isfinite(static_cast<float>(values[index]))) {
      return false;
    }
  }
  return true;
}

bool write_output(const Tensor& tensor) {
  if (FLAGS_output_path.empty()) {
    return true;
  }
  std::ofstream output(FLAGS_output_path, std::ios::binary);
  output.write(
      static_cast<const char*>(tensor.const_data_ptr()), tensor.nbytes());
  return output.good();
}

} // namespace

int main(int argc, char** argv) {
  executorch::runtime::runtime_init();
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  const bool is_attention = FLAGS_design == "attention";
  const bool is_edge = FLAGS_design == "edge";
  const bool is_baseline = FLAGS_design == "baseline";
  const bool is_decode = FLAGS_phase == "decode";
  const bool is_prefill = FLAGS_phase == "prefill";
  const bool measured_width_portfolio = !FLAGS_block_widths.empty();
  const std::vector<size_t> portfolio_widths = measured_width_portfolio
      ? parse_positive_sizes(FLAGS_block_widths)
      : std::vector<size_t>{FLAGS_block_width};
  const std::vector<double> first_costs = measured_width_portfolio
      ? parse_positive_costs(FLAGS_first_costs_ms)
      : std::vector<double>{1.0};
  const std::vector<double> merge_costs = measured_width_portfolio
      ? parse_positive_costs(FLAGS_merge_costs_ms)
      : std::vector<double>{1.0};
  if ((!is_attention && !is_edge && !is_baseline) ||
      (!is_decode && !is_prefill) || (is_prefill && !is_edge) ||
      FLAGS_logical_layers == 0 || FLAGS_iterations == 0 ||
      FLAGS_capacity != 40960 || FLAGS_block_width == 0 ||
      (!measured_width_portfolio && FLAGS_capacity % FLAGS_block_width != 0) ||
      portfolio_widths.empty() ||
      portfolio_widths.size() != first_costs.size() ||
      portfolio_widths.size() != merge_costs.size() ||
      (measured_width_portfolio && !is_prefill) || FLAGS_dim != 1024 ||
      FLAGS_hidden_dim != 3072 || FLAGS_query_heads != 16 ||
      FLAGS_kv_heads != 8 || FLAGS_head_dim != 128 || FLAGS_query_rows == 0 ||
      (is_decode && FLAGS_query_rows != 1) ||
      (is_prefill &&
       (FLAGS_logical_layers != 1 || FLAGS_prompt_tokens == 0 ||
        FLAGS_prompt_tokens > FLAGS_capacity ||
        FLAGS_prompt_tokens % FLAGS_query_rows != 0))) {
    std::cerr << "invalid benchmark configuration" << std::endl;
    return 2;
  }

  std::vector<StaticAttentionGraphCost> graph_costs;
  graph_costs.reserve(portfolio_widths.size());
  for (size_t index = 0; index < portfolio_widths.size(); ++index) {
    graph_costs.push_back(
        {portfolio_widths[index], first_costs[index], merge_costs[index]});
  }

  std::unique_ptr<Module> attention_module;
  std::unique_ptr<Module> pre_module;
  std::unique_ptr<Module> post_module;
  std::unique_ptr<Module> baseline_module;
  if (is_attention || is_edge) {
    attention_module = std::make_unique<Module>(
        FLAGS_attention_model_path, Module::LoadMode::Mmap);
  }
  if (is_edge) {
    pre_module =
        std::make_unique<Module>(FLAGS_pre_model_path, Module::LoadMode::Mmap);
    post_module =
        std::make_unique<Module>(FLAGS_post_model_path, Module::LoadMode::Mmap);
  }
  if (is_baseline) {
    baseline_module = std::make_unique<Module>(
        FLAGS_baseline_model_path, Module::LoadMode::Mmap);
  }
  CommonBuffers common;
  if (!common.valid()) {
    std::cerr << "failed to allocate common shared buffers" << std::endl;
    return 3;
  }
  std::unique_ptr<ComposableAttentionRunner> attention;
  std::optional<StaticAttentionPlan> prefill_layout;
  if (attention_module != nullptr) {
    auto attention_result = ComposableAttentionRunner::create(
        attention_module.get(),
        FLAGS_query_rows,
        graph_costs,
        FLAGS_assume_nonempty);
    if (!attention_result.ok()) {
      std::cerr << "failed to create composable Attention runner" << std::endl;
      return 3;
    }
    attention = std::make_unique<ComposableAttentionRunner>(
        std::move(*attention_result));
    if (is_prefill && measured_width_portfolio) {
      std::vector<size_t> visible_prefixes;
      visible_prefixes.reserve(
          (FLAGS_prompt_tokens + FLAGS_query_rows - 1) / FLAGS_query_rows);
      for (size_t end = FLAGS_query_rows; end < FLAGS_prompt_tokens;
           end += FLAGS_query_rows) {
        visible_prefixes.push_back(end);
      }
      visible_prefixes.push_back(FLAGS_prompt_tokens);
      auto layout = attention->plan_prefixes(visible_prefixes);
      if (!layout.ok()) {
        std::cerr << "failed to plan measured Prefill K/V layout" << std::endl;
        return 3;
      }
      prefill_layout.emplace(std::move(*layout));
    }
  }

  const std::vector<size_t> cache_widths = prefill_layout.has_value()
      ? prefill_layout->widths
      : uniform_widths(FLAGS_capacity, FLAGS_block_width);
  if ((is_attention || is_edge) &&
      (cache_widths.empty() ||
       width_coverage(cache_widths) <
           (is_prefill ? FLAGS_prompt_tokens : FLAGS_capacity))) {
    std::cerr << "invalid EdgeInfer KV layout" << std::endl;
    return 3;
  }

  std::vector<std::unique_ptr<EdgeCache>> edge_caches;
  std::vector<std::unique_ptr<BaselineCache>> baseline_caches;
  const size_t cache_count = is_attention ? 1 : FLAGS_logical_layers;
  if (is_attention || is_edge) {
    edge_caches.reserve(cache_count);
    for (size_t layer = 0; layer < cache_count; ++layer) {
      auto cache = std::make_unique<EdgeCache>(layer, is_prefill, cache_widths);
      if (cache->keys.size() != cache_widths.size()) {
        std::cerr << "failed to allocate EdgeInfer KV for layer " << layer
                  << std::endl;
        return 3;
      }
      cache->blocks.reserve(cache->keys.size());
      if (is_decode) {
        for (size_t block = 0; block < cache->keys.size(); ++block) {
          cache->blocks.push_back(
              {cache->keys[block].get(),
               cache->values[block].get(),
               common.visibility.get()});
        }
      }
      edge_caches.emplace_back(std::move(cache));
    }
  }
  if (is_baseline) {
    baseline_caches.reserve(cache_count);
    for (size_t layer = 0; layer < cache_count; ++layer) {
      auto cache = std::make_unique<BaselineCache>(layer);
      if (cache->key == nullptr || cache->value == nullptr) {
        std::cerr << "failed to allocate baseline KV for layer " << layer
                  << std::endl;
        return 3;
      }
      baseline_caches.emplace_back(std::move(cache));
    }
  }

  bool kv_independent = true;
  if (is_edge) {
    for (size_t left = 0; left < edge_caches.size(); ++left) {
      for (size_t right = left + 1; right < edge_caches.size(); ++right) {
        kv_independent &= edge_caches[left]->keys[0]->mutable_data_ptr() !=
            edge_caches[right]->keys[0]->mutable_data_ptr();
      }
    }
  } else if (is_baseline) {
    for (size_t left = 0; left < baseline_caches.size(); ++left) {
      for (size_t right = left + 1; right < baseline_caches.size(); ++right) {
        kv_independent &= baseline_caches[left]->key->mutable_data_ptr() !=
            baseline_caches[right]->key->mutable_data_ptr();
      }
    }
  }

  const std::string pre = "llama_layer_0_pre";
  const std::string post = "llama_layer_0_post";
  const std::string full = "llama_layer_0_full_c40960";
  if (is_edge &&
      (bind_outputs(
           *pre_module,
           pre,
           {common.pre_query, common.pre_key, common.pre_value}) != Error::Ok ||
       bind_outputs(*post_module, post, {common.layer_output}) != Error::Ok)) {
    std::cerr << "failed to bind pre/post outputs" << std::endl;
    return 4;
  }
  if (is_baseline &&
      bind_outputs(
          *baseline_module,
          full,
          {common.layer_output, common.full_new_key, common.full_new_value}) !=
          Error::Ok) {
    std::cerr << "failed to bind complete-layer outputs" << std::endl;
    return 4;
  }

  auto invoke = [&](Sample* sample) {
    if (is_prefill) {
      return run_prefill(
          *pre_module,
          *post_module,
          *attention,
          common,
          *edge_caches.front(),
          sample);
    }
    if (is_attention) {
      return run_attention(*attention, common, *edge_caches.front(), sample);
    }
    if (is_edge) {
      return run_edge(
          *pre_module, *post_module, *attention, common, edge_caches, sample);
    }
    return run_baseline(*baseline_module, common, baseline_caches, sample);
  };
  for (size_t index = 0; index < FLAGS_warmup; ++index) {
    if (!invoke(nullptr)) {
      std::cerr << "warm-up failed" << std::endl;
      return 5;
    }
  }
  std::vector<Sample> samples(FLAGS_iterations);
  for (Sample& sample : samples) {
    if (!invoke(&sample)) {
      std::cerr << "measured invocation failed" << std::endl;
      return 5;
    }
  }

  const Tensor& final_output = is_attention
      ? *common.attention_workspace.banks[0][0]
      : *common.layer_output;
  const bool finite = finite_tensor(final_output);
  const bool finite_required = FLAGS_fail_on_nonfinite;
  const bool output_written = write_output(final_output);
  const auto totals = field(samples, &Sample::total_ms);
  const auto pre_values = field(samples, &Sample::pre_ms);
  const auto attention_values = field(samples, &Sample::attention_ms);
  const auto post_values = field(samples, &Sample::post_ms);
  const auto host_values = field(samples, &Sample::host_ms);
  const auto full_values = field(samples, &Sample::full_ms);
  const size_t blocks = cache_widths.size();
  const size_t tiles = is_prefill ? FLAGS_prompt_tokens / FLAGS_query_rows : 1;
  const size_t attention_calls = is_prefill
      ? samples.front().attention_calls
      : (is_attention ? 1 : (is_edge ? FLAGS_logical_layers : 0)) * blocks;
  const size_t graph_calls = is_prefill ? attention_calls + 2 * tiles
      : is_attention
      ? attention_calls
      : (is_edge ? FLAGS_logical_layers * (blocks + 2) : FLAGS_logical_layers);
  const size_t rectangular_attention_calls = is_prefill ? tiles * blocks : 0;

  std::cout << "EDGEINFER_LAYER_JSON {\"design\":\"" << FLAGS_design
            << "\",\"phase\":\"" << FLAGS_phase
            << "\",\"device\":\"SM8650\",\"precision\":\"FP16\""
            << ",\"context\":" << FLAGS_capacity << ",\"logical_layers\":"
            << (is_attention ? 1 : FLAGS_logical_layers)
            << ",\"weights\":\"one_exported_layer_reused\""
            << ",\"independent_kv\":" << (kv_independent ? "true" : "false")
            << ",\"synthetic_history_timed\":false" << ",\"assume_nonempty\":"
            << (FLAGS_assume_nonempty ? "true" : "false")
            << ",\"warmup\":" << FLAGS_warmup
            << ",\"iterations\":" << FLAGS_iterations
            << ",\"query_rows\":" << FLAGS_query_rows
            << ",\"prompt_tokens\":" << (is_prefill ? FLAGS_prompt_tokens : 0)
            << ",\"query_tiles\":" << tiles << ",\"static_graph_count\":"
            << (is_attention ? 2 * portfolio_widths.size()
                             : (is_edge ? 2 * portfolio_widths.size() + 2 : 1))
            << ",\"graph_calls_per_invocation\":" << graph_calls
            << ",\"attention_graph_calls\":" << attention_calls
            << ",\"rectangular_attention_calls\":"
            << rectangular_attention_calls << ",\"upper_right_skipped\":"
            << (rectangular_attention_calls - attention_calls)
            << ",\"measured_width_portfolio\":"
            << (measured_width_portfolio ? "true" : "false")
            << ",\"predicted_attention_ms\":"
            << (prefill_layout.has_value() ? prefill_layout->predicted_cost
                                           : 0.0)
            << ",\"selected_kv_layout\":";
  print_size_array(cache_widths);
  std::cout << ",\"total_samples_ms\":";
  print_array(totals);
  std::cout << ",\"p50_ms\":" << std::fixed << std::setprecision(4)
            << quantile(totals, 0.50)
            << ",\"p95_ms\":" << quantile(totals, 0.95)
            << ",\"pre_p50_ms\":" << quantile(pre_values, 0.50)
            << ",\"attention_p50_ms\":" << quantile(attention_values, 0.50)
            << ",\"post_p50_ms\":" << quantile(post_values, 0.50)
            << ",\"host_p50_ms\":" << quantile(host_values, 0.50)
            << ",\"full_graph_p50_ms\":" << quantile(full_values, 0.50)
            << ",\"finite\":" << (finite ? "true" : "false")
            << ",\"finite_required\":" << (finite_required ? "true" : "false")
            << ",\"output_written\":" << (output_written ? "true" : "false")
            << "}" << std::endl;

  if (attention != nullptr) {
    attention->release_output_bindings();
  }
  if (pre_module != nullptr) {
    pre_module->unload_method(pre);
  }
  if (post_module != nullptr) {
    post_module->unload_method(post);
  }
  if (baseline_module != nullptr) {
    baseline_module->unload_method(full);
  }
  return (!finite_required || finite) && output_written && kv_independent ? 0
                                                                          : 6;
}
