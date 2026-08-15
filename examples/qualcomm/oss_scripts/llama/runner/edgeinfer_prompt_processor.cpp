/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/edgeinfer_prompt_processor.h>

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/edgeinfer_fallback.h>

#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/extension/llm/runner/causal_attention_block.h>
#include <executorch/extension/llm/runner/edgeinfer_decode_cache_transaction.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/platform/log.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace example {
namespace {

using executorch::aten::Half;
using executorch::aten::ScalarType;
using executorch::aten::SizesType;
using executorch::aten::Tensor;
using executorch::extension::TensorPtr;
using executorch::extension::llm::causal_block_visible_columns;
using executorch::extension::llm::causal_prefill_plan_stats;
using executorch::extension::llm::causal_visible_prefix_end;
using executorch::extension::llm::ComposableAttentionWorkspace;
using executorch::extension::llm::ComposableAttentionRunner;
using executorch::extension::llm::PreparedAttentionBlock;
using executorch::extension::llm::rollback_edgeinfer_cache_blocks;
using executorch::extension::llm::run_edgeinfer_decode_cache_transaction;
using executorch::extension::llm::StaticAttentionGraphCost;
using executorch::extension::llm::StaticAttentionPlan;
using executorch::extension::llm::StaticAttentionPrefillPlan;
using executorch::extension::llm::StaticAttentionShapeCost;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::Result;

bool is_power_of_two(size_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

std::string row_method_suffix(size_t rows, size_t primary_rows) {
  return rows == primary_rows ? std::string{} : "_r" + std::to_string(rows);
}

std::string stage_method_name(
    const char* kind,
    size_t rows,
    size_t primary_rows,
    std::optional<size_t> layer = std::nullopt) {
  if (layer.has_value()) {
    return "llama_layer_" + std::to_string(*layer) + "_" + kind +
        row_method_suffix(rows, primary_rows);
  }
  return std::string("llama_") + kind + row_method_suffix(rows, primary_rows);
}

std::optional<size_t> parse_positive_suffix(
    const std::string& value,
    const std::string& prefix) {
  if (value.compare(0, prefix.size(), prefix) != 0 ||
      value.size() == prefix.size()) {
    return std::nullopt;
  }
  const char* begin = value.c_str() + prefix.size();
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(begin, &end, 10);
  if (end == begin || *end != '\0' || parsed == 0 ||
      parsed > std::numeric_limits<size_t>::max()) {
    return std::nullopt;
  }
  return static_cast<size_t>(parsed);
}

std::vector<size_t> available_attention_rows(
    executorch::extension::Module* module) {
  std::vector<size_t> rows;
  if (module == nullptr) {
    return rows;
  }
  auto names = module->method_names();
  if (!names.ok()) {
    return rows;
  }
  constexpr const char* kPrefix = "attn_first_r";
  for (const std::string& name : *names) {
    const size_t width_marker = name.find("_c", std::strlen(kPrefix));
    if (name.compare(0, std::strlen(kPrefix), kPrefix) != 0 ||
        width_marker == std::string::npos) {
      continue;
    }
    const auto row = parse_positive_suffix(
        name.substr(0, width_marker), std::string(kPrefix));
    if (row.has_value() && is_power_of_two(*row)) {
      rows.push_back(*row);
    }
  }
  std::sort(rows.begin(), rows.end());
  rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
  return rows;
}

std::vector<size_t> available_attention_widths(
    executorch::extension::Module* module,
    size_t rows) {
  std::vector<size_t> widths;
  if (module == nullptr || rows == 0) {
    return widths;
  }
  auto names = module->method_names();
  if (!names.ok()) {
    return widths;
  }
  const std::string first_prefix =
      "attn_first_r" + std::to_string(rows) + "_c";
  const std::string merge_prefix =
      "attn_merge_r" + std::to_string(rows) + "_c";
  for (const std::string& name : *names) {
    const auto width = parse_positive_suffix(name, first_prefix);
    if (width.has_value() && is_power_of_two(*width) &&
        names->count(merge_prefix + std::to_string(*width)) != 0) {
      widths.push_back(*width);
    }
  }
  std::sort(widths.begin(), widths.end());
  widths.erase(std::unique(widths.begin(), widths.end()), widths.end());
  return widths;
}

bool checked_add(size_t lhs, size_t rhs, size_t* result) {
  if (result == nullptr || lhs > std::numeric_limits<size_t>::max() - rhs) {
    return false;
  }
  *result = lhs + rhs;
  return true;
}

bool checked_mul(size_t lhs, size_t rhs, size_t* result) {
  if (result == nullptr ||
      (rhs != 0 && lhs > std::numeric_limits<size_t>::max() / rhs)) {
    return false;
  }
  *result = lhs * rhs;
  return true;
}

class QnnSharedArena final {
 public:
  explicit QnnSharedArena(size_t bytes)
      : bytes_(bytes),
        base_(QnnExecuTorchAllocCustomMem(
            bytes,
            executorch::runtime::MemoryAllocator::kDefaultAlignment)) {}

  ~QnnSharedArena() {
    if (base_ != nullptr) {
      QnnExecuTorchFreeCustomMem(base_);
    }
  }

  QnnSharedArena(const QnnSharedArena&) = delete;
  QnnSharedArena& operator=(const QnnSharedArena&) = delete;

  bool valid() const {
    return base_ != nullptr;
  }

  TensorPtr make_tensor(std::vector<SizesType> sizes, ScalarType scalar_type) {
    if (scalar_type != ScalarType::Float && scalar_type != ScalarType::Half) {
      return nullptr;
    }
    size_t elements = 1;
    for (const SizesType size : sizes) {
      if (size <= 0 ||
          !checked_mul(elements, static_cast<size_t>(size), &elements)) {
        return nullptr;
      }
    }
    size_t tensor_bytes = 0;
    if (!checked_mul(
            elements,
            executorch::aten::elementSize(scalar_type),
            &tensor_bytes)) {
      return nullptr;
    }
    const size_t alignment =
        executorch::runtime::MemoryAllocator::kDefaultAlignment;
    const size_t padding =
        offset_ % alignment == 0 ? 0 : alignment - offset_ % alignment;
    size_t aligned_offset = 0;
    size_t end_offset = 0;
    if (!checked_add(offset_, padding, &aligned_offset) ||
        !checked_add(aligned_offset, tensor_bytes, &end_offset) ||
        end_offset > bytes_) {
      return nullptr;
    }
    auto* data = static_cast<std::byte*>(base_) + aligned_offset;
    offset_ = end_offset;
    std::memset(data, 0, tensor_bytes);
    QnnExecuTorchAddCustomMemTensorAddr(data, base_);
    return executorch::extension::make_tensor_ptr(
        std::move(sizes),
        data,
        scalar_type,
        executorch::aten::TensorShapeDynamism::STATIC,
        [](void*) {});
  }

 private:
  size_t bytes_;
  size_t offset_ = 0;
  void* base_;
};

size_t aligned_storage_bytes(
    size_t elements,
    size_t tensor_count,
    ScalarType scalar_type) {
  if (scalar_type != ScalarType::Float && scalar_type != ScalarType::Half) {
    return 0;
  }
  size_t data_bytes = 0;
  size_t alignment_bytes = 0;
  size_t result = 0;
  if (!checked_mul(
          elements, executorch::aten::elementSize(scalar_type), &data_bytes) ||
      !checked_mul(
          tensor_count,
          executorch::runtime::MemoryAllocator::kDefaultAlignment,
          &alignment_bytes) ||
      !checked_add(data_bytes, alignment_bytes, &result)) {
    return 0;
  }
  return result;
}

TensorPtr make_zero_tensor(
    std::vector<SizesType> sizes,
    ScalarType scalar_type) {
  if (scalar_type != ScalarType::Float && scalar_type != ScalarType::Half) {
    return nullptr;
  }
  size_t elements = 1;
  for (const SizesType size : sizes) {
    if (size <= 0 ||
        !checked_mul(elements, static_cast<size_t>(size), &elements)) {
      return nullptr;
    }
  }
  size_t bytes = 0;
  if (!checked_mul(
          elements, executorch::aten::elementSize(scalar_type), &bytes)) {
    return nullptr;
  }
  return executorch::extension::make_tensor_ptr(
      std::move(sizes),
      std::vector<uint8_t>(bytes, uint8_t{0}),
      scalar_type,
      executorch::aten::TensorShapeDynamism::STATIC);
}

TensorPtr make_int64_tensor(
    std::vector<SizesType> sizes,
    std::vector<int64_t> values) {
  return executorch::extension::make_tensor_ptr(
      std::move(sizes),
      std::move(values),
      {},
      {},
      ScalarType::Long,
      executorch::aten::TensorShapeDynamism::STATIC);
}

bool has_shape(
    const Tensor& tensor,
    const std::vector<SizesType>& sizes,
    ScalarType scalar_type) {
  return tensor.scalar_type() == scalar_type &&
      executorch::runtime::tensor_is_contiguous(tensor) &&
      tensor.dim() == sizes.size() &&
      std::equal(sizes.begin(), sizes.end(), tensor.sizes().begin());
}

bool fill_tensor(Tensor& tensor, size_t elements, float value) {
  if (static_cast<size_t>(tensor.numel()) < elements) {
    return false;
  }
  if (tensor.scalar_type() == ScalarType::Float) {
    std::fill_n(tensor.mutable_data_ptr<float>(), elements, value);
    return true;
  }
  if (tensor.scalar_type() == ScalarType::Half) {
    std::fill_n(tensor.mutable_data_ptr<Half>(), elements, Half(value));
    return true;
  }
  return false;
}

bool fill_tensor_range(
    Tensor& tensor,
    size_t offset,
    size_t elements,
    float value) {
  if (offset > static_cast<size_t>(tensor.numel()) ||
      elements > static_cast<size_t>(tensor.numel()) - offset) {
    return false;
  }
  if (tensor.scalar_type() == ScalarType::Float) {
    std::fill_n(tensor.mutable_data_ptr<float>() + offset, elements, value);
    return true;
  }
  if (tensor.scalar_type() == ScalarType::Half) {
    std::fill_n(
        tensor.mutable_data_ptr<Half>() + offset, elements, Half(value));
    return true;
  }
  return false;
}

bool has_method_tensor_contract(
    executorch::extension::Module* module,
    const std::string& method,
    const std::vector<ScalarType>& input_types,
    const std::vector<ScalarType>& output_types) {
  if (module == nullptr) {
    return false;
  }
  auto method_meta = module->method_meta(method);
  if (!method_meta.ok() || method_meta->num_inputs() != input_types.size() ||
      method_meta->num_outputs() != output_types.size()) {
    return false;
  }
  for (size_t index = 0; index < input_types.size(); ++index) {
    auto tensor_meta = method_meta->input_tensor_meta(index);
    if (!tensor_meta.ok() || tensor_meta->scalar_type() != input_types[index]) {
      return false;
    }
  }
  for (size_t index = 0; index < output_types.size(); ++index) {
    auto tensor_meta = method_meta->output_tensor_meta(index);
    if (!tensor_meta.ok() ||
        tensor_meta->scalar_type() != output_types[index]) {
      return false;
    }
  }
  return true;
}

template <typename T>
void fill_rope_values(
    T* cosine_data,
    T* sine_data,
    size_t query_rows,
    size_t causal_query_begin,
    size_t pairs,
    size_t rope_width,
    bool hf_rope,
    const std::vector<double>& inverse_frequencies) {
  for (size_t row = 0; row < query_rows; ++row) {
    const size_t position = causal_query_begin + row;
    for (size_t pair = 0; pair < pairs; ++pair) {
      const double angle =
          static_cast<double>(position) * inverse_frequencies[pair];
      const float cosine = static_cast<float>(std::cos(angle));
      const float sine = static_cast<float>(std::sin(angle));
      cosine_data[row * rope_width + pair] = T(cosine);
      sine_data[row * rope_width + pair] = T(sine);
      if (hf_rope) {
        cosine_data[row * rope_width + pairs + pair] = T(cosine);
        sine_data[row * rope_width + pairs + pair] = T(sine);
      }
    }
  }
}

template <typename T>
void unpack_and_rotate_qkv(
    const Tensor& packed,
    Tensor& query,
    Tensor& key,
    Tensor& value,
    const Tensor& rope_cosine,
    const Tensor& rope_sine,
    size_t query_rows,
    size_t valid_rows,
    size_t query_heads,
    size_t kv_heads,
    size_t head_dim,
    bool hf_rope) {
  const size_t query_features = query_heads * head_dim;
  const size_t kv_features = kv_heads * head_dim;
  const size_t packed_features = query_features + 2 * kv_features;
  const size_t rope_width = hf_rope ? head_dim : head_dim / 2;
  const size_t pairs = head_dim / 2;
  const T* packed_data = packed.const_data_ptr<T>();
  T* query_data = query.mutable_data_ptr<T>();
  T* key_data = key.mutable_data_ptr<T>();
  T* value_data = value.mutable_data_ptr<T>();
  const T* rope_cosine_data = rope_cosine.const_data_ptr<T>();
  const T* rope_sine_data = rope_sine.const_data_ptr<T>();

  for (size_t row = 0; row < query_rows; ++row) {
    const T* source = packed_data + row * packed_features;
    for (size_t head = 0; head < query_heads; ++head) {
      std::memcpy(
          query_data + (head * query_rows + row) * head_dim,
          source + head * head_dim,
          head_dim * sizeof(T));
    }
    for (size_t head = 0; head < kv_heads; ++head) {
      for (size_t dim = 0; dim < head_dim; ++dim) {
        key_data[(head * head_dim + dim) * query_rows + row] =
            source[query_features + head * head_dim + dim];
      }
      std::memcpy(
          value_data + (head * query_rows + row) * head_dim,
          source + query_features + kv_features + head * head_dim,
          head_dim * sizeof(T));
    }
  }

  for (size_t row = 0; row < valid_rows; ++row) {
    for (size_t pair = 0; pair < pairs; ++pair) {
      const float cosine =
          static_cast<float>(rope_cosine_data[row * rope_width + pair]);
      const float sine =
          static_cast<float>(rope_sine_data[row * rope_width + pair]);
      auto rotate = [&](T* data, size_t heads, bool key_layout) {
        for (size_t head = 0; head < heads; ++head) {
          const size_t base = key_layout ? head * head_dim * query_rows + row
                                         : (head * query_rows + row) * head_dim;
          const size_t stride = key_layout ? query_rows : 1;
          const size_t first = hf_rope ? pair : pair * 2;
          const size_t second = hf_rope ? pairs + pair : pair * 2 + 1;
          const float real = static_cast<float>(data[base + first * stride]);
          const float imaginary =
              static_cast<float>(data[base + second * stride]);
          data[base + first * stride] = T(real * cosine - imaginary * sine);
          data[base + second * stride] = T(real * sine + imaginary * cosine);
        }
      };
      rotate(query_data, query_heads, false);
      rotate(key_data, kv_heads, true);
    }
  }
}

bool has_same_tensor_contract(const Tensor& actual, const Tensor& expected) {
  return actual.scalar_type() == expected.scalar_type() &&
      executorch::runtime::tensor_is_contiguous(actual) &&
      executorch::runtime::tensor_is_contiguous(expected) &&
      actual.dim() == expected.dim() && actual.nbytes() == expected.nbytes() &&
      std::equal(
             actual.sizes().begin(),
             actual.sizes().end(),
             expected.sizes().begin()) &&
      std::equal(
             actual.dim_order().begin(),
             actual.dim_order().end(),
             expected.dim_order().begin()) &&
      std::equal(
             actual.strides().begin(),
             actual.strides().end(),
             expected.strides().begin());
}

bool has_memory_planned_output(
    executorch::extension::Module* module,
    const std::string& method) {
  if (module == nullptr) {
    return false;
  }
  auto method_meta = module->method_meta(method);
  if (!method_meta.ok()) {
    return false;
  }
  for (size_t index = 0; index < method_meta->num_outputs(); ++index) {
    auto output_meta = method_meta->output_tensor_meta(index);
    if (output_meta.ok() && output_meta->is_memory_planned()) {
      return true;
    }
  }
  return false;
}

void unload_module_methods(executorch::extension::Module* module) {
  if (module == nullptr) {
    return;
  }
  auto method_names = module->method_names();
  if (!method_names.ok()) {
    return;
  }
  for (const std::string& method : *method_names) {
    module->unload_method(method);
  }
}

} // namespace

struct EdgeInferPromptProcessor::Block {
  size_t width;
  size_t valid_width;
  TensorPtr key;
  TensorPtr value;
  TensorPtr visibility;
  TensorPtr decode_visibility;
  edgeinfer::detail::VisibilityMaskCache visibility_cache;
  edgeinfer::detail::VisibilityMaskCache decode_visibility_cache;
};

struct EdgeInferPromptProcessor::PreparedBlocks {
  std::vector<PreparedAttentionBlock> blocks;
};

struct EdgeInferPromptProcessor::StageWorkspace {
  struct Mode {
    size_t rows = 0;
    std::array<TensorPtr, 2> hidden;
    TensorPtr packed_qkv;
    TensorPtr query;
    TensorPtr key;
    TensorPtr value;
    TensorPtr rope_cosine;
    TensorPtr rope_sine;
    TensorPtr logits;
  };

  StageWorkspace(
      size_t query_heads,
      size_t kv_heads,
      size_t prefill_rows,
      size_t decode_rows,
      size_t head_dim,
      size_t dim,
      size_t vocab_size,
      bool hf_rope,
      ScalarType scalar_type)
      : scalar_type(scalar_type) {
    const size_t max_size =
        static_cast<size_t>(std::numeric_limits<SizesType>::max());
    if (query_heads == 0 || kv_heads == 0 || prefill_rows == 0 ||
        decode_rows == 0 || head_dim == 0 || dim == 0 || vocab_size == 0 ||
        query_heads > max_size || kv_heads > max_size ||
        prefill_rows > max_size || decode_rows > max_size ||
        head_dim > max_size || dim > max_size || vocab_size > max_size) {
      return;
    }
    const size_t rope_width = hf_rope ? head_dim : head_dim / 2;
    size_t query_features = 0;
    size_t kv_features = 0;
    size_t two_kv_features = 0;
    size_t packed_features = 0;
    if (rope_width == 0 ||
        !checked_mul(query_heads, head_dim, &query_features) ||
        !checked_mul(kv_heads, head_dim, &kv_features) ||
        !checked_mul(kv_features, size_t{2}, &two_kv_features) ||
        !checked_add(query_features, two_kv_features, &packed_features) ||
        packed_features > max_size) {
      return;
    }

    size_t elements = 0;
    auto add_mode_elements = [&](size_t rows) {
      size_t hidden = 0;
      size_t query = 0;
      size_t kv = 0;
      size_t packed = 0;
      size_t rope = 0;
      size_t logits = 0;
      size_t mode_elements = 0;
      return checked_mul(rows, dim, &hidden) &&
          checked_mul(hidden, size_t{2}, &hidden) &&
          checked_mul(query_heads, rows, &query) &&
          checked_mul(query, head_dim, &query) &&
          checked_mul(kv_heads, rows, &kv) && checked_mul(kv, head_dim, &kv) &&
          checked_mul(kv, size_t{2}, &kv) &&
          checked_mul(rows, packed_features, &packed) &&
          checked_mul(rows, rope_width, &rope) &&
          checked_mul(rope, size_t{2}, &rope) &&
          checked_mul(rows, vocab_size, &logits) &&
          checked_add(hidden, query, &mode_elements) &&
          checked_add(mode_elements, kv, &mode_elements) &&
          checked_add(mode_elements, packed, &mode_elements) &&
          checked_add(mode_elements, rope, &mode_elements) &&
          checked_add(mode_elements, logits, &mode_elements) &&
          checked_add(elements, mode_elements, &elements);
    };
    if (!add_mode_elements(prefill_rows) || !add_mode_elements(decode_rows)) {
      return;
    }
    if (scalar_type != ScalarType::Float && scalar_type != ScalarType::Half) {
      return;
    }
    const size_t storage_bytes =
        aligned_storage_bytes(elements, 18, scalar_type);
    if (storage_bytes == 0) {
      return;
    }
    arena = std::make_unique<QnnSharedArena>(storage_bytes);
    if (!arena->valid()) {
      arena.reset();
      return;
    }

    auto allocate_mode = [&](Mode& mode, size_t rows) {
      mode.rows = rows;
      const auto row_size = static_cast<SizesType>(rows);
      const auto dim_size = static_cast<SizesType>(dim);
      const auto query_heads_size = static_cast<SizesType>(query_heads);
      const auto kv_heads_size = static_cast<SizesType>(kv_heads);
      const auto head_dim_size = static_cast<SizesType>(head_dim);
      const auto rope_width_size = static_cast<SizesType>(rope_width);
      const auto vocab_size_value = static_cast<SizesType>(vocab_size);
      const auto packed_features_size = static_cast<SizesType>(packed_features);
      for (TensorPtr& hidden : mode.hidden) {
        hidden = arena->make_tensor({1, row_size, dim_size}, scalar_type);
      }
      mode.packed_qkv =
          arena->make_tensor({1, row_size, packed_features_size}, scalar_type);
      mode.query = arena->make_tensor(
          {1, query_heads_size, row_size, head_dim_size}, scalar_type);
      mode.key = arena->make_tensor(
          {1, kv_heads_size, head_dim_size, row_size}, scalar_type);
      mode.value = arena->make_tensor(
          {1, kv_heads_size, row_size, head_dim_size}, scalar_type);
      mode.rope_cosine =
          arena->make_tensor({row_size, rope_width_size}, scalar_type);
      mode.rope_sine =
          arena->make_tensor({row_size, rope_width_size}, scalar_type);
      mode.logits =
          arena->make_tensor({1, row_size, vocab_size_value}, scalar_type);
      return mode.hidden[0] != nullptr && mode.hidden[1] != nullptr &&
          mode.packed_qkv != nullptr && mode.query != nullptr &&
          mode.key != nullptr && mode.value != nullptr &&
          mode.rope_cosine != nullptr && mode.rope_sine != nullptr &&
          mode.logits != nullptr;
    };
    if (!allocate_mode(prefill, prefill_rows) ||
        !allocate_mode(decode, decode_rows)) {
      arena.reset();
    }
  }

  bool valid() const {
    return arena != nullptr;
  }

  Mode& for_decode(bool is_decode) {
    return is_decode ? decode : prefill;
  }

  std::unique_ptr<QnnSharedArena> arena;
  ScalarType scalar_type;
  Mode prefill;
  Mode decode;
};

struct EdgeInferPromptProcessor::PrefillScratchPool {
  struct Scratch {
    TensorPtr key;
    TensorPtr value;
  };

  PrefillScratchPool(
      size_t kv_heads,
      size_t head_dim,
      const std::vector<size_t>& widths,
      ScalarType scalar_type) {
    size_t elements = 0;
    for (const size_t width : widths) {
      size_t block_elements = 0;
      if (width == 0 ||
          !checked_mul(kv_heads, head_dim, &block_elements) ||
          !checked_mul(block_elements, width, &block_elements) ||
          !checked_mul(block_elements, size_t{2}, &block_elements) ||
          !checked_add(elements, block_elements, &elements)) {
        return;
      }
    }
    const size_t storage_bytes =
        aligned_storage_bytes(elements, widths.size() * 2, scalar_type);
    if (storage_bytes == 0) {
      return;
    }
    arena = std::make_unique<QnnSharedArena>(storage_bytes);
    if (!arena->valid()) {
      arena.reset();
      return;
    }
    for (const size_t width : widths) {
      Scratch scratch{
          arena->make_tensor(
              {1,
               static_cast<SizesType>(kv_heads),
               static_cast<SizesType>(head_dim),
               static_cast<SizesType>(width)},
              scalar_type),
          arena->make_tensor(
              {1,
               static_cast<SizesType>(kv_heads),
               static_cast<SizesType>(width),
               static_cast<SizesType>(head_dim)},
              scalar_type)};
      if (scratch.key == nullptr || scratch.value == nullptr) {
        blocks.clear();
        arena.reset();
        return;
      }
      blocks.emplace(width, std::move(scratch));
    }
  }

  bool valid() const {
    return arena != nullptr && !blocks.empty();
  }

  std::unique_ptr<QnnSharedArena> arena;
  std::unordered_map<size_t, Scratch> blocks;
};

struct EdgeInferPromptProcessor::PrefillRowRuntime {
  PrefillRowRuntime(
      size_t query_heads,
      size_t kv_heads,
      size_t rows,
      size_t head_dim,
      size_t dim,
      size_t vocab_size,
      bool hf_rope,
      ScalarType scalar_type,
      const std::vector<size_t>& widths,
      std::optional<ComposableAttentionRunner> owned_runner)
      : rows(rows), owned_runner(std::move(owned_runner)) {
    stage = std::make_unique<StageWorkspace>(
        query_heads,
        kv_heads,
        rows,
        1,
        head_dim,
        dim,
        vocab_size,
        hf_rope,
        scalar_type);
    if (!stage->valid()) {
      stage.reset();
      return;
    }

    size_t state_elements = 0;
    size_t mask_elements = 0;
    size_t query_state = 0;
    if (!checked_mul(query_heads, rows, &query_state) ||
        !checked_mul(query_state, head_dim, &state_elements) ||
        !checked_mul(state_elements, size_t{4}, &state_elements) ||
        !checked_mul(query_state, size_t{4}, &query_state) ||
        !checked_add(state_elements, query_state, &state_elements)) {
      stage.reset();
      return;
    }
    for (const size_t width : widths) {
      size_t current = 0;
      if (!checked_mul(rows, width, &current) ||
          !checked_add(mask_elements, current, &mask_elements)) {
        stage.reset();
        return;
      }
    }
    size_t elements = 0;
    if (!checked_add(state_elements, mask_elements, &elements)) {
      stage.reset();
      return;
    }
    const size_t storage_bytes =
        aligned_storage_bytes(elements, widths.size() + 8, scalar_type);
    if (storage_bytes == 0) {
      stage.reset();
      return;
    }
    attention_arena = std::make_unique<QnnSharedArena>(storage_bytes);
    if (!attention_arena->valid()) {
      attention_arena.reset();
      stage.reset();
      return;
    }
    for (auto& bank : workspace.banks) {
      bank[0] = attention_arena->make_tensor(
          {1,
           static_cast<SizesType>(query_heads),
           static_cast<SizesType>(rows),
           static_cast<SizesType>(head_dim)},
          scalar_type);
      bank[1] = attention_arena->make_tensor(
          {1,
           static_cast<SizesType>(query_heads),
           static_cast<SizesType>(rows),
           1},
          scalar_type);
      bank[2] = attention_arena->make_tensor(
          {1,
           static_cast<SizesType>(query_heads),
           static_cast<SizesType>(rows),
           1},
          scalar_type);
      bank[3] = attention_arena->make_tensor(
          {1,
           static_cast<SizesType>(query_heads),
           static_cast<SizesType>(rows),
           static_cast<SizesType>(head_dim)},
          scalar_type);
      if (std::any_of(bank.begin(), bank.end(), [](const TensorPtr& tensor) {
            return tensor == nullptr;
          })) {
        attention_arena.reset();
        stage.reset();
        return;
      }
    }
    for (const size_t width : widths) {
      auto mask = attention_arena->make_tensor(
          {1,
           1,
           static_cast<SizesType>(rows),
           static_cast<SizesType>(width)},
          scalar_type);
      if (mask == nullptr ||
          !visibility_caches[width].reset_known_zero(rows, width)) {
        masks.clear();
        visibility_caches.clear();
        attention_arena.reset();
        stage.reset();
        return;
      }
      masks.emplace(width, std::move(mask));
    }
  }

  bool valid() const {
    return rows != 0 && stage != nullptr && stage->valid() &&
        attention_arena != nullptr && !masks.empty();
  }

  size_t rows;
  std::optional<ComposableAttentionRunner> owned_runner;
  std::unique_ptr<StageWorkspace> stage;
  std::unique_ptr<QnnSharedArena> attention_arena;
  ComposableAttentionWorkspace workspace;
  std::unordered_map<size_t, TensorPtr> masks;
  std::unordered_map<size_t, edgeinfer::detail::VisibilityMaskCache>
      visibility_caches;
};

struct EdgeInferPromptProcessor::LayerCache {
  struct PreparedLayout {
    PreparedLayout() = default;
    PreparedLayout(const PreparedLayout&) = delete;
    PreparedLayout& operator=(const PreparedLayout&) = delete;
    PreparedLayout(PreparedLayout&&) noexcept = default;
    PreparedLayout& operator=(PreparedLayout&&) noexcept = default;

    bool needs_commit() const noexcept {
      return arena != nullptr;
    }

    std::unique_ptr<QnnSharedArena> arena;
    std::vector<Block> blocks;
    ComposableAttentionWorkspace workspace;
    ComposableAttentionWorkspace decode_workspace;
  };

  LayerCache(
      size_t query_heads,
      size_t kv_heads,
      size_t query_rows,
      size_t decode_query_rows,
      size_t head_dim,
      ScalarType scalar_type)
      : query_heads(query_heads),
        kv_heads(kv_heads),
        query_rows(query_rows),
        decode_query_rows(decode_query_rows),
        head_dim(head_dim),
        scalar_type(scalar_type) {}

  bool allocate_layout(
      const StaticAttentionPlan& plan,
      std::unique_ptr<QnnSharedArena>* new_arena,
      std::vector<Block>* new_blocks,
      ComposableAttentionWorkspace* new_workspace,
      ComposableAttentionWorkspace* new_decode_workspace) const {
    if (plan.widths.empty() || new_arena == nullptr || new_blocks == nullptr ||
        new_workspace == nullptr || new_decode_workspace == nullptr) {
      return false;
    }
    size_t elements = 0;
    for (const size_t width : plan.widths) {
      size_t kv_elements = 0;
      size_t visibility_elements = 0;
      size_t decode_visibility_elements = 0;
      size_t block_elements = 0;
      if (!checked_mul(kv_heads, head_dim, &kv_elements) ||
          !checked_mul(kv_elements, width, &kv_elements) ||
          !checked_mul(kv_elements, size_t{2}, &kv_elements) ||
          !checked_mul(query_rows, width, &visibility_elements) ||
          !checked_mul(decode_query_rows, width, &decode_visibility_elements) ||
          !checked_add(kv_elements, visibility_elements, &block_elements) ||
          !checked_add(
              block_elements, decode_visibility_elements, &block_elements) ||
          !checked_add(elements, block_elements, &elements)) {
        return false;
      }
    }
    auto add_workspace_elements = [&](size_t rows) {
      size_t state_elements = 0;
      size_t output_elements = 0;
      size_t scalar_state_elements = 0;
      size_t two_scalar_states = 0;
      return checked_mul(query_heads, rows, &scalar_state_elements) &&
          checked_mul(scalar_state_elements, head_dim, &output_elements) &&
          checked_mul(output_elements, size_t{2}, &state_elements) &&
          checked_mul(scalar_state_elements, size_t{2}, &two_scalar_states) &&
          checked_add(state_elements, two_scalar_states, &state_elements) &&
          checked_mul(state_elements, size_t{2}, &state_elements) &&
          checked_add(elements, state_elements, &elements);
    };
    if (!add_workspace_elements(query_rows) ||
        !add_workspace_elements(decode_query_rows)) {
      return false;
    }
    const size_t tensor_count = plan.widths.size() * 4 + 16;
    const size_t storage_bytes =
        aligned_storage_bytes(elements, tensor_count, scalar_type);
    if (storage_bytes == 0) {
      return false;
    }
    auto allocated_arena = std::make_unique<QnnSharedArena>(storage_bytes);
    if (!allocated_arena->valid()) {
      return false;
    }
    std::vector<Block> allocated_blocks;
    allocated_blocks.reserve(plan.widths.size());
    for (const size_t width : plan.widths) {
      Block block{
          width,
          0,
          allocated_arena->make_tensor(
              {1,
               static_cast<SizesType>(kv_heads),
               static_cast<SizesType>(head_dim),
               static_cast<SizesType>(width)},
              scalar_type),
          allocated_arena->make_tensor(
              {1,
               static_cast<SizesType>(kv_heads),
               static_cast<SizesType>(width),
               static_cast<SizesType>(head_dim)},
              scalar_type),
          allocated_arena->make_tensor(
              {1,
               1,
               static_cast<SizesType>(query_rows),
               static_cast<SizesType>(width)},
              scalar_type),
          allocated_arena->make_tensor(
              {1,
               1,
               static_cast<SizesType>(decode_query_rows),
               static_cast<SizesType>(width)},
              scalar_type)};
      if (block.key == nullptr || block.value == nullptr ||
          block.visibility == nullptr || block.decode_visibility == nullptr) {
        return false;
      }
      if (!block.visibility_cache.reset_known_zero(query_rows, width) ||
          !block.decode_visibility_cache.reset_known_zero(
              decode_query_rows, width)) {
        return false;
      }
      allocated_blocks.emplace_back(std::move(block));
    }
    ComposableAttentionWorkspace allocated_workspace;
    ComposableAttentionWorkspace allocated_decode_workspace;
    auto allocate_workspace = [&](ComposableAttentionWorkspace& workspace,
                                  size_t rows) {
      for (auto& bank : workspace.banks) {
        bank[0] = allocated_arena->make_tensor(
            {1,
             static_cast<SizesType>(query_heads),
             static_cast<SizesType>(rows),
             static_cast<SizesType>(head_dim)},
            scalar_type);
        bank[1] = allocated_arena->make_tensor(
            {1,
             static_cast<SizesType>(query_heads),
             static_cast<SizesType>(rows),
             1},
            scalar_type);
        bank[2] = allocated_arena->make_tensor(
            {1,
             static_cast<SizesType>(query_heads),
             static_cast<SizesType>(rows),
             1},
            scalar_type);
        bank[3] = allocated_arena->make_tensor(
            {1,
             static_cast<SizesType>(query_heads),
             static_cast<SizesType>(rows),
             static_cast<SizesType>(head_dim)},
            scalar_type);
        if (std::any_of(bank.begin(), bank.end(), [](const TensorPtr& tensor) {
              return tensor == nullptr;
            })) {
          return false;
        }
      }
      return true;
    };
    if (!allocate_workspace(allocated_workspace, query_rows) ||
        !allocate_workspace(allocated_decode_workspace, decode_query_rows)) {
      return false;
    }
    *new_blocks = std::move(allocated_blocks);
    *new_workspace = std::move(allocated_workspace);
    *new_decode_workspace = std::move(allocated_decode_workspace);
    *new_arena = std::move(allocated_arena);
    return true;
  }

  bool prepare_recompose(
      const StaticAttentionPlan& plan,
      PreparedLayout* prepared) const {
    if (prepared == nullptr || plan.coverage() < valid_length()) {
      return false;
    }
    *prepared = PreparedLayout{};
    if (arena != nullptr && layout() == plan.widths) {
      return true;
    }
    PreparedLayout replacement;
    if (!allocate_layout(
            plan,
            &replacement.arena,
            &replacement.blocks,
            &replacement.workspace,
            &replacement.decode_workspace)) {
      return false;
    }

    size_t source_block = 0;
    size_t source_offset = 0;
    size_t destination_block = 0;
    size_t destination_offset = 0;
    size_t remaining = valid_length();
    while (remaining > 0) {
      while (source_block < blocks.size() &&
             source_offset == blocks[source_block].valid_width) {
        ++source_block;
        source_offset = 0;
      }
      while (destination_block < replacement.blocks.size() &&
             destination_offset ==
                 replacement.blocks[destination_block].width) {
        ++destination_block;
        destination_offset = 0;
      }
      if (source_block == blocks.size() ||
          destination_block == replacement.blocks.size()) {
        return false;
      }
      const Block& source = blocks[source_block];
      Block& destination = replacement.blocks[destination_block];
      const size_t count = std::min(
          source.valid_width - source_offset,
          destination.width - destination_offset);
      const size_t element_size = executorch::aten::elementSize(scalar_type);
      const auto* source_key =
          static_cast<const uint8_t*>(source.key->const_data_ptr());
      auto* destination_key =
          static_cast<uint8_t*>(destination.key->mutable_data_ptr());
      for (size_t row = 0; row < kv_heads * head_dim; ++row) {
        std::memcpy(
            destination_key +
                (row * destination.width + destination_offset) * element_size,
            source_key + (row * source.width + source_offset) * element_size,
            count * element_size);
      }
      const auto* source_value =
          static_cast<const uint8_t*>(source.value->const_data_ptr());
      auto* destination_value =
          static_cast<uint8_t*>(destination.value->mutable_data_ptr());
      for (size_t head = 0; head < kv_heads; ++head) {
        std::memcpy(
            destination_value +
                (head * destination.width + destination_offset) * head_dim *
                    element_size,
            source_value +
                (head * source.width + source_offset) * head_dim * element_size,
            count * head_dim * element_size);
      }
      destination.valid_width += count;
      source_offset += count;
      destination_offset += count;
      remaining -= count;
    }
    *prepared = std::move(replacement);
    return true;
  }

  void commit_recompose(PreparedLayout&& prepared) noexcept {
    blocks.swap(prepared.blocks);
    for (size_t bank = 0; bank < 2; ++bank) {
      for (size_t tensor = 0; tensor < 4; ++tensor) {
        workspace.banks[bank][tensor].swap(
            prepared.workspace.banks[bank][tensor]);
        decode_workspace.banks[bank][tensor].swap(
            prepared.decode_workspace.banks[bank][tensor]);
      }
    }
    arena.swap(prepared.arena);
  }

  std::vector<size_t> layout() const {
    std::vector<size_t> result;
    result.reserve(blocks.size());
    for (const Block& block : blocks) {
      result.push_back(block.width);
    }
    return result;
  }

  size_t valid_length() const {
    size_t result = 0;
    for (const Block& block : blocks) {
      result += block.valid_width;
    }
    return result;
  }

  void rollback_to(size_t checkpoint) noexcept {
    rollback_edgeinfer_cache_blocks(blocks, checkpoint);
  }

  bool append(const Tensor& key, const Tensor& value, size_t valid_rows) {
    const size_t rows = static_cast<size_t>(key.size(3));
    if (valid_rows == 0 || valid_rows > rows ||
        key.scalar_type() != scalar_type ||
        value.scalar_type() != scalar_type ||
        !executorch::runtime::tensor_is_contiguous(key) ||
        !executorch::runtime::tensor_is_contiguous(value) || key.dim() != 4 ||
        value.dim() != 4 || key.size(0) != 1 || value.size(0) != 1 ||
        static_cast<size_t>(key.size(1)) != kv_heads ||
        static_cast<size_t>(value.size(1)) != kv_heads ||
        static_cast<size_t>(key.size(2)) != head_dim ||
        static_cast<size_t>(value.size(3)) != head_dim ||
        static_cast<size_t>(value.size(2)) != rows || blocks.empty()) {
      return false;
    }

    size_t source_row = 0;
    size_t block_index = 0;
    const size_t element_size = executorch::aten::elementSize(scalar_type);
    const auto* source_key = static_cast<const uint8_t*>(key.const_data_ptr());
    const auto* source_value =
        static_cast<const uint8_t*>(value.const_data_ptr());
    while (source_row < valid_rows) {
      while (block_index < blocks.size() &&
             blocks[block_index].valid_width == blocks[block_index].width) {
        ++block_index;
      }
      if (block_index == blocks.size()) {
        return false;
      }
      Block& block = blocks[block_index];
      const size_t count =
          std::min(valid_rows - source_row, block.width - block.valid_width);
      auto* destination_key =
          static_cast<uint8_t*>(block.key->mutable_data_ptr());
      auto* destination_value =
          static_cast<uint8_t*>(block.value->mutable_data_ptr());
      for (size_t head = 0; head < kv_heads; ++head) {
        for (size_t dim = 0; dim < head_dim; ++dim) {
          const size_t source_key_row = head * head_dim + dim;
          std::memcpy(
              destination_key +
                  (source_key_row * block.width + block.valid_width) *
                      element_size,
              source_key + (source_key_row * rows + source_row) * element_size,
              count * element_size);
        }
        for (size_t row = 0; row < count; ++row) {
          std::memcpy(
              destination_value +
                  (head * block.width + block.valid_width + row) * head_dim *
                      element_size,
              source_value +
                  (head * rows + source_row + row) * head_dim * element_size,
              head_dim * element_size);
        }
      }
      block.valid_width += count;
      source_row += count;
    }
    return true;
  }

  bool copy_range_to_scratch(
      size_t source_begin,
      size_t valid_width,
      size_t scratch_width,
      Tensor& key_scratch,
      Tensor& value_scratch) const {
    size_t source_end = 0;
    if (valid_width == 0 || valid_width > scratch_width ||
        !checked_add(source_begin, valid_width, &source_end) ||
        source_end > valid_length() || key_scratch.scalar_type() != scalar_type ||
        value_scratch.scalar_type() != scalar_type ||
        !has_shape(
            key_scratch,
            {1,
             static_cast<SizesType>(kv_heads),
             static_cast<SizesType>(head_dim),
             static_cast<SizesType>(scratch_width)},
            scalar_type) ||
        !has_shape(
            value_scratch,
            {1,
             static_cast<SizesType>(kv_heads),
             static_cast<SizesType>(scratch_width),
             static_cast<SizesType>(head_dim)},
            scalar_type)) {
      return false;
    }

    const size_t element_size = executorch::aten::elementSize(scalar_type);
    auto* destination_key =
        static_cast<uint8_t*>(key_scratch.mutable_data_ptr());
    auto* destination_value =
        static_cast<uint8_t*>(value_scratch.mutable_data_ptr());
    if (valid_width < scratch_width) {
      const size_t tail = scratch_width - valid_width;
      for (size_t row = 0; row < kv_heads * head_dim; ++row) {
        std::memset(
            destination_key +
                (row * scratch_width + valid_width) * element_size,
            0,
            tail * element_size);
      }
      for (size_t head = 0; head < kv_heads; ++head) {
        std::memset(
            destination_value +
                (head * scratch_width + valid_width) * head_dim * element_size,
            0,
            tail * head_dim * element_size);
      }
    }

    size_t logical_begin = 0;
    size_t copied = 0;
    for (const Block& block : blocks) {
      size_t logical_end = 0;
      if (!checked_add(logical_begin, block.valid_width, &logical_end)) {
        return false;
      }
      if (logical_end <= source_begin) {
        logical_begin = logical_end;
        continue;
      }
      if (logical_begin >= source_end) {
        break;
      }
      const size_t block_offset =
          source_begin > logical_begin ? source_begin - logical_begin : 0;
      const size_t available = block.valid_width - block_offset;
      const size_t count = std::min(valid_width - copied, available);
      const auto* source_key =
          static_cast<const uint8_t*>(block.key->const_data_ptr());
      const auto* source_value =
          static_cast<const uint8_t*>(block.value->const_data_ptr());
      for (size_t row = 0; row < kv_heads * head_dim; ++row) {
        std::memcpy(
            destination_key + (row * scratch_width + copied) * element_size,
            source_key + (row * block.width + block_offset) * element_size,
            count * element_size);
      }
      for (size_t head = 0; head < kv_heads; ++head) {
        std::memcpy(
            destination_value +
                (head * scratch_width + copied) * head_dim * element_size,
            source_value +
                (head * block.width + block_offset) * head_dim * element_size,
            count * head_dim * element_size);
      }
      copied += count;
      if (copied == valid_width) {
        break;
      }
      logical_begin = logical_end;
    }
    return copied == valid_width;
  }

  bool prepare_blocks(
      size_t query_rows,
      size_t causal_query_begin,
      size_t valid_query_rows,
      bool decode,
      PreparedBlocks* prepared) {
    if (prepared == nullptr || valid_query_rows == 0 ||
        valid_query_rows > query_rows) {
      return false;
    }
    const auto visible_end =
        causal_visible_prefix_end(causal_query_begin, valid_query_rows);
    if (!visible_end.has_value()) {
      return false;
    }
    prepared->blocks.clear();
    prepared->blocks.reserve(blocks.size());
    size_t block_begin = 0;
    for (Block& block : blocks) {
      if (causal_block_visible_columns(
              block_begin, block.valid_width, *visible_end) == 0) {
        break;
      }
      TensorPtr const& visibility =
          decode ? block.decode_visibility : block.visibility;
      auto& visibility_cache =
          decode ? block.decode_visibility_cache : block.visibility_cache;
      const bool visibility_ready =
          edgeinfer::detail::update_causal_visibility_mask_cache(
              visibility_cache,
              query_rows,
              block.width,
              block_begin,
              block.valid_width,
              causal_query_begin,
              valid_query_rows,
              [&](size_t offset, size_t elements, bool visible) {
                return fill_tensor_range(
                    *visibility, offset, elements, visible ? 1.0f : 0.0f);
              });
      if (!visibility_ready) {
        return false;
      }
      prepared->blocks.push_back(
          {block.key.get(), block.value.get(), visibility.get()});
      if (c10::add_overflows(block_begin, block.width, &block_begin)) {
        return false;
      }
    }
    return !prepared->blocks.empty();
  }

  Result<std::array<TensorPtr, 2>> materialize() const {
    const size_t length = valid_length();
    if (length == 0) {
      return Error::InvalidArgument;
    }
    auto key = make_zero_tensor(
        {1,
         static_cast<SizesType>(kv_heads),
         static_cast<SizesType>(head_dim),
         static_cast<SizesType>(length)},
        scalar_type);
    auto value = make_zero_tensor(
        {1,
         static_cast<SizesType>(kv_heads),
         static_cast<SizesType>(length),
         static_cast<SizesType>(head_dim)},
        scalar_type);
    if (key == nullptr || value == nullptr) {
      return Error::MemoryAllocationFailed;
    }
    const size_t element_size = executorch::aten::elementSize(scalar_type);
    size_t offset = 0;
    for (const Block& block : blocks) {
      const auto* source_key =
          static_cast<const uint8_t*>(block.key->const_data_ptr());
      const auto* source_value =
          static_cast<const uint8_t*>(block.value->const_data_ptr());
      auto* destination_key = static_cast<uint8_t*>(key->mutable_data_ptr());
      auto* destination_value =
          static_cast<uint8_t*>(value->mutable_data_ptr());
      for (size_t row = 0; row < kv_heads * head_dim; ++row) {
        std::memcpy(
            destination_key + (row * length + offset) * element_size,
            source_key + row * block.width * element_size,
            block.valid_width * element_size);
      }
      for (size_t head = 0; head < kv_heads; ++head) {
        std::memcpy(
            destination_value +
                (head * length + offset) * head_dim * element_size,
            source_value + head * block.width * head_dim * element_size,
            block.valid_width * head_dim * element_size);
      }
      offset += block.valid_width;
    }
    if (offset != length) {
      return Error::Internal;
    }
    return std::array<TensorPtr, 2>{std::move(key), std::move(value)};
  }

  size_t query_heads;
  size_t kv_heads;
  size_t query_rows;
  size_t decode_query_rows;
  size_t head_dim;
  ScalarType scalar_type;
  std::unique_ptr<QnnSharedArena> arena;
  std::vector<Block> blocks;
  ComposableAttentionWorkspace workspace;
  ComposableAttentionWorkspace decode_workspace;
};

EdgeInferPromptProcessor::~EdgeInferPromptProcessor() {
  release_pre_output_bindings();
  unload_all_composable_modules();
}

EdgeInferPromptProcessor::EdgeInferPromptProcessor(
    DecoderRunner* decoder_runner,
    KVManager* kv_manager,
    const std::string& method_name,
    Metadata metadata,
    std::unique_ptr<executorch::extension::MethodMeta> method_meta,
    executorch::extension::Module* composable_module,
    std::vector<executorch::extension::Module*> composable_layer_modules,
    executorch::extension::Module* decode_composable_module,
    std::vector<executorch::extension::Module*> decode_composable_layer_modules,
    int32_t native_decode_ar_len,
    EdgeInferPrefillOptions options)
    : PromptProcessor(
          decoder_runner,
          kv_manager,
          method_name,
          metadata,
          std::move(method_meta)),
      composable_module_(composable_module),
      composable_layer_modules_(std::move(composable_layer_modules)),
      decode_composable_module_(decode_composable_module),
      decode_composable_layer_modules_(
          std::move(decode_composable_layer_modules)),
      options_(std::move(options)),
      native_decode_ar_len_(native_decode_ar_len) {}

executorch::extension::Module* EdgeInferPromptProcessor::core_module(
    bool decode) const {
  return decode && decode_composable_module_ != nullptr
      ? decode_composable_module_
      : composable_module_;
}

executorch::extension::Module* EdgeInferPromptProcessor::layer_module(
    bool decode,
    size_t layer) const {
  const auto& layer_modules = decode && decode_composable_module_ != nullptr
      ? decode_composable_layer_modules_
      : composable_layer_modules_;
  const size_t layers_per_shard = decode && decode_composable_module_ != nullptr
      ? options_.decode_layers_per_shard
      : options_.layers_per_shard;
  if (layer_modules.empty()) {
    return core_module(decode);
  }
  if (layers_per_shard == 0 ||
      layer / layers_per_shard >= layer_modules.size()) {
    return nullptr;
  }
  return layer_modules[layer / layers_per_shard];
}

std::vector<executorch::extension::Module*>
EdgeInferPromptProcessor::portfolio_modules(bool decode) const {
  std::vector<executorch::extension::Module*> modules;
  auto append_unique = [&](executorch::extension::Module* module) {
    if (module != nullptr &&
        std::find(modules.begin(), modules.end(), module) == modules.end()) {
      modules.emplace_back(module);
    }
  };
  append_unique(core_module(decode));
  const auto& layer_modules = decode && decode_composable_module_ != nullptr
      ? decode_composable_layer_modules_
      : composable_layer_modules_;
  for (auto* module : layer_modules) {
    append_unique(module);
  }
  return modules;
}

bool EdgeInferPromptProcessor::has_separate_decode_portfolio() const {
  return decode_composable_module_ != nullptr;
}

size_t EdgeInferPromptProcessor::query_rows_for_phase(bool decode) const {
  return edgeinfer::detail::query_rows_for_phase(
      static_cast<size_t>(metadata_.ar_len),
      decode,
      has_separate_decode_portfolio());
}

void EdgeInferPromptProcessor::unload_all_composable_modules() {
  std::vector<executorch::extension::Module*> modules =
      portfolio_modules(false);
  for (auto* module : portfolio_modules(true)) {
    if (std::find(modules.begin(), modules.end(), module) == modules.end()) {
      modules.emplace_back(module);
    }
  }
  for (auto* module : modules) {
    unload_composable_module_methods(module);
  }
}

void EdgeInferPromptProcessor::unload_composable_module_methods(
    executorch::extension::Module* module) {
  unload_module_methods(module);
  forget_stage_output_bindings(module);
}

Error EdgeInferPromptProcessor::execute_into_outputs(
    executorch::extension::Module* module,
    const std::string& method,
    const std::vector<EValue>& inputs,
    const std::vector<TensorPtr>& outputs) {
  if (module == nullptr || outputs.empty() ||
      std::any_of(outputs.begin(), outputs.end(), [](const TensorPtr& output) {
        return output == nullptr;
      })) {
    return Error::InvalidArgument;
  }

  const auto fingerprint =
      edgeinfer::detail::make_stage_output_binding_fingerprint(outputs);
  if (!fingerprint.has_value()) {
    return Error::InvalidArgument;
  }

  bool outputs_bound = false;
  auto& copy_methods = copy_stage_methods_[module];
  if (copy_methods.count(method) == 0) {
    std::vector<EValue> values;
    values.reserve(outputs.size());
    for (const TensorPtr& output : outputs) {
      values.emplace_back(*output);
    }
    const auto disposition = edgeinfer::detail::prepare_stage_output_binding(
        stage_output_bindings_,
        module,
        method,
        *fingerprint,
        [&]() { return has_memory_planned_output(module, method); },
        [&]() { return module->set_outputs(method, values); },
        [&]() { module->unload_method(method); });
    outputs_bound =
        disposition == edgeinfer::detail::StageOutputBindingDisposition::Bound;
    if (!outputs_bound) {
      copy_methods.insert(method);
      ET_LOG(
          Info,
          "EdgeInfer method '%s' does not expose stable caller-owned outputs; "
          "using the "
          "compatible copy path.",
          method.c_str());
    }
  }

  auto executed = module->execute(method, inputs);
  if (!executed.ok()) {
    ET_LOG(
        Error,
        "EdgeInfer split method '%s' failed: 0x%x",
        method.c_str(),
        static_cast<unsigned int>(executed.error()));
    return executed.error();
  }
  if (executed->size() != outputs.size()) {
    return Error::InvalidArgument;
  }
  bool bound_address_mismatch = false;
  for (size_t index = 0; index < outputs.size(); ++index) {
    if (!(*executed)[index].isTensor()) {
      return Error::InvalidArgument;
    }
    const Tensor& actual = (*executed)[index].toTensor();
    Tensor& expected = *outputs[index];
    if (!has_same_tensor_contract(actual, expected)) {
      return Error::InvalidArgument;
    }
    if (outputs_bound) {
      if (actual.mutable_data_ptr() != expected.mutable_data_ptr()) {
        bound_address_mismatch = true;
        if (actual.nbytes() > 0) {
          std::memcpy(
              expected.mutable_data_ptr(),
              actual.const_data_ptr(),
              actual.nbytes());
        }
      }
    } else if (
        actual.const_data_ptr() != expected.const_data_ptr() &&
        actual.nbytes() > 0) {
      std::memcpy(
          expected.mutable_data_ptr(),
          actual.const_data_ptr(),
          actual.nbytes());
    }
  }
  if (bound_address_mismatch) {
    ET_LOG(
        Info,
        "EdgeInfer method '%s' retained a planned output after set_outputs(); "
        "using the compatible copy path.",
        method.c_str());
    edgeinfer::detail::reject_stage_output_binding(
        stage_output_bindings_,
        module,
        method,
        [&]() { copy_methods.insert(method); },
        [&]() { module->unload_method(method); });
  }
  return Error::Ok;
}

void EdgeInferPromptProcessor::release_pre_output_bindings() {
  for (const auto& entry : stage_output_bindings_.bindings()) {
    if (entry.first == nullptr) {
      continue;
    }
    for (const auto& method : entry.second) {
      entry.first->unload_method(method.first);
    }
  }
  stage_output_bindings_.clear();
  copy_stage_methods_.clear();
}

void EdgeInferPromptProcessor::forget_stage_output_bindings(
    executorch::extension::Module* module) {
  stage_output_bindings_.forget_module(module);
  copy_stage_methods_.erase(module);
}

void EdgeInferPromptProcessor::clear_local_state() {
  release_pre_output_bindings();
  unload_all_composable_modules();
  layer_cache_view_.clear();
  layer_caches_.clear();
  stage_workspace_.reset();
  rope_inverse_frequencies_.clear();
  attention_runner_.reset();
  decode_attention_runner_.reset();
  prefill_row_runtimes_.clear();
  prefill_scratch_pool_.reset();
  prefill_planner_.reset();
  sequence_length_ = 0;
  scalar_type_ = ScalarType::Undefined;
  ready_ = false;
}

Result<executorch::extension::llm::ComposableAttentionRunner>
EdgeInferPromptProcessor::profile_attention_runner(
    executorch::extension::Module* module,
    size_t query_rows,
    const std::vector<size_t>& widths) {
  if (module == nullptr || query_rows == 0 || widths.empty() ||
      (scalar_type_ != ScalarType::Float && scalar_type_ != ScalarType::Half)) {
    return Error::InvalidArgument;
  }
  const size_t iterations = std::max<size_t>(options_.profile_iterations, 1);
  auto profile_method = [&](const std::string& method,
                            const std::vector<EValue>& inputs,
                            const std::vector<TensorPtr>& outputs,
                            double* p50_ms,
                            double* p90_ms) {
    if (outputs.empty() || p50_ms == nullptr || p90_ms == nullptr) {
      return Error::InvalidArgument;
    }
    Error error = execute_into_outputs(module, method, inputs, outputs);
    if (error != Error::Ok) {
      return error;
    }
    for (size_t index = 0; index < options_.profile_warmup; ++index) {
      error = execute_into_outputs(module, method, inputs, outputs);
      if (error != Error::Ok) {
        return error;
      }
    }
    std::vector<double> samples;
    samples.reserve(iterations);
    for (size_t index = 0; index < iterations; ++index) {
      const auto start = std::chrono::steady_clock::now();
      error = execute_into_outputs(module, method, inputs, outputs);
      if (error != Error::Ok) {
        return error;
      }
      samples.push_back(std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start)
                            .count());
    }
    std::sort(samples.begin(), samples.end());
    const size_t p90_index = static_cast<size_t>(
        std::ceil(0.90 * static_cast<double>(samples.size() - 1)));
    *p50_ms = std::max(samples[samples.size() / 2], 1.0e-6);
    *p90_ms = std::max(samples[p90_index], *p50_ms);
    return Error::Ok;
  };

  std::vector<StaticAttentionGraphCost> costs;
  costs.reserve(widths.size());
  for (const size_t width : widths) {
    const std::string first_method =
        executorch::extension::llm::static_attention_method_name(
            true, query_rows, width);
    const std::string merge_method =
        executorch::extension::llm::static_attention_method_name(
            false, query_rows, width);
    if (!has_method_tensor_contract(
            module,
            first_method,
            std::vector<ScalarType>(4, scalar_type_),
            std::vector<ScalarType>(4, scalar_type_)) ||
        !has_method_tensor_contract(
            module,
            merge_method,
            std::vector<ScalarType>(7, scalar_type_),
            std::vector<ScalarType>(4, scalar_type_))) {
      ET_LOG(
          Error,
          "EdgeInfer Attention method dtype contract mismatch for R=%zu "
          "C=%zu (expected %s).",
          query_rows,
          width,
          executorch::runtime::toString(scalar_type_));
      return Error::NotSupported;
    }
    auto release_profile_methods = [&]() {
      module->unload_method(first_method);
      module->unload_method(merge_method);
      stage_output_bindings_.forget(module, first_method);
      stage_output_bindings_.forget(module, merge_method);
      copy_stage_methods_[module].erase(first_method);
      copy_stage_methods_[module].erase(merge_method);
    };

    size_t query_elements = 0;
    size_t kv_elements = 0;
    size_t visibility_elements = 0;
    size_t scalar_state_elements = 0;
    size_t total_elements = 0;
    auto add_elements = [&](size_t count) {
      return checked_add(total_elements, count, &total_elements);
    };
    if (!checked_mul(options_.query_heads, query_rows, &query_elements) ||
        !checked_mul(query_elements, options_.head_dim, &query_elements) ||
        !checked_mul(options_.kv_heads, options_.head_dim, &kv_elements) ||
        !checked_mul(kv_elements, width, &kv_elements) ||
        !checked_mul(query_rows, width, &visibility_elements) ||
        !checked_mul(
            options_.query_heads, query_rows, &scalar_state_elements) ||
        !add_elements(query_elements) || !add_elements(kv_elements) ||
        !add_elements(kv_elements) || !add_elements(visibility_elements) ||
        !add_elements(scalar_state_elements) ||
        !add_elements(scalar_state_elements) || !add_elements(query_elements) ||
        !add_elements(query_elements) || !add_elements(scalar_state_elements) ||
        !add_elements(scalar_state_elements) || !add_elements(query_elements)) {
      return Error::InvalidArgument;
    }
    const size_t storage_bytes =
        aligned_storage_bytes(total_elements, 11, scalar_type_);
    if (storage_bytes == 0) {
      return Error::InvalidArgument;
    }
    QnnSharedArena arena(storage_bytes);
    if (!arena.valid()) {
      return Error::MemoryAllocationFailed;
    }
    auto q = arena.make_tensor(
        {1,
         static_cast<SizesType>(options_.query_heads),
         static_cast<SizesType>(query_rows),
         static_cast<SizesType>(options_.head_dim)},
        scalar_type_);
    auto key = arena.make_tensor(
        {1,
         static_cast<SizesType>(options_.kv_heads),
         static_cast<SizesType>(options_.head_dim),
         static_cast<SizesType>(width)},
        scalar_type_);
    auto value = arena.make_tensor(
        {1,
         static_cast<SizesType>(options_.kv_heads),
         static_cast<SizesType>(width),
         static_cast<SizesType>(options_.head_dim)},
        scalar_type_);
    auto visibility = arena.make_tensor(
        {1,
         1,
         static_cast<SizesType>(query_rows),
         static_cast<SizesType>(width)},
        scalar_type_);
    auto state_max = arena.make_tensor(
        {1,
         static_cast<SizesType>(options_.query_heads),
         static_cast<SizesType>(query_rows),
         1},
        scalar_type_);
    auto state_denom = arena.make_tensor(
        {1,
         static_cast<SizesType>(options_.query_heads),
         static_cast<SizesType>(query_rows),
         1},
        scalar_type_);
    auto state_numer = arena.make_tensor(
        {1,
         static_cast<SizesType>(options_.query_heads),
         static_cast<SizesType>(query_rows),
         static_cast<SizesType>(options_.head_dim)},
        scalar_type_);
    std::vector<TensorPtr> outputs{
        arena.make_tensor(
            {1,
             static_cast<SizesType>(options_.query_heads),
             static_cast<SizesType>(query_rows),
             static_cast<SizesType>(options_.head_dim)},
            scalar_type_),
        arena.make_tensor(
            {1,
             static_cast<SizesType>(options_.query_heads),
             static_cast<SizesType>(query_rows),
             1},
            scalar_type_),
        arena.make_tensor(
            {1,
             static_cast<SizesType>(options_.query_heads),
             static_cast<SizesType>(query_rows),
             1},
            scalar_type_),
        arena.make_tensor(
            {1,
             static_cast<SizesType>(options_.query_heads),
             static_cast<SizesType>(query_rows),
             static_cast<SizesType>(options_.head_dim)},
            scalar_type_)};
    if (q == nullptr || key == nullptr || value == nullptr ||
        visibility == nullptr || state_max == nullptr ||
        state_denom == nullptr || state_numer == nullptr ||
        std::any_of(
            outputs.begin(), outputs.end(), [](const TensorPtr& tensor) {
              return tensor == nullptr;
            })) {
      return Error::MemoryAllocationFailed;
    }
    if (!fill_tensor(*visibility, visibility_elements, 1.0f) ||
        !fill_tensor(*state_max, scalar_state_elements, -1.0f) ||
        !fill_tensor(*state_denom, scalar_state_elements, 1.0f)) {
      return Error::InvalidArgument;
    }

    std::vector<EValue> first_inputs{*q, *key, *value, *visibility};
    double first_ms = 0.0;
    double first_p90_ms = 0.0;
    Error error = profile_method(
        first_method, first_inputs, outputs, &first_ms, &first_p90_ms);
    if (error != Error::Ok) {
      release_profile_methods();
      return error;
    }
    std::vector<EValue> merge_inputs{
        *q, *key, *value, *visibility, *state_max, *state_denom, *state_numer};
    double merge_ms = 0.0;
    double merge_p90_ms = 0.0;
    error = profile_method(
        merge_method, merge_inputs, outputs, &merge_ms, &merge_p90_ms);
    if (error != Error::Ok) {
      release_profile_methods();
      return error;
    }
    ET_LOG(
        Info,
        "EdgeInfer R=%zu profile width=%zu first_p50=%.4f ms "
        "first_p90=%.4f ms merge_p50=%.4f ms merge_p90=%.4f ms",
        query_rows,
        width,
        first_ms,
        first_p90_ms,
        merge_ms,
        merge_p90_ms);
    costs.push_back({width, first_ms, merge_ms});
    release_profile_methods();
  }
  return executorch::extension::llm::ComposableAttentionRunner::create(
      module, query_rows, std::move(costs));
}

Error EdgeInferPromptProcessor::initialize_prefill_portfolio() {
  executorch::extension::Module* module = core_module(false);
  const size_t primary_rows = query_rows_for_phase(false);
  if (module == nullptr || primary_rows == 0) {
    return Error::InvalidArgument;
  }

  std::vector<size_t> rows = available_attention_rows(module);
  if (!options_.prefill_query_rows.empty()) {
    std::vector<size_t> requested = options_.prefill_query_rows;
    requested.push_back(primary_rows);
    std::sort(requested.begin(), requested.end());
    requested.erase(std::unique(requested.begin(), requested.end()), requested.end());
    rows.erase(
        std::remove_if(
            rows.begin(),
            rows.end(),
            [&](size_t value) {
              return !std::binary_search(
                  requested.begin(), requested.end(), value);
            }),
        rows.end());
  }
  if (!std::binary_search(rows.begin(), rows.end(), primary_rows)) {
    ET_LOG(
        Error,
        "EdgeInfer Prefill PTE does not contain the primary R=%zu Attention "
        "portfolio.",
        primary_rows);
    return Error::NotSupported;
  }

  std::vector<StaticAttentionShapeCost> shape_costs;
  std::vector<size_t> scratch_widths;
  prefill_row_runtimes_.clear();
  prefill_planner_.reset();
  prefill_scratch_pool_.reset();
  attention_runner_.reset();

  const char* projection_kind = options_.pre_attention_rope ? "pre" : "qkv";
  for (const size_t row_count : rows) {
    std::vector<size_t> widths = available_attention_widths(module, row_count);
    if (row_count == primary_rows) {
      widths.erase(
          std::remove_if(
              widths.begin(),
              widths.end(),
              [&](size_t width) {
                return !std::binary_search(
                    options_.widths.begin(), options_.widths.end(), width);
              }),
          widths.end());
      if (widths != options_.widths) {
        ET_LOG(
            Error,
            "EdgeInfer primary R=%zu PTE methods do not match the declared "
            "width portfolio.",
            row_count);
        return Error::NotSupported;
      }
    }
    if (widths.empty()) {
      return Error::NotSupported;
    }

    const std::string embedding =
        stage_method_name("embedding", row_count, primary_rows);
    const std::string output =
        stage_method_name("output", row_count, primary_rows);
    if (!has_method_tensor_contract(
            module, embedding, {ScalarType::Long}, {scalar_type_}) ||
        !has_method_tensor_contract(
            module, output, {scalar_type_}, {scalar_type_})) {
      ET_LOG(
          Error,
          "EdgeInfer R=%zu embedding/output ABI is unavailable.",
          row_count);
      return Error::NotSupported;
    }
    for (size_t layer = 0; layer < options_.layers; ++layer) {
      auto* stage_module = layer_module(false, layer);
      const std::string projection = stage_method_name(
          projection_kind, row_count, primary_rows, layer);
      const std::string post =
          stage_method_name("post", row_count, primary_rows, layer);
      const std::vector<ScalarType> projection_inputs(
          options_.pre_attention_rope ? 3 : 1, scalar_type_);
      const std::vector<ScalarType> projection_outputs(
          options_.pre_attention_rope ? 3 : 1, scalar_type_);
      if (stage_module == nullptr ||
          !has_method_tensor_contract(
              stage_module,
              projection,
              projection_inputs,
              projection_outputs) ||
          !has_method_tensor_contract(
              stage_module,
              post,
              {scalar_type_, scalar_type_},
              {scalar_type_})) {
        ET_LOG(
            Error,
            "EdgeInfer R=%zu layer=%zu stage ABI is unavailable.",
            row_count,
            layer);
        return Error::NotSupported;
      }
    }

    auto profiled = profile_attention_runner(module, row_count, widths);
    if (!profiled.ok()) {
      return profiled.error();
    }
    for (const auto& cost : profiled->graph_costs()) {
      shape_costs.push_back(
          {row_count, cost.width, cost.first_cost, cost.merge_cost});
      scratch_widths.push_back(cost.width);
    }

    std::optional<ComposableAttentionRunner> owned_runner;
    if (row_count == primary_rows) {
      attention_runner_.emplace(std::move(*profiled));
    } else {
      owned_runner.emplace(std::move(*profiled));
    }
    auto runtime = std::make_unique<PrefillRowRuntime>(
        options_.query_heads,
        options_.kv_heads,
        row_count,
        options_.head_dim,
        options_.dim,
        options_.vocab_size,
        options_.hf_rope,
        scalar_type_,
        widths,
        std::move(owned_runner));
    if (!runtime->valid()) {
      return Error::MemoryAllocationFailed;
    }
    prefill_row_runtimes_.emplace(row_count, std::move(runtime));
  }

  auto planner =
      executorch::extension::llm::StaticAttentionPrefillPlanner::create(
          std::move(shape_costs));
  if (!planner.has_value()) {
    return Error::InvalidArgument;
  }
  prefill_planner_.emplace(std::move(*planner));

  std::sort(scratch_widths.begin(), scratch_widths.end());
  scratch_widths.erase(
      std::unique(scratch_widths.begin(), scratch_widths.end()),
      scratch_widths.end());
  prefill_scratch_pool_ = std::make_unique<PrefillScratchPool>(
      options_.kv_heads, options_.head_dim, scratch_widths, scalar_type_);
  if (!prefill_scratch_pool_->valid()) {
    prefill_scratch_pool_.reset();
    return Error::MemoryAllocationFailed;
  }
  options_.prefill_query_rows = std::move(rows);
  ET_LOG(
      Info,
      "EdgeInfer initialized measured Prefill portfolio: query_shapes=%zu "
      "kv_shapes=%zu.",
      options_.prefill_query_rows.size(),
      scratch_widths.size());
  return Error::Ok;
}

Error EdgeInferPromptProcessor::ensure_ready() {
  if (ready_) {
    return Error::Ok;
  }
  if (options_.layers == 0) {
    options_.layers = static_cast<size_t>(metadata_.num_layers);
  }
  if (options_.vocab_size == 0) {
    options_.vocab_size = static_cast<size_t>(metadata_.vocab_size);
  }
  if (options_.decode_widths.empty()) {
    options_.decode_widths = options_.widths;
  }
  if (composable_module_ == nullptr || options_.layers == 0 ||
      options_.dim == 0 || options_.query_heads == 0 ||
      options_.kv_heads == 0 || options_.head_dim == 0 ||
      options_.vocab_size == 0 || metadata_.ar_len <= 0 ||
      options_.query_heads % options_.kv_heads != 0 ||
      options_.widths.empty() || options_.decode_widths.empty() ||
      !std::all_of(
          options_.widths.begin(), options_.widths.end(), is_power_of_two) ||
      !std::all_of(
          options_.decode_widths.begin(),
          options_.decode_widths.end(),
          is_power_of_two)) {
    return Error::InvalidArgument;
  }
  std::sort(options_.widths.begin(), options_.widths.end());
  options_.widths.erase(
      std::unique(options_.widths.begin(), options_.widths.end()),
      options_.widths.end());
  std::sort(options_.decode_widths.begin(), options_.decode_widths.end());
  options_.decode_widths.erase(
      std::unique(options_.decode_widths.begin(), options_.decode_widths.end()),
      options_.decode_widths.end());
  const ScalarType cache_type = kv_manager_->cache_scalar_type();
  const Error cache_type_error =
      edgeinfer::validate_edgeinfer_cache_scalar_type(cache_type);
  if (options_.vocab_size != static_cast<size_t>(metadata_.vocab_size) ||
      options_.layers != static_cast<size_t>(metadata_.num_layers) ||
      options_.head_dim != static_cast<size_t>(kv_manager_->get_head_dim()) ||
      cache_type_error != Error::Ok) {
    ET_LOG(
        Info,
        "EdgeInfer split/native contract mismatch: split layers=%zu vocab=%zu "
        "head_dim=%zu; native layers=%lld vocab=%d head_dim=%lld cache=%s.",
        options_.layers,
        options_.vocab_size,
        options_.head_dim,
        static_cast<long long>(metadata_.num_layers),
        metadata_.vocab_size,
        static_cast<long long>(kv_manager_->get_head_dim()),
        executorch::runtime::toString(cache_type));
    return Error::NotSupported;
  }

  auto valid_shard_layout = [&](bool decode) {
    const auto& layer_modules = decode && has_separate_decode_portfolio()
        ? decode_composable_layer_modules_
        : composable_layer_modules_;
    const size_t layers_per_shard = decode && has_separate_decode_portfolio()
        ? options_.decode_layers_per_shard
        : options_.layers_per_shard;
    if (layer_modules.empty()) {
      return layers_per_shard == 0;
    }
    if (layers_per_shard == 0) {
      return false;
    }
    const size_t expected_shards =
        (options_.layers + layers_per_shard - 1) / layers_per_shard;
    return layer_modules.size() == expected_shards &&
        std::none_of(
               layer_modules.begin(),
               layer_modules.end(),
               [](const auto* module) { return module == nullptr; });
  };
  if (!valid_shard_layout(false) || !valid_shard_layout(true)) {
    ET_LOG(
        Error,
        "EdgeInfer layer-shard layout does not cover exactly %zu layers.",
        options_.layers);
    return Error::InvalidArgument;
  }

  executorch::extension::Module* decode_module = core_module(true);
  const size_t query_rows = query_rows_for_phase(false);
  const size_t decode_query_rows = query_rows_for_phase(true);
  const std::string layer_stage = options_.pre_attention_rope ? "_pre" : "_qkv";
  auto infer_stage_type = [&](bool decode) -> std::optional<ScalarType> {
    executorch::extension::Module* module = core_module(decode);
    if (module == nullptr) {
      return std::nullopt;
    }
    auto names = module->method_names();
    if (!names.ok() || names->count("llama_embedding") == 0 ||
        names->count("llama_output") == 0) {
      return std::nullopt;
    }
    auto embedding_meta = module->method_meta("llama_embedding");
    if (!embedding_meta.ok() || embedding_meta->num_inputs() != 1 ||
        embedding_meta->num_outputs() != 1) {
      return std::nullopt;
    }
    auto token_meta = embedding_meta->input_tensor_meta(0);
    auto hidden_meta = embedding_meta->output_tensor_meta(0);
    if (!token_meta.ok() || !hidden_meta.ok() ||
        token_meta->scalar_type() != ScalarType::Long) {
      return std::nullopt;
    }
    const ScalarType scalar_type = hidden_meta->scalar_type();
    if (scalar_type != ScalarType::Float && scalar_type != ScalarType::Half) {
      return std::nullopt;
    }
    if (!has_method_tensor_contract(
            module, "llama_output", {scalar_type}, {scalar_type})) {
      return std::nullopt;
    }
    for (size_t layer = 0; layer < options_.layers; ++layer) {
      executorch::extension::Module* stage_module = layer_module(decode, layer);
      if (stage_module == nullptr) {
        return std::nullopt;
      }
      auto stage_names = stage_module->method_names();
      const std::string prefix = "llama_layer_" + std::to_string(layer);
      if (!stage_names.ok() || stage_names->count(prefix + layer_stage) == 0 ||
          stage_names->count(prefix + "_post") == 0) {
        return std::nullopt;
      }
      const std::vector<ScalarType> stage_inputs(
          options_.pre_attention_rope ? 3 : 1, scalar_type);
      const std::vector<ScalarType> stage_outputs(
          options_.pre_attention_rope ? 3 : 1, scalar_type);
      if (!has_method_tensor_contract(
              stage_module,
              prefix + layer_stage,
              stage_inputs,
              stage_outputs) ||
          !has_method_tensor_contract(
              stage_module,
              prefix + "_post",
              {scalar_type, scalar_type},
              {scalar_type})) {
        return std::nullopt;
      }
    }
    return scalar_type;
  };
  const auto prefill_type = infer_stage_type(false);
  const auto decode_type =
      !has_separate_decode_portfolio() ? prefill_type : infer_stage_type(true);
  if (!prefill_type.has_value() || !decode_type.has_value() ||
      *prefill_type != *decode_type) {
    ET_LOG(
        Error,
        "EdgeInfer split PTE does not implement one consistent Float/Half %s "
        "layer ABI.",
        options_.pre_attention_rope ? "pre-Attention RoPE" : "packed QKV");
    return Error::NotSupported;
  }
  scalar_type_ = *prefill_type;
  if (scalar_type_ == ScalarType::Half && !options_.pre_attention_rope) {
    ET_LOG(
        Info,
        "EdgeInfer FP16 is using the compatible host QKV/RoPE path; export "
        "with pre-Attention RoPE for the optimized graph-native path.");
  }

  Error error = initialize_prefill_portfolio();
  if (error != Error::Ok) {
    return error;
  }
  if (decode_module == composable_module_ && decode_query_rows == query_rows &&
      options_.decode_widths == options_.widths) {
    decode_attention_runner_.reset();
  } else {
    auto decode_attention = profile_attention_runner(
        decode_module, decode_query_rows, options_.decode_widths);
    if (!decode_attention.ok()) {
      return decode_attention.error();
    }
    decode_attention_runner_.emplace(std::move(*decode_attention));
  }

  stage_workspace_ = std::make_unique<StageWorkspace>(
      options_.query_heads,
      options_.kv_heads,
      query_rows,
      decode_query_rows,
      options_.head_dim,
      options_.dim,
      options_.vocab_size,
      options_.hf_rope,
      scalar_type_);
  if (!stage_workspace_->valid()) {
    stage_workspace_.reset();
    return Error::MemoryAllocationFailed;
  }
  const size_t frequency_count = options_.head_dim / 2;
  rope_inverse_frequencies_.resize(frequency_count);
  for (size_t pair = 0; pair < frequency_count; ++pair) {
    const double exponent = static_cast<double>(pair * 2) / options_.head_dim;
    rope_inverse_frequencies_[pair] =
        1.0 / std::pow(options_.rope_theta, exponent);
    if (!std::isfinite(rope_inverse_frequencies_[pair])) {
      return Error::InvalidArgument;
    }
  }
  layer_cache_view_.clear();
  layer_caches_.clear();
  layer_caches_.reserve(options_.layers);
  for (size_t layer = 0; layer < options_.layers; ++layer) {
    layer_caches_.emplace_back(std::make_unique<LayerCache>(
        options_.query_heads,
        options_.kv_heads,
        query_rows,
        decode_query_rows,
        options_.head_dim,
        scalar_type_));
  }
  layer_cache_view_.reserve(layer_caches_.size());
  for (const auto& cache : layer_caches_) {
    layer_cache_view_.push_back(cache.get());
  }
  ready_ = true;
  return Error::Ok;
}

Error EdgeInferPromptProcessor::bridge_kv_cache() {
  if (sequence_length_ == 0) {
    return Error::InvalidArgument;
  }
  std::vector<std::array<TensorPtr, 2>> materialized_layers;
  materialized_layers.reserve(layer_caches_.size());
  for (size_t layer = 0; layer < layer_caches_.size(); ++layer) {
    auto materialized = layer_caches_[layer]->materialize();
    if (!materialized.ok()) {
      return materialized.error();
    }
    const Error error = kv_manager_->validate_layer_cache(
        static_cast<int64_t>(layer),
        *(*materialized)[0],
        *(*materialized)[1],
        static_cast<int32_t>(sequence_length_));
    if (error != Error::Ok) {
      return error;
    }
    materialized_layers.emplace_back(std::move(*materialized));
  }

  const Error layout_error = edgeinfer::prepare_native_decode_cache_layout(
      *kv_manager_,
      metadata_.context_len,
      native_decode_ar_len_,
      sequence_length_);
  if (layout_error != Error::Ok) {
    return layout_error;
  }
  for (size_t layer = 0; layer < materialized_layers.size(); ++layer) {
    const Error error = kv_manager_->replace_layer_cache(
        static_cast<int64_t>(layer),
        *materialized_layers[layer][0],
        *materialized_layers[layer][1],
        static_cast<int32_t>(sequence_length_));
    if (error != Error::Ok) {
      return error;
    }
  }
  return Error::Ok;
}

Error EdgeInferPromptProcessor::run_edgeinfer_prefill(
    const std::vector<uint64_t>& prompt_tokens,
    int64_t start_pos,
    bool dump_logits,
    uint64_t* next_token) {
  if (next_token == nullptr || start_pos < 0 || prompt_tokens.empty()) {
    return Error::InvalidArgument;
  }
  const Error error = ensure_ready();
  if (error != Error::Ok) {
    return error;
  }
  // A single-PTE portfolio and a layer-sharded portfolio share the same
  // measured two-level Prefill planner. The original decoder-PTE path remains
  // the transactional fallback in prefill().
  return run_layer_major_prefill(
      prompt_tokens, start_pos, dump_logits, next_token);
}

Error EdgeInferPromptProcessor::run_layer_major_prefill(
    const std::vector<uint64_t>& prompt_tokens,
    int64_t start_pos,
    bool dump_logits,
    uint64_t* next_token) {
  if (start_pos < 0 || prefill_planner_ == std::nullopt) {
    return Error::InvalidArgument;
  }
  auto plan = prefill_planner_->plan(
      prompt_tokens.size(), static_cast<size_t>(start_pos), true);
  if (!plan.has_value()) {
    return Error::NotSupported;
  }
  return run_layer_major_prefill_plan(
      prompt_tokens, start_pos, dump_logits, next_token, *plan);
}

Error EdgeInferPromptProcessor::run_layer_major_prefill_plan(
    const std::vector<uint64_t>& prompt_tokens,
    int64_t start_pos,
    bool dump_logits,
    uint64_t* next_token,
    const StaticAttentionPrefillPlan& plan) {
  if (next_token == nullptr || start_pos < 0 || prompt_tokens.empty() ||
      plan.sequence_length != prompt_tokens.size() ||
      plan.history_length != static_cast<size_t>(start_pos) ||
      plan.query_tiles.empty() || prefill_scratch_pool_ == nullptr ||
      !prefill_scratch_pool_->valid()) {
    return Error::InvalidArgument;
  }
  Error error = ensure_ready();
  if (error != Error::Ok) {
    return error;
  }
  if (sequence_length_ != static_cast<size_t>(start_pos)) {
    return Error::InvalidArgument;
  }
  if (prompt_tokens.size() >
      static_cast<size_t>(metadata_.context_len - metadata_.ar_len) -
          sequence_length_) {
    return Error::InvalidArgument;
  }
  const size_t prompt_end = sequence_length_ + prompt_tokens.size();
  error = recompose_for_target(prompt_end, false);
  if (error != Error::Ok) {
    return error;
  }

  size_t hidden_elements = 0;
  for (const auto& tile : plan.query_tiles) {
    size_t tile_elements = 0;
    if (tile.valid_query_rows == 0 ||
        tile.valid_query_rows > tile.graph_query_rows ||
        !checked_mul(tile.graph_query_rows, options_.dim, &tile_elements) ||
        !checked_add(hidden_elements, tile_elements, &hidden_elements)) {
      return Error::InvalidArgument;
    }
  }
  const size_t hidden_storage_bytes =
      aligned_storage_bytes(hidden_elements, plan.query_tiles.size(), scalar_type_);
  if (hidden_storage_bytes == 0) {
    return Error::MemoryAllocationFailed;
  }
  QnnSharedArena hidden_arena(hidden_storage_bytes);
  if (!hidden_arena.valid()) {
    return Error::MemoryAllocationFailed;
  }
  std::vector<TensorPtr> hidden_states;
  hidden_states.reserve(plan.query_tiles.size());
  for (const auto& tile : plan.query_tiles) {
    auto hidden = hidden_arena.make_tensor(
        {1,
         static_cast<SizesType>(tile.graph_query_rows),
         static_cast<SizesType>(options_.dim)},
        scalar_type_);
    if (hidden == nullptr) {
      return Error::MemoryAllocationFailed;
    }
    hidden_states.emplace_back(std::move(hidden));
  }

  executorch::extension::Module* core = core_module(false);
  if (core == nullptr) {
    return Error::InvalidState;
  }
  const size_t primary_rows = query_rows_for_phase(false);
  for (size_t tile_index = 0; tile_index < plan.query_tiles.size(); ++tile_index) {
    const auto& tile = plan.query_tiles[tile_index];
    const auto runtime_it = prefill_row_runtimes_.find(tile.graph_query_rows);
    if (runtime_it == prefill_row_runtimes_.end() ||
        runtime_it->second == nullptr || !runtime_it->second->valid()) {
      return Error::InvalidState;
    }
    PrefillRowRuntime& runtime = *runtime_it->second;
    StageWorkspace::Mode& stage = runtime.stage->for_decode(false);
    std::vector<int64_t> token_values(tile.graph_query_rows, 0);
    const size_t offset = tile.row_begin;
    const size_t valid_rows = tile.valid_query_rows;
    for (size_t row = 0; row < valid_rows; ++row) {
      token_values[row] = static_cast<int64_t>(prompt_tokens[offset + row]);
    }
    auto tokens = make_int64_tensor(
        {1, static_cast<SizesType>(tile.graph_query_rows)},
        std::move(token_values));
    error = execute_into_outputs(
        core,
        stage_method_name(
            "embedding", tile.graph_query_rows, primary_rows),
        std::vector<EValue>{*tokens},
        std::vector<TensorPtr>{stage.hidden[0]});
    if (error != Error::Ok) {
      return error;
    }
    std::memcpy(
        hidden_states[tile_index]->mutable_data_ptr(),
        stage.hidden[0]->const_data_ptr(),
        stage.hidden[0]->nbytes());
  }

  const size_t rope_width =
      options_.hf_rope ? options_.head_dim : options_.head_dim / 2;
  const size_t pairs = options_.head_dim / 2;
  if (rope_width == 0 ||
      rope_inverse_frequencies_.size() != options_.head_dim / 2) {
    return Error::InvalidState;
  }
  for (size_t layer = 0; layer < options_.layers; ++layer) {
    executorch::extension::Module* stage_module = layer_module(false, layer);
    if (stage_module == nullptr) {
      return Error::InvalidState;
    }
    for (size_t tile_index = 0; tile_index < plan.query_tiles.size(); ++tile_index) {
      const auto& tile = plan.query_tiles[tile_index];
      PrefillRowRuntime& runtime =
          *prefill_row_runtimes_.at(tile.graph_query_rows);
      StageWorkspace::Mode& stage = runtime.stage->for_decode(false);
      const size_t valid_rows = tile.valid_query_rows;
      const size_t query_rows = tile.graph_query_rows;
      const size_t causal_query_begin =
          static_cast<size_t>(start_pos) + tile.row_begin;
      if (hidden_states[tile_index]->nbytes() != stage.hidden[0]->nbytes()) {
        return Error::InvalidState;
      }
      std::memcpy(
          stage.hidden[0]->mutable_data_ptr(),
          hidden_states[tile_index]->const_data_ptr(),
          stage.hidden[0]->nbytes());
      if (scalar_type_ == ScalarType::Float) {
        fill_rope_values(
            stage.rope_cosine->mutable_data_ptr<float>(),
            stage.rope_sine->mutable_data_ptr<float>(),
            query_rows,
            causal_query_begin,
            pairs,
            rope_width,
            options_.hf_rope,
            rope_inverse_frequencies_);
      } else if (scalar_type_ == ScalarType::Half) {
        fill_rope_values(
            stage.rope_cosine->mutable_data_ptr<Half>(),
            stage.rope_sine->mutable_data_ptr<Half>(),
            query_rows,
            causal_query_begin,
            pairs,
            rope_width,
            options_.hf_rope,
            rope_inverse_frequencies_);
      } else {
        return Error::InvalidState;
      }

      TensorPtr& hidden = stage.hidden[0];
      const std::string stage_method = stage_method_name(
          options_.pre_attention_rope ? "pre" : "qkv",
          query_rows,
          primary_rows,
          layer);
      if (options_.pre_attention_rope) {
        error = execute_into_outputs(
            stage_module,
            stage_method,
            std::vector<EValue>{*hidden, *stage.rope_cosine, *stage.rope_sine},
            std::vector<TensorPtr>{stage.query, stage.key, stage.value});
      } else {
        error = execute_into_outputs(
            stage_module,
            stage_method,
            std::vector<EValue>{*hidden},
            std::vector<TensorPtr>{stage.packed_qkv});
        if (error == Error::Ok) {
          if (scalar_type_ == ScalarType::Float) {
            unpack_and_rotate_qkv<float>(
                *stage.packed_qkv,
                *stage.query,
                *stage.key,
                *stage.value,
                *stage.rope_cosine,
                *stage.rope_sine,
                query_rows,
                valid_rows,
                options_.query_heads,
                options_.kv_heads,
                options_.head_dim,
                options_.hf_rope);
          } else {
            unpack_and_rotate_qkv<Half>(
                *stage.packed_qkv,
                *stage.query,
                *stage.key,
                *stage.value,
                *stage.rope_cosine,
                *stage.rope_sine,
                query_rows,
                valid_rows,
                options_.query_heads,
                options_.kv_heads,
                options_.head_dim,
                options_.hf_rope);
          }
        }
      }
      if (error != Error::Ok ||
          !layer_caches_[layer]->append(*stage.key, *stage.value, valid_rows)) {
        return error == Error::Ok ? Error::InvalidArgument : error;
      }

      ComposableAttentionRunner* attention =
          query_rows == primary_rows
          ? (attention_runner_.has_value() ? &*attention_runner_ : nullptr)
          : (runtime.owned_runner.has_value() ? &*runtime.owned_runner : nullptr);
      if (attention == nullptr) {
        return Error::InvalidState;
      }
      size_t block_begin = 0;
      for (size_t block_index = 0;
           block_index < tile.key_plan.widths.size();
           ++block_index) {
        const size_t width = tile.key_plan.widths[block_index];
        const auto scratch_it = prefill_scratch_pool_->blocks.find(width);
        const auto mask_it = runtime.masks.find(width);
        const auto cache_it = runtime.visibility_caches.find(width);
        if (scratch_it == prefill_scratch_pool_->blocks.end() ||
            mask_it == runtime.masks.end() ||
            cache_it == runtime.visibility_caches.end() ||
            block_begin >= tile.visible_prefix) {
          return Error::InvalidState;
        }
        const size_t valid_width =
            std::min(width, tile.visible_prefix - block_begin);
        if (!layer_caches_[layer]->copy_range_to_scratch(
                block_begin,
                valid_width,
                width,
                *scratch_it->second.key,
                *scratch_it->second.value) ||
            !edgeinfer::detail::update_causal_visibility_mask_cache(
                cache_it->second,
                query_rows,
                width,
                block_begin,
                valid_width,
                causal_query_begin,
                valid_rows,
                [&](size_t mask_offset, size_t elements, bool visible) {
                  return fill_tensor_range(
                      *mask_it->second,
                      mask_offset,
                      elements,
                      visible ? 1.0f : 0.0f);
                })) {
          return Error::Internal;
        }
        const PreparedAttentionBlock block{
            scratch_it->second.key.get(),
            scratch_it->second.value.get(),
            mask_it->second.get()};
        error = attention->run_prepared_block(
            *stage.query, block, block_index, runtime.workspace);
        if (error != Error::Ok) {
          return error;
        }
        block_begin += valid_width;
      }
      if (block_begin != tile.visible_prefix) {
        return Error::Internal;
      }
      error = execute_into_outputs(
          stage_module,
          stage_method_name("post", query_rows, primary_rows, layer),
          std::vector<EValue>{
              *hidden,
              *ComposableAttentionRunner::prepared_output(runtime.workspace)},
          std::vector<TensorPtr>{stage.hidden[1]});
      if (error != Error::Ok) {
        return error;
      }
      std::memcpy(
          hidden_states[tile_index]->mutable_data_ptr(),
          stage.hidden[1]->const_data_ptr(),
          stage.hidden[1]->nbytes());
    }
    const bool sharded_portfolio = stage_module != core;
    const bool shard_complete = layer + 1 == options_.layers ||
        layer_module(false, layer + 1) != stage_module;
    if (sharded_portfolio && shard_complete) {
      unload_composable_module_methods(stage_module);
    }
  }

  TensorPtr final_logits;
  size_t final_valid_rows = 0;
  for (size_t tile_index = 0; tile_index < plan.query_tiles.size(); ++tile_index) {
    const auto& tile = plan.query_tiles[tile_index];
    PrefillRowRuntime& runtime =
        *prefill_row_runtimes_.at(tile.graph_query_rows);
    StageWorkspace::Mode& stage = runtime.stage->for_decode(false);
    error = execute_into_outputs(
        core,
        stage_method_name("output", tile.graph_query_rows, primary_rows),
        std::vector<EValue>{*hidden_states[tile_index]},
        std::vector<TensorPtr>{stage.logits});
    if (error != Error::Ok) {
      return error;
    }
    final_logits = stage.logits;
    final_valid_rows = tile.valid_query_rows;
    if (dump_logits) {
      const auto* raw =
          reinterpret_cast<const std::byte*>(stage.logits->const_data_ptr());
      const size_t valid_bytes = tile.valid_query_rows * options_.vocab_size *
          executorch::aten::elementSize(scalar_type_);
      prompt_all_logits_.insert(
          prompt_all_logits_.end(), raw, raw + valid_bytes);
    }
  }
  if (final_logits == nullptr || final_valid_rows == 0) {
    return Error::Internal;
  }
  sequence_length_ = prompt_end;
  ET_LOG(
      Info,
      "EdgeInfer measured Prefill plan: tokens=%zu query_tiles=%zu "
      "attention_calls=%zu predicted_attention=%.4f ms score_waste=%zu.",
      prompt_tokens.size(),
      plan.query_tiles.size(),
      plan.graph_calls,
      plan.predicted_cost,
      plan.wasted_score_elements());
  for (size_t index = 0; index < plan.query_tiles.size(); ++index) {
    const auto& tile = plan.query_tiles[index];
    std::string widths;
    for (const size_t width : tile.key_plan.widths) {
      if (!widths.empty()) {
        widths.append("+");
      }
      widths.append(std::to_string(width));
    }
    ET_LOG(
        Debug,
        "EdgeInfer Prefill tile=%zu row_begin=%zu valid_R=%zu graph_R=%zu "
        "visible_C=%zu plan=%s.",
        index,
        tile.row_begin,
        tile.valid_query_rows,
        tile.graph_query_rows,
        tile.visible_prefix,
        widths.c_str());
  }
  *next_token = static_cast<uint64_t>(decoder_runner_->logits_to_token(
      *final_logits, static_cast<int64_t>(final_valid_rows - 1)));
  return Error::Ok;
}

Error EdgeInferPromptProcessor::recompose_for_target(
    size_t target_context,
    bool decode) {
  auto* planner = decode && decode_attention_runner_.has_value()
      ? &*decode_attention_runner_
      : attention_runner_.has_value() ? &*attention_runner_
                                      : nullptr;
  if (planner == nullptr || target_context < sequence_length_ ||
      target_context > static_cast<size_t>(metadata_.context_len)) {
    return Error::InvalidArgument;
  }
  auto persistent_plan = planner->plan(target_context);
  if (!persistent_plan.ok()) {
    return persistent_plan.error();
  }
  return apply_persistent_plan(
      *persistent_plan, decode ? "Decode" : "Prefill", false);
}

Error EdgeInferPromptProcessor::recompose_for_prefill(
    const std::vector<size_t>& visible_prefixes) {
  if (!attention_runner_.has_value() || visible_prefixes.empty() ||
      visible_prefixes.back() < sequence_length_ ||
      visible_prefixes.back() > static_cast<size_t>(metadata_.context_len)) {
    return Error::InvalidArgument;
  }
  auto persistent_plan = attention_runner_->plan_prefixes(visible_prefixes);
  if (!persistent_plan.ok()) {
    return persistent_plan.error();
  }
  const Error error = apply_persistent_plan(*persistent_plan, "Prefill", true);
  if (error != Error::Ok) {
    return error;
  }

  const auto stats =
      causal_prefill_plan_stats(persistent_plan->widths, visible_prefixes);
  if (!stats.has_value()) {
    return Error::InvalidArgument;
  }
  for (size_t tile = 0; tile < stats->tiles.size(); ++tile) {
    ET_LOG(
        Debug,
        "EdgeInfer Prefill tile=%zu visible_prefix=%zu "
        "visible_graph_calls=%zu boundary_padding=%zu.",
        tile,
        visible_prefixes[tile],
        stats->tiles[tile].visible_graph_calls,
        stats->tiles[tile].boundary_padding);
  }
  ET_LOG(
      Info,
      "EdgeInfer Prefill causal staircase: tiles=%zu physical_blocks=%zu "
      "full_graph_calls=%zu visible_graph_calls=%zu "
      "upper_right_skipped=%zu aggregate_boundary_padding=%zu.",
      visible_prefixes.size(),
      stats->physical_blocks,
      stats->full_graph_calls,
      stats->visible_graph_calls,
      stats->skipped_graph_calls,
      stats->aggregate_boundary_padding);
  return Error::Ok;
}

Error EdgeInferPromptProcessor::apply_persistent_plan(
    const StaticAttentionPlan& persistent_plan,
    const char* phase,
    bool aggregate_cost) {
  if (phase == nullptr || persistent_plan.coverage() < sequence_length_) {
    return Error::InvalidArgument;
  }
  const bool recomposed = edgeinfer::detail::apply_layer_plan_atomically(
      layer_caches_, persistent_plan, [&]() {
        if (attention_runner_.has_value()) {
          attention_runner_->release_output_bindings();
        }
        if (decode_attention_runner_.has_value()) {
          decode_attention_runner_->release_output_bindings();
        }
      });
  if (!recomposed) {
    return Error::MemoryAllocationFailed;
  }
  std::string widths;
  for (const size_t width : persistent_plan.widths) {
    if (!widths.empty()) {
      widths.append("+");
    }
    widths.append(std::to_string(width));
  }
  ET_LOG(
      Info,
      "EdgeInfer %s persistent KV plan: target_context=%zu widths=%s "
      "physical_blocks=%zu coverage=%zu padding=%zu predicted_%s=%.4f ms.",
      phase,
      persistent_plan.sequence_length,
      widths.c_str(),
      persistent_plan.graph_calls(),
      persistent_plan.coverage(),
      persistent_plan.padding(),
      aggregate_cost ? "prefill_total" : "route",
      persistent_plan.predicted_cost);
  return Error::Ok;
}

Error EdgeInferPromptProcessor::run_chunk(
    const uint64_t* chunk_tokens,
    size_t valid_rows,
    bool dump_logits,
    TensorPtr* logits,
    bool decode) {
  Error error = Error::Ok;
  if (decode) {
    error = run_edgeinfer_decode_cache_transaction(
        layer_cache_view_, sequence_length_, valid_rows, [&]() {
          return run_chunk_graphs(chunk_tokens, valid_rows, logits, true);
        });
  } else {
    error = run_chunk_graphs(chunk_tokens, valid_rows, logits, false);
  }
  if (error != Error::Ok) {
    return error;
  }
  if (dump_logits) {
    const auto* raw =
        reinterpret_cast<const std::byte*>((*logits)->const_data_ptr());
    prompt_all_logits_.insert(
        prompt_all_logits_.end(), raw, raw + (*logits)->nbytes());
  }
  sequence_length_ += valid_rows;
  return Error::Ok;
}

Error EdgeInferPromptProcessor::run_chunk_graphs(
    const uint64_t* chunk_tokens,
    size_t valid_rows,
    TensorPtr* logits,
    bool decode) {
  executorch::extension::Module* module = core_module(decode);
  auto* attention = decode && decode_attention_runner_.has_value()
      ? &*decode_attention_runner_
      : attention_runner_.has_value() ? &*attention_runner_
                                      : nullptr;
  const size_t query_rows = query_rows_for_phase(decode);
  if (chunk_tokens == nullptr || logits == nullptr || valid_rows == 0 ||
      valid_rows > query_rows || layer_caches_.size() != options_.layers ||
      module == nullptr || attention == nullptr) {
    return Error::InvalidArgument;
  }
  if (stage_workspace_ == nullptr || !stage_workspace_->valid()) {
    return Error::InvalidState;
  }
  StageWorkspace::Mode& stage = stage_workspace_->for_decode(decode);
  if (stage.rows != query_rows) {
    return Error::InvalidState;
  }

  std::vector<int64_t> token_values(query_rows, 0);
  for (size_t row = 0; row < valid_rows; ++row) {
    token_values[row] = static_cast<int64_t>(chunk_tokens[row]);
  }
  auto tokens = make_int64_tensor(
      {1, static_cast<SizesType>(query_rows)}, std::move(token_values));
  Error error = execute_into_outputs(
      module,
      "llama_embedding",
      std::vector<EValue>{*tokens},
      std::vector<TensorPtr>{stage.hidden[0]});
  if (error != Error::Ok) {
    return error;
  }
  size_t hidden_bank = 0;
  const size_t causal_query_begin = sequence_length_;
  const size_t rope_width =
      options_.hf_rope ? options_.head_dim : options_.head_dim / 2;
  if (rope_width == 0 ||
      rope_inverse_frequencies_.size() != options_.head_dim / 2) {
    return Error::InvalidState;
  }
  const size_t pairs = options_.head_dim / 2;
  if (scalar_type_ == ScalarType::Float) {
    fill_rope_values(
        stage.rope_cosine->mutable_data_ptr<float>(),
        stage.rope_sine->mutable_data_ptr<float>(),
        query_rows,
        causal_query_begin,
        pairs,
        rope_width,
        options_.hf_rope,
        rope_inverse_frequencies_);
  } else if (scalar_type_ == ScalarType::Half) {
    fill_rope_values(
        stage.rope_cosine->mutable_data_ptr<Half>(),
        stage.rope_sine->mutable_data_ptr<Half>(),
        query_rows,
        causal_query_begin,
        pairs,
        rope_width,
        options_.hf_rope,
        rope_inverse_frequencies_);
  } else {
    return Error::InvalidState;
  }

  for (size_t layer = 0; layer < options_.layers; ++layer) {
    executorch::extension::Module* stage_module = layer_module(decode, layer);
    if (stage_module == nullptr) {
      return Error::InvalidState;
    }
    const std::string prefix = "llama_layer_" + std::to_string(layer);
    TensorPtr& hidden = stage.hidden[hidden_bank];
    if (options_.pre_attention_rope) {
      error = execute_into_outputs(
          stage_module,
          prefix + "_pre",
          std::vector<EValue>{*hidden, *stage.rope_cosine, *stage.rope_sine},
          std::vector<TensorPtr>{stage.query, stage.key, stage.value});
      if (error != Error::Ok) {
        return error;
      }
    } else {
      error = execute_into_outputs(
          stage_module,
          prefix + "_qkv",
          std::vector<EValue>{*hidden},
          std::vector<TensorPtr>{stage.packed_qkv});
      if (error != Error::Ok) {
        return error;
      }
      if (scalar_type_ == ScalarType::Float) {
        unpack_and_rotate_qkv<float>(
            *stage.packed_qkv,
            *stage.query,
            *stage.key,
            *stage.value,
            *stage.rope_cosine,
            *stage.rope_sine,
            query_rows,
            valid_rows,
            options_.query_heads,
            options_.kv_heads,
            options_.head_dim,
            options_.hf_rope);
      } else {
        unpack_and_rotate_qkv<Half>(
            *stage.packed_qkv,
            *stage.query,
            *stage.key,
            *stage.value,
            *stage.rope_cosine,
            *stage.rope_sine,
            query_rows,
            valid_rows,
            options_.query_heads,
            options_.kv_heads,
            options_.head_dim,
            options_.hf_rope);
      }
    }

    if (!layer_caches_[layer]->append(*stage.key, *stage.value, valid_rows)) {
      return Error::InvalidArgument;
    }
    PreparedBlocks prepared;
    if (!layer_caches_[layer]->prepare_blocks(
            query_rows, causal_query_begin, valid_rows, decode, &prepared)) {
      return Error::Internal;
    }
    // Attention is sequential across decoder layers. Reusing one workspace
    // keeps every static method's caller-owned output addresses stable and
    // avoids rebinding QNN graph outputs once per layer.
    auto& workspace = decode ? layer_caches_.front()->decode_workspace
                             : layer_caches_.front()->workspace;
    auto attention_output =
        attention->run_blocks(*stage.query, prepared.blocks, workspace);
    if (!attention_output.ok() ||
        !has_shape(
            **attention_output,
            {1,
             static_cast<SizesType>(options_.query_heads),
             static_cast<SizesType>(query_rows),
             static_cast<SizesType>(options_.head_dim)},
            scalar_type_)) {
      return attention_output.ok() ? Error::InvalidArgument
                                   : attention_output.error();
    }
    const size_t next_hidden_bank = 1 - hidden_bank;
    error = execute_into_outputs(
        stage_module,
        prefix + "_post",
        std::vector<EValue>{*hidden, **attention_output},
        std::vector<TensorPtr>{stage.hidden[next_hidden_bank]});
    if (error != Error::Ok) {
      return error;
    }
    hidden_bank = next_hidden_bank;
    const bool sharded_portfolio = stage_module != module;
    const bool shard_complete = layer + 1 == options_.layers ||
        layer_module(decode, layer + 1) != stage_module;
    if (!decode && sharded_portfolio && shard_complete) {
      unload_composable_module_methods(stage_module);
    }
  }
  error = execute_into_outputs(
      module,
      "llama_output",
      std::vector<EValue>{*stage.hidden[hidden_bank]},
      std::vector<TensorPtr>{stage.logits});
  if (error != Error::Ok) {
    return error;
  }
  *logits = stage.logits;
  return Error::Ok;
}

Error EdgeInferPromptProcessor::prepare_decode(size_t target_context) {
  Error error = ensure_ready();
  if (error != Error::Ok || sequence_length_ == 0) {
    return error == Error::Ok ? Error::InvalidState : error;
  }
  const bool separate_decode_module = has_separate_decode_portfolio();
  if (separate_decode_module) {
    for (auto* module : portfolio_modules(false)) {
      unload_composable_module_methods(module);
    }
    ET_LOG(
        Info,
        "EdgeInfer released Prefill QNN method contexts before switching to "
        "the independent R=1 Decode PTE.");
  }
  error = recompose_for_target(target_context, true);
  if (error != Error::Ok || layer_caches_.empty()) {
    return error;
  }

  const auto initialize_start = std::chrono::steady_clock::now();
  executorch::extension::Module* decode_core = core_module(true);
  if (decode_core == nullptr) {
    return Error::InvalidState;
  }
  std::vector<std::pair<executorch::extension::Module*, std::string>>
      decode_methods{
          {decode_core, "llama_embedding"}, {decode_core, "llama_output"}};
  decode_methods.reserve(
      2 + options_.layers * 2 + layer_caches_.front()->layout().size());
  for (size_t layer = 0; layer < options_.layers; ++layer) {
    auto* stage_module = layer_module(true, layer);
    if (stage_module == nullptr) {
      return Error::InvalidState;
    }
    const std::string prefix = "llama_layer_" + std::to_string(layer);
    decode_methods.emplace_back(
        stage_module, prefix + (options_.pre_attention_rope ? "_pre" : "_qkv"));
    decode_methods.emplace_back(stage_module, prefix + "_post");
  }
  const std::vector<size_t> decode_layout = layer_caches_.front()->layout();
  const size_t decode_query_rows = query_rows_for_phase(true);
  for (size_t index = 0; index < decode_layout.size(); ++index) {
    const size_t width = decode_layout[index];
    decode_methods.emplace_back(
        decode_core,
        executorch::extension::llm::static_attention_method_name(
            index == 0, decode_query_rows, width));
  }
  for (const auto& entry : decode_methods) {
    error = entry.first->load_method(entry.second);
    if (error != Error::Ok) {
      return error;
    }
  }
  ET_LOG(
      Info,
      "EdgeInfer initialized %zu R=%zu Decode methods in %.3f ms before "
      "steady-state token execution.",
      decode_methods.size(),
      decode_query_rows,
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - initialize_start)
          .count());
  return Error::Ok;
}

Error EdgeInferPromptProcessor::bridge_to_native_decode() {
  if (!has_edgeinfer_state()) {
    return Error::InvalidState;
  }
  const auto bridge_start = std::chrono::steady_clock::now();
  const Error error = bridge_kv_cache();
  if (error == Error::Ok) {
    ET_LOG(
        Info,
        "EdgeInfer materialized block-native K/V for native Decode fallback "
        "in %.3f ms.",
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - bridge_start)
            .count());
    release_pre_output_bindings();
    unload_all_composable_modules();
  }
  return error;
}

Result<uint64_t> EdgeInferPromptProcessor::decode_token(
    uint64_t token,
    int64_t position,
    bool dump_logits) {
  if (position < 0 || static_cast<size_t>(position) != sequence_length_) {
    return Error::InvalidArgument;
  }
  TensorPtr logits;
  const Error error = run_chunk(&token, 1, dump_logits, &logits, true);
  if (error != Error::Ok) {
    return error;
  }
  return static_cast<uint64_t>(decoder_runner_->logits_to_token(*logits, 0));
}

Result<uint64_t> EdgeInferPromptProcessor::prefill(
    std::vector<uint64_t> prompt_tokens,
    int64_t start_pos,
    bool dump_logits,
    AttentionSinkRopeRunner* attention_sink_rope_runner) {
  // Attention sink mutates the native cache with model-specific RoPE state.
  // Retain its established decoder path until the split graph exports the
  // corresponding eviction contract.
  if (attention_sink_rope_runner != nullptr) {
    return PromptProcessor::prefill(
        std::move(prompt_tokens),
        start_pos,
        dump_logits,
        attention_sink_rope_runner);
  }
  const Error ready_error = ensure_ready();
  if (ready_error == Error::Ok && start_pos >= 0 &&
      sequence_length_ == static_cast<size_t>(start_pos)) {
    const size_t checkpoint = sequence_length_;
    const size_t logits_checkpoint = prompt_all_logits_.size();
    return edgeinfer::run_prefill_with_native_fallback(
        layer_cache_view_,
        checkpoint,
        prompt_tokens.size(),
        [&]() -> Result<uint64_t> {
          uint64_t next_token = 0;
          const Error error = run_edgeinfer_prefill(
              prompt_tokens, start_pos, dump_logits, &next_token);
          if (error != Error::Ok) {
            return error;
          }
          ET_LOG(
              Info,
              "EdgeInfer Prefill completed through split "
              "QKV/Attention/Post graphs: tokens=%zu context=%zu.",
              prompt_tokens.size(),
              sequence_length_);
          return next_token;
        },
        [&](Error error) {
          sequence_length_ = checkpoint;
          prompt_all_logits_.resize(logits_checkpoint);
          ET_LOG(
              Info,
              "EdgeInfer Prefill unavailable (0x%x); restored %zu historical "
              "K/V entries before decoder-PTE fallback.",
              static_cast<unsigned int>(error),
              checkpoint);
        },
        [&]() { return bridge_to_native_decode(); },
        [&]() { clear_local_state(); },
        [&]() {
          return PromptProcessor::prefill(
              std::move(prompt_tokens),
              start_pos,
              dump_logits,
              attention_sink_rope_runner);
        });
  }

  ET_LOG(
      Info,
      "EdgeInfer Prefill unavailable before cache mutation (0x%x); preserving "
      "the decoder-PTE baseline for this request.",
      static_cast<unsigned int>(
          ready_error == Error::Ok ? Error::InvalidState : ready_error));
  return edgeinfer::dispatch_prefill_after_preflight(
      ready_error == Error::Ok ? Error::InvalidState : ready_error,
      []() -> Result<uint64_t> { return Error::InvalidState; },
      [&]() { clear_local_state(); },
      [&]() {
        return PromptProcessor::prefill(
            std::move(prompt_tokens),
            start_pos,
            dump_logits,
            attention_sink_rope_runner);
      });
}

} // namespace example
