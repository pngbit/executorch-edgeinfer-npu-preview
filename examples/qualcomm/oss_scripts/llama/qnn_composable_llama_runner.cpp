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
 * Runs a Llama model as fixed-shape stages plus host-composed Attention. The
 * host owns growing per-layer K/V state; no exported method contains a maximum
 * context length. A finite QNN Attention portfolio is profiled once and its
 * merge methods are then reused for every requested token position.
 */

#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/extension/llm/runner/composable_attention_runner.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
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
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

DEFINE_string(
    model_path,
    "composable_llama_qnn.pte",
    "Composable Llama core PTE, or the complete PTE in single-file mode.");
DEFINE_string(
    layer_model_paths,
    "",
    "Comma-separated QKV/post layer-shard PTEs in ascending layer order.");
DEFINE_uint32(
    layers_per_shard,
    0,
    "Number of contiguous Transformer layers in each layer-shard PTE. Zero "
    "selects single-file mode and requires layer_model_paths to be empty.");
DEFINE_string(
    output_path,
    "composable_llama_logits.bin",
    "Raw FP32 logits for every input token.");
DEFINE_bool(
    output_all_logits,
    true,
    "Write logits for every position. If false, retain only the final position.");
DEFINE_string(
    debug_output_prefix,
    "",
    "Optional prefix for first-token intermediate FP32 tensors.");
DEFINE_uint32(sequence_length, 160, "Number of deterministic tokens to run.");
DEFINE_bool(
    layer_major_prefill,
    false,
    "Process a known prompt one Transformer layer at a time and unload each "
    "layer's QKV/post methods after use. This bounds live QNN method state for "
    "deep models while preserving causal prefill dependencies.");
DEFINE_uint32(token_seed, 11, "First deterministic token before modulo.");
DEFINE_uint32(token_stride, 37, "Increment between deterministic tokens.");
DEFINE_uint32(layers, 5, "Number of Transformer layers.");
DEFINE_uint32(dim, 64, "Transformer hidden dimension.");
DEFINE_uint32(query_heads, 8, "Number of query heads.");
DEFINE_uint32(kv_heads, 4, "Number of key/value heads.");
DEFINE_uint32(head_dim, 8, "Attention head dimension.");
DEFINE_uint32(vocab_size, 512, "Vocabulary size.");
DEFINE_double(rope_theta, 10000.0, "RoPE frequency base.");
DEFINE_string(rope_style, "llama", "RoPE layout: llama or hf.");
DEFINE_string(widths, "16,32,64", "Comma-separated static K/V graph widths.");
DEFINE_uint32(warmup, 1, "Warm-up executions per Attention method.");
DEFINE_uint32(iterations, 3, "Timed executions per Attention method.");
DEFINE_uint32(
    fixed_width,
    0,
    "Optional Fixed-Max Attention width for an end-to-end comparison.");
DEFINE_uint32(
    sequence_warmup,
    1,
    "Warm-up sequences for each end-to-end design.");
DEFINE_uint32(
    sequence_iterations,
    5,
    "Timed sequences for each end-to-end design.");

namespace {

using executorch::aten::ScalarType;
using executorch::aten::SizesType;
using executorch::aten::Tensor;
using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::extension::llm::ComposableAttentionRunner;
using executorch::extension::llm::PreparedAttentionBlock;
using executorch::extension::llm::StaticAttentionGraphCost;
using executorch::extension::llm::StaticAttentionPlan;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::Result;

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

std::vector<std::string> parse_paths(const std::string& values) {
  std::vector<std::string> result;
  if (values.empty()) {
    return result;
  }
  std::stringstream stream(values);
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (token.empty()) {
      return {};
    }
    result.emplace_back(std::move(token));
  }
  return result;
}

bool fits_tensor_size(size_t value) {
  return value <= static_cast<size_t>(std::numeric_limits<SizesType>::max());
}

bool is_power_of_two(size_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

TensorPtr make_float_tensor(
    std::vector<SizesType> sizes,
    std::vector<float> values) {
  return executorch::extension::make_tensor_ptr(
      std::move(sizes),
      std::move(values),
      {},
      {},
      ScalarType::Float,
      executorch::aten::TensorShapeDynamism::STATIC);
}

Result<TensorPtr> clone_contiguous_tensor(const Tensor& tensor) {
  if (!executorch::runtime::tensor_is_contiguous(tensor)) {
    return Error::InvalidArgument;
  }
  std::vector<SizesType> sizes(tensor.sizes().begin(), tensor.sizes().end());
  std::vector<uint8_t> data(tensor.nbytes());
  if (!data.empty()) {
    std::memcpy(data.data(), tensor.const_data_ptr(), data.size());
  }
  return executorch::extension::make_tensor_ptr(
      std::move(sizes),
      std::move(data),
      tensor.scalar_type(),
      executorch::aten::TensorShapeDynamism::STATIC);
}

Result<TensorPtr> execute_single_output(
    Module& module,
    const std::string& method,
    const std::vector<EValue>& inputs) {
  auto outputs = module.execute(method, inputs);
  if (!outputs.ok()) {
    ET_LOG(
        Error,
        "Method '%s' failed: 0x%x",
        method.c_str(),
        static_cast<unsigned int>(outputs.error()));
    return outputs.error();
  }
  if (outputs->size() != 1 || !outputs->front().isTensor()) {
    ET_LOG(Error, "Method '%s' returned an invalid output.", method.c_str());
    return Error::InvalidArgument;
  }
  return clone_contiguous_tensor(outputs->front().toTensor());
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

ProfileInputs make_profile_inputs(size_t width, bool merge) {
  const size_t query_elements = FLAGS_query_heads * FLAGS_head_dim;
  auto q = make_float_tensor(
      {1,
       static_cast<SizesType>(FLAGS_query_heads),
       1,
       static_cast<SizesType>(FLAGS_head_dim)},
      deterministic_values(query_elements, 0.017f));
  auto k = make_float_tensor(
      {1,
       static_cast<SizesType>(FLAGS_kv_heads),
       static_cast<SizesType>(FLAGS_head_dim),
       static_cast<SizesType>(width)},
      deterministic_values(FLAGS_kv_heads * FLAGS_head_dim * width, 0.013f));
  auto v = make_float_tensor(
      {1,
       static_cast<SizesType>(FLAGS_kv_heads),
       static_cast<SizesType>(width),
       static_cast<SizesType>(FLAGS_head_dim)},
      deterministic_values(FLAGS_kv_heads * width * FLAGS_head_dim, 0.011f));
  auto visibility = make_float_tensor(
      {1, 1, 1, static_cast<SizesType>(width)},
      std::vector<float>(width, 1.0f));

  ProfileInputs inputs;
  inputs.tensors.reserve(merge ? 7 : 4);
  inputs.tensors.emplace_back(std::move(q));
  inputs.tensors.emplace_back(std::move(k));
  inputs.tensors.emplace_back(std::move(v));
  inputs.tensors.emplace_back(std::move(visibility));
  if (merge) {
    inputs.tensors.emplace_back(make_float_tensor(
        {1, static_cast<SizesType>(FLAGS_query_heads), 1, 1},
        std::vector<float>(FLAGS_query_heads, -1.0f)));
    inputs.tensors.emplace_back(make_float_tensor(
        {1, static_cast<SizesType>(FLAGS_query_heads), 1, 1},
        std::vector<float>(FLAGS_query_heads, 1.0f)));
    inputs.tensors.emplace_back(make_float_tensor(
        {1,
         static_cast<SizesType>(FLAGS_query_heads),
         1,
         static_cast<SizesType>(FLAGS_head_dim)},
        deterministic_values(query_elements, 0.007f)));
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
  // Every exported method is executed once before timing so lazy QNN setup is
  // never charged to the first candidate in the planner's cost portfolio.
  auto priming_outputs = module.execute(method, inputs);
  if (!priming_outputs.ok()) {
    ET_LOG(
        Error,
        "Priming failed for method '%s': 0x%x",
        method.c_str(),
        static_cast<unsigned int>(priming_outputs.error()));
    return -1.0;
  }
  for (size_t index = 0; index < FLAGS_warmup; ++index) {
    auto outputs = module.execute(method, inputs);
    if (!outputs.ok()) {
      ET_LOG(
          Error,
          "Warm-up failed for method '%s': 0x%x",
          method.c_str(),
          static_cast<unsigned int>(outputs.error()));
      return -1.0;
    }
  }

  const auto start = std::chrono::steady_clock::now();
  for (size_t index = 0; index < FLAGS_iterations; ++index) {
    auto outputs = module.execute(method, inputs);
    if (!outputs.ok()) {
      ET_LOG(
          Error,
          "Timed execution failed for method '%s': 0x%x",
          method.c_str(),
          static_cast<unsigned int>(outputs.error()));
      return -1.0;
    }
  }
  const double elapsed_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - start)
                                .count();
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
            true, 1, width);
    const std::string merge_name =
        executorch::extension::llm::static_attention_method_name(
            false, 1, width);
    auto first_inputs = make_profile_inputs(width, false);
    auto merge_inputs = make_profile_inputs(width, true);
    const double first_ms =
        profile_method(module, first_name, first_inputs.values);
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

struct LayerCache {
  struct Block {
    size_t width;
    size_t valid_width;
    TensorPtr key;
    TensorPtr value;
    TensorPtr visibility;
  };

  std::vector<Block> blocks;

  Block make_block(size_t width) const {
    return {
        width,
        0,
        make_float_tensor(
            {1,
             static_cast<SizesType>(FLAGS_kv_heads),
             static_cast<SizesType>(FLAGS_head_dim),
             static_cast<SizesType>(width)},
            std::vector<float>(FLAGS_kv_heads * FLAGS_head_dim * width, 0.0f)),
        make_float_tensor(
            {1,
             static_cast<SizesType>(FLAGS_kv_heads),
             static_cast<SizesType>(width),
             static_cast<SizesType>(FLAGS_head_dim)},
            std::vector<float>(FLAGS_kv_heads * width * FLAGS_head_dim, 0.0f)),
        make_float_tensor(
            {1, 1, 1, static_cast<SizesType>(width)},
            std::vector<float>(width, 0.0f))};
  }

  std::vector<size_t> layout() const {
    std::vector<size_t> result;
    result.reserve(blocks.size());
    for (const Block& block : blocks) {
      result.push_back(block.width);
    }
    return result;
  }

  size_t valid_width() const {
    size_t result = 0;
    for (const Block& block : blocks) {
      result += block.valid_width;
    }
    return result;
  }

  bool recompose(const StaticAttentionPlan& plan) {
    if (plan.widths.empty() || plan.coverage() < valid_width()) {
      return false;
    }
    if (layout() == plan.widths) {
      return true;
    }
    std::vector<Block> replacement;
    replacement.reserve(plan.widths.size());
    for (const size_t width : plan.widths) {
      replacement.emplace_back(make_block(width));
    }

    size_t source_index = 0;
    size_t source_offset = 0;
    size_t destination_index = 0;
    size_t destination_offset = 0;
    size_t remaining = valid_width();
    while (remaining > 0) {
      while (source_index < blocks.size() &&
             source_offset == blocks[source_index].valid_width) {
        ++source_index;
        source_offset = 0;
      }
      while (destination_index < replacement.size() &&
             destination_offset == replacement[destination_index].width) {
        ++destination_index;
        destination_offset = 0;
      }
      if (source_index == blocks.size() ||
          destination_index == replacement.size()) {
        return false;
      }
      const Block& source = blocks[source_index];
      Block& destination = replacement[destination_index];
      const size_t count = std::min(
          source.valid_width - source_offset,
          destination.width - destination_offset);
      const size_t key_rows = FLAGS_kv_heads * FLAGS_head_dim;
      const float* source_key = source.key->const_data_ptr<float>();
      float* destination_key = destination.key->mutable_data_ptr<float>();
      for (size_t row = 0; row < key_rows; ++row) {
        std::memcpy(
            destination_key + row * destination.width + destination_offset,
            source_key + row * source.width + source_offset,
            count * sizeof(float));
      }
      const float* source_value = source.value->const_data_ptr<float>();
      float* destination_value = destination.value->mutable_data_ptr<float>();
      for (size_t head = 0; head < FLAGS_kv_heads; ++head) {
        std::memcpy(
            destination_value +
                (head * destination.width + destination_offset) *
                    FLAGS_head_dim,
            source_value +
                (head * source.width + source_offset) * FLAGS_head_dim,
            count * FLAGS_head_dim * sizeof(float));
      }
      std::memcpy(
          destination.visibility->mutable_data_ptr<float>() +
              destination_offset,
          source.visibility->const_data_ptr<float>() + source_offset,
          count * sizeof(float));
      destination.valid_width += count;
      source_offset += count;
      destination_offset += count;
      remaining -= count;
    }
    blocks = std::move(replacement);
    return true;
  }

  bool append(
      const Tensor& key,
      const Tensor& value,
      const StaticAttentionPlan& plan) {
    if (key.scalar_type() != ScalarType::Float ||
        value.scalar_type() != ScalarType::Float ||
        !executorch::runtime::tensor_is_contiguous(key) ||
        !executorch::runtime::tensor_is_contiguous(value) || key.dim() != 4 ||
        value.dim() != 4 || key.size(0) != 1 || value.size(0) != 1 ||
        key.size(1) != static_cast<SizesType>(FLAGS_kv_heads) ||
        value.size(1) != static_cast<SizesType>(FLAGS_kv_heads) ||
        key.size(2) != static_cast<SizesType>(FLAGS_head_dim) ||
        key.size(3) != 1 || value.size(2) != 1 ||
        value.size(3) != static_cast<SizesType>(FLAGS_head_dim) ||
        !recompose(plan)) {
      return false;
    }
    auto tail =
        std::find_if(blocks.begin(), blocks.end(), [](const Block& block) {
          return block.valid_width < block.width;
        });
    if (tail == blocks.end()) {
      return false;
    }
    const size_t offset = tail->valid_width;
    const float* key_data = key.const_data_ptr<float>();
    const float* value_data = value.const_data_ptr<float>();
    float* cached_key = tail->key->mutable_data_ptr<float>();
    float* cached_value = tail->value->mutable_data_ptr<float>();
    for (size_t head = 0; head < FLAGS_kv_heads; ++head) {
      for (size_t dim = 0; dim < FLAGS_head_dim; ++dim) {
        const size_t source = head * FLAGS_head_dim + dim;
        cached_key[source * tail->width + offset] = key_data[source];
        cached_value[(head * tail->width + offset) * FLAGS_head_dim + dim] =
            value_data[source];
      }
    }
    tail->visibility->mutable_data_ptr<float>()[offset] = 1.0f;
    ++tail->valid_width;
    return true;
  }

  std::vector<PreparedAttentionBlock> prepared_blocks() const {
    std::vector<PreparedAttentionBlock> result;
    result.reserve(blocks.size());
    for (const Block& block : blocks) {
      result.push_back(
          {block.key.get(), block.value.get(), block.visibility.get()});
    }
    return result;
  }
};

Result<std::array<TensorPtr, 3>> unpack_qkv(const Tensor& packed) {
  const size_t query_features = FLAGS_query_heads * FLAGS_head_dim;
  const size_t kv_features = FLAGS_kv_heads * FLAGS_head_dim;
  const size_t total_features = query_features + 2 * kv_features;
  if (packed.scalar_type() != ScalarType::Float ||
      !executorch::runtime::tensor_is_contiguous(packed) || packed.dim() != 3 ||
      packed.size(0) != 1 || packed.size(1) != 1 ||
      packed.size(2) != static_cast<SizesType>(total_features)) {
    return Error::InvalidArgument;
  }
  const float* data = packed.const_data_ptr<float>();
  std::vector<float> query(data, data + query_features);
  std::vector<float> key(
      data + query_features, data + query_features + kv_features);
  std::vector<float> value(
      data + query_features + kv_features, data + total_features);
  return std::array<TensorPtr, 3>{
      make_float_tensor(
          {1,
           static_cast<SizesType>(FLAGS_query_heads),
           1,
           static_cast<SizesType>(FLAGS_head_dim)},
          std::move(query)),
      make_float_tensor(
          {1,
           static_cast<SizesType>(FLAGS_kv_heads),
           static_cast<SizesType>(FLAGS_head_dim),
           1},
          std::move(key)),
      make_float_tensor(
          {1,
           static_cast<SizesType>(FLAGS_kv_heads),
           1,
           static_cast<SizesType>(FLAGS_head_dim)},
          std::move(value))};
}

Result<std::pair<TensorPtr, TensorPtr>>
apply_rope(const Tensor& query, const Tensor& key, size_t position) {
  auto query_clone = clone_contiguous_tensor(query);
  auto key_clone = clone_contiguous_tensor(key);
  if (!query_clone.ok() || !key_clone.ok() ||
      query.scalar_type() != ScalarType::Float ||
      key.scalar_type() != ScalarType::Float) {
    return Error::InvalidArgument;
  }
  const size_t pairs = FLAGS_head_dim / 2;
  std::vector<float> cosine(pairs);
  std::vector<float> sine(pairs);
  for (size_t pair = 0; pair < pairs; ++pair) {
    const double exponent =
        static_cast<double>(pair * 2) / static_cast<double>(FLAGS_head_dim);
    const double frequency = 1.0 / std::pow(FLAGS_rope_theta, exponent);
    const double angle = static_cast<double>(position) * frequency;
    cosine[pair] = static_cast<float>(std::cos(angle));
    sine[pair] = static_cast<float>(std::sin(angle));
  }

  auto rotate_llama = [&](float* data, size_t heads) {
    for (size_t head = 0; head < heads; ++head) {
      for (size_t pair = 0; pair < pairs; ++pair) {
        const size_t real_index = head * FLAGS_head_dim + pair * 2;
        const size_t imaginary_index = real_index + 1;
        const float real = data[real_index];
        const float imaginary = data[imaginary_index];
        data[real_index] = real * cosine[pair] - imaginary * sine[pair];
        data[imaginary_index] = real * sine[pair] + imaginary * cosine[pair];
      }
    }
  };
  auto rotate_hf = [&](float* data, size_t heads) {
    for (size_t head = 0; head < heads; ++head) {
      const size_t offset = head * FLAGS_head_dim;
      for (size_t pair = 0; pair < pairs; ++pair) {
        const size_t first_index = offset + pair;
        const size_t second_index = offset + pairs + pair;
        const float first = data[first_index];
        const float second = data[second_index];
        data[first_index] = first * cosine[pair] - second * sine[pair];
        data[second_index] = second * cosine[pair] + first * sine[pair];
      }
    }
  };
  if (FLAGS_rope_style == "hf") {
    rotate_hf((*query_clone)->mutable_data_ptr<float>(), FLAGS_query_heads);
    rotate_hf((*key_clone)->mutable_data_ptr<float>(), FLAGS_kv_heads);
  } else {
    rotate_llama((*query_clone)->mutable_data_ptr<float>(), FLAGS_query_heads);
    rotate_llama((*key_clone)->mutable_data_ptr<float>(), FLAGS_kv_heads);
  }
  return std::make_pair(std::move(*query_clone), std::move(*key_clone));
}

bool has_shape(
    const Tensor& tensor,
    const std::vector<SizesType>& expected_sizes) {
  if (tensor.scalar_type() != ScalarType::Float ||
      !executorch::runtime::tensor_is_contiguous(tensor) ||
      tensor.dim() != expected_sizes.size()) {
    return false;
  }
  return std::equal(
      expected_sizes.begin(), expected_sizes.end(), tensor.sizes().begin());
}

bool dump_debug_tensor(const std::string& name, const Tensor& tensor) {
  if (FLAGS_debug_output_prefix.empty()) {
    return true;
  }
  if (tensor.scalar_type() != ScalarType::Float ||
      !executorch::runtime::tensor_is_contiguous(tensor)) {
    ET_LOG(Error, "Debug tensor '%s' is not contiguous FP32.", name.c_str());
    return false;
  }
  const std::string path = FLAGS_debug_output_prefix + name + ".bin";
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    ET_LOG(Error, "Could not open debug output '%s'.", path.c_str());
    return false;
  }
  output.write(
      static_cast<const char*>(tensor.const_data_ptr()),
      static_cast<std::streamsize>(tensor.nbytes()));
  return static_cast<bool>(output);
}

std::vector<int64_t> make_tokens() {
  std::vector<int64_t> tokens(FLAGS_sequence_length);
  for (size_t position = 0; position < tokens.size(); ++position) {
    const uint64_t value = static_cast<uint64_t>(FLAGS_token_seed) +
        static_cast<uint64_t>(position) * FLAGS_token_stride;
    tokens[position] = static_cast<int64_t>(value % FLAGS_vocab_size);
  }
  return tokens;
}

class ComposableLlamaModules final {
 public:
  ComposableLlamaModules(
      const std::string& core_path,
      const std::vector<std::string>& layer_paths,
      size_t layers_per_shard)
      : core_(core_path, Module::LoadMode::MmapUseMlockIgnoreErrors),
        layers_per_shard_(layers_per_shard) {
    layer_shards_.reserve(layer_paths.size());
    for (const std::string& path : layer_paths) {
      layer_shards_.emplace_back(std::make_unique<Module>(
          path, Module::LoadMode::MmapUseMlockIgnoreErrors));
    }
  }

  Module& core() {
    return core_;
  }

  Module& layer(size_t layer_index) {
    if (layer_shards_.empty()) {
      return core_;
    }
    return *layer_shards_[layer_index / layers_per_shard_];
  }

 private:
  Module core_;
  std::vector<std::unique_ptr<Module>> layer_shards_;
  size_t layers_per_shard_;
};

Result<TensorPtr> run_embedding(
    Module& core,
    int64_t token_value,
    size_t position,
    bool emit_observability) {
  auto token = executorch::extension::make_tensor_ptr(
      {1, 1},
      std::vector<int64_t>{token_value},
      {},
      {},
      ScalarType::Long,
      executorch::aten::TensorShapeDynamism::STATIC);
  auto hidden = execute_single_output(
      core, "llama_embedding", std::vector<EValue>{*token});
  if (!hidden.ok() ||
      !has_shape(**hidden, {1, 1, static_cast<SizesType>(FLAGS_dim)})) {
    ET_LOG(Error, "Invalid embedding output at position %zu.", position);
    return Error::InvalidArgument;
  }
  if (emit_observability && position == 0 &&
      !dump_debug_tensor("embedding", **hidden)) {
    return Error::InvalidArgument;
  }
  return hidden;
}

Result<TensorPtr> run_layer_position(
    Module& layer_module,
    ComposableAttentionRunner& attention,
    LayerCache& cache,
    const Tensor& hidden,
    size_t position,
    size_t layer,
    bool emit_observability) {
  const std::string prefix = "llama_layer_" + std::to_string(layer);
  auto qkv_outputs =
      layer_module.execute(prefix + "_qkv", std::vector<EValue>{hidden});
  if (!qkv_outputs.ok() || qkv_outputs->size() != 1 ||
      !qkv_outputs->front().isTensor()) {
    ET_LOG(Error, "Invalid Q/K/V output at layer %zu.", layer);
    return Error::InvalidArgument;
  }
  auto qkv = unpack_qkv(qkv_outputs->front().toTensor());
  if (!qkv.ok()) {
    ET_LOG(Error, "Unexpected packed Q/K/V shape at layer %zu.", layer);
    return Error::InvalidArgument;
  }
  auto rotated = apply_rope(*(*qkv)[0], *(*qkv)[1], position);
  auto plan = attention.plan(position + 1);
  const bool layout_changed = plan.ok() && cache.layout() != plan->widths;
  if (!rotated.ok() || !plan.ok() ||
      !cache.append(*rotated->second, *(*qkv)[2], *plan)) {
    ET_LOG(Error, "Could not rotate or cache Q/K/V at layer %zu.", layer);
    return Error::InvalidArgument;
  }
  if (emit_observability && layer == 0 &&
      (layout_changed || position < 3 || ((position + 1) & position) == 0 ||
       position + 1 == FLAGS_sequence_length)) {
    std::string widths;
    for (const size_t width : plan->widths) {
      if (!widths.empty()) {
        widths += "+";
      }
      widths += std::to_string(width);
    }
    ET_LOG(
        Info,
        "Planner context=%zu blocks=%s calls=%zu predicted=%.4f ms "
        "padding=%zu layout_changed=%s",
        position + 1,
        widths.c_str(),
        plan->graph_calls(),
        plan->predicted_cost,
        plan->padding(),
        layout_changed ? "true" : "false");
  }
  if (emit_observability && position == 0 &&
      (!dump_debug_tensor(prefix + "_q", *rotated->first) ||
       !dump_debug_tensor(prefix + "_k", *rotated->second) ||
       !dump_debug_tensor(prefix + "_v", *(*qkv)[2]))) {
    return Error::InvalidArgument;
  }
  const auto prepared_blocks = cache.prepared_blocks();
  auto attention_output =
      attention.run_blocks(*rotated->first, prepared_blocks);
  if (!attention_output.ok()) {
    ET_LOG(
        Error,
        "Attention failed at position %zu, layer %zu: 0x%x",
        position,
        layer,
        static_cast<unsigned int>(attention_output.error()));
    return attention_output.error();
  }
  if (emit_observability && position == 0 &&
      !dump_debug_tensor(prefix + "_attention", **attention_output)) {
    return Error::InvalidArgument;
  }
  auto post = execute_single_output(
      layer_module,
      prefix + "_post",
      std::vector<EValue>{hidden, **attention_output});
  if (!post.ok() ||
      !has_shape(**post, {1, 1, static_cast<SizesType>(FLAGS_dim)})) {
    ET_LOG(Error, "Invalid post output at layer %zu.", layer);
    return Error::InvalidArgument;
  }
  if (emit_observability && position == 0 &&
      !dump_debug_tensor(prefix + "_post", **post)) {
    return Error::InvalidArgument;
  }
  return post;
}

bool append_logits(
    Module& core,
    const Tensor& hidden,
    const std::vector<int64_t>& tokens,
    size_t position,
    std::vector<float>& all_logits,
    bool emit_observability) {
  auto logits =
      execute_single_output(core, "llama_output", std::vector<EValue>{hidden});
  if (!logits.ok() ||
      !has_shape(**logits, {1, 1, static_cast<SizesType>(FLAGS_vocab_size)})) {
    ET_LOG(Error, "Invalid logits at position %zu.", position);
    return false;
  }
  if (emit_observability && position == 0 &&
      !dump_debug_tensor("logits", **logits)) {
    return false;
  }
  const float* logits_data = (*logits)->const_data_ptr<float>();
  if (!FLAGS_output_all_logits) {
    all_logits.clear();
  }
  all_logits.insert(
      all_logits.end(), logits_data, logits_data + FLAGS_vocab_size);
  const size_t argmax = static_cast<size_t>(std::distance(
      logits_data,
      std::max_element(logits_data, logits_data + FLAGS_vocab_size)));
  if (emit_observability &&
      (position < 3 || position + 1 == tokens.size() ||
       (position + 1) % 100 == 0)) {
    ET_LOG(
        Info,
        "Position=%zu token=%lld argmax=%zu context=%zu",
        position,
        static_cast<long long>(tokens[position]),
        argmax,
        position + 1);
  }
  return true;
}

bool run_sequence_token_major(
    ComposableLlamaModules& modules,
    ComposableAttentionRunner& attention,
    const std::vector<int64_t>& tokens,
    std::vector<float>& all_logits,
    bool emit_observability = true) {
  std::vector<LayerCache> caches;
  caches.reserve(FLAGS_layers);
  for (size_t layer = 0; layer < FLAGS_layers; ++layer) {
    caches.emplace_back();
  }
  all_logits.clear();
  all_logits.reserve(
      (FLAGS_output_all_logits ? tokens.size() : size_t{1}) * FLAGS_vocab_size);

  for (size_t position = 0; position < tokens.size(); ++position) {
    auto hidden_result = run_embedding(
        modules.core(), tokens[position], position, emit_observability);
    if (!hidden_result.ok()) {
      return false;
    }
    TensorPtr hidden = std::move(*hidden_result);
    for (size_t layer = 0; layer < FLAGS_layers; ++layer) {
      Module& layer_module = modules.layer(layer);
      auto post_result = run_layer_position(
          layer_module,
          attention,
          caches[layer],
          *hidden,
          position,
          layer,
          emit_observability);
      if (!post_result.ok()) {
        return false;
      }
      hidden = std::move(*post_result);
    }
    if (!append_logits(
            modules.core(),
            *hidden,
            tokens,
            position,
            all_logits,
            emit_observability)) {
      return false;
    }
  }
  return true;
}

bool run_sequence_layer_major(
    ComposableLlamaModules& modules,
    ComposableAttentionRunner& attention,
    const std::vector<int64_t>& tokens,
    std::vector<float>& all_logits,
    bool emit_observability) {
  all_logits.clear();
  all_logits.reserve(
      (FLAGS_output_all_logits ? tokens.size() : size_t{1}) * FLAGS_vocab_size);

  std::vector<TensorPtr> hidden_states;
  hidden_states.reserve(tokens.size());
  for (size_t position = 0; position < tokens.size(); ++position) {
    auto hidden = run_embedding(
        modules.core(), tokens[position], position, emit_observability);
    if (!hidden.ok()) {
      return false;
    }
    hidden_states.emplace_back(std::move(*hidden));
  }

  for (size_t layer = 0; layer < FLAGS_layers; ++layer) {
    Module& layer_module = modules.layer(layer);
    LayerCache cache;
    for (size_t position = 0; position < tokens.size(); ++position) {
      auto post = run_layer_position(
          layer_module,
          attention,
          cache,
          *hidden_states[position],
          position,
          layer,
          emit_observability);
      if (!post.ok()) {
        return false;
      }
      hidden_states[position] = std::move(*post);
    }
    const std::string prefix = "llama_layer_" + std::to_string(layer);
    const bool unloaded_qkv = layer_module.unload_method(prefix + "_qkv");
    const bool unloaded_post = layer_module.unload_method(prefix + "_post");
    if (!unloaded_qkv || !unloaded_post) {
      ET_LOG(Error, "Could not unload methods for layer %zu.", layer);
      return false;
    }
  }

  for (size_t position = 0; position < tokens.size(); ++position) {
    if (!append_logits(
            modules.core(),
            *hidden_states[position],
            tokens,
            position,
            all_logits,
            emit_observability)) {
      return false;
    }
  }
  return true;
}

bool run_sequence(
    ComposableLlamaModules& modules,
    ComposableAttentionRunner& attention,
    const std::vector<int64_t>& tokens,
    std::vector<float>& all_logits,
    bool emit_observability = true) {
  if (FLAGS_layer_major_prefill) {
    return run_sequence_layer_major(
        modules, attention, tokens, all_logits, emit_observability);
  }
  return run_sequence_token_major(
      modules, attention, tokens, all_logits, emit_observability);
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

bool time_sequence(
    ComposableLlamaModules& modules,
    ComposableAttentionRunner& attention,
    const std::vector<int64_t>& tokens,
    std::vector<float>& logits,
    double& elapsed_ms) {
  const auto start = std::chrono::steady_clock::now();
  const bool passed = run_sequence(modules, attention, tokens, logits, false);
  elapsed_ms = std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - start)
                   .count();
  return passed;
}

struct LogitComparisonStats {
  double finite_maximum_error{0.0};
  double finite_mean_error{0.0};
  size_t finite_comparison_count{0};
  size_t composed_nan_count{0};
  size_t composed_positive_infinity_count{0};
  size_t composed_negative_infinity_count{0};
  size_t fixed_nan_count{0};
  size_t fixed_positive_infinity_count{0};
  size_t fixed_negative_infinity_count{0};
  size_t argmax_comparison_count{0};
  size_t matching_argmax_count{0};
  size_t mismatching_argmax_count{0};
  size_t uncomparable_argmax_count{0};

  double argmax_agreement() const {
    return argmax_comparison_count == 0
        ? 0.0
        : static_cast<double>(matching_argmax_count) /
            static_cast<double>(argmax_comparison_count);
  }
};

void record_nonfinite_logit(
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

bool compare_logits(
    const std::vector<float>& composed,
    const std::vector<float>& fixed,
    LogitComparisonStats& result) {
  if (composed.empty() || composed.size() != fixed.size() ||
      composed.size() % FLAGS_vocab_size != 0) {
    return false;
  }
  result = {};
  double error_sum = 0.0;
  const size_t positions = composed.size() / FLAGS_vocab_size;
  for (size_t position = 0; position < positions; ++position) {
    const size_t offset = position * FLAGS_vocab_size;
    size_t composed_argmax = 0;
    size_t fixed_argmax = 0;
    bool composed_argmax_valid = false;
    bool fixed_argmax_valid = false;
    bool position_has_nonfinite = false;
    for (size_t token = 0; token < FLAGS_vocab_size; ++token) {
      const size_t index = offset + token;
      const double composed_value = composed[index];
      const double fixed_value = fixed[index];
      if (std::isfinite(composed_value)) {
        if (!composed_argmax_valid ||
            composed_value > composed[offset + composed_argmax]) {
          composed_argmax = token;
          composed_argmax_valid = true;
        }
      } else {
        position_has_nonfinite = true;
        record_nonfinite_logit(
            composed_value,
            result.composed_nan_count,
            result.composed_positive_infinity_count,
            result.composed_negative_infinity_count);
      }
      if (std::isfinite(fixed_value)) {
        if (!fixed_argmax_valid || fixed_value > fixed[offset + fixed_argmax]) {
          fixed_argmax = token;
          fixed_argmax_valid = true;
        }
      } else {
        position_has_nonfinite = true;
        record_nonfinite_logit(
            fixed_value,
            result.fixed_nan_count,
            result.fixed_positive_infinity_count,
            result.fixed_negative_infinity_count);
      }
      if (std::isfinite(composed_value) && std::isfinite(fixed_value)) {
        const double error = std::abs(composed_value - fixed_value);
        result.finite_maximum_error =
            std::max(result.finite_maximum_error, error);
        error_sum += error;
        ++result.finite_comparison_count;
      }
    }
    if (position_has_nonfinite || !composed_argmax_valid ||
        !fixed_argmax_valid) {
      ++result.uncomparable_argmax_count;
    } else {
      ++result.argmax_comparison_count;
      if (composed_argmax == fixed_argmax) {
        ++result.matching_argmax_count;
      } else {
        ++result.mismatching_argmax_count;
      }
    }
  }
  if (result.finite_comparison_count == 0) {
    result.finite_maximum_error = std::numeric_limits<double>::infinity();
    result.finite_mean_error = std::numeric_limits<double>::infinity();
  } else {
    result.finite_mean_error =
        error_sum / static_cast<double>(result.finite_comparison_count);
  }
  return true;
}

bool benchmark_sequences(
    ComposableLlamaModules& modules,
    ComposableAttentionRunner& attention,
    ComposableAttentionRunner& fixed_attention,
    const std::vector<int64_t>& tokens,
    std::vector<float>& final_logits) {
  std::vector<float> composed_logits;
  std::vector<float> fixed_logits;
  double ignored = 0.0;
  for (size_t index = 0; index < FLAGS_sequence_warmup; ++index) {
    if (!time_sequence(modules, attention, tokens, composed_logits, ignored) ||
        !time_sequence(
            modules, fixed_attention, tokens, fixed_logits, ignored)) {
      return false;
    }
  }

  std::vector<double> composed_samples;
  std::vector<double> fixed_samples;
  composed_samples.reserve(FLAGS_sequence_iterations);
  fixed_samples.reserve(FLAGS_sequence_iterations);
  for (size_t index = 0; index < FLAGS_sequence_iterations; ++index) {
    double composed_ms = 0.0;
    double fixed_ms = 0.0;
    if (index % 2 == 0) {
      if (!time_sequence(
              modules, attention, tokens, composed_logits, composed_ms) ||
          !time_sequence(
              modules, fixed_attention, tokens, fixed_logits, fixed_ms)) {
        return false;
      }
    } else if (
        !time_sequence(
            modules, fixed_attention, tokens, fixed_logits, fixed_ms) ||
        !time_sequence(
            modules, attention, tokens, composed_logits, composed_ms)) {
      return false;
    }
    composed_samples.push_back(composed_ms);
    fixed_samples.push_back(fixed_ms);
  }

  LogitComparisonStats comparison;
  if (!compare_logits(composed_logits, fixed_logits, comparison)) {
    ET_LOG(Error, "Could not compare end-to-end logits.");
    return false;
  }
  const TimingStats composed = summarize_timings(std::move(composed_samples));
  const TimingStats fixed = summarize_timings(std::move(fixed_samples));
  ET_LOG(
      Info,
      "End-to-end benchmark tokens=%zu: EdgeInfer median=%.3f ms "
      "mean=%.3f ms range=[%.3f,%.3f]; Fixed-Max(C=%u) median=%.3f ms "
      "mean=%.3f ms range=[%.3f,%.3f]; speedup=%.3fx",
      tokens.size(),
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
  ET_LOG(
      Info,
      "End-to-end correctness: finite_logit_max_abs=%.9f "
      "finite_logit_mean_abs=%.9f finite_logit_pairs=%zu "
      "composed_nan=%zu composed_pos_inf=%zu composed_neg_inf=%zu "
      "fixed_nan=%zu fixed_pos_inf=%zu fixed_neg_inf=%zu "
      "argmax_compared=%zu argmax_matches=%zu argmax_mismatches=%zu "
      "argmax_uncomparable=%zu argmax_agreement=%.6f",
      comparison.finite_maximum_error,
      comparison.finite_mean_error,
      comparison.finite_comparison_count,
      comparison.composed_nan_count,
      comparison.composed_positive_infinity_count,
      comparison.composed_negative_infinity_count,
      comparison.fixed_nan_count,
      comparison.fixed_positive_infinity_count,
      comparison.fixed_negative_infinity_count,
      comparison.argmax_comparison_count,
      comparison.matching_argmax_count,
      comparison.mismatching_argmax_count,
      comparison.uncomparable_argmax_count,
      comparison.argmax_agreement());
  final_logits = std::move(composed_logits);
  return true;
}

bool write_logits(const std::vector<float>& logits) {
  std::ofstream output(FLAGS_output_path, std::ios::binary);
  if (!output) {
    ET_LOG(
        Error, "Could not open logits output '%s'.", FLAGS_output_path.c_str());
    return false;
  }
  output.write(
      reinterpret_cast<const char*>(logits.data()),
      static_cast<std::streamsize>(logits.size() * sizeof(float)));
  if (!output) {
    ET_LOG(
        Error,
        "Could not write logits output '%s'.",
        FLAGS_output_path.c_str());
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char** argv) {
  executorch::runtime::runtime_init();
  QnnExecuTorchBackendRegister(
      reinterpret_cast<void*>(executorch::runtime::register_backend));
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const bool dimensions_valid = FLAGS_sequence_length > 0 && FLAGS_layers > 0 &&
      FLAGS_dim > 0 && FLAGS_query_heads > 0 && FLAGS_kv_heads > 0 &&
      FLAGS_head_dim > 0 && FLAGS_head_dim % 2 == 0 &&
      FLAGS_query_heads % FLAGS_kv_heads == 0 && FLAGS_vocab_size > 0 &&
      FLAGS_rope_theta > 0.0 && FLAGS_iterations > 0 &&
      (FLAGS_rope_style == "llama" || FLAGS_rope_style == "hf") &&
      (FLAGS_fixed_width == 0 || FLAGS_sequence_iterations > 0) &&
      fits_tensor_size(FLAGS_sequence_length) && fits_tensor_size(FLAGS_dim) &&
      fits_tensor_size(FLAGS_query_heads) && fits_tensor_size(FLAGS_kv_heads) &&
      fits_tensor_size(FLAGS_head_dim) && fits_tensor_size(FLAGS_vocab_size);
  if (argc != 1 || !dimensions_valid || FLAGS_output_path.empty()) {
    ET_LOG(Error, "Invalid composable Llama runner arguments.");
    return 1;
  }
  const auto widths = parse_positive_sizes(FLAGS_widths);
  if (widths.empty() ||
      !std::all_of(widths.begin(), widths.end(), [](size_t width) {
        return fits_tensor_size(width) && is_power_of_two(width);
      })) {
    ET_LOG(Error, "Graph widths must be powers of two that fit SizesType.");
    return 1;
  }
  const auto layer_paths = parse_paths(FLAGS_layer_model_paths);
  if (FLAGS_layers_per_shard == 0) {
    if (!FLAGS_layer_model_paths.empty()) {
      ET_LOG(
          Error,
          "layer_model_paths requires a positive layers_per_shard value.");
      return 1;
    }
  } else {
    const size_t expected_shards =
        (FLAGS_layers + FLAGS_layers_per_shard - 1) / FLAGS_layers_per_shard;
    if (layer_paths.size() != expected_shards) {
      ET_LOG(
          Error,
          "Expected %zu layer-shard PTE paths but received %zu.",
          expected_shards,
          layer_paths.size());
      return 1;
    }
  }
  if (FLAGS_fixed_width > 0 &&
      (FLAGS_fixed_width < FLAGS_sequence_length ||
       !std::binary_search(
           widths.begin(),
           widths.end(),
           static_cast<size_t>(FLAGS_fixed_width)))) {
    ET_LOG(
        Error,
        "fixed_width must be an exported width no smaller than the sequence.");
    return 1;
  }

  ComposableLlamaModules modules(
      FLAGS_model_path, layer_paths, FLAGS_layers_per_shard);
  ET_LOG(
      Info,
      "Composable Llama PTE layout: %s (%zu layer shards).",
      layer_paths.empty() ? "single" : "layer-sharded",
      layer_paths.size());
  ET_LOG(
      Info,
      "Composable Llama prefill order: %s.",
      FLAGS_layer_major_prefill ? "layer-major with layer-method unloading"
                                : "token-major");
  auto costs = profile_portfolio(modules.core(), widths);
  if (costs.size() != widths.size()) {
    return 2;
  }
  auto runner_result =
      ComposableAttentionRunner::create(&modules.core(), 1, costs);
  if (!runner_result.ok()) {
    ET_LOG(Error, "Failed to create composable Attention runner.");
    return 2;
  }
  auto attention = std::move(*runner_result);
  std::optional<ComposableAttentionRunner> fixed_attention;
  if (FLAGS_fixed_width > 0) {
    const auto fixed_cost = std::find_if(
        costs.begin(), costs.end(), [](const StaticAttentionGraphCost& cost) {
          return cost.width == FLAGS_fixed_width;
        });
    if (fixed_cost == costs.end()) {
      ET_LOG(Error, "Could not find the profiled Fixed-Max graph.");
      return 2;
    }
    auto fixed_result = ComposableAttentionRunner::create(
        &modules.core(), 1, std::vector<StaticAttentionGraphCost>{*fixed_cost});
    if (!fixed_result.ok()) {
      ET_LOG(Error, "Failed to create Fixed-Max Attention runner.");
      return 2;
    }
    fixed_attention.emplace(std::move(*fixed_result));
  }
  const auto tokens = make_tokens();
  std::vector<float> logits;
  double elapsed_ms = 0.0;
  if (fixed_attention) {
    if (!benchmark_sequences(
            modules, attention, *fixed_attention, tokens, logits)) {
      return 3;
    }
  } else {
    const auto start = std::chrono::steady_clock::now();
    if (!run_sequence(modules, attention, tokens, logits)) {
      return 3;
    }
    elapsed_ms = std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - start)
                     .count();
  }
  if (!write_logits(logits)) {
    return 4;
  }
  if (!fixed_attention) {
    ET_LOG(
        Info,
        "Composable Llama passed through %u tokens in %.3f ms "
        "(largest static width: %zu, logits: %s).",
        FLAGS_sequence_length,
        elapsed_ms,
        widths.back(),
        FLAGS_output_path.c_str());
  }
  return 0;
}
