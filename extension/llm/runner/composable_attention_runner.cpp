/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/runner/composable_attention_runner.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>

#include <c10/util/safe_numerics.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/platform/log.h>

namespace executorch {
namespace extension {
namespace llm {
namespace {

using executorch::aten::ScalarType;
using executorch::aten::SizesType;
using executorch::aten::Tensor;
using runtime::Error;
using runtime::EValue;
using runtime::Result;

Result<size_t> checked_product(const std::vector<SizesType>& sizes) {
  size_t result = 1;
  for (const SizesType size : sizes) {
    if (size < 0 ||
        c10::mul_overflows(result, static_cast<size_t>(size), &result)) {
      return Error::InvalidArgument;
    }
  }
  return result;
}

Result<TensorPtr> make_zero_tensor(
    std::vector<SizesType> sizes,
    ScalarType scalar_type) {
  auto numel = checked_product(sizes);
  if (!numel.ok()) {
    return numel.error();
  }
  size_t nbytes = 0;
  if (c10::mul_overflows(
          *numel,
          static_cast<size_t>(executorch::aten::elementSize(scalar_type)),
          &nbytes)) {
    return Error::InvalidArgument;
  }
  auto storage = std::make_shared<std::vector<uint8_t>>(nbytes, uint8_t{0});
  void* data = nbytes == 0 ? nullptr : storage->data();
  return make_tensor_ptr(
      std::move(sizes),
      data,
      scalar_type,
      executorch::aten::TensorShapeDynamism::STATIC,
      [storage = std::move(storage)](void*) {});
}

Result<TensorPtr> clone_contiguous_tensor(const Tensor& tensor) {
  if (!runtime::tensor_is_contiguous(tensor)) {
    return Error::InvalidArgument;
  }
  std::vector<SizesType> sizes(tensor.sizes().begin(), tensor.sizes().end());
  auto clone = make_zero_tensor(std::move(sizes), tensor.scalar_type());
  if (!clone.ok()) {
    return clone.error();
  }
  if (tensor.nbytes() > 0) {
    std::memcpy(
        (*clone)->mutable_data_ptr(), tensor.const_data_ptr(), tensor.nbytes());
  }
  return std::move(*clone);
}

Result<SizesType> checked_size(size_t value) {
  if (value > static_cast<size_t>(std::numeric_limits<SizesType>::max())) {
    return Error::InvalidArgument;
  }
  return static_cast<SizesType>(value);
}

bool has_expected_shapes(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& visibility,
    size_t query_rows) {
  if (q.dim() != 4 || k.dim() != 4 || v.dim() != 4 || visibility.dim() != 4) {
    return false;
  }
  if (q.size(0) <= 0 || q.size(1) <= 0 || q.size(2) <= 0 || q.size(3) <= 0 ||
      k.size(1) <= 0 || k.size(3) <= 0 || visibility.size(1) <= 0) {
    return false;
  }
  return static_cast<size_t>(q.size(2)) == query_rows &&
      q.size(0) == k.size(0) && q.size(0) == v.size(0) &&
      q.size(0) == visibility.size(0) && k.size(1) == v.size(1) &&
      q.size(3) == k.size(2) && q.size(3) == v.size(3) &&
      k.size(3) == v.size(2) && visibility.size(2) == q.size(2) &&
      visibility.size(3) == k.size(3) && q.size(1) % k.size(1) == 0 &&
      (visibility.size(1) == 1 || visibility.size(1) == q.size(1));
}

bool has_expected_output_shape(
    const Tensor& output,
    const Tensor& q,
    size_t output_index) {
  if (output.scalar_type() != q.scalar_type() ||
      !runtime::tensor_is_contiguous(output) || output.dim() != 4 ||
      output.size(0) != q.size(0) || output.size(1) != q.size(1) ||
      output.size(2) != q.size(2)) {
    return false;
  }
  const SizesType expected_last_dimension =
      output_index == 1 || output_index == 2 ? 1 : q.size(3);
  return output.size(3) == expected_last_dimension;
}

Result<std::array<TensorPtr, 4>> clone_method_outputs(
    const std::vector<EValue>& outputs,
    const Tensor& q) {
  if (outputs.size() != 4) {
    return Error::InvalidArgument;
  }
  std::array<TensorPtr, 4> result;
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (!outputs[i].isTensor() ||
        !has_expected_output_shape(outputs[i].toTensor(), q, i)) {
      return Error::InvalidArgument;
    }
    auto clone = clone_contiguous_tensor(outputs[i].toTensor());
    if (!clone.ok()) {
      return clone.error();
    }
    result[i] = std::move(*clone);
  }
  return result;
}

Error copy_padded_rows(
    const Tensor& source,
    Tensor& target,
    size_t rows,
    size_t source_width,
    size_t target_width,
    size_t offset,
    size_t valid_width,
    size_t trailing_elements) {
  if (source.scalar_type() != target.scalar_type() || rows == 0 ||
      trailing_elements == 0 || offset > source_width ||
      valid_width > source_width - offset || valid_width > target_width) {
    return Error::InvalidArgument;
  }

  const size_t element_size =
      executorch::aten::elementSize(source.scalar_type());
  size_t source_row_elements = 0;
  size_t target_row_elements = 0;
  size_t source_elements = 0;
  size_t target_elements = 0;
  size_t source_row_bytes = 0;
  size_t target_row_bytes = 0;
  size_t source_bytes = 0;
  size_t target_bytes = 0;
  size_t source_offset_bytes = 0;
  size_t copy_bytes = 0;
  if (element_size == 0 ||
      c10::mul_overflows(
          source_width, trailing_elements, &source_row_elements) ||
      c10::mul_overflows(
          target_width, trailing_elements, &target_row_elements) ||
      c10::mul_overflows(rows, source_row_elements, &source_elements) ||
      c10::mul_overflows(rows, target_row_elements, &target_elements) ||
      c10::mul_overflows(
          source_row_elements, element_size, &source_row_bytes) ||
      c10::mul_overflows(
          target_row_elements, element_size, &target_row_bytes) ||
      c10::mul_overflows(source_elements, element_size, &source_bytes) ||
      c10::mul_overflows(target_elements, element_size, &target_bytes) ||
      c10::mul_overflows(offset, trailing_elements, &source_offset_bytes) ||
      c10::mul_overflows(
          source_offset_bytes, element_size, &source_offset_bytes) ||
      c10::mul_overflows(valid_width, trailing_elements, &copy_bytes) ||
      c10::mul_overflows(copy_bytes, element_size, &copy_bytes) ||
      source.nbytes() != source_bytes || target.nbytes() != target_bytes) {
    return Error::InvalidArgument;
  }

  const auto* source_data =
      static_cast<const uint8_t*>(source.const_data_ptr());
  auto* target_data = static_cast<uint8_t*>(target.mutable_data_ptr());
  for (size_t row = 0; row < rows; ++row) {
    std::memcpy(
        target_data + row * target_row_bytes,
        source_data + row * source_row_bytes + source_offset_bytes,
        copy_bytes);
  }
  return Error::Ok;
}

Result<std::array<TensorPtr, 3>> make_attention_blocks(
    const Tensor& k,
    const Tensor& v,
    const Tensor& visibility,
    size_t offset,
    size_t valid_width,
    size_t block_width) {
  const size_t source_width = static_cast<size_t>(k.size(3));
  if (offset > source_width || valid_width > source_width - offset ||
      valid_width > block_width) {
    return Error::InvalidArgument;
  }
  auto block_width_size = checked_size(block_width);
  if (!block_width_size.ok()) {
    return block_width_size.error();
  }
  const SizesType width = *block_width_size;
  auto k_block = make_zero_tensor(
      {static_cast<SizesType>(k.size(0)),
       static_cast<SizesType>(k.size(1)),
       static_cast<SizesType>(k.size(2)),
       width},
      k.scalar_type());
  auto v_block = make_zero_tensor(
      {static_cast<SizesType>(v.size(0)),
       static_cast<SizesType>(v.size(1)),
       width,
       static_cast<SizesType>(v.size(3))},
      v.scalar_type());
  auto visibility_block = make_zero_tensor(
      {static_cast<SizesType>(visibility.size(0)),
       static_cast<SizesType>(visibility.size(1)),
       static_cast<SizesType>(visibility.size(2)),
       width},
      visibility.scalar_type());
  if (!k_block.ok()) {
    return k_block.error();
  }
  if (!v_block.ok()) {
    return v_block.error();
  }
  if (!visibility_block.ok()) {
    return visibility_block.error();
  }

  auto k_rows = checked_product(
      {static_cast<SizesType>(k.size(0)),
       static_cast<SizesType>(k.size(1)),
       static_cast<SizesType>(k.size(2))});
  auto v_rows = checked_product(
      {static_cast<SizesType>(v.size(0)), static_cast<SizesType>(v.size(1))});
  auto visibility_rows = checked_product(
      {static_cast<SizesType>(visibility.size(0)),
       static_cast<SizesType>(visibility.size(1)),
       static_cast<SizesType>(visibility.size(2))});
  if (!k_rows.ok() || !v_rows.ok() || !visibility_rows.ok()) {
    return Error::InvalidArgument;
  }

  Error error = copy_padded_rows(
      k, **k_block, *k_rows, source_width, block_width, offset, valid_width, 1);
  if (error != Error::Ok) {
    return error;
  }
  error = copy_padded_rows(
      v,
      **v_block,
      *v_rows,
      source_width,
      block_width,
      offset,
      valid_width,
      static_cast<size_t>(v.size(3)));
  if (error != Error::Ok) {
    return error;
  }
  error = copy_padded_rows(
      visibility,
      **visibility_block,
      *visibility_rows,
      source_width,
      block_width,
      offset,
      valid_width,
      1);
  if (error != Error::Ok) {
    return error;
  }

  return std::array<TensorPtr, 3>{
      std::move(*k_block), std::move(*v_block), std::move(*visibility_block)};
}

bool is_supported_width(const StaticAttentionPlanner& planner, size_t width) {
  const auto& widths = planner.graph_widths();
  return std::binary_search(widths.begin(), widths.end(), width);
}

bool has_expected_workspace(
    const ComposableAttentionWorkspace& workspace,
    const Tensor& q) {
  for (const auto& bank : workspace.banks) {
    for (size_t index = 0; index < bank.size(); ++index) {
      if (bank[index] == nullptr ||
          !has_expected_output_shape(*bank[index], q, index)) {
        return false;
      }
    }
  }
  return true;
}

enum class OutputBindingContract {
  kCallerOwned,
  kMemoryPlanned,
  kMetadataUnavailable,
};

OutputBindingContract output_binding_contract(
    Module* module,
    const std::string& method) {
  if (module == nullptr) {
    return OutputBindingContract::kMetadataUnavailable;
  }
  auto method_meta = module->method_meta(method);
  if (!method_meta.ok()) {
    return OutputBindingContract::kMetadataUnavailable;
  }
  if (method_meta->num_outputs() == 0) {
    return OutputBindingContract::kMetadataUnavailable;
  }
  for (size_t index = 0; index < method_meta->num_outputs(); ++index) {
    auto output_meta = method_meta->output_tensor_meta(index);
    if (!output_meta.ok()) {
      return OutputBindingContract::kMetadataUnavailable;
    }
    if (output_meta->is_memory_planned()) {
      return OutputBindingContract::kMemoryPlanned;
    }
  }
  return OutputBindingContract::kCallerOwned;
}

const char* output_binding_contract_name(OutputBindingContract contract) {
  switch (contract) {
    case OutputBindingContract::kCallerOwned:
      return "caller-owned";
    case OutputBindingContract::kMemoryPlanned:
      return "memory-planned";
    case OutputBindingContract::kMetadataUnavailable:
      return "metadata-unavailable";
  }
  return "unknown";
}

Error copy_output_to_workspace(
    const Tensor& source,
    Tensor& destination,
    const Tensor& q,
    size_t output_index) {
  if (!has_expected_output_shape(source, q, output_index) ||
      !has_expected_output_shape(destination, q, output_index) ||
      source.nbytes() != destination.nbytes()) {
    return Error::InvalidArgument;
  }
  if (source.nbytes() > 0 &&
      source.const_data_ptr() != destination.const_data_ptr()) {
    std::memcpy(
        destination.mutable_data_ptr(),
        source.const_data_ptr(),
        source.nbytes());
  }
  return Error::Ok;
}

} // namespace

std::string
static_attention_method_name(bool first, size_t query_rows, size_t width) {
  return static_attention_method_name(first, query_rows, width, false);
}

std::string static_attention_method_name(
    bool first,
    size_t query_rows,
    size_t width,
    bool assume_nonempty) {
  return std::string(first ? "attn_first" : "attn_merge") +
      (assume_nonempty ? "_nonempty_r" : "_r") + std::to_string(query_rows) +
      "_c" + std::to_string(width);
}

Result<ComposableAttentionRunner> ComposableAttentionRunner::create(
    Module* module,
    size_t query_rows,
    std::vector<StaticAttentionGraphCost> graph_costs,
    bool assume_nonempty,
    CallerOwnedOutputPolicy caller_owned_output_policy) {
  const size_t max_tensor_size =
      static_cast<size_t>(std::numeric_limits<SizesType>::max());
  if (module == nullptr || query_rows == 0 || query_rows > max_tensor_size ||
      (assume_nonempty && query_rows != 1) ||
      std::any_of(
          graph_costs.begin(),
          graph_costs.end(),
          [max_tensor_size](const StaticAttentionGraphCost& graph_cost) {
            return graph_cost.width > max_tensor_size;
          })) {
    return Error::InvalidArgument;
  }
  auto planner = StaticAttentionPlanner::create(std::move(graph_costs));
  if (!planner.has_value()) {
    return Error::InvalidArgument;
  }
  if (caller_owned_output_policy ==
      CallerOwnedOutputPolicy::kRequireUnplannedOutputs) {
    for (const size_t width : planner->graph_widths()) {
      for (const bool first : {true, false}) {
        const std::string method = static_attention_method_name(
            first, query_rows, width, assume_nonempty);
        const OutputBindingContract contract =
            output_binding_contract(module, method);
        if (contract != OutputBindingContract::kCallerOwned) {
          ET_LOG(
              Error,
              "Static Attention method '%s' has %s output metadata; "
              "RequireUnplannedOutputs rejects this PTE.",
              method.c_str(),
              output_binding_contract_name(contract));
          return Error::NotSupported;
        }
      }
    }
  }
  return ComposableAttentionRunner(
      module,
      query_rows,
      std::move(*planner),
      assume_nonempty,
      caller_owned_output_policy);
}

ComposableAttentionRunner::ComposableAttentionRunner(
    Module* module,
    size_t query_rows,
    StaticAttentionPlanner planner,
    bool assume_nonempty,
    CallerOwnedOutputPolicy caller_owned_output_policy)
    : module_(module),
      query_rows_(query_rows),
      planner_(std::move(planner)),
      assume_nonempty_(assume_nonempty),
      caller_owned_output_policy_(caller_owned_output_policy) {}

Result<StaticAttentionPlan> ComposableAttentionRunner::plan(
    size_t sequence_length) {
  auto plan = planner_.plan(sequence_length);
  if (!plan.has_value()) {
    return Error::InvalidArgument;
  }
  return std::move(*plan);
}

Result<StaticAttentionPlan> ComposableAttentionRunner::plan_causal_tile(
    size_t full_sequence_length,
    size_t causal_query_begin,
    size_t valid_query_rows) {
  auto plan = planner_.plan_causal_tile(
      full_sequence_length, causal_query_begin, query_rows_, valid_query_rows);
  if (!plan.has_value()) {
    return Error::InvalidArgument;
  }
  return std::move(*plan);
}

Result<StaticAttentionPlan> ComposableAttentionRunner::plan_prefixes(
    const std::vector<size_t>& visible_prefixes) {
  auto plan = planner_.plan_prefixes(visible_prefixes);
  if (!plan.has_value()) {
    return Error::InvalidArgument;
  }
  return std::move(*plan);
}

Result<TensorPtr> ComposableAttentionRunner::run(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& visibility) {
  if (!has_expected_shapes(q, k, v, visibility, query_rows_) ||
      !runtime::isFloatingType(q.scalar_type()) ||
      q.scalar_type() != k.scalar_type() ||
      q.scalar_type() != v.scalar_type() ||
      q.scalar_type() != visibility.scalar_type() ||
      !runtime::tensor_is_contiguous(q) || !runtime::tensor_is_contiguous(k) ||
      !runtime::tensor_is_contiguous(v) ||
      !runtime::tensor_is_contiguous(visibility)) {
    return Error::InvalidArgument;
  }

  const size_t sequence_length = static_cast<size_t>(k.size(3));
  auto plan_result = plan(sequence_length);
  if (!plan_result.ok()) {
    return plan_result.error();
  }
  return run_plan(q, k, v, visibility, *plan_result);
}

Result<TensorPtr> ComposableAttentionRunner::run_causal_tile(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& visibility,
    size_t causal_query_begin,
    size_t valid_query_rows) {
  if (!has_expected_shapes(q, k, v, visibility, query_rows_) ||
      !runtime::isFloatingType(q.scalar_type()) ||
      q.scalar_type() != k.scalar_type() ||
      q.scalar_type() != v.scalar_type() ||
      q.scalar_type() != visibility.scalar_type() ||
      !runtime::tensor_is_contiguous(q) || !runtime::tensor_is_contiguous(k) ||
      !runtime::tensor_is_contiguous(v) ||
      !runtime::tensor_is_contiguous(visibility)) {
    return Error::InvalidArgument;
  }

  auto plan_result = plan_causal_tile(
      static_cast<size_t>(k.size(3)), causal_query_begin, valid_query_rows);
  if (!plan_result.ok()) {
    return plan_result.error();
  }
  return run_plan(q, k, v, visibility, *plan_result);
}

Result<TensorPtr> ComposableAttentionRunner::run_plan(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& visibility,
    const StaticAttentionPlan& plan) {
  const size_t sequence_length = plan.sequence_length;
  if (sequence_length == 0 ||
      sequence_length > static_cast<size_t>(k.size(3))) {
    return Error::InvalidArgument;
  }

  size_t offset = 0;
  std::vector<std::array<TensorPtr, 3>> owned_blocks;
  std::vector<PreparedAttentionBlock> prepared_blocks;
  owned_blocks.reserve(plan.widths.size());
  prepared_blocks.reserve(plan.widths.size());
  for (const size_t width : plan.widths) {
    if (offset > sequence_length) {
      return Error::Internal;
    }
    const size_t valid_width = std::min(width, sequence_length - offset);
    auto blocks =
        make_attention_blocks(k, v, visibility, offset, valid_width, width);
    if (!blocks.ok()) {
      return blocks.error();
    }
    owned_blocks.emplace_back(std::move(*blocks));
    const auto& owned = owned_blocks.back();
    prepared_blocks.push_back({owned[0].get(), owned[1].get(), owned[2].get()});
    offset += valid_width;
  }

  if (prepared_blocks.empty() || offset != sequence_length) {
    return Error::Internal;
  }
  return run_blocks(q, prepared_blocks);
}

Result<TensorPtr> ComposableAttentionRunner::run_blocks(
    const Tensor& q,
    const std::vector<PreparedAttentionBlock>& blocks) {
  if (blocks.empty() || q.dim() != 4 ||
      static_cast<size_t>(q.size(2)) != query_rows_ ||
      !runtime::isFloatingType(q.scalar_type()) ||
      !runtime::tensor_is_contiguous(q)) {
    return Error::InvalidArgument;
  }

  std::array<TensorPtr, 4> state;
  bool first = true;
  for (const PreparedAttentionBlock& block : blocks) {
    if (block.key == nullptr || block.value == nullptr ||
        block.visibility == nullptr ||
        !has_expected_shapes(
            q, *block.key, *block.value, *block.visibility, query_rows_) ||
        q.scalar_type() != block.key->scalar_type() ||
        q.scalar_type() != block.value->scalar_type() ||
        q.scalar_type() != block.visibility->scalar_type() ||
        !runtime::tensor_is_contiguous(*block.key) ||
        !runtime::tensor_is_contiguous(*block.value) ||
        !runtime::tensor_is_contiguous(*block.visibility)) {
      return Error::InvalidArgument;
    }
    const size_t width = static_cast<size_t>(block.key->size(3));
    if (!is_supported_width(planner_, width)) {
      return Error::InvalidArgument;
    }

    std::vector<EValue> inputs{q, *block.key, *block.value, *block.visibility};
    if (!first) {
      inputs.emplace_back(*state[1]);
      inputs.emplace_back(*state[2]);
      inputs.emplace_back(*state[3]);
    }
    const std::string method = static_attention_method_name(
        first, query_rows_, width, assume_nonempty_);
    auto outputs = module_->execute(method, inputs);
    if (!outputs.ok()) {
      ET_LOG(
          Error,
          "Failed to execute static Attention method '%s'.",
          method.c_str());
      return outputs.error();
    }
    auto cloned_outputs = clone_method_outputs(*outputs, q);
    if (!cloned_outputs.ok()) {
      ET_LOG(
          Error,
          "Static Attention method '%s' returned invalid outputs.",
          method.c_str());
      return cloned_outputs.error();
    }
    state = std::move(*cloned_outputs);
    first = false;
  }
  return state[0];
}

Result<TensorPtr> ComposableAttentionRunner::run_blocks(
    const Tensor& q,
    const std::vector<PreparedAttentionBlock>& blocks,
    ComposableAttentionWorkspace& workspace) {
  if (blocks.empty() || q.dim() != 4 ||
      static_cast<size_t>(q.size(2)) != query_rows_ ||
      !runtime::isFloatingType(q.scalar_type()) ||
      !runtime::tensor_is_contiguous(q) ||
      !has_expected_workspace(workspace, q)) {
    return Error::InvalidArgument;
  }

  for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
    const Error error =
        run_prepared_block(q, blocks[block_index], block_index, workspace);
    if (error != Error::Ok) {
      return error;
    }
  }
  return workspace.banks[0][0];
}

Error ComposableAttentionRunner::run_prepared_block(
    const Tensor& q,
    const PreparedAttentionBlock& block,
    size_t block_index,
    ComposableAttentionWorkspace& workspace) {
  if (q.dim() != 4 || static_cast<size_t>(q.size(2)) != query_rows_ ||
      !runtime::isFloatingType(q.scalar_type()) ||
      !runtime::tensor_is_contiguous(q) ||
      !has_expected_workspace(workspace, q) || block.key == nullptr ||
      block.value == nullptr || block.visibility == nullptr ||
      !has_expected_shapes(
          q, *block.key, *block.value, *block.visibility, query_rows_) ||
      q.scalar_type() != block.key->scalar_type() ||
      q.scalar_type() != block.value->scalar_type() ||
      q.scalar_type() != block.visibility->scalar_type() ||
      !runtime::tensor_is_contiguous(*block.key) ||
      !runtime::tensor_is_contiguous(*block.value) ||
      !runtime::tensor_is_contiguous(*block.visibility)) {
    return Error::InvalidArgument;
  }
  const size_t width = static_cast<size_t>(block.key->size(3));
  if (!is_supported_width(planner_, width)) {
    return Error::InvalidArgument;
  }

  const bool first = block_index == 0;
  const std::string method = static_attention_method_name(
      first, query_rows_, width, assume_nonempty_);
  std::vector<EValue> inputs{q, *block.key, *block.value, *block.visibility};
  if (!first) {
    const auto& previous_state = workspace.banks[(block_index - 1) % 2];
    inputs.emplace_back(*previous_state[1]);
    inputs.emplace_back(*previous_state[2]);
    inputs.emplace_back(*previous_state[3]);
  }

  const auto& state_destination = workspace.banks[block_index % 2];
  const std::array<TensorPtr, 4> destination{
      workspace.banks[0][0],
      state_destination[1],
      state_destination[2],
      state_destination[3]};

  bool outputs_bound = false;
  if (copy_fallback_methods_.count(method) == 0) {
    if (has_output_binding(module_, method, destination)) {
      outputs_bound = true;
    } else {
      const OutputBindingContract contract = output_binding_contract(module_, method);
      if (contract != OutputBindingContract::kCallerOwned) {
        if (caller_owned_output_policy_ ==
            CallerOwnedOutputPolicy::kRequireUnplannedOutputs) {
          ET_LOG(
              Error,
              "Static Attention method '%s' has %s output metadata after "
              "strict contract validation.",
              method.c_str(),
              output_binding_contract_name(contract));
          return Error::InvalidState;
        }
        copy_fallback_methods_.insert(method);
        ET_LOG(
            Info,
            "Static Attention method '%s' has %s output metadata; using the "
            "compatible copy path.",
            method.c_str(),
            output_binding_contract_name(contract));
      } else {
        const Error bind_error = bind_outputs_if_needed(
            module_,
            method,
            destination,
            [&]() {
              std::vector<EValue> values;
              values.reserve(destination.size());
              for (const TensorPtr& output : destination) {
                values.emplace_back(*output);
              }
              return module_->set_outputs(method, values);
            },
            outputs_bound);
        if (bind_error != Error::Ok) {
          module_->unload_method(method);
          bound_methods_.erase(method);
          forget_output_binding(method);
          copy_fallback_methods_.insert(method);
          ET_LOG(
              Info,
              "Static Attention method '%s' does not accept caller-owned "
              "outputs; using the compatible copy path.",
              method.c_str());
        }
      }
    }
  }
  auto executed_outputs = module_->execute(method, inputs);
  if (!executed_outputs.ok()) {
    ET_LOG(
        Error,
        "Failed to execute static Attention method '%s'.",
        method.c_str());
    return executed_outputs.error();
  }
  if (executed_outputs->size() != destination.size()) {
    return Error::InvalidArgument;
  }
  for (size_t index = 0; index < destination.size(); ++index) {
    if (!(*executed_outputs)[index].isTensor() ||
        !has_expected_output_shape((*executed_outputs)[index].toTensor(), q, index)) {
      return Error::InvalidArgument;
    }
  }

  bool bound_address_mismatch = false;
  for (size_t index = 0; index < destination.size(); ++index) {
    const Tensor& actual = (*executed_outputs)[index].toTensor();
    if (outputs_bound &&
        actual.mutable_data_ptr() == destination[index]->mutable_data_ptr() &&
        actual.nbytes() == destination[index]->nbytes()) {
      continue;
    }
    bound_address_mismatch = bound_address_mismatch || outputs_bound;
    const Error error =
        copy_output_to_workspace(actual, *destination[index], q, index);
    if (error != Error::Ok) {
      return error;
    }
  }
  if (bound_address_mismatch) {
    ET_LOG(
        Info,
        "Static Attention method '%s' retained a planned output after "
        "set_outputs(); using the compatible copy path.",
        method.c_str());
    module_->unload_method(method);
    bound_methods_.erase(method);
    forget_output_binding(method);
    copy_fallback_methods_.insert(method);
  }
  return Error::Ok;
}

bool ComposableAttentionRunner::has_output_binding(
    Module* module,
    const std::string& method,
    const std::array<TensorPtr, 4>& outputs) const {
  const auto binding = output_bindings_.find(method);
  if (binding == output_bindings_.end() || binding->second.module != module) {
    return false;
  }

  for (size_t index = 0; index < outputs.size(); ++index) {
    const TensorPtr& output = outputs[index];
    if (output == nullptr) {
      return false;
    }
    const OutputTensorBinding& expected = binding->second.outputs[index];
    const std::shared_ptr<Tensor> owner = expected.owner.lock();
    if (owner == nullptr || owner.get() != output.get() ||
        expected.owner.owner_before(output) ||
        output.owner_before(expected.owner) ||
        expected.tensor != output.get() ||
        expected.data != output->mutable_data_ptr() ||
        expected.dtype != output->scalar_type() ||
        expected.nbytes != output->nbytes() ||
        expected.sizes.size() != static_cast<size_t>(output->dim())) {
      return false;
    }
    for (size_t dim = 0; dim < expected.sizes.size(); ++dim) {
      if (expected.sizes[dim] != output->size(dim)) {
        return false;
      }
    }
  }
  return true;
}

void ComposableAttentionRunner::remember_output_binding(
    Module* module,
    const std::string& method,
    const std::array<TensorPtr, 4>& outputs) {
  OutputBinding binding;
  binding.module = module;
  for (size_t index = 0; index < outputs.size(); ++index) {
    const TensorPtr& output = outputs[index];
    OutputTensorBinding& destination = binding.outputs[index];
    destination.owner = output;
    destination.tensor = output.get();
    destination.data = output->mutable_data_ptr();
    destination.dtype = output->scalar_type();
    destination.nbytes = output->nbytes();
    destination.sizes.reserve(static_cast<size_t>(output->dim()));
    for (size_t dim = 0; dim < static_cast<size_t>(output->dim()); ++dim) {
      destination.sizes.push_back(output->size(dim));
    }
  }
  output_bindings_.insert_or_assign(method, std::move(binding));
}

void ComposableAttentionRunner::forget_output_binding(
    const std::string& method) {
  output_bindings_.erase(method);
}

void ComposableAttentionRunner::release_output_bindings() {
  for (const std::string& method : bound_methods_) {
    module_->unload_method(method);
  }
  bound_methods_.clear();
  output_bindings_.clear();
  copy_fallback_methods_.clear();
}

} // namespace llm
} // namespace extension
} // namespace executorch
