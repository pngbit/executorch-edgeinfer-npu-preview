/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/llm/runner/edgeinfer_decode_cache_transaction.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/result.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>
#include <vector>

namespace example {
namespace edgeinfer {

/** Runtime selection policy used by Runner::load(). */
inline bool should_use_edgeinfer_prompt_processor(
    const void* composable_module,
    bool lookahead_decoding) {
  return composable_module != nullptr && !lookahead_decoding;
}

/** Split attention currently owns only IEEE Float/Half K/V state. */
inline executorch::runtime::Error validate_edgeinfer_cache_scalar_type(
    executorch::aten::ScalarType scalar_type) {
  return scalar_type == executorch::aten::ScalarType::Float ||
          scalar_type == executorch::aten::ScalarType::Half
      ? executorch::runtime::Error::Ok
      : executorch::runtime::Error::NotSupported;
}

/**
 * Dispatches to the native Prefill path when EdgeInfer rejects the request
 * before mutating any local state.
 */
template <
    typename EdgeInferOperation,
    typename ClearOperation,
    typename NativeOperation>
executorch::runtime::Result<uint64_t> dispatch_prefill_after_preflight(
    executorch::runtime::Error preflight_error,
    EdgeInferOperation&& edgeinfer_operation,
    ClearOperation&& clear_operation,
    NativeOperation&& native_operation) {
  if (preflight_error == executorch::runtime::Error::Ok) {
    return std::forward<EdgeInferOperation>(edgeinfer_operation)();
  }
  std::forward<ClearOperation>(clear_operation)();
  return std::forward<NativeOperation>(native_operation)();
}

/**
 * Copies block-composed K/V into a native cache, including Float/Half bridge
 * conversion and zero-filled capacity beyond the valid prefix.
 */
template <typename Source, typename Destination>
void copy_layer_cache_data(
    const Source* source_key,
    const Source* source_value,
    Destination* destination_key,
    Destination* destination_value,
    size_t heads,
    size_t head_dim,
    size_t valid,
    size_t cache_length) {
  const size_t cache_elements = heads * head_dim * cache_length;
  std::fill_n(destination_key, cache_elements, Destination(0.0f));
  std::fill_n(destination_value, cache_elements, Destination(0.0f));
  for (size_t head = 0; head < heads; ++head) {
    for (size_t dim = 0; dim < head_dim; ++dim) {
      const size_t row = head * head_dim + dim;
      if constexpr (std::is_same_v<Source, Destination>) {
        std::memcpy(
            destination_key + row * cache_length,
            source_key + row * valid,
            valid * sizeof(Source));
      } else {
        for (size_t index = 0; index < valid; ++index) {
          destination_key[row * cache_length + index] =
              Destination(source_key[row * valid + index]);
        }
      }
    }
    const size_t source_offset = head * valid * head_dim;
    const size_t destination_offset = head * cache_length * head_dim;
    if constexpr (std::is_same_v<Source, Destination>) {
      std::memcpy(
          destination_value + destination_offset,
          source_value + source_offset,
          valid * head_dim * sizeof(Source));
    } else {
      for (size_t index = 0; index < valid * head_dim; ++index) {
        destination_value[destination_offset + index] =
            Destination(source_value[source_offset + index]);
      }
    }
  }
}

inline executorch::runtime::Result<size_t> native_decode_cache_capacity(
    int32_t context_len,
    int32_t native_decode_ar_len) {
  if (context_len <= 0 || native_decode_ar_len <= 0 ||
      native_decode_ar_len > context_len) {
    return executorch::runtime::Error::InvalidArgument;
  }
  return static_cast<size_t>(context_len - native_decode_ar_len);
}

/** Selects the physical cache layout consumed by the native Decode method. */
template <typename CacheManager>
executorch::runtime::Error prepare_native_decode_cache_layout(
    CacheManager& cache_manager,
    int32_t context_len,
    int32_t native_decode_ar_len,
    size_t sequence_length) {
  auto capacity =
      native_decode_cache_capacity(context_len, native_decode_ar_len);
  if (!capacity.ok() || sequence_length > capacity.get()) {
    return executorch::runtime::Error::InvalidArgument;
  }
  cache_manager.rearrange_cache(native_decode_ar_len);
  return executorch::runtime::Error::Ok;
}

/**
 * Runs Prefill transactionally and invokes native recovery only after every
 * layer has been restored to the pre-call checkpoint.
 */
template <
    typename Cache,
    typename EdgeInferOperation,
    typename RestoreOperation,
    typename BridgeOperation,
    typename ClearOperation,
    typename NativePrefillOperation>
executorch::runtime::Result<uint64_t> run_prefill_with_native_fallback(
    const std::vector<Cache*>& layer_caches,
    size_t checkpoint,
    size_t appended_rows,
    EdgeInferOperation&& edgeinfer_operation,
    RestoreOperation&& restore_operation,
    BridgeOperation&& bridge_operation,
    ClearOperation&& clear_operation,
    NativePrefillOperation&& native_prefill_operation) {
  using executorch::extension::llm::EdgeInferDecodeCacheTransaction;
  using executorch::runtime::Error;

  Error failure = Error::Ok;
  {
    EdgeInferDecodeCacheTransaction<Cache> transaction(
        layer_caches, checkpoint);
    if (!transaction.valid()) {
      return Error::InvalidState;
    }

    auto edgeinfer_result =
        std::forward<EdgeInferOperation>(edgeinfer_operation)();
    if (edgeinfer_result.ok()) {
      const uint64_t next_token = edgeinfer_result.get();
      if (transaction.commit(appended_rows)) {
        return next_token;
      }
      failure = Error::InvalidState;
    } else {
      failure = edgeinfer_result.error();
    }
  }

  std::forward<RestoreOperation>(restore_operation)(failure);
  if (checkpoint > 0) {
    const Error bridge_error =
        std::forward<BridgeOperation>(bridge_operation)();
    if (bridge_error != Error::Ok) {
      return bridge_error;
    }
  }
  std::forward<ClearOperation>(clear_operation)();
  return std::forward<NativePrefillOperation>(native_prefill_operation)();
}

} // namespace edgeinfer
} // namespace example
