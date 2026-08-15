/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace executorch {
namespace extension {
namespace llm {

/** Removes an appended suffix while preserving the cache's block layout. */
template <typename Block>
void rollback_edgeinfer_cache_blocks(
    std::vector<Block>& blocks,
    size_t checkpoint) noexcept {
  size_t retained = 0;
  for (Block& block : blocks) {
    size_t keep = 0;
    if (retained < checkpoint) {
      keep = std::min(block.valid_width, checkpoint - retained);
      retained += keep;
    }
    block.valid_width = keep;
  }
}

/** Restores synchronized per-layer K/V lengths unless decode commits. */
template <typename Cache>
class EdgeInferDecodeCacheTransaction final {
 public:
  explicit EdgeInferDecodeCacheTransaction(
      const std::vector<Cache*>& layer_caches,
      size_t checkpoint)
      : layer_caches_(layer_caches), checkpoint_(checkpoint) {
    static_assert(
        noexcept(std::declval<Cache&>().rollback_to(size_t{})),
        "Decode cache rollback must not fail");
    valid_ = !layer_caches_.empty();
    for (const Cache* cache : layer_caches_) {
      if (cache == nullptr || cache->valid_length() != checkpoint_) {
        valid_ = false;
      }
    }
  }

  ~EdgeInferDecodeCacheTransaction() {
    if (!committed_) {
      rollback();
    }
  }

  EdgeInferDecodeCacheTransaction(const EdgeInferDecodeCacheTransaction&) =
      delete;
  EdgeInferDecodeCacheTransaction& operator=(
      const EdgeInferDecodeCacheTransaction&) = delete;

  bool valid() const {
    return valid_;
  }

  bool commit(size_t appended_rows) {
    if (!valid_ || appended_rows == 0) {
      return false;
    }
    if (checkpoint_ > std::numeric_limits<size_t>::max() - appended_rows) {
      return false;
    }
    for (const Cache* cache : layer_caches_) {
      if (cache->valid_length() != checkpoint_ + appended_rows) {
        return false;
      }
    }
    committed_ = true;
    return true;
  }

 private:
  void rollback() noexcept {
    if (!valid_) {
      return;
    }
    for (Cache* cache : layer_caches_) {
      cache->rollback_to(checkpoint_);
    }
  }

  const std::vector<Cache*>& layer_caches_;
  size_t checkpoint_;
  bool valid_ = false;
  bool committed_ = false;
};

/** Runs one decode mutation and commits only one equal append per layer. */
template <typename Cache, typename Operation>
runtime::Error run_edgeinfer_decode_cache_transaction(
    const std::vector<Cache*>& layer_caches,
    size_t checkpoint,
    size_t appended_rows,
    Operation&& operation) {
  EdgeInferDecodeCacheTransaction<Cache> transaction(layer_caches, checkpoint);
  if (!transaction.valid()) {
    return runtime::Error::InvalidState;
  }
  const runtime::Error error = std::forward<Operation>(operation)();
  if (error != runtime::Error::Ok) {
    return error;
  }
  return transaction.commit(appended_rows) ? runtime::Error::Ok
                                           : runtime::Error::InvalidState;
}

/** Bridges committed K/V state and resumes a failed EdgeInfer decode natively.
 */
template <typename BridgeOperation, typename NativeDecodeOperation>
runtime::Result<int64_t> resume_native_decode_after_edgeinfer_failure(
    int64_t successful_edgeinfer_tokens,
    uint64_t failed_token,
    int64_t failed_position,
    BridgeOperation&& bridge_operation,
    NativeDecodeOperation&& native_decode_operation) {
  if (successful_edgeinfer_tokens < 0 || failed_position < 0) {
    return runtime::Error::InvalidArgument;
  }
  const runtime::Error bridge_error =
      std::forward<BridgeOperation>(bridge_operation)();
  if (bridge_error != runtime::Error::Ok) {
    return bridge_error;
  }
  auto native_result =
      std::forward<NativeDecodeOperation>(native_decode_operation)(
          std::vector<uint64_t>{failed_token}, failed_position);
  if (!native_result.ok()) {
    return native_result.error();
  }
  const int64_t native_tokens = native_result.get();
  if (native_tokens < 0 ||
      successful_edgeinfer_tokens >
          std::numeric_limits<int64_t>::max() - native_tokens) {
    return runtime::Error::InvalidState;
  }
  return successful_edgeinfer_tokens + native_tokens;
}

} // namespace llm
} // namespace extension
} // namespace executorch
