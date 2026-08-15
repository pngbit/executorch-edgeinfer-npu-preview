/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace executorch {
namespace backends {
namespace qnn {

struct RpcMemAllocationLayout {
  int32_t allocation_bytes;
  size_t padding_bytes;
};

/// Validates the arguments accepted by rpcmem_alloc(), whose byte count is a
/// signed int, and reserves enough padding to align any returned address.
inline std::optional<RpcMemAllocationLayout> GetRpcMemAllocationLayout(
    size_t bytes,
    size_t alignment) {
  if (bytes == 0 || alignment == 0 || (alignment & (alignment - 1)) != 0) {
    return std::nullopt;
  }
  const size_t padding = alignment - 1;
  if (bytes > std::numeric_limits<size_t>::max() - padding) {
    return std::nullopt;
  }
  const size_t allocation_bytes = bytes + padding;
  constexpr size_t kMaxRpcMemBytes =
      static_cast<size_t>(std::numeric_limits<int32_t>::max());
  if (allocation_bytes > kMaxRpcMemBytes) {
    return std::nullopt;
  }
  return RpcMemAllocationLayout{
      static_cast<int32_t>(allocation_bytes), padding};
}

/// Computes the address returned to the caller without overflowing uintptr_t.
inline std::optional<uintptr_t> GetAlignedRpcMemAddress(
    uintptr_t raw_base,
    size_t alignment) {
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    return std::nullopt;
  }
  const uintptr_t mask = static_cast<uintptr_t>(alignment - 1);
  if (raw_base > std::numeric_limits<uintptr_t>::max() - mask) {
    return std::nullopt;
  }
  return (raw_base + mask) & ~mask;
}

struct RpcMemDescriptorRange {
  size_t total_bytes;
  size_t tensor_offset;
};

inline std::optional<size_t> GetAddressOffset(
    const void* base,
    const void* address) {
  if (base == nullptr || address == nullptr) {
    return std::nullopt;
  }
  const uintptr_t begin = reinterpret_cast<uintptr_t>(base);
  const uintptr_t value = reinterpret_cast<uintptr_t>(address);
  if (value < begin) {
    return std::nullopt;
  }
  const uintptr_t offset = value - begin;
  if constexpr (
      std::numeric_limits<uintptr_t>::max() >
      std::numeric_limits<size_t>::max()) {
    if (offset > static_cast<uintptr_t>(std::numeric_limits<size_t>::max())) {
      return std::nullopt;
    }
  }
  return static_cast<size_t>(offset);
}

/// Builds the QNN shared-buffer descriptor contract for a file descriptor
/// obtained from raw_base. tensor_offset_from_aligned is relative to the
/// aligned pointer exposed by QnnExecuTorchAllocCustomMem().
inline std::optional<RpcMemDescriptorRange> GetRpcMemDescriptorRange(
    const void* raw_base,
    const void* aligned_base,
    size_t allocation_bytes,
    size_t usable_bytes,
    size_t tensor_offset_from_aligned,
    size_t tensor_bytes) {
  if (raw_base == nullptr || aligned_base == nullptr || allocation_bytes == 0) {
    return std::nullopt;
  }
  auto prefix = GetAddressOffset(raw_base, aligned_base);
  if (!prefix.has_value()) {
    return std::nullopt;
  }
  if (*prefix > allocation_bytes || usable_bytes > allocation_bytes - *prefix ||
      tensor_offset_from_aligned > usable_bytes) {
    return std::nullopt;
  }
  const size_t final_offset = *prefix + tensor_offset_from_aligned;
  if (tensor_bytes > usable_bytes - tensor_offset_from_aligned) {
    return std::nullopt;
  }
  return RpcMemDescriptorRange{allocation_bytes, final_offset};
}

inline bool
IsAddressInRange(const void* address, const void* base, size_t bytes) {
  if (address == nullptr || base == nullptr || bytes == 0) {
    return false;
  }
  const uintptr_t begin = reinterpret_cast<uintptr_t>(base);
  if (bytes > std::numeric_limits<uintptr_t>::max() - begin) {
    return false;
  }
  const uintptr_t value = reinterpret_cast<uintptr_t>(address);
  return value >= begin && value < begin + bytes;
}

template <typename RestoreMap>
void* ResolveRpcAllocationBase(
    const RestoreMap& restore_map,
    const void* aligned_address) {
  auto it = restore_map.find(const_cast<void*>(aligned_address));
  return it == restore_map.end() ? nullptr : it->second;
}

template <typename HandleMap, typename Deregister, typename OnReleased>
bool DeregisterHandlesInRange(
    HandleMap& handles,
    void* base,
    size_t bytes,
    Deregister&& deregister,
    OnReleased&& on_released) {
  bool success = true;
  for (auto it = handles.begin(); it != handles.end();) {
    if (!IsAddressInRange(it->second, base, bytes)) {
      ++it;
      continue;
    }
    if (!deregister(it->first)) {
      success = false;
      ++it;
      continue;
    }
    on_released(it->first);
    it = handles.erase(it);
  }
  return success;
}

} // namespace qnn
} // namespace backends
} // namespace executorch
