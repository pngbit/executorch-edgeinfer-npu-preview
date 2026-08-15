/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <QnnTypes.h>
#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/runtime/core/error.h>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <unordered_set>

using RpcMemAllocFn_t = void* (*)(int, uint32_t, int);
using RpcMemFreeFn_t = void (*)(void*);
using RpcMemToFdFn_t = int (*)(void*);

// TODO Finad a better file to place CustomMemTensorInfo
bool operator==(const CustomMemTensorInfo& lhs, const CustomMemTensorInfo& rhs);
template <>
struct std::hash<CustomMemTensorInfo> {
  std::size_t operator()(const CustomMemTensorInfo& info) const noexcept;
};

namespace executorch {
namespace backends {
namespace qnn {

class SharedBuffer final {
 public:
  class AllocationLease final {
   public:
    AllocationLease() = default;
    AllocationLease(const AllocationLease&) = delete;
    AllocationLease& operator=(const AllocationLease&) = delete;
    AllocationLease(AllocationLease&& other) noexcept;
    AllocationLease& operator=(AllocationLease&& other) noexcept;
    ~AllocationLease();

    explicit operator bool() const {
      return manager_ != nullptr;
    }

   private:
    friend class SharedBuffer;
    AllocationLease(SharedBuffer* manager, void* allocation)
        : manager_(manager), allocation_(allocation) {}
    void Reset();

    SharedBuffer* manager_{nullptr};
    void* allocation_{nullptr};
  };

  SharedBuffer(const SharedBuffer&) = delete;
  SharedBuffer& operator=(const SharedBuffer&) = delete;
  SharedBuffer(SharedBuffer&&) = delete;
  SharedBuffer& operator=(SharedBuffer&&) = delete;
  ~SharedBuffer();

  static SharedBuffer& GetSharedBufferManager();
  void* AllocMem(size_t bytes, size_t alignment);
  // map a buffer allocated via RPCMem to a file descriptor so it can be
  // registered with a backend via QnnMem_register()
  int32_t MemToFd(void* buf);
  int32_t MemToFd(void* buf, const AllocationLease& lease);

  void FreeMem(void* buf);

  bool IsAllocated(void* buf);

  // Pins an allocation against FreeMem while a caller performs registration or
  // another external operation that uses the underlying fd.
  AllocationLease AcquireAllocation(void* buf);

  bool GetInitialize() {
    return initialize_;
  }
  void SetInitialize(bool initialize) {
    initialize_ = initialize;
  }

  // memory handle is registered during execution
  void AddCusomMemTensorAddr(void* tensor_addr, void* custom_mem);

  size_t GetAllocatedSize(void* buf);

  size_t GetAllocationRangeSize(void* buf);

  /// Returns QNN descriptor size/offset for the raw allocation represented by
  /// buf. tensor_offset is relative to the aligned pointer returned by
  /// AllocMem(); the returned offset is relative to the raw-base fd.
  bool GetCustomMemDescriptorRange(
      void* buf,
      size_t tensor_offset,
      size_t tensor_bytes,
      size_t* total_bytes,
      size_t* fd_offset);

  void* GetCustomMemBase(void* buf);

  // Transactional release for callers that must deregister external QNN
  // handles before freeing the RPC allocation.
  std::optional<size_t> BeginFree(void* buf);
  void CancelFree(void* buf);
  void CommitFree(void* buf);

 private:
  friend class SharedBufferTestPeer;
  SharedBuffer() = default;

  // dlopen RPCMem library and dlysm required functions
  executorch::runtime::Error Load();

  executorch::runtime::Error UnLoad();

  void ReleaseAllocation(void* buf);

  // Pointer to the dlopen'd libcdsprpc.so shared library which contains
  // rpcmem_alloc, rpcmem_free, rpcmem_to_fd APIs
  void* lib_cdsp_rpc_{nullptr};
  // Function pointer to rpcmem_alloc
  RpcMemAllocFn_t rpc_mem_alloc_{nullptr};
  // Function pointer to rpcmem_free
  RpcMemFreeFn_t rpc_mem_free_{nullptr};
  // Function pointer to rpcmem_to_fd
  RpcMemToFdFn_t rpc_mem_to_fd_{nullptr};
  std::unordered_map<void*, void*> restore_map_;
  std::unordered_map<void*, size_t> allocated_size_map_;
  std::unordered_map<void*, size_t> allocation_range_size_map_;
  // Maps for the custom memory
  std::unordered_map<void*, void*> tensor_addr_to_custom_mem_;
  // After the custom memory is freed, we will ensure that no tensor addresses
  // remain linked to this custom memory.
  std::unordered_map<void*, std::unordered_set<void*>>
      custom_mem_to_tensor_addr_;
  // Protects all allocation and tensor-address maps. Vendor RPC functions are
  // always called after releasing this lock.
  mutable std::mutex state_mutex_;
  std::condition_variable no_active_allocations_;
  std::unordered_map<void*, size_t> active_allocations_;
  std::unordered_set<void*> freeing_allocations_;
  std::atomic_bool initialize_{false};
  static std::mutex init_mutex_;
};

} // namespace qnn
} // namespace backends
} // namespace executorch
