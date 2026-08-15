/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/runtime/Logging.h>
#include <executorch/backends/qualcomm/runtime/QnnMemoryRange.h>
#include <executorch/backends/qualcomm/runtime/SharedBuffer.h>

#include <pal/DynamicLoading.h>

// Refer to the QNN HTP Shared Buffer Tutorial
// in Qualcomm® AI Engine Direct document
constexpr uint8_t RPCMEM_HEAP_ID_SYSTEM = 25;
constexpr uint8_t RPCMEM_DEFAULT_FLAGS = 1;

std::size_t std::hash<CustomMemTensorInfo>::operator()(
    const CustomMemTensorInfo& info) const noexcept {
  size_t hash_val = 0;
  hash_val ^= std::hash<void*>()(info.tensor_addr);
  hash_val ^= std::hash<void*>()(info.custom_mem);
  hash_val ^= std::hash<size_t>()(info.pos);
  hash_val ^= std::hash<size_t>()(info.tensor_bytes);
  for (size_t i = 0; i < info.rank; ++i) {
    hash_val ^= std::hash<uint32_t>()(info.shape[i]);
  }
  hash_val ^= std::hash<uint32_t>()(info.rank);
  hash_val ^= std::hash<executorch::aten::ScalarType>()(info.dtype);
  return hash_val;
}

bool operator==(
    const CustomMemTensorInfo& lhs,
    const CustomMemTensorInfo& rhs) {
  bool is_same =
      (lhs.tensor_addr == rhs.tensor_addr && lhs.custom_mem == rhs.custom_mem &&
       lhs.pos == rhs.pos && lhs.tensor_bytes == rhs.tensor_bytes &&
       lhs.rank == rhs.rank && lhs.dtype == rhs.dtype);
  for (size_t i = 0; i < lhs.rank; ++i) {
    is_same &= lhs.shape[i] == rhs.shape[i];
  }
  return is_same;
}

namespace executorch {
namespace backends {
namespace qnn {

using executorch::runtime::Error;

std::mutex SharedBuffer::init_mutex_;

SharedBuffer::AllocationLease::AllocationLease(AllocationLease&& other) noexcept
    : manager_(other.manager_), allocation_(other.allocation_) {
  other.manager_ = nullptr;
  other.allocation_ = nullptr;
}

SharedBuffer::AllocationLease& SharedBuffer::AllocationLease::operator=(
    AllocationLease&& other) noexcept {
  if (this != &other) {
    Reset();
    manager_ = other.manager_;
    allocation_ = other.allocation_;
    other.manager_ = nullptr;
    other.allocation_ = nullptr;
  }
  return *this;
}

SharedBuffer::AllocationLease::~AllocationLease() {
  Reset();
}

void SharedBuffer::AllocationLease::Reset() {
  if (manager_ != nullptr) {
    manager_->ReleaseAllocation(allocation_);
  }
  manager_ = nullptr;
  allocation_ = nullptr;
}

void* SharedBuffer::GetCustomMemBase(void* buf) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  auto it = tensor_addr_to_custom_mem_.find(buf);
  if (it == tensor_addr_to_custom_mem_.end()) {
    return nullptr;
  }
  return it->second;
}

size_t SharedBuffer::GetAllocatedSize(void* buf) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (freeing_allocations_.count(buf) != 0) {
    return 0;
  }
  auto it = allocated_size_map_.find(buf);
  if (it == allocated_size_map_.end()) {
    return 0;
  }
  return it->second;
}

size_t SharedBuffer::GetAllocationRangeSize(void* buf) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (freeing_allocations_.count(buf) != 0) {
    return 0;
  }
  auto it = allocation_range_size_map_.find(buf);
  return it == allocation_range_size_map_.end() ? 0 : it->second;
}

bool SharedBuffer::GetCustomMemDescriptorRange(
    void* buf,
    size_t tensor_offset,
    size_t tensor_bytes,
    size_t* total_bytes,
    size_t* fd_offset) {
  if (total_bytes == nullptr || fd_offset == nullptr) {
    return false;
  }
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (freeing_allocations_.count(buf) != 0) {
    return false;
  }
  void* raw_base = ResolveRpcAllocationBase(restore_map_, buf);
  auto allocated = allocated_size_map_.find(buf);
  auto usable = allocation_range_size_map_.find(buf);
  if (raw_base == nullptr || allocated == allocated_size_map_.end() ||
      usable == allocation_range_size_map_.end()) {
    return false;
  }
  auto range = GetRpcMemDescriptorRange(
      raw_base,
      buf,
      allocated->second,
      usable->second,
      tensor_offset,
      tensor_bytes);
  if (!range.has_value()) {
    return false;
  }
  *total_bytes = range->total_bytes;
  *fd_offset = range->tensor_offset;
  return true;
}

SharedBuffer& SharedBuffer::GetSharedBufferManager() {
  std::lock_guard<std::mutex> lk(init_mutex_);
  static SharedBuffer shared_buffer_manager;
  if (!shared_buffer_manager.GetInitialize()) {
#if defined(__ANDROID__)
    Error status = shared_buffer_manager.Load();
    if (status == Error::Ok) {
      shared_buffer_manager.SetInitialize(true);
    }
#else
    // libcdsprpc.so is Android-only. Keep the manager unavailable so every
    // custom-memory API fails closed instead of calling null RPC pointers.
#endif
  }
  return shared_buffer_manager;
}

// FastRPC listener/HwBinder threads can outlive C++ static destruction, so the
// process-lifetime libcdsprpc.so reference must remain loaded during teardown.
SharedBuffer::~SharedBuffer() = default;

void* SharedBuffer::AllocMem(size_t bytes, size_t alignment) {
  RpcMemAllocFn_t alloc_fn = nullptr;
  RpcMemFreeFn_t free_fn = nullptr;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (!initialize_ || rpc_mem_alloc_ == nullptr || rpc_mem_free_ == nullptr) {
      QNN_EXECUTORCH_LOG_ERROR("Shared memory not initialized.");
      return nullptr;
    }
    alloc_fn = rpc_mem_alloc_;
    free_fn = rpc_mem_free_;
  }
  auto layout = GetRpcMemAllocationLayout(bytes, alignment);
  if (!layout.has_value()) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Invalid RPC memory allocation: bytes=%zu alignment=%zu.",
        bytes,
        alignment);
    return nullptr;
  }
  void* buf = alloc_fn(
      RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, layout->allocation_bytes);
  if (buf == nullptr) {
    QNN_EXECUTORCH_LOG_WARN("Failed to allocate the tensor by RPC memory.");
    return nullptr;
  }
  auto aligned_address =
      GetAlignedRpcMemAddress(reinterpret_cast<uintptr_t>(buf), alignment);
  if (!aligned_address.has_value()) {
    QNN_EXECUTORCH_LOG_ERROR("RPC memory address alignment overflow.");
    free_fn(buf);
    return nullptr;
  }
  auto* aligned_buf = reinterpret_cast<void*>(*aligned_address);
  bool inserted = false;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    inserted = restore_map_.insert({aligned_buf, buf}).second;
    if (inserted) {
      allocated_size_map_.insert(
          {aligned_buf, static_cast<size_t>(layout->allocation_bytes)});
      allocation_range_size_map_.insert({aligned_buf, bytes});
      active_allocations_.insert({aligned_buf, 0});
    }
  }
  if (!inserted) {
    QNN_EXECUTORCH_LOG_ERROR("Failed to allocate the tensor by RPC memory.");
    free_fn(buf);
    return nullptr;
  }
  return aligned_buf;
}

int32_t SharedBuffer::MemToFd(void* buf) {
  auto allocation_lease = AcquireAllocation(buf);
  if (!allocation_lease) {
    return -1;
  }
  return MemToFd(buf, allocation_lease);
}

int32_t SharedBuffer::MemToFd(void* buf, const AllocationLease& lease) {
  if (lease.manager_ != this || lease.allocation_ != buf) {
    return -1;
  }
  void* allocation_base = nullptr;
  RpcMemToFdFn_t to_fd_fn = nullptr;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (!initialize_ || rpc_mem_to_fd_ == nullptr) {
      QNN_EXECUTORCH_LOG_ERROR("Shared memory not initialized.");
      return -1;
    }
    allocation_base = ResolveRpcAllocationBase(restore_map_, buf);
    if (allocation_base == nullptr) {
      QNN_EXECUTORCH_LOG_ERROR("Cannot resolve RPC memory base for %p.", buf);
      return -1;
    }
    to_fd_fn = rpc_mem_to_fd_;
  }
  return to_fd_fn(allocation_base);
}

void SharedBuffer::FreeMem(void* buf) {
  auto bytes = BeginFree(buf);
  if (!bytes.has_value()) {
    QNN_EXECUTORCH_LOG_WARN("Don't free an unallocated tensor.");
    return;
  }
  CommitFree(buf);
}

std::optional<size_t> SharedBuffer::BeginFree(void* buf) {
  std::unique_lock<std::mutex> lock(state_mutex_);
  if (!initialize_ || rpc_mem_free_ == nullptr) {
    QNN_EXECUTORCH_LOG_ERROR("Shared memory not initialized.");
    return std::nullopt;
  }
  auto allocation = restore_map_.find(buf);
  auto range = allocation_range_size_map_.find(buf);
  if (allocation == restore_map_.end() ||
      range == allocation_range_size_map_.end() ||
      freeing_allocations_.count(buf) != 0) {
    return std::nullopt;
  }
  freeing_allocations_.insert(buf);
  no_active_allocations_.wait(lock, [&]() {
    auto active = active_allocations_.find(buf);
    return active == active_allocations_.end() || active->second == 0;
  });
  return range->second;
}

void SharedBuffer::CancelFree(void* buf) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  freeing_allocations_.erase(buf);
}

void SharedBuffer::CommitFree(void* buf) {
  void* allocation_base = nullptr;
  RpcMemFreeFn_t free_fn = nullptr;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (!initialize_ || rpc_mem_free_ == nullptr) {
      QNN_EXECUTORCH_LOG_ERROR("Shared memory not initialized.");
      return;
    }
    auto allocation = restore_map_.find(buf);
    if (allocation == restore_map_.end() ||
        freeing_allocations_.count(buf) == 0) {
      QNN_EXECUTORCH_LOG_WARN("Custom-memory free was not begun.");
      return;
    }
    allocation_base = allocation->second;
    free_fn = rpc_mem_free_;
    restore_map_.erase(buf);
    allocated_size_map_.erase(buf);
    allocation_range_size_map_.erase(buf);
    active_allocations_.erase(buf);
    // Unbind the custom memory from tensor address.
    auto mit = custom_mem_to_tensor_addr_.find(buf);
    if (mit != custom_mem_to_tensor_addr_.end()) {
      for (auto it = mit->second.begin(); it != mit->second.end(); ++it) {
        tensor_addr_to_custom_mem_.erase(*it);
      }
      custom_mem_to_tensor_addr_.erase(buf);
    }
    freeing_allocations_.erase(buf);
  }
  free_fn(allocation_base);
}

bool SharedBuffer::IsAllocated(void* buf) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return restore_map_.count(buf) != 0U && freeing_allocations_.count(buf) == 0;
}

SharedBuffer::AllocationLease SharedBuffer::AcquireAllocation(void* buf) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (!initialize_) {
    return {};
  }
  void* allocation = buf;
  if (restore_map_.count(allocation) == 0) {
    auto custom = tensor_addr_to_custom_mem_.find(buf);
    if (custom == tensor_addr_to_custom_mem_.end()) {
      return {};
    }
    allocation = custom->second;
  }
  if (freeing_allocations_.count(allocation) != 0) {
    return {};
  }
  ++active_allocations_[allocation];
  return AllocationLease(this, allocation);
}

void SharedBuffer::ReleaseAllocation(void* buf) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  auto active = active_allocations_.find(buf);
  if (active == active_allocations_.end() || active->second == 0) {
    return;
  }
  if (--active->second == 0 && freeing_allocations_.count(buf) != 0) {
    no_active_allocations_.notify_all();
  }
}

Error SharedBuffer::Load() {
  // On Android, 32-bit and 64-bit libcdsprpc.so can be found at /vendor/lib/
  // and /vendor/lib64/ respectively.
  lib_cdsp_rpc_ = pal::dynamic_loading::DlOpen(
      "libcdsprpc.so",
      pal::dynamic_loading::DL_NOW | pal::dynamic_loading::DL_LOCAL);
  if (lib_cdsp_rpc_ == nullptr) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Unable to load shared buffer. dlerror(): %s",
        pal::dynamic_loading::DlError());
    return Error::Internal;
  }
  rpc_mem_alloc_ = reinterpret_cast<RpcMemAllocFn_t>( // NOLINT
      pal::dynamic_loading::DlSym(lib_cdsp_rpc_, "rpcmem_alloc"));
  rpc_mem_free_ = reinterpret_cast<RpcMemFreeFn_t>( // NOLINT
      pal::dynamic_loading::DlSym(lib_cdsp_rpc_, "rpcmem_free"));
  rpc_mem_to_fd_ = reinterpret_cast<RpcMemToFdFn_t>( // NOLINT
      pal::dynamic_loading::DlSym(lib_cdsp_rpc_, "rpcmem_to_fd"));
  if (nullptr == rpc_mem_alloc_ || nullptr == rpc_mem_free_ ||
      nullptr == rpc_mem_to_fd_) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Unable to access symbols in shared buffer. dlerror(): %s",
        pal::dynamic_loading::DlError());
    pal::dynamic_loading::DlClose(lib_cdsp_rpc_);
    lib_cdsp_rpc_ = nullptr;
    rpc_mem_alloc_ = nullptr;
    rpc_mem_free_ = nullptr;
    rpc_mem_to_fd_ = nullptr;
    return Error::Internal;
  }
  return Error::Ok;
}

void SharedBuffer::AddCusomMemTensorAddr(void* tensor_addr, void* custom_mem) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (restore_map_.count(custom_mem) == 0 ||
      freeing_allocations_.count(custom_mem) != 0) {
    QNN_EXECUTORCH_LOG_WARN(
        "Cannot associate tensor address %p with unknown custom memory %p",
        tensor_addr,
        custom_mem);
    return;
  }
  bool status =
      tensor_addr_to_custom_mem_.insert({tensor_addr, custom_mem}).second;
  if (!status) {
    QNN_EXECUTORCH_LOG_WARN(
        "Tensor address %p already associated with custom memory %p",
        tensor_addr,
        custom_mem);
    return;
  }
  custom_mem_to_tensor_addr_[custom_mem].insert(tensor_addr);
};

Error SharedBuffer::UnLoad() {
  if (lib_cdsp_rpc_ == nullptr) {
    return Error::Ok;
  }
  if (pal::dynamic_loading::DlClose(lib_cdsp_rpc_) != 0) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Unable to close shared buffer. dlerror(): %s",
        pal::dynamic_loading::DlError());
    return Error::Internal;
  };
  lib_cdsp_rpc_ = nullptr;
  rpc_mem_alloc_ = nullptr;
  rpc_mem_free_ = nullptr;
  rpc_mem_to_fd_ = nullptr;
  initialize_ = false;
  return Error::Ok;
}
} // namespace qnn
} // namespace backends
} // namespace executorch
