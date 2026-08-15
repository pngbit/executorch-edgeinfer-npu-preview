/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/runtime/backends/QnnMemManager.h>

#include <executorch/backends/qualcomm/runtime/QnnMemoryRange.h>

#include <mutex>
#include <unordered_set>

namespace executorch {
namespace backends {
namespace qnn {

using executorch::runtime::Error;

namespace {

struct ActiveMemManagers {
  std::mutex mutex;
  std::unordered_set<QnnMemManager*> managers;
};

ActiveMemManagers& GetActiveMemManagers() {
  static ActiveMemManagers managers;
  return managers;
}

} // namespace

QnnMemManager::QnnMemManager(
    QnnImplementation* implementation,
    QnnContext* context,
    QnnExecuTorchLogLevel log_level)
    : implementation_(implementation),
      context_(context),
      log_level_(log_level) {
  ActiveMemManagers& active = GetActiveMemManagers();
  std::lock_guard<std::mutex> lock(active.mutex);
  active.managers.insert(this);
}

QnnMemManager::~QnnMemManager() {
  ActiveMemManagers& active = GetActiveMemManagers();
  std::lock_guard<std::mutex> active_lock(active.mutex);
  DeRegisterMem();
  active.managers.erase(this);
}

bool QnnMemManager::IsRegistered(Qnn_MemHandle_t handle, void* mem_ptr) {
  auto it = registered_map_.find(handle);
  if (it != registered_map_.end()) {
    return it->second == mem_ptr;
  }
  return false;
}

Error QnnMemManager::RegisterIonMem(
    const std::shared_ptr<TensorWrapper>& tensor_wrapper,
    int32_t mem_fd,
    void* mem_ptr) {
  ActiveMemManagers& active = GetActiveMemManagers();
  std::lock_guard<std::mutex> active_lock(active.mutex);
  const QnnInterface& qnn_interface = implementation_->GetQnnInterface();
  Qnn_MemDescriptor_t descriptor = {
      {tensor_wrapper->GetRank(), tensor_wrapper->GetDims(), nullptr},
      tensor_wrapper->GetDataType(),
      QNN_MEM_TYPE_ION,
      {{mem_fd}}};
  Qnn_MemHandle_t handle = nullptr;
  Qnn_ErrorHandle_t error = QNN_SUCCESS;
  error = qnn_interface.qnn_mem_register(
      context_->GetHandle(),
      &descriptor,
      /*numDescriptors=*/1,
      &handle);
  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_WARN(
        "Tensor %s is failed to register shared memory. Error %d",
        tensor_wrapper->GetName().c_str(),
        QNN_GET_ERROR_CODE(error));
    return Error::Internal;
  }
  tensor_wrapper->SetMemHandle(handle);
  registered_map_.insert({handle, mem_ptr});
  registered_tensors_[handle].push_back(tensor_wrapper);
  if (log_level_ >= QnnExecuTorchLogLevel::kLogLevelInfo) {
    QNN_EXECUTORCH_LOG_INFO(
        "Tensor %s is successfully registered to ION shared memory.",
        tensor_wrapper->GetName().c_str());
  }

  return Error::Ok;
}

Error QnnMemManager::RegisterCustomMem(
    const std::shared_ptr<TensorWrapper>& tensor_wrapper,
    int32_t mem_fd,
    void* mem_ptr,
    size_t total_custom_mem_size,
    size_t tensor_offset,
    const CustomMemTensorInfo& info) {
  ActiveMemManagers& active = GetActiveMemManagers();
  std::lock_guard<std::mutex> active_lock(active.mutex);
  const QnnInterface& qnn_interface = implementation_->GetQnnInterface();
  Qnn_MemDescriptor_t descriptor = {
      {tensor_wrapper->GetRank(), tensor_wrapper->GetDims(), nullptr},
      tensor_wrapper->GetDataType(),
      QNN_MEM_TYPE_CUSTOM,
      {{mem_fd}}};
  Qnn_MemHandle_t handle = nullptr;
  Qnn_ErrorHandle_t error = QNN_SUCCESS;

  QnnMemHtp_Descriptor_t htp_descriptor;
  htp_descriptor.type = QNN_HTP_MEM_SHARED_BUFFER;
  htp_descriptor.size = total_custom_mem_size;

  QnnHtpMem_SharedBufferConfig_t htpSharedBuffConfig = {mem_fd, tensor_offset};
  htp_descriptor.sharedBufferConfig = htpSharedBuffConfig;

  descriptor.customInfo = &htp_descriptor;

  error = qnn_interface.qnn_mem_register(
      context_->GetHandle(),
      &descriptor,
      /*numDescriptors=*/1,
      &handle);
  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_WARN(
        "Tensor %s is failed to register shared memory. Error %d",
        tensor_wrapper->GetName().c_str(),
        QNN_GET_ERROR_CODE(error));
    return Error::Internal;
  }
  tensor_wrapper->SetMemHandle(handle);
  pre_registered_handles_.insert({info, handle});
  registered_map_.insert({handle, mem_ptr});
  registered_tensors_[handle].push_back(tensor_wrapper);
  if (log_level_ >= QnnExecuTorchLogLevel::kLogLevelInfo) {
    QNN_EXECUTORCH_LOG_INFO(
        "Tensor %s is successfully registered to custom shared memory.",
        tensor_wrapper->GetName().c_str());
  }
  return Error::Ok;
}

void* QnnMemManager::GetPreRegisteredHandle(const CustomMemTensorInfo& info) {
  auto it = pre_registered_handles_.find(info);
  if (it == pre_registered_handles_.end()) {
    return nullptr;
  }
  return it->second;
}

Error QnnMemManager::SetMemHandle(
    const std::shared_ptr<TensorWrapper>& tensor_wrapper,
    void* mem_ptr,
    Qnn_MemHandle_t handle) {
  ActiveMemManagers& active = GetActiveMemManagers();
  std::lock_guard<std::mutex> active_lock(active.mutex);
  tensor_wrapper->SetMemHandle(handle);
  registered_map_.insert({handle, mem_ptr});
  registered_tensors_[handle].push_back(tensor_wrapper);
  return Error::Ok;
}

Error QnnMemManager::DeregisterMemRange(void* base_ptr, size_t bytes) {
  if (base_ptr == nullptr || bytes == 0) {
    return Error::InvalidArgument;
  }

  const QnnInterface& qnn_interface = implementation_->GetQnnInterface();
  std::unordered_set<Qnn_MemHandle_t> released;
  const bool success = DeregisterHandlesInRange(
      registered_map_,
      base_ptr,
      bytes,
      [&](Qnn_MemHandle_t handle) {
        const Qnn_ErrorHandle_t error =
            qnn_interface.qnn_mem_de_register(&handle, /*numHandles=*/1);
        if (error != QNN_SUCCESS) {
          QNN_EXECUTORCH_LOG_WARN(
              "Failed to de-register shared memory range handle. Error %d",
              QNN_GET_ERROR_CODE(error));
          return false;
        }
        return true;
      },
      [&](Qnn_MemHandle_t handle) {
        released.insert(handle);
        auto tensors = registered_tensors_.find(handle);
        if (tensors != registered_tensors_.end()) {
          for (auto& weak_tensor : tensors->second) {
            if (auto tensor = weak_tensor.lock()) {
              tensor->SetMemHandle(nullptr);
            }
          }
          registered_tensors_.erase(tensors);
        }
      });

  for (auto it = pre_registered_handles_.begin();
       it != pre_registered_handles_.end();) {
    if (released.count(it->second) != 0U) {
      it = pre_registered_handles_.erase(it);
    } else {
      ++it;
    }
  }
  return success ? Error::Ok : Error::Internal;
}

Error QnnMemManager::DeregisterAllMemRange(
    void* base_ptr,
    size_t bytes,
    ReleaseMemoryFn release_memory) {
  if (base_ptr == nullptr || bytes == 0) {
    return Error::InvalidArgument;
  }
  ActiveMemManagers& active = GetActiveMemManagers();
  std::lock_guard<std::mutex> lock(active.mutex);
  bool failed = false;
  for (QnnMemManager* manager : active.managers) {
    if (manager->DeregisterMemRange(base_ptr, bytes) != Error::Ok) {
      failed = true;
    }
  }
  if (failed) {
    return Error::Internal;
  }
  if (release_memory != nullptr) {
    release_memory(base_ptr);
  }
  return Error::Ok;
}

void QnnMemManager::DeRegisterMem() {
  const QnnInterface& qnn_interface = implementation_->GetQnnInterface();
  Qnn_ErrorHandle_t error = QNN_SUCCESS;

  for (auto& it : registered_map_) {
    error = qnn_interface.qnn_mem_de_register(&it.first, /*numHandles=*/1);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_WARN(
          "Failed to de-register shared memory. Error %d",
          QNN_GET_ERROR_CODE(error));
    }
  }
  registered_map_.clear();
  registered_tensors_.clear();
}

} // namespace qnn
} // namespace backends
} // namespace executorch
