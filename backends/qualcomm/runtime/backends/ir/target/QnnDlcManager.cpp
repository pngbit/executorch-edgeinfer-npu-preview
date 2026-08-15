/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/runtime/backends/QnnDlcManager.h>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <fstream>

namespace executorch {
namespace backends {
namespace qnn {

QnnDlcManager::QnnDlcManager(
    const QnnExecuTorchContextBinary& qnn_context_blob,
    const QnnExecuTorchOptions* options)
    : qnn_context_blob_(qnn_context_blob), options_(options) {
  if (options_ == nullptr) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Fail to create QnnDlcManager, options is nullptr");
  }
}

Error QnnDlcManager::LoadQnnIrLibrary() {
  return Error::Ok;
}

Error QnnDlcManager::Create() {
  return Error::Ok;
}

Error QnnDlcManager::Configure(const std::vector<std::string>& graph_names) {
  return Error::Ok;
}

Error QnnDlcManager::SetUpDlcEnvironment(
    const Qnn_Version_t& coreApiVersion,
    const std::vector<std::string>& graph_names) {
  return Error::Ok;
}

void QnnDlcManager::Destroy() {
  // The target-side DLC logger is created from the execution backend's
  // QnnImplementation. Release it before QnnManager drops that implementation;
  // otherwise QnnLogger's destructor calls through an unloaded interface.
  detail::ReleaseQnnDlcOwnedResources(backend_params_ptr_, backend_bundle_ptr_);
  // Preserve the reusable empty state expected after DestroyContext().
  backend_params_ptr_ = std::make_unique<BackendConfigParameters>();
  backend_bundle_ptr_ = std::make_unique<QnnBackendBundle>();
}

} // namespace qnn
} // namespace backends
} // namespace executorch
