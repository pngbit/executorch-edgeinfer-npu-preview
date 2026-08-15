/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/runtime/Logging.h>
#include <executorch/backends/qualcomm/runtime/QnnBackendOptions.h>
#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnBackendUnifiedRegistry.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnLogger.h>
#include <executorch/backends/qualcomm/runtime/backends/gpu/GpuBackend.h>
#include <executorch/backends/qualcomm/runtime/backends/gpu/GpuDevice.h>
#include <executorch/backends/qualcomm/runtime/backends/htp/HtpBackend.h>
#include <executorch/backends/qualcomm/runtime/backends/htp/HtpDevice.h>
#include <executorch/backends/qualcomm/runtime/backends/lpai/LpaiBackend.h>
#include <executorch/backends/qualcomm/runtime/backends/lpai/LpaiDevice.h>

#include <pal/Path.h>

#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <type_traits>

namespace executorch {
namespace backends {
namespace qnn {
using executorch::runtime::Error;

namespace {

template <typename T>
void AppendScalar(std::string& identity, T value) {
  static_assert(std::is_trivially_copyable_v<T>);
  identity.append(reinterpret_cast<const char*>(&value), sizeof(value));
}

void AppendString(std::string& identity, const flatbuffers::String* value) {
  const bool present = value != nullptr;
  AppendScalar(identity, present);
  if (!present) {
    return;
  }
  const uint64_t size = value->size();
  AppendScalar(identity, size);
  identity.append(value->c_str(), value->size());
}

std::optional<std::string> ResolveBackendLibraryPath(
    const QnnExecuTorchOptions* options) {
  if (options == nullptr || options->backend_options() == nullptr) {
    return std::nullopt;
  }
  if (options->library_path() != nullptr &&
      !options->library_path()->str().empty()) {
    return options->library_path()->str();
  }
  switch (options->backend_options()->backend_type()) {
    case QnnExecuTorchBackendType::kHtpBackend:
#ifdef __hexagon__
      return std::string(HEXAGON_LIB);
#else
      return pal::path::GetLibraryName("QnnHtp");
#endif
    case QnnExecuTorchBackendType::kGpuBackend:
      return pal::path::GetLibraryName("QnnGpu");
    case QnnExecuTorchBackendType::kLpaiBackend:
      return pal::path::GetLibraryName("QnnLpai");
    case QnnExecuTorchBackendType::kDspBackend:
    case QnnExecuTorchBackendType::kUndefinedBackend:
      return std::nullopt;
  }
  return std::nullopt;
}

} // namespace

std::optional<std::string> BuildQnnBackendCacheIdentity(
    const QnnExecuTorchOptions* options) {
  auto library_path = ResolveBackendLibraryPath(options);
  if (!library_path.has_value() || options->soc_info() == nullptr) {
    return std::nullopt;
  }
  const auto* backend_options = options->backend_options();
  switch (backend_options->backend_type()) {
    case QnnExecuTorchBackendType::kHtpBackend:
      if (backend_options->htp_options() == nullptr ||
          options->soc_info()->htp_info() == nullptr) {
        return std::nullopt;
      }
      break;
    case QnnExecuTorchBackendType::kGpuBackend:
      if (backend_options->gpu_options() == nullptr) {
        return std::nullopt;
      }
      break;
    case QnnExecuTorchBackendType::kLpaiBackend:
      if (backend_options->lpai_options() == nullptr ||
          options->soc_info()->lpai_info() == nullptr) {
        return std::nullopt;
      }
      break;
    case QnnExecuTorchBackendType::kDspBackend:
    case QnnExecuTorchBackendType::kUndefinedBackend:
      return std::nullopt;
  }
  if (options->saver() && options->saver_output_dir() == nullptr) {
    return std::nullopt;
  }

  std::string identity;
  identity.reserve(256);
  constexpr uint32_t kIdentityVersion = 1;
  AppendScalar(identity, kIdentityVersion);
  AppendScalar(identity, backend_options->backend_type());
  const uint64_t library_size = library_path->size();
  AppendScalar(identity, library_size);
  identity.append(*library_path);

  // Logger lifetime is tied to Backend/Device. A different log level must use
  // a different immutable bundle rather than replacing a live logger.
  AppendScalar(
      identity, get_option(options->log_level(), QNN_RUNTIME_LOG_LEVEL));
  AppendScalar(identity, options->online_prepare());
  AppendScalar(identity, options->dump_intermediate_outputs());
  AppendScalar(
      identity,
      get_option(options->profile_level(), QNN_RUNTIME_PROFILE_LEVEL));
  AppendScalar(identity, options->shared_buffer());
  AppendScalar(identity, options->is_from_context_binary());
  AppendScalar(identity, options->saver());
  AppendString(identity, options->saver_output_dir());
  AppendScalar(identity, options->use_mha2sha());

  const auto* soc = options->soc_info();
  AppendScalar(identity, soc->soc_model());
  const bool has_htp_info = soc->htp_info() != nullptr;
  AppendScalar(identity, has_htp_info);
  if (has_htp_info) {
    AppendScalar(identity, soc->htp_info()->htp_arch());
    AppendScalar(identity, soc->htp_info()->vtcm_size_in_mb());
  }
  const bool has_lpai_info = soc->lpai_info() != nullptr;
  AppendScalar(identity, has_lpai_info);
  if (has_lpai_info) {
    AppendScalar(identity, soc->lpai_info()->lpai_hardware_version());
  }

  const auto* htp = backend_options->htp_options();
  const bool has_htp_options = htp != nullptr;
  AppendScalar(identity, has_htp_options);
  if (has_htp_options) {
    AppendScalar(identity, htp->max_sf_buf_size());
    AppendScalar(
        identity,
        get_option(htp->performance_mode(), QNN_RUNTIME_HTP_PERFORMANCE_MODE));
    AppendScalar(identity, htp->precision());
    AppendScalar(identity, htp->pd_session());
    AppendScalar(identity, htp->use_conv_hmx());
    AppendScalar(identity, htp->use_dlbc());
    AppendScalar(identity, htp->use_fold_relu());
    AppendScalar(identity, htp->use_multi_contexts());
    AppendScalar(identity, htp->use_weight_sharing());
    AppendScalar(identity, htp->use_slc_allocator());
  }

  const auto* gpu = backend_options->gpu_options();
  const bool has_gpu_options = gpu != nullptr;
  AppendScalar(identity, has_gpu_options);
  if (has_gpu_options) {
    AppendScalar(identity, gpu->performance_mode());
    AppendScalar(identity, gpu->precision());
    AppendScalar(identity, gpu->use_memory_optimizations());
    AppendScalar(identity, gpu->use_node_optimizations());
    AppendScalar(identity, gpu->use_queue_recording());
    AppendScalar(identity, gpu->use_weight_sharing());
  }

  const auto* lpai = backend_options->lpai_options();
  const bool has_lpai_options = lpai != nullptr;
  AppendScalar(identity, has_lpai_options);
  if (has_lpai_options) {
    AppendScalar(identity, get_option(lpai->fps(), QNN_RUNTIME_LPAI_FPS));
    AppendScalar(
        identity, get_option(lpai->ftrt_ratio(), QNN_RUNTIME_LPAI_FTRT_RATIO));
    AppendScalar(
        identity,
        get_option(
            lpai->client_perf_type(), QNN_RUNTIME_LPAI_CLIENT_PERF_TYPE));
    AppendScalar(
        identity, get_option(lpai->affinity(), QNN_RUNTIME_LPAI_AFFINITY));
    AppendScalar(
        identity,
        get_option(lpai->core_selection(), QNN_RUNTIME_LPAI_CORE_SELECTION));
    AppendScalar(identity, lpai->target_env());
  }

  const auto* package_options = options->op_package_options();
  const auto* packages = package_options == nullptr
      ? nullptr
      : package_options->op_package_infos();
  const uint64_t package_count = packages == nullptr ? 0 : packages->size();
  AppendScalar(identity, package_count);
  if (packages != nullptr) {
    for (const auto* package : *packages) {
      if (package == nullptr) {
        AppendScalar(identity, false);
        continue;
      }
      AppendScalar(identity, true);
      AppendString(identity, package->op_package_name());
      AppendString(identity, package->op_package_path());
      AppendString(identity, package->interface_provider());
      AppendScalar(identity, package->target());
      AppendString(identity, package->custom_op_name());
      AppendString(identity, package->qnn_op_type_name());
      AppendScalar(identity, package->platform());
    }
  }
  return identity;
}

std::optional<std::string> BuildQnnDelegateCacheIdentity(
    std::int64_t context_signature,
    const QnnExecuTorchOptions* options) {
  auto backend_identity = BuildQnnBackendCacheIdentity(options);
  if (!backend_identity.has_value()) {
    return std::nullopt;
  }
  std::string identity;
  identity.reserve(sizeof(context_signature) + backend_identity->size());
  AppendScalar(identity, context_signature);
  identity.append(*backend_identity);
  return identity;
}

// Static instance for the singleton
QnnBackendUnifiedRegistry& QnnBackendUnifiedRegistry::GetInstance() {
  static QnnBackendUnifiedRegistry instance;
  return instance;
}

// Private constructor
QnnBackendUnifiedRegistry::QnnBackendUnifiedRegistry() = default;

// Destructor
QnnBackendUnifiedRegistry::~QnnBackendUnifiedRegistry() {
  CleanupExpired();
}

Error QnnBackendUnifiedRegistry::GetOrCreateBackendBundle(
    const QnnExecuTorchOptions* options,
    std::shared_ptr<QnnBackendBundle>& bundle) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto cache_identity = BuildQnnBackendCacheIdentity(options);
  ET_CHECK_OR_RETURN_ERROR(
      cache_identity.has_value(),
      InvalidArgument,
      "Incomplete or unsupported QNN backend options");
  auto current_lib_path = ResolveBackendLibraryPath(options).value();
  QnnExecuTorchLogLevel current_log_level =
      get_option(options->log_level(), QNN_RUNTIME_LOG_LEVEL);
  QnnExecuTorchBackendType backend_type =
      options->backend_options()->backend_type();

  // Check if resources already exist
  auto it = qnn_backend_bundles_map_.find(*cache_identity);
  if (it != qnn_backend_bundles_map_.end()) {
    // Create new shared_ptr that shares ownership of the managed object.
    if (auto existing_bundle = it->second.lock()) {
      bundle = existing_bundle;
      QNN_EXECUTORCH_LOG_INFO(
          "Use cached backend bundle for current backend: %s",
          EnumNameQnnExecuTorchBackendType(backend_type));
      return Error::Ok;
    }
  }

  QNN_EXECUTORCH_LOG_INFO("Creating new backend bundle.");

  // 1. Create QnnImplementation and load qnn library
  std::unique_ptr<QnnImplementation> implementation =
      std::make_unique<QnnImplementation>(current_lib_path);
  auto config = GetImplementationConfig(options);
  Error ret = implementation->Load(config.get());
  ET_CHECK_OR_RETURN_ERROR(
      ret == Error::Ok, Internal, "Fail to load Qnn library");

  // 2. Create QnnLogger
  std::unique_ptr<QnnLogger> logger = std::make_unique<QnnLogger>(
      implementation.get(), LoggingCallback, current_log_level);

  // 3. Create QnnBackend (specific type based on options)
  // 4. Create QnnDevice (specific type based on options)
  std::unique_ptr<QnnBackend> backend = nullptr;
  std::unique_ptr<QnnDevice> device = nullptr;

  switch (backend_type) {
    case QnnExecuTorchBackendType::kHtpBackend: {
      auto htp_options = options->backend_options()->htp_options();
      backend =
          std::make_unique<HtpBackend>(implementation.get(), logger.get());
      device = std::make_unique<HtpDevice>(
          implementation.get(), logger.get(), options->soc_info(), htp_options);
      break;
    }
    case QnnExecuTorchBackendType::kGpuBackend: {
      auto gpu_options = options->backend_options()->gpu_options();
      backend = std::make_unique<GpuBackend>(
          implementation.get(), logger.get(), gpu_options);
      device = std::make_unique<GpuDevice>(implementation.get(), logger.get());
      break;
    }
    case QnnExecuTorchBackendType::kLpaiBackend: {
      auto lpai_options = options->backend_options()->lpai_options();
      backend = std::make_unique<LpaiBackend>(
          implementation.get(),
          logger.get(),
          options->soc_info(),
          lpai_options);
      device = std::make_unique<LpaiDevice>(implementation.get(), logger.get());
      break;
    }
    case QnnExecuTorchBackendType::kDspBackend:
    case QnnExecuTorchBackendType::kUndefinedBackend:
    default:
      return Error::NotFound;
  }
  ET_CHECK_OR_RETURN_ERROR(
      backend->Configure(options->op_package_options()) == Error::Ok,
      Internal,
      "Fail to configure Qnn backend");
  ET_CHECK_OR_RETURN_ERROR(
      device->Configure() == Error::Ok,
      Internal,
      "Fail to configure Qnn device");

  if (backend->VerifyQNNSDKVersion() != Error::Ok) {
    return Error::Internal;
  }
  // 5. Create QnnSystemImplementation and load qnn library
  std::unique_ptr<QnnSystemImplementation> system_implementation =
      std::make_unique<QnnSystemImplementation>(
          pal::path::GetLibraryName("QnnSystem"));
  ret = system_implementation->Load();
  ET_CHECK_OR_RETURN_ERROR(
      ret == Error::Ok, Internal, "Fail to load Qnn system library");

  auto new_bundle = std::make_shared<QnnBackendBundle>();
  new_bundle->implementation = std::move(implementation);
  new_bundle->system_implementation = std::move(system_implementation);
  new_bundle->qnn_logger_ptr = std::move(logger);
  new_bundle->qnn_backend_ptr = std::move(backend);
  new_bundle->qnn_device_ptr = std::move(device);
  bundle = new_bundle;
  qnn_backend_bundles_map_.insert_or_assign(*cache_identity, new_bundle);

  return Error::Ok;
}

void QnnBackendUnifiedRegistry::CleanupExpired() {
  std::lock_guard<std::mutex> lock(mutex_);

  for (auto it = qnn_backend_bundles_map_.begin();
       it != qnn_backend_bundles_map_.end();) {
    if (it->second.expired()) {
      it = qnn_backend_bundles_map_.erase(it);
    } else {
      ++it;
    }
  }
}

} // namespace qnn
} // namespace backends
} // namespace executorch
