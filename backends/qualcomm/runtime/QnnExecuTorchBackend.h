/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/backends/qualcomm/runtime/QnnBackendOptions.h>
#include <executorch/backends/qualcomm/runtime/QnnDelegateHandleRegistry.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>

#include <mutex>

namespace executorch {
namespace backends {
namespace qnn {

class QnnExecuTorchBackend final
    : public ::executorch::runtime::BackendInterface {
 public:
  ~QnnExecuTorchBackend(){};

  executorch::runtime::Result<executorch::runtime::DelegateHandle*> init(
      executorch::runtime::BackendInitContext& context,
      executorch::runtime::FreeableBuffer* processed,
      executorch::runtime::ArrayRef<executorch::runtime::CompileSpec>
          compile_specs) const override;

  executorch::runtime::Error execute(
      ET_UNUSED executorch::runtime::BackendExecutionContext& context,
      executorch::runtime::DelegateHandle* handle,
      executorch::runtime::Span<executorch::runtime::EValue*> args)
      const override;

  ET_NODISCARD executorch::runtime::Error set_option(
      executorch::runtime::BackendOptionContext& context,
      const executorch::runtime::Span<executorch::runtime::BackendOption>&
          backend_options) override;

  executorch::runtime::Error get_option(
      executorch::runtime::BackendOptionContext& context,
      executorch::runtime::Span<executorch::runtime::BackendOption>&
          backend_options) override;

  void destroy(executorch::runtime::DelegateHandle* handle) const override;

  bool is_available() const override;

 private:
  mutable std::mutex runtime_option_mutex_;
  mutable QnnDelegateHandleRegistry<executorch::runtime::DelegateHandle>
      delegate_registry_;

  RuntimeOption qnn_runtime_log_level_{false, 0};
  RuntimeOption qnn_runtime_performance_mode_{false, 0};
  RuntimeOption qnn_runtime_profile_level_{false, 0};
  RuntimeOption qnn_runtime_lpai_fps_{false, 0};
  RuntimeOption qnn_runtime_lpai_ftrt_ratio_{false, 0};
  RuntimeOption qnn_runtime_lpai_client_perf_type_{false, 0};
  RuntimeOption qnn_runtime_lpai_affinity_{false, 0};
  RuntimeOption qnn_runtime_lpai_core_selection_{false, 0};
  RuntimeOption qnn_runtime_heap_profiling_path_{false, {}};
};

} // namespace qnn
} // namespace backends
} // namespace executorch
