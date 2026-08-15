/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

namespace executorch {
namespace backends {
namespace qnn {
namespace detail {

// Backend parameters can refer to bundle-owned resources. The bundle can also
// contain a DLC logger backed by the outer QNN implementation, so release these
// resources in dependency order before that implementation is unloaded.
template <typename BackendParameters, typename BackendBundle>
void ReleaseQnnDlcOwnedResources(
    std::unique_ptr<BackendParameters>& backend_parameters,
    std::unique_ptr<BackendBundle>& backend_bundle) noexcept {
  backend_parameters.reset();
  backend_bundle.reset();
}

} // namespace detail
} // namespace qnn
} // namespace backends
} // namespace executorch
