/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/backends/QnnDlcResourceLifecycle.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace executorch {
namespace backends {
namespace qnn {
namespace {

struct DestructionState {
  bool implementation_alive{true};
  bool logger_observed_live_implementation{false};
  std::vector<std::string> events;
};

struct FakeExternalImplementation {
  explicit FakeExternalImplementation(DestructionState* state) : state(state) {}
  ~FakeExternalImplementation() {
    state->events.emplace_back("implementation");
    state->implementation_alive = false;
  }
  DestructionState* state;
};

struct FakeBackendParameters {
  explicit FakeBackendParameters(DestructionState* state) : state(state) {}
  ~FakeBackendParameters() {
    state->events.emplace_back("parameters");
  }
  DestructionState* state;
};

struct FakeDlcLogger {
  explicit FakeDlcLogger(DestructionState* state) : state(state) {}
  ~FakeDlcLogger() {
    state->logger_observed_live_implementation = state->implementation_alive;
    state->events.emplace_back("dlc_logger");
  }
  DestructionState* state;
};

struct FakeBackendBundle {
  explicit FakeBackendBundle(DestructionState* state)
      : logger(std::make_unique<FakeDlcLogger>(state)) {}
  std::unique_ptr<FakeDlcLogger> logger;
};

TEST(
    QnnDlcManagerLifecycleTest,
    ReleasesDlcLoggerBeforeExternalImplementationUnload) {
  DestructionState state;
  auto implementation = std::make_unique<FakeExternalImplementation>(&state);
  auto parameters = std::make_unique<FakeBackendParameters>(&state);
  auto bundle = std::make_unique<FakeBackendBundle>(&state);

  detail::ReleaseQnnDlcOwnedResources(parameters, bundle);

  EXPECT_EQ(parameters, nullptr);
  EXPECT_EQ(bundle, nullptr);
  EXPECT_TRUE(implementation != nullptr);
  EXPECT_TRUE(state.logger_observed_live_implementation);
  ASSERT_EQ(state.events.size(), std::size_t{2});
  EXPECT_EQ(state.events[0], "parameters");
  EXPECT_EQ(state.events[1], "dlc_logger");

  implementation.reset();
  ASSERT_EQ(state.events.size(), std::size_t{3});
  EXPECT_EQ(state.events[2], "implementation");
}

TEST(QnnDlcManagerLifecycleTest, ResourceReleaseIsIdempotent) {
  DestructionState state;
  auto parameters = std::make_unique<FakeBackendParameters>(&state);
  auto bundle = std::make_unique<FakeBackendBundle>(&state);

  detail::ReleaseQnnDlcOwnedResources(parameters, bundle);
  ASSERT_EQ(state.events.size(), std::size_t{2});

  detail::ReleaseQnnDlcOwnedResources(parameters, bundle);
  EXPECT_EQ(state.events.size(), std::size_t{2});
}

} // namespace
} // namespace qnn
} // namespace backends
} // namespace executorch
