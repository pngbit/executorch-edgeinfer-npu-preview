/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/runner/composable_attention_runner.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/result.h>

using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::extension::llm::CallerOwnedOutputPolicy;
using executorch::extension::llm::ComposableAttentionRunner;
using executorch::extension::llm::ComposableAttentionWorkspace;
using executorch::extension::llm::PreparedAttentionBlock;
using executorch::extension::llm::StaticAttentionGraphCost;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::Result;
using testing::_;
using testing::Invoke;

namespace executorch {
namespace extension {
namespace llm {

struct ComposableAttentionRunnerTestPeer {
  template <typename BindOutputs>
  static Error bind_outputs_if_needed(
      ComposableAttentionRunner& runner,
      Module* module,
      const std::string& method,
      const std::array<TensorPtr, 4>& outputs,
      BindOutputs&& bind_outputs,
      bool& outputs_bound) {
    return runner.bind_outputs_if_needed(
        module,
        method,
        outputs,
        std::forward<BindOutputs>(bind_outputs),
        outputs_bound);
  }

  static void corrupt_cached_dtype(
      ComposableAttentionRunner& runner,
      const std::string& method,
      size_t output_index,
      executorch::aten::ScalarType dtype) {
    runner.output_bindings_.at(method).outputs.at(output_index).dtype = dtype;
  }
};

} // namespace llm
} // namespace extension
} // namespace executorch

using executorch::extension::llm::ComposableAttentionRunnerTestPeer;

namespace {

class MockModule : public Module {
 public:
  MockModule() : Module("") {}

  MOCK_METHOD(
      Result<std::vector<EValue>>,
      execute,
      (const std::string&, const std::vector<EValue>&),
      (override));
};

class ComposableAttentionRunnerTest : public testing::Test {
 protected:
  std::vector<EValue>
  make_outputs(float output_value, float state_value, size_t query_rows = 1) {
    const auto rows = static_cast<executorch::aten::SizesType>(query_rows);
    std::vector<TensorPtr> tensors{
        executorch::extension::make_tensor_ptr(
            {1, 4, rows, 16},
            std::vector<float>(64 * query_rows, output_value)),
        executorch::extension::make_tensor_ptr(
            {1, 4, rows, 1}, std::vector<float>(4 * query_rows, state_value)),
        executorch::extension::make_tensor_ptr(
            {1, 4, rows, 1},
            std::vector<float>(4 * query_rows, state_value + 1.0f)),
        executorch::extension::make_tensor_ptr(
            {1, 4, rows, 16},
            std::vector<float>(64 * query_rows, state_value + 2.0f)),
    };
    std::vector<EValue> outputs;
    for (const auto& tensor : tensors) {
      outputs.emplace_back(*tensor);
      owned_tensors_.push_back(tensor);
    }
    return outputs;
  }

  std::vector<TensorPtr> owned_tensors_;

  ComposableAttentionWorkspace make_workspace(size_t query_rows = 1) {
    ComposableAttentionWorkspace workspace;
    const auto rows = static_cast<executorch::aten::SizesType>(query_rows);
    for (auto& bank : workspace.banks) {
      bank[0] = executorch::extension::make_tensor_ptr(
          {1, 4, rows, 16}, std::vector<float>(64 * query_rows, -1.0f));
      bank[1] = executorch::extension::make_tensor_ptr(
          {1, 4, rows, 1}, std::vector<float>(4 * query_rows, -1.0f));
      bank[2] = executorch::extension::make_tensor_ptr(
          {1, 4, rows, 1}, std::vector<float>(4 * query_rows, -1.0f));
      bank[3] = executorch::extension::make_tensor_ptr(
          {1, 4, rows, 16}, std::vector<float>(64 * query_rows, -1.0f));
    }
    return workspace;
  }

  std::array<TensorPtr, 4> workspace_destination(
      const ComposableAttentionWorkspace& workspace,
      size_t state_bank) {
    return {
        workspace.banks[0][0],
        workspace.banks[state_bank][1],
        workspace.banks[state_bank][2],
        workspace.banks[state_bank][3]};
  }
};

TEST_F(
    ComposableAttentionRunnerTest,
    ReusesStableOutputBindingAndRebindsWhenBankChanges) {
  MockModule module;
  auto runner_result = ComposableAttentionRunner::create(
      &module, 1, std::vector<StaticAttentionGraphCost>{{8, 1.0, 1.0}});
  ASSERT_TRUE(runner_result.ok());
  auto runner = std::move(*runner_result);
  auto workspace = make_workspace();
  const auto bank0 = workspace_destination(workspace, 0);
  const auto bank1 = workspace_destination(workspace, 1);
  size_t bind_calls = 0;
  auto bind = [&]() {
    ++bind_calls;
    return Error::Ok;
  };
  bool outputs_bound = false;

  EXPECT_EQ(
      ComposableAttentionRunnerTestPeer::bind_outputs_if_needed(
          runner, &module, "attn_merge_r1_c8", bank0, bind, outputs_bound),
      Error::Ok);
  EXPECT_TRUE(outputs_bound);
  EXPECT_EQ(bind_calls, size_t{1});

  EXPECT_EQ(
      ComposableAttentionRunnerTestPeer::bind_outputs_if_needed(
          runner, &module, "attn_merge_r1_c8", bank0, bind, outputs_bound),
      Error::Ok);
  EXPECT_TRUE(outputs_bound);
  EXPECT_EQ(bind_calls, size_t{1});

  EXPECT_EQ(
      ComposableAttentionRunnerTestPeer::bind_outputs_if_needed(
          runner, &module, "attn_merge_r1_c8", bank1, bind, outputs_bound),
      Error::Ok);
  EXPECT_TRUE(outputs_bound);
  EXPECT_EQ(bind_calls, size_t{2});

  EXPECT_EQ(
      ComposableAttentionRunnerTestPeer::bind_outputs_if_needed(
          runner, &module, "attn_merge_r1_c8", bank1, bind, outputs_bound),
      Error::Ok);
  EXPECT_TRUE(outputs_bound);
  EXPECT_EQ(bind_calls, size_t{2});

  MockModule other_module;
  EXPECT_EQ(
      ComposableAttentionRunnerTestPeer::bind_outputs_if_needed(
          runner,
          &other_module,
          "attn_merge_r1_c8",
          bank1,
          bind,
          outputs_bound),
      Error::Ok);
  EXPECT_EQ(bind_calls, size_t{3});

  EXPECT_EQ(
      ComposableAttentionRunnerTestPeer::bind_outputs_if_needed(
          runner,
          &other_module,
          "attn_first_r1_c8",
          bank1,
          bind,
          outputs_bound),
      Error::Ok);
  EXPECT_EQ(bind_calls, size_t{4});
}

TEST_F(
    ComposableAttentionRunnerTest,
    RebindsWhenOutputOwnerAddressShapeOrDtypeChanges) {
  MockModule module;
  auto runner_result = ComposableAttentionRunner::create(
      &module, 1, std::vector<StaticAttentionGraphCost>{{8, 1.0, 1.0}});
  ASSERT_TRUE(runner_result.ok());
  auto runner = std::move(*runner_result);
  auto workspace = make_workspace();
  auto outputs = workspace_destination(workspace, 0);
  size_t bind_calls = 0;
  auto bind = [&]() {
    ++bind_calls;
    return Error::Ok;
  };
  bool outputs_bound = false;
  const std::string method = "attn_merge_r1_c8";

  ASSERT_EQ(
      ComposableAttentionRunnerTestPeer::bind_outputs_if_needed(
          runner, &module, method, outputs, bind, outputs_bound),
      Error::Ok);
  EXPECT_EQ(bind_calls, size_t{1});

  std::vector<float> shared_storage(64);
  outputs[0] = executorch::extension::make_tensor_ptr(
      {1, 4, 1, 16}, shared_storage.data());
  ASSERT_EQ(
      ComposableAttentionRunnerTestPeer::bind_outputs_if_needed(
          runner, &module, method, outputs, bind, outputs_bound),
      Error::Ok);
  EXPECT_EQ(bind_calls, size_t{2});

  // A distinct Tensor owner over the same address still requires rebinding.
  outputs[0] = executorch::extension::make_tensor_ptr(
      {1, 4, 1, 16}, shared_storage.data());
  ASSERT_EQ(
      ComposableAttentionRunnerTestPeer::bind_outputs_if_needed(
          runner, &module, method, outputs, bind, outputs_bound),
      Error::Ok);
  EXPECT_EQ(bind_calls, size_t{3});

  ASSERT_EQ(
      executorch::extension::resize_tensor_ptr(outputs[0], {1, 4, 1, 8}),
      Error::Ok);
  ASSERT_EQ(
      ComposableAttentionRunnerTestPeer::bind_outputs_if_needed(
          runner, &module, method, outputs, bind, outputs_bound),
      Error::Ok);
  EXPECT_EQ(bind_calls, size_t{4});

  ComposableAttentionRunnerTestPeer::corrupt_cached_dtype(
      runner, method, 0, executorch::aten::ScalarType::Half);
  EXPECT_EQ(
      ComposableAttentionRunnerTestPeer::bind_outputs_if_needed(
          runner, &module, method, outputs, bind, outputs_bound),
      Error::Ok);
  EXPECT_EQ(bind_calls, size_t{5});
}

TEST_F(
    ComposableAttentionRunnerTest,
    ReleaseClearsBindingCacheAndBindingErrorsAreNotCached) {
  MockModule module;
  auto runner_result = ComposableAttentionRunner::create(
      &module, 1, std::vector<StaticAttentionGraphCost>{{8, 1.0, 1.0}});
  ASSERT_TRUE(runner_result.ok());
  auto runner = std::move(*runner_result);
  auto workspace = make_workspace();
  const auto outputs = workspace_destination(workspace, 0);
  const std::string method = "attn_merge_r1_c8";
  size_t bind_calls = 0;
  bool outputs_bound = false;

  EXPECT_EQ(
      ComposableAttentionRunnerTestPeer::bind_outputs_if_needed(
          runner,
          &module,
          method,
          outputs,
          [&]() {
            ++bind_calls;
            return Error::Ok;
          },
          outputs_bound),
      Error::Ok);
  EXPECT_EQ(bind_calls, size_t{1});

  runner.release_output_bindings();
  EXPECT_EQ(
      ComposableAttentionRunnerTestPeer::bind_outputs_if_needed(
          runner,
          &module,
          method,
          outputs,
          [&]() {
            ++bind_calls;
            return Error::AccessFailed;
          },
          outputs_bound),
      Error::AccessFailed);
  EXPECT_FALSE(outputs_bound);
  EXPECT_EQ(bind_calls, size_t{2});

  EXPECT_EQ(
      ComposableAttentionRunnerTestPeer::bind_outputs_if_needed(
          runner,
          &module,
          method,
          outputs,
          [&]() {
            ++bind_calls;
            return Error::Ok;
          },
          outputs_bound),
      Error::Ok);
  EXPECT_TRUE(outputs_bound);
  EXPECT_EQ(bind_calls, size_t{3});
}

TEST_F(
    ComposableAttentionRunnerTest,
    DefaultCloneOverloadPreservesExecuteErrors) {
  MockModule module;
  auto runner_result = ComposableAttentionRunner::create(
      &module, 1, std::vector<StaticAttentionGraphCost>{{8, 1.0, 1.0}});
  ASSERT_TRUE(runner_result.ok());
  auto runner = std::move(*runner_result);
  auto q = executorch::extension::make_tensor_ptr(
      {1, 4, 1, 16}, std::vector<float>(64, 1.0f));
  auto k = executorch::extension::make_tensor_ptr(
      {1, 2, 16, 8}, std::vector<float>(256, 2.0f));
  auto v = executorch::extension::make_tensor_ptr(
      {1, 2, 8, 16}, std::vector<float>(256, 3.0f));
  auto visibility = executorch::extension::make_tensor_ptr(
      {1, 1, 1, 8}, std::vector<float>(8, 1.0f));
  const std::vector<PreparedAttentionBlock> blocks{
      {k.get(), v.get(), visibility.get()},
  };

  EXPECT_CALL(module, execute("attn_first_r1_c8", _))
      .WillOnce(Invoke([](const std::string&, const std::vector<EValue>&) {
        return Result<std::vector<EValue>>(Error::AccessFailed);
      }));
  const auto output = runner.run_blocks(*q, blocks);
  ASSERT_FALSE(output.ok());
  EXPECT_EQ(output.error(), Error::AccessFailed);
}

TEST_F(ComposableAttentionRunnerTest, SlicesAndPadsAPlanBeyondGraphWidths) {
  MockModule module;
  auto runner_result = ComposableAttentionRunner::create(
      &module,
      1,
      std::vector<StaticAttentionGraphCost>{
          {4, 1.5, 0.5},
          {8, 1.0, 2.0},
      });
  ASSERT_TRUE(runner_result.ok());
  auto runner = std::move(*runner_result);

  std::vector<float> k_data(1 * 2 * 16 * 10);
  for (size_t row = 0; row < 32; ++row) {
    for (size_t column = 0; column < 10; ++column) {
      k_data[row * 10 + column] = static_cast<float>(row * 100 + column);
    }
  }
  std::vector<float> v_data(1 * 2 * 10 * 16);
  for (size_t i = 0; i < v_data.size(); ++i) {
    v_data[i] = static_cast<float>(i);
  }
  auto q = executorch::extension::make_tensor_ptr(
      {1, 4, 1, 16}, std::vector<float>(64, 1.0f));
  auto k =
      executorch::extension::make_tensor_ptr({1, 2, 16, 10}, std::move(k_data));
  auto v =
      executorch::extension::make_tensor_ptr({1, 2, 10, 16}, std::move(v_data));
  auto visibility = executorch::extension::make_tensor_ptr(
      {1, 1, 1, 10}, std::vector<float>(10, 1.0f));

  EXPECT_CALL(module, execute("attn_first_r1_c8", _))
      .WillOnce(
          Invoke([&](const std::string&, const std::vector<EValue>& inputs) {
            EXPECT_EQ(inputs.size(), size_t{4});
            const auto& k_block = inputs[1].toTensor();
            EXPECT_EQ(k_block.size(3), 8);
            const float* values = k_block.const_data_ptr<float>();
            for (size_t i = 0; i < 8; ++i) {
              EXPECT_EQ(values[i], static_cast<float>(i));
            }
            EXPECT_EQ(values[8], 100.0f);
            return Result<std::vector<EValue>>(make_outputs(1.0f, 2.0f));
          }));
  EXPECT_CALL(module, execute("attn_merge_r1_c4", _))
      .WillOnce(Invoke([&](const std::string&,
                           const std::vector<EValue>& inputs) {
        EXPECT_EQ(inputs.size(), size_t{7});
        const float* k_values = inputs[1].toTensor().const_data_ptr<float>();
        EXPECT_EQ(k_values[0], 8.0f);
        EXPECT_EQ(k_values[1], 9.0f);
        EXPECT_EQ(k_values[2], 0.0f);
        EXPECT_EQ(k_values[3], 0.0f);
        EXPECT_EQ(k_values[4], 108.0f);
        const float* mask_values = inputs[3].toTensor().const_data_ptr<float>();
        EXPECT_EQ(mask_values[0], 1.0f);
        EXPECT_EQ(mask_values[1], 1.0f);
        EXPECT_EQ(mask_values[2], 0.0f);
        EXPECT_EQ(mask_values[3], 0.0f);
        EXPECT_EQ(inputs[4].toTensor().const_data_ptr<float>()[0], 2.0f);
        EXPECT_EQ(inputs[5].toTensor().const_data_ptr<float>()[0], 3.0f);
        EXPECT_EQ(inputs[6].toTensor().const_data_ptr<float>()[0], 4.0f);
        return Result<std::vector<EValue>>(make_outputs(9.0f, 10.0f));
      }));

  auto output = runner.run(*q, *k, *v, *visibility);
  ASSERT_TRUE(output.ok());
  EXPECT_EQ((*output)->const_data_ptr<float>()[0], 9.0f);
  EXPECT_EQ(runner.planned_through(), size_t{10});
}

TEST_F(ComposableAttentionRunnerTest, RunsOnlyTheVisibleCausalPrefix) {
  MockModule module;
  auto runner_result = ComposableAttentionRunner::create(
      &module,
      4,
      std::vector<StaticAttentionGraphCost>{
          {4, 1.0, 1.0},
          {8, 3.0, 3.0},
      });
  ASSERT_TRUE(runner_result.ok());
  auto runner = std::move(*runner_result);

  auto q = executorch::extension::make_tensor_ptr(
      {1, 4, 4, 16}, std::vector<float>(256, 1.0f));
  std::vector<float> k_data(1 * 2 * 16 * 16);
  for (size_t row = 0; row < 32; ++row) {
    for (size_t column = 0; column < 16; ++column) {
      k_data[row * 16 + column] = static_cast<float>(row * 100 + column);
    }
  }
  auto k =
      executorch::extension::make_tensor_ptr({1, 2, 16, 16}, std::move(k_data));
  auto v = executorch::extension::make_tensor_ptr(
      {1, 2, 16, 16}, std::vector<float>(512, 2.0f));
  auto visibility = executorch::extension::make_tensor_ptr(
      {1, 1, 4, 16}, std::vector<float>(64, 1.0f));

  EXPECT_CALL(module, execute("attn_first_r4_c4", _))
      .WillOnce(Invoke([&](const std::string&,
                           const std::vector<EValue>& inputs) {
        const float* k_values = inputs[1].toTensor().const_data_ptr<float>();
        EXPECT_EQ(k_values[0], 0.0f);
        EXPECT_EQ(k_values[1], 1.0f);
        EXPECT_EQ(k_values[2], 2.0f);
        EXPECT_EQ(k_values[3], 3.0f);
        return Result<std::vector<EValue>>(make_outputs(1.0f, 2.0f, 4));
      }));
  EXPECT_CALL(module, execute("attn_merge_r4_c4", _))
      .WillOnce(Invoke([&](const std::string&,
                           const std::vector<EValue>& inputs) {
        const float* k_values = inputs[1].toTensor().const_data_ptr<float>();
        EXPECT_EQ(k_values[0], 4.0f);
        EXPECT_EQ(k_values[1], 5.0f);
        EXPECT_EQ(k_values[2], 0.0f);
        EXPECT_EQ(k_values[3], 0.0f);
        EXPECT_EQ(k_values[4], 104.0f);
        EXPECT_EQ(k_values[5], 105.0f);
        EXPECT_EQ(k_values[6], 0.0f);
        EXPECT_EQ(k_values[7], 0.0f);
        const float* mask_values = inputs[3].toTensor().const_data_ptr<float>();
        EXPECT_EQ(mask_values[0], 1.0f);
        EXPECT_EQ(mask_values[1], 1.0f);
        EXPECT_EQ(mask_values[2], 0.0f);
        EXPECT_EQ(mask_values[3], 0.0f);
        return Result<std::vector<EValue>>(make_outputs(9.0f, 10.0f, 4));
      }));

  auto output = runner.run_causal_tile(*q, *k, *v, *visibility, 4, 2);
  ASSERT_TRUE(output.ok());
  EXPECT_EQ((*output)->const_data_ptr<float>()[0], 9.0f);
  EXPECT_EQ(runner.planned_through(), size_t{6});
}

TEST_F(ComposableAttentionRunnerTest, RejectsInvalidCausalTileMetadata) {
  MockModule module;
  auto runner_result = ComposableAttentionRunner::create(
      &module, 4, std::vector<StaticAttentionGraphCost>{{4, 1.0, 1.0}});
  ASSERT_TRUE(runner_result.ok());
  auto runner = std::move(*runner_result);

  auto q = executorch::extension::make_tensor_ptr(
      {1, 4, 4, 16}, std::vector<float>(256, 1.0f));
  auto k = executorch::extension::make_tensor_ptr(
      {1, 2, 16, 16}, std::vector<float>(512, 1.0f));
  auto v = executorch::extension::make_tensor_ptr(
      {1, 2, 16, 16}, std::vector<float>(512, 1.0f));
  auto visibility = executorch::extension::make_tensor_ptr(
      {1, 1, 4, 16}, std::vector<float>(64, 1.0f));

  EXPECT_EQ(
      runner.run_causal_tile(*q, *k, *v, *visibility, 15).error(),
      Error::InvalidArgument);
  EXPECT_EQ(
      runner.run_causal_tile(*q, *k, *v, *visibility, 0, 5).error(),
      Error::InvalidArgument);
}

TEST_F(ComposableAttentionRunnerTest, RejectsMismatchedDynamicInputs) {
  MockModule module;
  auto runner_result = ComposableAttentionRunner::create(
      &module, 1, std::vector<StaticAttentionGraphCost>{{8, 1.0, 1.0}});
  ASSERT_TRUE(runner_result.ok());
  auto runner = std::move(*runner_result);

  auto q = executorch::extension::make_tensor_ptr(
      {1, 4, 2, 16}, std::vector<float>(128, 0.0f));
  auto k = executorch::extension::make_tensor_ptr(
      {1, 2, 16, 8}, std::vector<float>(256, 0.0f));
  auto v = executorch::extension::make_tensor_ptr(
      {1, 2, 8, 16}, std::vector<float>(256, 0.0f));
  auto visibility = executorch::extension::make_tensor_ptr(
      {1, 1, 2, 8}, std::vector<float>(16, 1.0f));

  auto output = runner.run(*q, *k, *v, *visibility);
  EXPECT_EQ(output.error(), Error::InvalidArgument);
}

TEST_F(
    ComposableAttentionRunnerTest,
    ExposesMeasuredCostPlanForPersistentCache) {
  MockModule module;
  auto runner_result = ComposableAttentionRunner::create(
      &module,
      1,
      std::vector<StaticAttentionGraphCost>{
          {4, 1.0, 1.0},
          {8, 3.0, 3.0},
      });
  ASSERT_TRUE(runner_result.ok());
  auto runner = std::move(*runner_result);

  auto plan = runner.plan(8);
  ASSERT_TRUE(plan.ok());
  EXPECT_EQ(plan->widths, (std::vector<size_t>{4, 4}));
  EXPECT_DOUBLE_EQ(plan->predicted_cost, 2.0);
  EXPECT_EQ(runner.plan(0).error(), Error::InvalidArgument);
}

TEST_F(ComposableAttentionRunnerTest, ExecutesPreparedBlocksWithoutRepacking) {
  MockModule module;
  auto runner_result = ComposableAttentionRunner::create(
      &module,
      1,
      std::vector<StaticAttentionGraphCost>{
          {4, 1.0, 1.0},
          {8, 1.0, 1.0},
      });
  ASSERT_TRUE(runner_result.ok());
  auto runner = std::move(*runner_result);

  auto q = executorch::extension::make_tensor_ptr(
      {1, 4, 1, 16}, std::vector<float>(64, 1.0f));
  auto k8 = executorch::extension::make_tensor_ptr(
      {1, 2, 16, 8}, std::vector<float>(256, 2.0f));
  auto v8 = executorch::extension::make_tensor_ptr(
      {1, 2, 8, 16}, std::vector<float>(256, 3.0f));
  auto mask8 = executorch::extension::make_tensor_ptr(
      {1, 1, 1, 8}, std::vector<float>(8, 1.0f));
  auto k4 = executorch::extension::make_tensor_ptr(
      {1, 2, 16, 4}, std::vector<float>(128, 4.0f));
  auto v4 = executorch::extension::make_tensor_ptr(
      {1, 2, 4, 16}, std::vector<float>(128, 5.0f));
  auto mask4 = executorch::extension::make_tensor_ptr(
      {1, 1, 1, 4}, std::vector<float>{1.0f, 1.0f, 0.0f, 0.0f});

  EXPECT_CALL(module, execute("attn_first_r1_c8", _))
      .WillOnce(Invoke([&](const std::string&,
                           const std::vector<EValue>& inputs) {
        EXPECT_EQ(inputs.size(), size_t{4});
        EXPECT_EQ(inputs[1].toTensor().const_data_ptr(), k8->const_data_ptr());
        return Result<std::vector<EValue>>(make_outputs(1.0f, 2.0f));
      }));
  EXPECT_CALL(module, execute("attn_merge_r1_c4", _))
      .WillOnce(Invoke([&](const std::string&,
                           const std::vector<EValue>& inputs) {
        EXPECT_EQ(inputs.size(), size_t{7});
        EXPECT_EQ(inputs[1].toTensor().const_data_ptr(), k4->const_data_ptr());
        EXPECT_EQ(inputs[3].toTensor().const_data_ptr<float>()[2], 0.0f);
        return Result<std::vector<EValue>>(make_outputs(9.0f, 10.0f));
      }));

  const std::vector<PreparedAttentionBlock> blocks{
      {k8.get(), v8.get(), mask8.get()},
      {k4.get(), v4.get(), mask4.get()},
  };
  auto output = runner.run_blocks(*q, blocks);
  ASSERT_TRUE(output.ok());
  EXPECT_EQ((*output)->const_data_ptr<float>()[0], 9.0f);
}

TEST_F(
    ComposableAttentionRunnerTest,
    RefillsOneScratchBlockAcrossFirstAndMergeCalls) {
  MockModule module;
  auto runner_result = ComposableAttentionRunner::create(
      &module, 1, std::vector<StaticAttentionGraphCost>{{4, 1.0, 1.0}});
  ASSERT_TRUE(runner_result.ok());
  auto runner = std::move(*runner_result);

  auto q = executorch::extension::make_tensor_ptr(
      {1, 4, 1, 16}, std::vector<float>(64, 1.0f));
  auto key_scratch = executorch::extension::make_tensor_ptr(
      {1, 2, 16, 4}, std::vector<float>(128, 2.0f));
  auto value_scratch = executorch::extension::make_tensor_ptr(
      {1, 2, 4, 16}, std::vector<float>(128, 3.0f));
  auto mask_scratch = executorch::extension::make_tensor_ptr(
      {1, 1, 1, 4}, std::vector<float>(4, 1.0f));
  auto workspace = make_workspace();
  const PreparedAttentionBlock scratch{
      key_scratch.get(), value_scratch.get(), mask_scratch.get()};

  EXPECT_CALL(module, execute("attn_first_r1_c4", _))
      .WillOnce(Invoke([&](const std::string&,
                           const std::vector<EValue>& inputs) {
        EXPECT_EQ(inputs.size(), size_t{4});
        EXPECT_EQ(inputs[1].toTensor().const_data_ptr<float>()[0], 2.0f);
        return Result<std::vector<EValue>>(make_outputs(5.0f, 6.0f));
      }));
  EXPECT_EQ(
      runner.run_prepared_block(*q, scratch, 0, workspace), Error::Ok);

  key_scratch->mutable_data_ptr<float>()[0] = 12.0f;
  value_scratch->mutable_data_ptr<float>()[0] = 13.0f;
  mask_scratch->mutable_data_ptr<float>()[3] = 0.0f;
  EXPECT_CALL(module, execute("attn_merge_r1_c4", _))
      .WillOnce(Invoke([&](const std::string&,
                           const std::vector<EValue>& inputs) {
        EXPECT_EQ(inputs.size(), size_t{7});
        EXPECT_EQ(inputs[1].toTensor().const_data_ptr<float>()[0], 12.0f);
        EXPECT_EQ(inputs[2].toTensor().const_data_ptr<float>()[0], 13.0f);
        EXPECT_EQ(inputs[3].toTensor().const_data_ptr<float>()[3], 0.0f);
        EXPECT_EQ(inputs[4].toTensor().const_data_ptr<float>()[0], 6.0f);
        EXPECT_EQ(inputs[5].toTensor().const_data_ptr<float>()[0], 7.0f);
        EXPECT_EQ(inputs[6].toTensor().const_data_ptr<float>()[0], 8.0f);
        return Result<std::vector<EValue>>(make_outputs(15.0f, 16.0f));
      }));
  EXPECT_EQ(
      runner.run_prepared_block(*q, scratch, 1, workspace), Error::Ok);
  EXPECT_EQ(
      ComposableAttentionRunner::prepared_output(workspace)
          ->const_data_ptr<float>()[0],
      15.0f);
}

TEST_F(
    ComposableAttentionRunnerTest,
    FallsBackToCopiesWhenExternalOutputsAreUnsupported) {
  MockModule module;
  auto runner_result = ComposableAttentionRunner::create(
      &module,
      1,
      std::vector<StaticAttentionGraphCost>{{4, 1.0, 1.0}, {8, 1.0, 1.0}});
  ASSERT_TRUE(runner_result.ok());
  auto runner = std::move(*runner_result);
  auto q = executorch::extension::make_tensor_ptr(
      {1, 4, 1, 16}, std::vector<float>(64, 1.0f));
  auto k8 = executorch::extension::make_tensor_ptr(
      {1, 2, 16, 8}, std::vector<float>(256, 2.0f));
  auto v8 = executorch::extension::make_tensor_ptr(
      {1, 2, 8, 16}, std::vector<float>(256, 3.0f));
  auto mask8 = executorch::extension::make_tensor_ptr(
      {1, 1, 1, 8}, std::vector<float>(8, 1.0f));
  auto k4 = executorch::extension::make_tensor_ptr(
      {1, 2, 16, 4}, std::vector<float>(128, 4.0f));
  auto v4 = executorch::extension::make_tensor_ptr(
      {1, 2, 4, 16}, std::vector<float>(128, 5.0f));
  auto mask4 = executorch::extension::make_tensor_ptr(
      {1, 1, 1, 4}, std::vector<float>(4, 1.0f));
  auto workspace = make_workspace();

  EXPECT_CALL(module, execute("attn_first_r1_c8", _))
      .WillOnce(
          Invoke([&](const std::string&, const std::vector<EValue>& inputs) {
            EXPECT_EQ(inputs.size(), size_t{4});
            return Result<std::vector<EValue>>(make_outputs(11.0f, 2.0f));
          }));
  EXPECT_CALL(module, execute("attn_merge_r1_c4", _))
      .WillOnce(
          Invoke([&](const std::string&, const std::vector<EValue>& inputs) {
            EXPECT_EQ(inputs.size(), size_t{7});
            EXPECT_EQ(inputs[4].toTensor().const_data_ptr<float>()[0], 2.0f);
            EXPECT_EQ(inputs[5].toTensor().const_data_ptr<float>()[0], 3.0f);
            EXPECT_EQ(inputs[6].toTensor().const_data_ptr<float>()[0], 4.0f);
            return Result<std::vector<EValue>>(make_outputs(9.0f, 10.0f));
          }));

  const std::vector<PreparedAttentionBlock> blocks{
      {k8.get(), v8.get(), mask8.get()},
      {k4.get(), v4.get(), mask4.get()},
  };
  auto output = runner.run_blocks(*q, blocks, workspace);
  ASSERT_TRUE(output.ok());
  EXPECT_EQ(output->get(), workspace.banks[0][0].get());
  EXPECT_EQ((*output)->const_data_ptr<float>()[0], 9.0f);
  EXPECT_EQ(workspace.banks[1][1]->const_data_ptr<float>()[0], 10.0f);
}

TEST_F(ComposableAttentionRunnerTest, RejectsMalformedReusableWorkspace) {
  MockModule module;
  auto runner_result = ComposableAttentionRunner::create(
      &module, 1, std::vector<StaticAttentionGraphCost>{{8, 1.0, 1.0}});
  ASSERT_TRUE(runner_result.ok());
  auto runner = std::move(*runner_result);
  auto q = executorch::extension::make_tensor_ptr(
      {1, 4, 1, 16}, std::vector<float>(64, 1.0f));
  auto k = executorch::extension::make_tensor_ptr(
      {1, 2, 16, 8}, std::vector<float>(256, 2.0f));
  auto v = executorch::extension::make_tensor_ptr(
      {1, 2, 8, 16}, std::vector<float>(256, 3.0f));
  auto mask = executorch::extension::make_tensor_ptr(
      {1, 1, 1, 8}, std::vector<float>(8, 1.0f));
  auto workspace = make_workspace();
  workspace.banks[1][2] = executorch::extension::make_tensor_ptr(
      {1, 4, 1, 2}, std::vector<float>(8, 0.0f));
  const std::vector<PreparedAttentionBlock> blocks{
      {k.get(), v.get(), mask.get()},
  };
  EXPECT_EQ(
      runner.run_blocks(*q, blocks, workspace).error(), Error::InvalidArgument);
}

TEST_F(ComposableAttentionRunnerTest, RejectsUnsupportedPreparedBlockWidth) {
  MockModule module;
  auto runner_result = ComposableAttentionRunner::create(
      &module, 1, std::vector<StaticAttentionGraphCost>{{8, 1.0, 1.0}});
  ASSERT_TRUE(runner_result.ok());
  auto runner = std::move(*runner_result);

  auto q = executorch::extension::make_tensor_ptr(
      {1, 4, 1, 16}, std::vector<float>(64, 0.0f));
  auto k = executorch::extension::make_tensor_ptr(
      {1, 2, 16, 4}, std::vector<float>(128, 0.0f));
  auto v = executorch::extension::make_tensor_ptr(
      {1, 2, 4, 16}, std::vector<float>(128, 0.0f));
  auto visibility = executorch::extension::make_tensor_ptr(
      {1, 1, 1, 4}, std::vector<float>(4, 1.0f));
  const std::vector<PreparedAttentionBlock> blocks{
      {k.get(), v.get(), visibility.get()},
  };
  EXPECT_EQ(runner.run_blocks(*q, blocks).error(), Error::InvalidArgument);
}

TEST_F(ComposableAttentionRunnerTest, RejectsDimensionsOutsideTensorAbi) {
  MockModule module;
  const size_t too_large =
      static_cast<size_t>(
          std::numeric_limits<executorch::aten::SizesType>::max()) +
      1;
  auto rows_result =
      ComposableAttentionRunner::create(&module, too_large, {{8, 1.0, 1.0}});
  EXPECT_EQ(rows_result.error(), Error::InvalidArgument);

  auto width_result =
      ComposableAttentionRunner::create(&module, 1, {{too_large, 1.0, 1.0}});
  EXPECT_EQ(width_result.error(), Error::InvalidArgument);
}

TEST_F(
    ComposableAttentionRunnerTest,
    StrictCallerOwnedContractRejectsPteWithoutOutputMetadata) {
  MockModule module;
  auto runner_result = ComposableAttentionRunner::create(
      &module,
      1,
      std::vector<StaticAttentionGraphCost>{{8, 1.0, 1.0}},
      false,
      CallerOwnedOutputPolicy::kRequireUnplannedOutputs);
  EXPECT_FALSE(runner_result.ok());
  EXPECT_EQ(runner_result.error(), Error::NotSupported);
}

TEST_F(ComposableAttentionRunnerTest, RejectsMalformedMethodState) {
  MockModule module;
  auto runner_result = ComposableAttentionRunner::create(
      &module, 1, std::vector<StaticAttentionGraphCost>{{8, 1.0, 1.0}});
  ASSERT_TRUE(runner_result.ok());
  auto runner = std::move(*runner_result);

  auto q = executorch::extension::make_tensor_ptr(
      {1, 4, 1, 16}, std::vector<float>(64, 0.0f));
  auto k = executorch::extension::make_tensor_ptr(
      {1, 2, 16, 8}, std::vector<float>(256, 0.0f));
  auto v = executorch::extension::make_tensor_ptr(
      {1, 2, 8, 16}, std::vector<float>(256, 0.0f));
  auto visibility = executorch::extension::make_tensor_ptr(
      {1, 1, 1, 8}, std::vector<float>(8, 1.0f));

  EXPECT_CALL(module, execute("attn_first_r1_c8", _))
      .WillOnce(Invoke([&](const std::string&, const std::vector<EValue>&) {
        auto outputs = make_outputs(1.0f, 2.0f);
        auto malformed = executorch::extension::make_tensor_ptr(
            {1, 4, 1, 2}, std::vector<float>(8, 0.0f));
        outputs[1] = EValue(*malformed);
        owned_tensors_.push_back(std::move(malformed));
        return Result<std::vector<EValue>>(std::move(outputs));
      }));

  auto output = runner.run(*q, *k, *v, *visibility);
  EXPECT_EQ(output.error(), Error::InvalidArgument);
}

} // namespace
