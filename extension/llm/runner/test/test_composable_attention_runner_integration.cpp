/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/runner/composable_attention_runner.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr.h>

using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::extension::llm::CallerOwnedOutputPolicy;
using executorch::extension::llm::ComposableAttentionRunner;
using executorch::extension::llm::ComposableAttentionWorkspace;
using executorch::extension::llm::PreparedAttentionBlock;
using executorch::extension::llm::StaticAttentionGraphCost;
using executorch::runtime::Error;

namespace {

constexpr size_t kQueryHeads = 4;
constexpr size_t kKvHeads = 2;
constexpr size_t kHeadDim = 8;

std::vector<float> reference_attention(
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    size_t sequence_length) {
  const size_t groups = kQueryHeads / kKvHeads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
  std::vector<float> output(kQueryHeads * kHeadDim);
  std::vector<float> scores(sequence_length);
  for (size_t head = 0; head < kQueryHeads; ++head) {
    const size_t kv_head = head / groups;
    float maximum = -INFINITY;
    for (size_t token = 0; token < sequence_length; ++token) {
      float score = 0.0f;
      for (size_t dim = 0; dim < kHeadDim; ++dim) {
        score += q[head * kHeadDim + dim] *
            k[(kv_head * kHeadDim + dim) * sequence_length + token];
      }
      scores[token] = score * scale;
      maximum = std::max(maximum, scores[token]);
    }

    float denominator = 0.0f;
    for (float& score : scores) {
      score = std::exp(score - maximum);
      denominator += score;
    }
    for (size_t dim = 0; dim < kHeadDim; ++dim) {
      float numerator = 0.0f;
      for (size_t token = 0; token < sequence_length; ++token) {
        numerator += scores[token] *
            v[(kv_head * sequence_length + token) * kHeadDim + dim];
      }
      output[head * kHeadDim + dim] = numerator / denominator;
    }
  }
  return output;
}

ComposableAttentionWorkspace make_workspace() {
  ComposableAttentionWorkspace workspace;
  for (auto& bank : workspace.banks) {
    bank[0] = executorch::extension::make_tensor_ptr(
        {1, kQueryHeads, 1, kHeadDim},
        std::vector<float>(kQueryHeads * kHeadDim));
    bank[1] = executorch::extension::make_tensor_ptr(
        {1, kQueryHeads, 1, 1}, std::vector<float>(kQueryHeads));
    bank[2] = executorch::extension::make_tensor_ptr(
        {1, kQueryHeads, 1, 1}, std::vector<float>(kQueryHeads));
    bank[3] = executorch::extension::make_tensor_ptr(
        {1, kQueryHeads, 1, kHeadDim},
        std::vector<float>(kQueryHeads * kHeadDim));
  }
  return workspace;
}

TEST(ComposableAttentionRunnerIntegrationTest, ExecutesRealMultimethodPte) {
  const char* pte_path = std::getenv("ET_COMPOSABLE_ATTENTION_PTE");
  if (pte_path == nullptr) {
    GTEST_SKIP() << "ET_COMPOSABLE_ATTENTION_PTE is not configured";
  }
  Module module{std::string(pte_path)};
  auto runner_result = ComposableAttentionRunner::create(
      &module,
      1,
      std::vector<StaticAttentionGraphCost>{
          {4, 1.5, 0.5},
          {8, 1.0, 2.0},
      });
  ASSERT_TRUE(runner_result.ok());
  auto runner = std::move(*runner_result);

  std::vector<float> q_data(kQueryHeads * kHeadDim);
  for (size_t i = 0; i < q_data.size(); ++i) {
    q_data[i] = std::sin(static_cast<float>(i + 1) * 0.17f);
  }
  auto q = executorch::extension::make_tensor_ptr(
      {1, kQueryHeads, 1, kHeadDim}, q_data);

  for (const size_t sequence_length : {size_t{3}, size_t{10}, size_t{21}}) {
    const int32_t length_size = static_cast<int32_t>(sequence_length);
    std::vector<float> k_data(kKvHeads * kHeadDim * sequence_length);
    std::vector<float> v_data(kKvHeads * sequence_length * kHeadDim);
    for (size_t i = 0; i < k_data.size(); ++i) {
      k_data[i] = std::cos(static_cast<float>(i + 3) * 0.11f);
    }
    for (size_t i = 0; i < v_data.size(); ++i) {
      v_data[i] = std::sin(static_cast<float>(i + 5) * 0.07f);
    }
    const auto expected =
        reference_attention(q_data, k_data, v_data, sequence_length);
    auto k = executorch::extension::make_tensor_ptr(
        {1, kKvHeads, kHeadDim, length_size}, std::move(k_data));
    auto v = executorch::extension::make_tensor_ptr(
        {1, kKvHeads, length_size, kHeadDim}, std::move(v_data));
    auto visibility = executorch::extension::make_tensor_ptr(
        {1, 1, 1, length_size}, std::vector<float>(sequence_length, 1.0f));

    auto output = runner.run(*q, *k, *v, *visibility);
    ASSERT_TRUE(output.ok()) << "sequence length " << sequence_length;
    ASSERT_EQ((*output)->numel(), expected.size());
    const float* actual = (*output)->const_data_ptr<float>();
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_NEAR(actual[i], expected[i], 2.0e-5f)
          << "sequence length " << sequence_length << ", element " << i;
    }
  }
  EXPECT_EQ(runner.planned_through(), size_t{21});
}

TEST(
    ComposableAttentionRunnerIntegrationTest,
    StrictCallerOwnedContractRejectsMemoryPlannedPte) {
  const char* pte_path = std::getenv("ET_COMPOSABLE_ATTENTION_PTE");
  if (pte_path == nullptr) {
    GTEST_SKIP() << "ET_COMPOSABLE_ATTENTION_PTE is not configured";
  }
  Module module{std::string(pte_path)};
  auto runner_result = ComposableAttentionRunner::create(
      &module,
      1,
      std::vector<StaticAttentionGraphCost>{
          {4, 1.0, 1.0},
          {8, 1.0, 1.0},
      },
      false,
      CallerOwnedOutputPolicy::kRequireUnplannedOutputs);
  EXPECT_FALSE(runner_result.ok());
  EXPECT_EQ(runner_result.error(), Error::NotSupported);
}

TEST(
    ComposableAttentionRunnerIntegrationTest,
    WritesRealMethodResultsIntoCallerWorkspace) {
  const char* pte_path = std::getenv("ET_COMPOSABLE_ATTENTION_PTE");
  if (pte_path == nullptr) {
    GTEST_SKIP() << "ET_COMPOSABLE_ATTENTION_PTE is not configured";
  }
  Module module{std::string(pte_path)};
  auto runner_result = ComposableAttentionRunner::create(
      &module,
      1,
      std::vector<StaticAttentionGraphCost>{{4, 1.0, 1.0}, {8, 1.0, 1.0}});
  ASSERT_TRUE(runner_result.ok());
  auto runner = std::move(*runner_result);
  auto q = executorch::extension::make_tensor_ptr(
      {1, kQueryHeads, 1, kHeadDim},
      std::vector<float>(kQueryHeads * kHeadDim, 0.25f));
  auto k8 = executorch::extension::make_tensor_ptr(
      {1, kKvHeads, kHeadDim, 8},
      std::vector<float>(kKvHeads * kHeadDim * 8, 0.5f));
  auto v8 = executorch::extension::make_tensor_ptr(
      {1, kKvHeads, 8, kHeadDim},
      std::vector<float>(kKvHeads * 8 * kHeadDim, 0.75f));
  auto mask8 = executorch::extension::make_tensor_ptr(
      {1, 1, 1, 8}, std::vector<float>(8, 1.0f));
  auto k4 = executorch::extension::make_tensor_ptr(
      {1, kKvHeads, kHeadDim, 4},
      std::vector<float>(kKvHeads * kHeadDim * 4, 0.125f));
  auto v4 = executorch::extension::make_tensor_ptr(
      {1, kKvHeads, 4, kHeadDim},
      std::vector<float>(kKvHeads * 4 * kHeadDim, 0.875f));
  auto mask4 = executorch::extension::make_tensor_ptr(
      {1, 1, 1, 4}, std::vector<float>(4, 1.0f));
  auto workspace = make_workspace();
  void* expected_output = workspace.banks[0][0]->mutable_data_ptr();
  const std::vector<PreparedAttentionBlock> blocks{
      {k8.get(), v8.get(), mask8.get()},
      {k4.get(), v4.get(), mask4.get()},
  };

  auto output = runner.run_blocks(*q, blocks, workspace);
  ASSERT_TRUE(output.ok());
  EXPECT_EQ((*output)->mutable_data_ptr(), expected_output);
  EXPECT_GT((*output)->const_data_ptr<float>()[0], 0.0f);
  runner.release_output_bindings();
}

} // namespace
