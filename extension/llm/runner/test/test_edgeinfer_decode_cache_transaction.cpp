/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/runner/edgeinfer_decode_cache_transaction.h>

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/edgeinfer_fallback.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/edgeinfer_prompt_processor.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace executorch {
namespace extension {
namespace llm {
namespace {

using runtime::Error;
using runtime::Result;

class FakeLayerCache {
 public:
  explicit FakeLayerCache(size_t valid_length) : valid_length_(valid_length) {}

  size_t valid_length() const {
    return valid_length_;
  }

  void append(size_t rows) {
    valid_length_ += rows;
  }

  void rollback_to(size_t checkpoint) noexcept {
    valid_length_ = checkpoint;
    ++rollback_calls_;
  }

  size_t rollback_calls() const {
    return rollback_calls_;
  }

 private:
  size_t valid_length_;
  size_t rollback_calls_ = 0;
};

struct FakeBlock {
  size_t width;
  size_t valid_width;
};

class FakeCacheManager {
 public:
  explicit FakeCacheManager(int32_t ar_len) : ar_len_(ar_len) {}

  void rearrange_cache(int32_t ar_len) {
    ++calls_;
    if (ar_len_ != ar_len) {
      ar_len_ = ar_len;
      ++physical_rearrangements_;
    }
  }

  int32_t ar_len() const {
    return ar_len_;
  }

  size_t calls() const {
    return calls_;
  }

  size_t physical_rearrangements() const {
    return physical_rearrangements_;
  }

 private:
  int32_t ar_len_;
  size_t calls_ = 0;
  size_t physical_rearrangements_ = 0;
};

struct FakePersistentPlan {
  std::vector<size_t> widths;
};

class FakePersistentLayerCache {
 public:
  struct PreparedLayout {
    bool needs_commit() const noexcept {
      return replace;
    }

    bool replace = false;
    std::vector<size_t> layout;
    std::vector<int> key;
    std::vector<int> value;
  };

  FakePersistentLayerCache(
      size_t layer,
      std::vector<size_t> layout,
      std::vector<int> key,
      std::vector<int> value,
      std::vector<int>* events)
      : layer_(layer),
        layout_(std::move(layout)),
        key_(std::move(key)),
        value_(std::move(value)),
        events_(events) {}

  bool prepare_recompose(
      const FakePersistentPlan& plan,
      PreparedLayout* prepared) const {
    events_->push_back(static_cast<int>(10 + layer_));
    if (prepared == nullptr || fail_prepare_) {
      return false;
    }
    *prepared = PreparedLayout{};
    if (layout_ == plan.widths) {
      return true;
    }
    prepared->replace = true;
    prepared->layout = plan.widths;
    prepared->key = key_;
    prepared->value = value_;
    return true;
  }

  void commit_recompose(PreparedLayout&& prepared) noexcept {
    events_->push_back(static_cast<int>(30 + layer_));
    layout_.swap(prepared.layout);
    key_.swap(prepared.key);
    value_.swap(prepared.value);
  }

  void fail_prepare(bool fail) {
    fail_prepare_ = fail;
  }

  const std::vector<size_t>& layout() const {
    return layout_;
  }

  const std::vector<int>& key() const {
    return key_;
  }

  const std::vector<int>& value() const {
    return value_;
  }

 private:
  size_t layer_;
  std::vector<size_t> layout_;
  std::vector<int> key_;
  std::vector<int> value_;
  std::vector<int>* events_;
  bool fail_prepare_ = false;
};

std::unique_ptr<FakePersistentLayerCache> make_persistent_layer(
    size_t layer,
    std::vector<int>* events) {
  return std::make_unique<FakePersistentLayerCache>(
      layer,
      std::vector<size_t>{2, 2},
      std::vector<int>{
          static_cast<int>(100 + layer), static_cast<int>(110 + layer)},
      std::vector<int>{
          static_cast<int>(200 + layer), static_cast<int>(210 + layer)},
      events);
}

struct MaskWrite {
  size_t offset;
  size_t count;
  bool visible;

  bool operator==(const MaskWrite& other) const {
    return offset == other.offset && count == other.count &&
        visible == other.visible;
  }
};

class FakeVisibilityMask {
 public:
  explicit FakeVisibilityMask(size_t elements) : values_(elements, false) {}

  bool fill(size_t offset, size_t count, bool visible) {
    if (offset > values_.size() || count > values_.size() - offset) {
      return false;
    }
    std::fill_n(values_.begin() + offset, count, visible);
    writes_.push_back({offset, count, visible});
    return true;
  }

  const std::vector<bool>& values() const {
    return values_;
  }

  const std::vector<MaskWrite>& writes() const {
    return writes_;
  }

  void clear_writes() {
    writes_.clear();
  }

 private:
  std::vector<bool> values_;
  std::vector<MaskWrite> writes_;
};

bool update_fake_visibility(
    ::example::edgeinfer::detail::VisibilityMaskCache& cache,
    FakeVisibilityMask& mask,
    size_t query_rows,
    size_t block_width,
    const std::vector<size_t>& visible_columns) {
  if (visible_columns.size() != query_rows) {
    return false;
  }
  return ::example::edgeinfer::detail::update_visibility_mask_cache(
      cache,
      query_rows,
      block_width,
      [&](size_t row) { return visible_columns[row]; },
      [&](size_t offset, size_t count, bool visible) {
        return mask.fill(offset, count, visible);
      });
}

std::vector<bool> legacy_causal_visibility_mask(
    size_t query_rows,
    size_t block_width,
    size_t block_begin,
    size_t block_valid_width,
    size_t causal_query_begin,
    size_t valid_query_rows) {
  std::vector<bool> mask(query_rows * block_width, false);
  for (size_t row = 0; row < valid_query_rows; ++row) {
    const size_t row_visible_end = causal_query_begin + row + size_t{1};
    const size_t visible_columns = block_begin >= row_visible_end
        ? size_t{0}
        : std::min(block_valid_width, row_visible_end - block_begin);
    std::fill_n(mask.begin() + row * block_width, visible_columns, true);
  }
  return mask;
}

bool update_fake_causal_visibility(
    ::example::edgeinfer::detail::VisibilityMaskCache& cache,
    FakeVisibilityMask& mask,
    size_t query_rows,
    size_t block_width,
    size_t block_begin,
    size_t block_valid_width,
    size_t causal_query_begin,
    size_t valid_query_rows) {
  return ::example::edgeinfer::detail::update_causal_visibility_mask_cache(
      cache,
      query_rows,
      block_width,
      block_begin,
      block_valid_width,
      causal_query_begin,
      valid_query_rows,
      [&](size_t offset, size_t count, bool visible) {
        return mask.fill(offset, count, visible);
      });
}

TEST(
    EdgeInferVisibilityMaskCacheTest,
    PrefillMatchesLegacyRewriteAcrossBoundaryAndPartialRows) {
  constexpr size_t kQueryRows = 4;
  constexpr size_t kBlockWidth = 8;
  constexpr size_t kBlockBegin = 8;
  ::example::edgeinfer::detail::VisibilityMaskCache cache;
  FakeVisibilityMask mask(kQueryRows * kBlockWidth);
  ASSERT_TRUE(cache.reset_known_zero(kQueryRows, kBlockWidth));

  struct Case {
    size_t block_valid_width;
    size_t causal_query_begin;
    size_t valid_query_rows;
  };
  const std::vector<Case> cases{
      // Staircase begins before this K/V block, so the first Query row is zero.
      {3, 7, 4},
      // The boundary advances and newly appended K/V columns become visible.
      {6, 10, 4},
      // A partial final Query tile must clear stale rows from the prior call.
      {8, 12, 2},
      // Repeating the same boundary must be a cache hit.
      {8, 12, 2},
      // A lower boundary verifies differential clearing against the old path.
      {8, 8, 4},
  };

  for (const Case& test_case : cases) {
    ASSERT_TRUE(update_fake_causal_visibility(
        cache,
        mask,
        kQueryRows,
        kBlockWidth,
        kBlockBegin,
        test_case.block_valid_width,
        test_case.causal_query_begin,
        test_case.valid_query_rows));
    EXPECT_EQ(
        mask.values(),
        legacy_causal_visibility_mask(
            kQueryRows,
            kBlockWidth,
            kBlockBegin,
            test_case.block_valid_width,
            test_case.causal_query_begin,
            test_case.valid_query_rows));
  }
}

TEST(
    EdgeInferVisibilityMaskCacheTest,
    DecodeMatchesLegacyRewriteAsContextGrowsAndContracts) {
  constexpr size_t kQueryRows = 1;
  constexpr size_t kBlockWidth = 8;
  constexpr size_t kBlockBegin = 32;
  ::example::edgeinfer::detail::VisibilityMaskCache cache;
  FakeVisibilityMask mask(kQueryRows * kBlockWidth);
  ASSERT_TRUE(cache.reset_known_zero(kQueryRows, kBlockWidth));

  const std::vector<size_t> query_positions{32, 34, 39, 35, 35};
  for (const size_t query_position : query_positions) {
    ASSERT_TRUE(update_fake_causal_visibility(
        cache,
        mask,
        kQueryRows,
        kBlockWidth,
        kBlockBegin,
        /*block_valid_width=*/8,
        query_position,
        /*valid_query_rows=*/1));
    EXPECT_EQ(
        mask.values(),
        legacy_causal_visibility_mask(
            kQueryRows,
            kBlockWidth,
            kBlockBegin,
            /*block_valid_width=*/8,
            query_position,
            /*valid_query_rows=*/1));
  }
}

TEST(
    EdgeInferVisibilityMaskCacheTest,
    ExhaustiveSmallCausalMasksMatchLegacyRewriteElementwise) {
  for (size_t query_rows = 1; query_rows <= 4; ++query_rows) {
    for (size_t block_width = 1; block_width <= 8; ++block_width) {
      ::example::edgeinfer::detail::VisibilityMaskCache cache;
      FakeVisibilityMask mask(query_rows * block_width);
      ASSERT_TRUE(cache.reset_known_zero(query_rows, block_width));
      for (size_t block_begin = 0; block_begin <= 12; ++block_begin) {
        for (size_t block_valid_width = 0; block_valid_width <= block_width;
             ++block_valid_width) {
          for (size_t causal_query_begin = 0; causal_query_begin <= 12;
               ++causal_query_begin) {
            for (size_t valid_query_rows = 1; valid_query_rows <= query_rows;
                 ++valid_query_rows) {
              SCOPED_TRACE(
                  ::testing::Message()
                  << "query_rows=" << query_rows << " block_width="
                  << block_width << " block_begin=" << block_begin
                  << " block_valid_width=" << block_valid_width
                  << " causal_query_begin=" << causal_query_begin
                  << " valid_query_rows=" << valid_query_rows);
              ASSERT_TRUE(update_fake_causal_visibility(
                  cache,
                  mask,
                  query_rows,
                  block_width,
                  block_begin,
                  block_valid_width,
                  causal_query_begin,
                  valid_query_rows));
              EXPECT_EQ(
                  mask.values(),
                  legacy_causal_visibility_mask(
                      query_rows,
                      block_width,
                      block_begin,
                      block_valid_width,
                      causal_query_begin,
                      valid_query_rows));
            }
          }
        }
      }
    }
  }
}

TEST(EdgeInferVisibilityMaskCacheTest, CacheHitPerformsNoPhysicalWrites) {
  ::example::edgeinfer::detail::VisibilityMaskCache cache;
  FakeVisibilityMask mask(/*elements=*/12);
  ASSERT_TRUE(cache.reset_known_zero(/*query_rows=*/3, /*block_width=*/4));

  ASSERT_TRUE(update_fake_visibility(cache, mask, 3, 4, {2, 3, 4}));
  EXPECT_EQ(
      mask.writes(),
      (std::vector<MaskWrite>{{0, 2, true}, {4, 3, true}, {8, 4, true}}));
  EXPECT_EQ(
      mask.values(),
      (std::vector<bool>{
          true,
          true,
          false,
          false,
          true,
          true,
          true,
          false,
          true,
          true,
          true,
          true}));

  mask.clear_writes();
  ASSERT_TRUE(update_fake_visibility(cache, mask, 3, 4, {2, 3, 4}));
  EXPECT_TRUE(mask.writes().empty());
}

TEST(
    EdgeInferVisibilityMaskCacheTest,
    BoundaryGrowthAndShrinkWriteOnlyChangedColumns) {
  ::example::edgeinfer::detail::VisibilityMaskCache cache;
  FakeVisibilityMask mask(/*elements=*/8);
  ASSERT_TRUE(cache.reset_known_zero(/*query_rows=*/1, /*block_width=*/8));

  ASSERT_TRUE(update_fake_visibility(cache, mask, 1, 8, {3}));
  mask.clear_writes();
  ASSERT_TRUE(update_fake_visibility(cache, mask, 1, 8, {5}));
  EXPECT_EQ(mask.writes(), (std::vector<MaskWrite>{{3, 2, true}}));

  mask.clear_writes();
  ASSERT_TRUE(update_fake_visibility(cache, mask, 1, 8, {2}));
  EXPECT_EQ(mask.writes(), (std::vector<MaskWrite>{{2, 3, false}}));
  EXPECT_EQ(
      mask.values(),
      (std::vector<bool>{
          true, true, false, false, false, false, false, false}));

  mask.clear_writes();
  ASSERT_TRUE(update_fake_visibility(cache, mask, 1, 8, {8}));
  EXPECT_EQ(mask.writes(), (std::vector<MaskWrite>{{2, 6, true}}));
  mask.clear_writes();
  ASSERT_TRUE(update_fake_visibility(cache, mask, 1, 8, {8}));
  EXPECT_TRUE(mask.writes().empty());
}

TEST(
    EdgeInferVisibilityMaskCacheTest,
    InvalidationClearsAndRebuildsThePhysicalMask) {
  ::example::edgeinfer::detail::VisibilityMaskCache cache;
  FakeVisibilityMask mask(/*elements=*/8);
  ASSERT_TRUE(cache.reset_known_zero(/*query_rows=*/2, /*block_width=*/4));
  ASSERT_TRUE(update_fake_visibility(cache, mask, 2, 4, {4, 4}));

  cache.invalidate();
  mask.clear_writes();
  ASSERT_TRUE(update_fake_visibility(cache, mask, 2, 4, {1, 3}));
  EXPECT_EQ(
      mask.writes(),
      (std::vector<MaskWrite>{{0, 8, false}, {0, 1, true}, {4, 3, true}}));
  EXPECT_EQ(
      mask.values(),
      (std::vector<bool>{true, false, false, false, true, true, true, false}));
}

TEST(
    EdgeInferVisibilityMaskCacheTest,
    LayoutReplacementStartsWithIndependentKnownZeroState) {
  ::example::edgeinfer::detail::VisibilityMaskCache old_cache;
  FakeVisibilityMask old_mask(/*elements=*/8);
  ASSERT_TRUE(old_cache.reset_known_zero(/*query_rows=*/2, /*block_width=*/4));
  ASSERT_TRUE(update_fake_visibility(old_cache, old_mask, 2, 4, {4, 4}));

  // allocate_layout() creates a zeroed tensor and a new cache for every
  // replacement Block; commit_recompose() swaps that complete Block in.
  ::example::edgeinfer::detail::VisibilityMaskCache replacement_cache;
  FakeVisibilityMask replacement_mask(/*elements=*/16);
  ASSERT_TRUE(
      replacement_cache.reset_known_zero(/*query_rows=*/2, /*block_width=*/8));
  ASSERT_TRUE(update_fake_visibility(
      replacement_cache, replacement_mask, 2, 8, {2, 6}));

  EXPECT_EQ(
      replacement_mask.writes(),
      (std::vector<MaskWrite>{{0, 2, true}, {8, 6, true}}));
  EXPECT_EQ(
      replacement_mask.values(),
      (std::vector<bool>{
          true,
          true,
          false,
          false,
          false,
          false,
          false,
          false,
          true,
          true,
          true,
          true,
          true,
          true,
          false,
          false}));
  EXPECT_EQ(
      old_mask.values(),
      (std::vector<bool>{true, true, true, true, true, true, true, true}));
}

TEST(
    EdgeInferVisibilityMaskCacheTest,
    QueryShapeChangeInvalidatesEvenWhenElementCountIsUnchanged) {
  ::example::edgeinfer::detail::VisibilityMaskCache cache;
  FakeVisibilityMask mask(/*elements=*/8);
  ASSERT_TRUE(cache.reset_known_zero(/*query_rows=*/2, /*block_width=*/4));
  ASSERT_TRUE(update_fake_visibility(cache, mask, 2, 4, {4, 4}));

  mask.clear_writes();
  ASSERT_TRUE(update_fake_visibility(cache, mask, 1, 8, {3}));
  EXPECT_EQ(
      mask.writes(), (std::vector<MaskWrite>{{0, 8, false}, {0, 3, true}}));
  EXPECT_EQ(
      mask.values(),
      (std::vector<bool>{true, true, true, false, false, false, false, false}));
}

TEST(
    EdgeInferVisibilityMaskCacheTest,
    PrefillAndDecodeModesKeepIndependentMaterializedState) {
  ::example::edgeinfer::detail::VisibilityMaskCache prefill_cache;
  ::example::edgeinfer::detail::VisibilityMaskCache decode_cache;
  FakeVisibilityMask prefill_mask(/*elements=*/8);
  FakeVisibilityMask decode_mask(/*elements=*/4);
  ASSERT_TRUE(
      prefill_cache.reset_known_zero(/*query_rows=*/2, /*block_width=*/4));
  ASSERT_TRUE(
      decode_cache.reset_known_zero(/*query_rows=*/1, /*block_width=*/4));

  ASSERT_TRUE(
      update_fake_visibility(prefill_cache, prefill_mask, 2, 4, {2, 3}));
  ASSERT_TRUE(update_fake_visibility(decode_cache, decode_mask, 1, 4, {4}));
  prefill_mask.clear_writes();
  decode_mask.clear_writes();

  ASSERT_TRUE(
      update_fake_visibility(prefill_cache, prefill_mask, 2, 4, {2, 3}));
  EXPECT_TRUE(prefill_mask.writes().empty());
  EXPECT_TRUE(decode_mask.writes().empty());

  ASSERT_TRUE(update_fake_visibility(decode_cache, decode_mask, 1, 4, {2}));
  EXPECT_TRUE(prefill_mask.writes().empty());
  EXPECT_EQ(decode_mask.writes(), (std::vector<MaskWrite>{{2, 2, false}}));
}

TEST(
    EdgeInferPersistentPlanTransactionTest,
    MidPrepareFailurePreservesEveryLayerLayoutAndKv) {
  std::vector<int> events;
  std::vector<std::unique_ptr<FakePersistentLayerCache>> layers;
  for (size_t layer = 0; layer < 3; ++layer) {
    layers.emplace_back(make_persistent_layer(layer, &events));
  }
  layers[1]->fail_prepare(true);

  std::vector<std::vector<size_t>> old_layouts;
  std::vector<std::vector<int>> old_keys;
  std::vector<std::vector<int>> old_values;
  for (const auto& layer : layers) {
    old_layouts.push_back(layer->layout());
    old_keys.push_back(layer->key());
    old_values.push_back(layer->value());
  }
  bool bindings_released = false;
  EXPECT_FALSE(::example::edgeinfer::detail::apply_layer_plan_atomically(
      layers, FakePersistentPlan{{4}}, [&]() { bindings_released = true; }));

  EXPECT_FALSE(bindings_released);
  EXPECT_EQ(events, (std::vector<int>{10, 11}));
  for (size_t layer = 0; layer < layers.size(); ++layer) {
    EXPECT_EQ(layers[layer]->layout(), old_layouts[layer]);
    EXPECT_EQ(layers[layer]->key(), old_keys[layer]);
    EXPECT_EQ(layers[layer]->value(), old_values[layer]);
  }
}

TEST(
    EdgeInferPersistentPlanTransactionTest,
    SameLayoutNeedsNoReplacementOrBindingRelease) {
  std::vector<int> events;
  std::vector<std::unique_ptr<FakePersistentLayerCache>> layers;
  layers.emplace_back(make_persistent_layer(0, &events));
  layers.emplace_back(make_persistent_layer(1, &events));
  bool bindings_released = false;

  EXPECT_TRUE(::example::edgeinfer::detail::apply_layer_plan_atomically(
      layers, FakePersistentPlan{{2, 2}}, [&]() { bindings_released = true; }));

  EXPECT_FALSE(bindings_released);
  EXPECT_EQ(events, (std::vector<int>{10, 11}));
  EXPECT_EQ(layers[0]->layout(), (std::vector<size_t>{2, 2}));
  EXPECT_EQ(layers[1]->layout(), (std::vector<size_t>{2, 2}));
}

TEST(
    EdgeInferPersistentPlanTransactionTest,
    CommitsOnlyAfterEveryLayerIsPrepared) {
  std::vector<int> events;
  std::vector<std::unique_ptr<FakePersistentLayerCache>> layers;
  for (size_t layer = 0; layer < 3; ++layer) {
    layers.emplace_back(make_persistent_layer(layer, &events));
  }

  EXPECT_TRUE(::example::edgeinfer::detail::apply_layer_plan_atomically(
      layers, FakePersistentPlan{{4}}, [&]() { events.push_back(20); }));

  EXPECT_EQ(events, (std::vector<int>{10, 11, 12, 20, 30, 31, 32}));
  for (size_t layer = 0; layer < layers.size(); ++layer) {
    EXPECT_EQ(layers[layer]->layout(), (std::vector<size_t>{4}));
    EXPECT_EQ(
        layers[layer]->key(),
        (std::vector<int>{
            static_cast<int>(100 + layer), static_cast<int>(110 + layer)}));
    EXPECT_EQ(
        layers[layer]->value(),
        (std::vector<int>{
            static_cast<int>(200 + layer), static_cast<int>(210 + layer)}));
  }
}

TEST(EdgeInferDecodeCacheTransactionTest, RollsBackAcrossBlockBoundary) {
  std::vector<FakeBlock> blocks{{4, 4}, {4, 4}, {2, 2}};

  rollback_edgeinfer_cache_blocks(blocks, 7);

  EXPECT_EQ(blocks[0].valid_width, size_t{4});
  EXPECT_EQ(blocks[1].valid_width, size_t{3});
  EXPECT_EQ(blocks[2].valid_width, size_t{0});
}

TEST(
    EdgeInferDecodeCacheTransactionTest,
    RollsBackEveryLayerAfterMidStepFailure) {
  FakeLayerCache first(4096);
  FakeLayerCache second(4096);
  FakeLayerCache third(4096);
  {
    const Error error = run_edgeinfer_decode_cache_transaction(
        std::vector<FakeLayerCache*>{&first, &second, &third}, 4096, 1, [&]() {
          first.append(1);
          second.append(1);
          // The third layer represents a failure before its K/V append.
          return Error::Internal;
        });
    EXPECT_EQ(error, Error::Internal);
  }

  EXPECT_EQ(first.valid_length(), size_t{4096});
  EXPECT_EQ(second.valid_length(), size_t{4096});
  EXPECT_EQ(third.valid_length(), size_t{4096});
  EXPECT_EQ(first.rollback_calls(), size_t{1});
  EXPECT_EQ(second.rollback_calls(), size_t{1});
  EXPECT_EQ(third.rollback_calls(), size_t{1});
}

TEST(EdgeInferDecodeCacheTransactionTest, CommitKeepsOneAppendPerLayer) {
  FakeLayerCache first(1024);
  FakeLayerCache second(1024);
  {
    const Error error = run_edgeinfer_decode_cache_transaction(
        std::vector<FakeLayerCache*>{&first, &second}, 1024, 1, [&]() {
          first.append(1);
          second.append(1);
          return Error::Ok;
        });
    EXPECT_EQ(error, Error::Ok);
  }

  EXPECT_EQ(first.valid_length(), size_t{1025});
  EXPECT_EQ(second.valid_length(), size_t{1025});
  EXPECT_EQ(first.rollback_calls(), size_t{0});
  EXPECT_EQ(second.rollback_calls(), size_t{0});
}

TEST(
    EdgeInferDecodeCacheTransactionTest,
    RejectsUnevenLayerCommitAndRollsBack) {
  FakeLayerCache first(128);
  FakeLayerCache second(128);
  {
    const Error error = run_edgeinfer_decode_cache_transaction(
        std::vector<FakeLayerCache*>{&first, &second}, 128, 1, [&]() {
          first.append(1);
          return Error::Ok;
        });
    EXPECT_EQ(error, Error::InvalidState);
  }

  EXPECT_EQ(first.valid_length(), size_t{128});
  EXPECT_EQ(second.valid_length(), size_t{128});
}

TEST(EdgeInferDecodeCacheTransactionTest, RejectsNullCache) {
  FakeLayerCache cache(64);
  bool called = false;
  const Error error = run_edgeinfer_decode_cache_transaction(
      std::vector<FakeLayerCache*>{&cache, nullptr}, 64, 1, [&]() {
        called = true;
        return Error::Ok;
      });
  EXPECT_EQ(error, Error::InvalidState);
  EXPECT_FALSE(called);
}

TEST(
    EdgeInferDecodeCacheTransactionTest,
    RejectsUnsynchronizedInitialLengthsWithoutMutation) {
  FakeLayerCache first(64);
  FakeLayerCache second(63);
  bool called = false;

  const Error error = run_edgeinfer_decode_cache_transaction(
      std::vector<FakeLayerCache*>{&first, &second}, 64, 1, [&]() {
        called = true;
        return Error::Ok;
      });

  EXPECT_EQ(error, Error::InvalidState);
  EXPECT_FALSE(called);
  EXPECT_EQ(first.valid_length(), size_t{64});
  EXPECT_EQ(second.valid_length(), size_t{63});
  EXPECT_EQ(first.rollback_calls(), size_t{0});
  EXPECT_EQ(second.rollback_calls(), size_t{0});
}

TEST(
    EdgeInferDecodeCacheTransactionTest,
    RollsBackBeforeBridgeAndContinuesWithNativeDecode) {
  FakeLayerCache first(4096);
  FakeLayerCache second(4096);
  FakeLayerCache third(4096);
  const Error decode_error = run_edgeinfer_decode_cache_transaction(
      std::vector<FakeLayerCache*>{&first, &second, &third}, 4096, 1, [&]() {
        first.append(1);
        second.append(1);
        return Error::Internal;
      });
  ASSERT_EQ(decode_error, Error::Internal);

  bool bridge_called = false;
  bool native_called = false;
  auto recovered = resume_native_decode_after_edgeinfer_failure(
      2,
      17,
      4096,
      [&]() {
        bridge_called = true;
        EXPECT_EQ(first.valid_length(), size_t{4096});
        EXPECT_EQ(second.valid_length(), size_t{4096});
        EXPECT_EQ(third.valid_length(), size_t{4096});
        return Error::Ok;
      },
      [&](std::vector<uint64_t> tokens, int64_t start_pos) -> Result<int64_t> {
        native_called = true;
        EXPECT_TRUE(bridge_called);
        EXPECT_EQ(tokens.size(), size_t{1});
        if (tokens.size() != 1) {
          return Error::InvalidState;
        }
        EXPECT_EQ(tokens[0], uint64_t{17});
        EXPECT_EQ(start_pos, int64_t{4096});
        return int64_t{3};
      });

  ASSERT_TRUE(recovered.ok());
  EXPECT_EQ(recovered.get(), int64_t{5});
  EXPECT_TRUE(bridge_called);
  EXPECT_TRUE(native_called);
}

TEST(EdgeInferDecodeCacheTransactionTest, BridgeFailurePreventsNativeDecode) {
  bool native_called = false;
  auto recovered = resume_native_decode_after_edgeinfer_failure(
      2,
      17,
      4096,
      []() { return Error::MemoryAllocationFailed; },
      [&](std::vector<uint64_t>, int64_t) -> Result<int64_t> {
        native_called = true;
        return int64_t{3};
      });

  EXPECT_FALSE(recovered.ok());
  EXPECT_EQ(recovered.error(), Error::MemoryAllocationFailed);
  EXPECT_FALSE(native_called);
}

TEST(
    EdgeInferDecodeCacheTransactionTest,
    NativeDecodeFailureIsPropagatedAfterBridge) {
  bool bridge_called = false;
  auto recovered = resume_native_decode_after_edgeinfer_failure(
      2,
      17,
      4096,
      [&]() {
        bridge_called = true;
        return Error::Ok;
      },
      [](std::vector<uint64_t>, int64_t) -> Result<int64_t> {
        return Error::Internal;
      });

  EXPECT_TRUE(bridge_called);
  EXPECT_FALSE(recovered.ok());
  EXPECT_EQ(recovered.error(), Error::Internal);
}

TEST(
    EdgeInferDecodeCacheTransactionTest,
    HybridBridgeUsesDecodeArAndDecodeTailCapacity) {
  FakeCacheManager cache_manager(/*prompt_ar_len=*/128);

  EXPECT_EQ(
      ::example::edgeinfer::prepare_native_decode_cache_layout(
          cache_manager,
          /*context_len=*/4096,
          /*native_decode_ar_len=*/1,
          /*sequence_length=*/4095),
      Error::Ok);
  EXPECT_EQ(cache_manager.ar_len(), 1);
  EXPECT_EQ(cache_manager.physical_rearrangements(), size_t{1});

  // TokenGenerator's existing Decode-AR rearrangement is now a no-op.
  cache_manager.rearrange_cache(1);
  EXPECT_EQ(cache_manager.calls(), size_t{2});
  EXPECT_EQ(cache_manager.physical_rearrangements(), size_t{1});

  EXPECT_EQ(
      ::example::edgeinfer::prepare_native_decode_cache_layout(
          cache_manager,
          /*context_len=*/4096,
          /*native_decode_ar_len=*/1,
          /*sequence_length=*/4096),
      Error::InvalidArgument);
  EXPECT_EQ(cache_manager.calls(), size_t{2});
}

TEST(
    EdgeInferDecodeCacheTransactionTest,
    PrefillFailureRestoresHistoryBeforeBridgeAndNativePrefill) {
  FakeLayerCache first(1024);
  FakeLayerCache second(1024);
  size_t sequence_length = 1024;
  std::vector<int> order;

  auto result = ::example::edgeinfer::run_prefill_with_native_fallback(
      std::vector<FakeLayerCache*>{&first, &second},
      /*checkpoint=*/1024,
      /*appended_rows=*/128,
      [&]() -> Result<uint64_t> {
        first.append(128);
        second.append(64);
        sequence_length += 64;
        return Error::Internal;
      },
      [&](Error failure) {
        EXPECT_EQ(failure, Error::Internal);
        EXPECT_EQ(first.valid_length(), size_t{1024});
        EXPECT_EQ(second.valid_length(), size_t{1024});
        sequence_length = 1024;
        order.push_back(1);
      },
      [&]() {
        EXPECT_EQ(sequence_length, size_t{1024});
        EXPECT_EQ(first.valid_length(), size_t{1024});
        EXPECT_EQ(second.valid_length(), size_t{1024});
        order.push_back(2);
        return Error::Ok;
      },
      [&]() { order.push_back(3); },
      [&]() -> Result<uint64_t> {
        order.push_back(4);
        return uint64_t{77};
      });

  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.get(), uint64_t{77});
  EXPECT_EQ(order, (std::vector<int>{1, 2, 3, 4}));
}

TEST(
    EdgeInferDecodeCacheTransactionTest,
    PrefillBridgeFailurePreventsClearAndNativePrefill) {
  FakeLayerCache cache(64);
  bool clear_called = false;
  bool native_called = false;

  auto result = ::example::edgeinfer::run_prefill_with_native_fallback(
      std::vector<FakeLayerCache*>{&cache},
      /*checkpoint=*/64,
      /*appended_rows=*/8,
      [&]() -> Result<uint64_t> {
        cache.append(4);
        return Error::Internal;
      },
      [](Error) {},
      []() { return Error::MemoryAllocationFailed; },
      [&]() { clear_called = true; },
      [&]() -> Result<uint64_t> {
        native_called = true;
        return uint64_t{1};
      });

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.error(), Error::MemoryAllocationFailed);
  EXPECT_EQ(cache.valid_length(), size_t{64});
  EXPECT_FALSE(clear_called);
  EXPECT_FALSE(native_called);
}

TEST(
    EdgeInferDecodeCacheTransactionTest,
    FirstTurnPrefillFallbackDoesNotRequireHistoryBridge) {
  FakeLayerCache cache(0);
  bool bridge_called = false;
  bool clear_called = false;
  bool native_called = false;

  auto result = ::example::edgeinfer::run_prefill_with_native_fallback(
      std::vector<FakeLayerCache*>{&cache},
      /*checkpoint=*/0,
      /*appended_rows=*/4,
      [&]() -> Result<uint64_t> {
        cache.append(2);
        return Error::Internal;
      },
      [](Error) {},
      [&]() {
        bridge_called = true;
        return Error::Ok;
      },
      [&]() { clear_called = true; },
      [&]() -> Result<uint64_t> {
        native_called = true;
        return uint64_t{9};
      });

  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.get(), uint64_t{9});
  EXPECT_EQ(cache.valid_length(), size_t{0});
  EXPECT_FALSE(bridge_called);
  EXPECT_TRUE(clear_called);
  EXPECT_TRUE(native_called);
}

TEST(
    EdgeInferDecodeCacheTransactionTest,
    SuccessfulPrefillCommitsWithoutNativeFallback) {
  FakeLayerCache first(32);
  FakeLayerCache second(32);
  bool fallback_called = false;

  auto result = ::example::edgeinfer::run_prefill_with_native_fallback(
      std::vector<FakeLayerCache*>{&first, &second},
      /*checkpoint=*/32,
      /*appended_rows=*/4,
      [&]() -> Result<uint64_t> {
        first.append(4);
        second.append(4);
        return uint64_t{11};
      },
      [&](Error) { fallback_called = true; },
      [&]() {
        fallback_called = true;
        return Error::Ok;
      },
      [&]() { fallback_called = true; },
      [&]() -> Result<uint64_t> {
        fallback_called = true;
        return uint64_t{0};
      });

  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.get(), uint64_t{11});
  EXPECT_EQ(first.valid_length(), size_t{36});
  EXPECT_EQ(second.valid_length(), size_t{36});
  EXPECT_FALSE(fallback_called);
}

TEST(EdgeInferLegacyPathTest, EmptyConfigurationPreservesPromptProcessor) {
  EXPECT_FALSE(::example::edgeinfer::should_use_edgeinfer_prompt_processor(
      nullptr, /*lookahead_decoding=*/false));
  EXPECT_FALSE(::example::edgeinfer::should_use_edgeinfer_prompt_processor(
      reinterpret_cast<void*>(uintptr_t{1}), /*lookahead_decoding=*/true));
  EXPECT_TRUE(::example::edgeinfer::should_use_edgeinfer_prompt_processor(
      reinterpret_cast<void*>(uintptr_t{1}), /*lookahead_decoding=*/false));
}

TEST(EdgeInferDecodeRowsTest, SharedPortfolioUsesPrefillQueryRows) {
  EXPECT_EQ(
      ::example::edgeinfer::detail::query_rows_for_phase(
          /*prefill_query_rows=*/32,
          /*decode=*/false,
          /*has_separate_decode_portfolio=*/false),
      size_t{32});
  EXPECT_EQ(
      ::example::edgeinfer::detail::query_rows_for_phase(
          /*prefill_query_rows=*/32,
          /*decode=*/true,
          /*has_separate_decode_portfolio=*/false),
      size_t{32});
}

TEST(EdgeInferDecodeRowsTest, SeparatePortfolioUsesOneDecodeQueryRow) {
  EXPECT_EQ(
      ::example::edgeinfer::detail::query_rows_for_phase(
          /*prefill_query_rows=*/32,
          /*decode=*/true,
          /*has_separate_decode_portfolio=*/true),
      size_t{1});
}

TEST(
    EdgeInferLegacyPathTest,
    UnsupportedCacheFallsBackBeforeMutationAndNeverDoubleExecutes) {
  using executorch::aten::ScalarType;
  for (const ScalarType scalar_type : {ScalarType::Byte, ScalarType::UInt16}) {
    SCOPED_TRACE(executorch::runtime::toString(scalar_type));
    const Error preflight_error =
        ::example::edgeinfer::validate_edgeinfer_cache_scalar_type(scalar_type);
    ASSERT_EQ(preflight_error, Error::NotSupported);

    size_t edgeinfer_calls = 0;
    size_t clear_calls = 0;
    size_t native_calls = 0;
    size_t mutations = 0;
    auto result = ::example::edgeinfer::dispatch_prefill_after_preflight(
        preflight_error,
        [&]() -> Result<uint64_t> {
          ++edgeinfer_calls;
          ++mutations;
          return uint64_t{1};
        },
        [&]() {
          ++clear_calls;
          EXPECT_EQ(mutations, size_t{0});
        },
        [&]() -> Result<uint64_t> {
          ++native_calls;
          EXPECT_EQ(mutations, size_t{0});
          return uint64_t{73};
        });

    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result.get(), uint64_t{73});
    EXPECT_EQ(edgeinfer_calls, size_t{0});
    EXPECT_EQ(clear_calls, size_t{1});
    EXPECT_EQ(native_calls, size_t{1});
    EXPECT_EQ(mutations, size_t{0});
  }
}

TEST(EdgeInferLegacyPathTest, FloatAndHalfPassCachePreflight) {
  using executorch::aten::ScalarType;
  EXPECT_EQ(
      ::example::edgeinfer::validate_edgeinfer_cache_scalar_type(
          ScalarType::Float),
      Error::Ok);
  EXPECT_EQ(
      ::example::edgeinfer::validate_edgeinfer_cache_scalar_type(
          ScalarType::Half),
      Error::Ok);
}

TEST(
    EdgeInferFallbackFaultInjectionTest,
    DecodeProviderFailuresRollbackBeforeSingleNativeResume) {
  const std::array<Error, 4> failures{
      Error::DelegateInvalidHandle,
      Error::DelegateMemoryAllocationFailed,
      Error::AccessFailed,
      Error::Internal};
  for (const Error failure : failures) {
    SCOPED_TRACE(static_cast<unsigned int>(failure));
    FakeLayerCache first(2048);
    FakeLayerCache second(2048);
    size_t bridge_calls = 0;
    size_t native_calls = 0;

    const Error decode_error = run_edgeinfer_decode_cache_transaction(
        std::vector<FakeLayerCache*>{&first, &second}, 2048, 1, [&]() {
          first.append(1);
          return failure;
        });
    ASSERT_EQ(decode_error, failure);
    ASSERT_EQ(first.valid_length(), size_t{2048});
    ASSERT_EQ(second.valid_length(), size_t{2048});

    auto recovered = resume_native_decode_after_edgeinfer_failure(
        /*successful_edgeinfer_tokens=*/
        0,
        /*failed_token=*/31,
        /*failed_position=*/2048,
        [&]() {
          ++bridge_calls;
          EXPECT_EQ(first.valid_length(), size_t{2048});
          EXPECT_EQ(second.valid_length(), size_t{2048});
          return Error::Ok;
        },
        [&](std::vector<uint64_t> tokens, int64_t position) -> Result<int64_t> {
          ++native_calls;
          EXPECT_EQ(tokens, (std::vector<uint64_t>{31}));
          EXPECT_EQ(position, int64_t{2048});
          return int64_t{2};
        });
    ASSERT_TRUE(recovered.ok());
    EXPECT_EQ(recovered.get(), int64_t{2});
    EXPECT_EQ(bridge_calls, size_t{1});
    EXPECT_EQ(native_calls, size_t{1});
  }
}

TEST(
    EdgeInferFallbackFaultInjectionTest,
    PrefillProviderFailuresRestoreAllLayersBeforeSingleFallback) {
  const std::array<Error, 4> failures{
      Error::DelegateInvalidHandle,
      Error::DelegateMemoryAllocationFailed,
      Error::AccessFailed,
      Error::Internal};
  for (const Error failure : failures) {
    SCOPED_TRACE(static_cast<unsigned int>(failure));
    FakeLayerCache first(512);
    FakeLayerCache second(512);
    size_t restore_calls = 0;
    size_t bridge_calls = 0;
    size_t clear_calls = 0;
    size_t native_calls = 0;

    auto result = ::example::edgeinfer::run_prefill_with_native_fallback(
        std::vector<FakeLayerCache*>{&first, &second},
        /*checkpoint=*/512,
        /*appended_rows=*/32,
        [&]() -> Result<uint64_t> {
          first.append(32);
          second.append(16);
          return failure;
        },
        [&](Error actual) {
          ++restore_calls;
          EXPECT_EQ(actual, failure);
          EXPECT_EQ(first.valid_length(), size_t{512});
          EXPECT_EQ(second.valid_length(), size_t{512});
        },
        [&]() {
          ++bridge_calls;
          EXPECT_EQ(restore_calls, size_t{1});
          return Error::Ok;
        },
        [&]() {
          ++clear_calls;
          EXPECT_EQ(bridge_calls, size_t{1});
        },
        [&]() -> Result<uint64_t> {
          ++native_calls;
          EXPECT_EQ(clear_calls, size_t{1});
          return uint64_t{99};
        });

    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result.get(), uint64_t{99});
    EXPECT_EQ(restore_calls, size_t{1});
    EXPECT_EQ(bridge_calls, size_t{1});
    EXPECT_EQ(clear_calls, size_t{1});
    EXPECT_EQ(native_calls, size_t{1});
  }
}

template <typename Source, typename Destination>
void expect_layer_cache_bridge_conversion() {
  using executorch::aten::Half;
  constexpr size_t kHeads = 1;
  constexpr size_t kHeadDim = 2;
  constexpr size_t kValid = 2;
  constexpr size_t kCapacity = 3;
  const std::array<float, 4> key_values{1.0f, 2.0f, 3.0f, 4.0f};
  const std::array<float, 4> value_values{5.0f, 6.0f, 7.0f, 8.0f};
  std::array<Source, 4> source_key{};
  std::array<Source, 4> source_value{};
  for (size_t index = 0; index < source_key.size(); ++index) {
    source_key[index] = Source(key_values[index]);
    source_value[index] = Source(value_values[index]);
  }
  std::array<Destination, 6> destination_key{};
  std::array<Destination, 6> destination_value{};
  destination_key.fill(Destination(42.0f));
  destination_value.fill(Destination(42.0f));

  ::example::edgeinfer::copy_layer_cache_data<Source, Destination>(
      source_key.data(),
      source_value.data(),
      destination_key.data(),
      destination_value.data(),
      kHeads,
      kHeadDim,
      kValid,
      kCapacity);

  const std::array<float, 6> expected_key{1, 2, 0, 3, 4, 0};
  const std::array<float, 6> expected_value{5, 6, 7, 8, 0, 0};
  for (size_t index = 0; index < expected_key.size(); ++index) {
    EXPECT_FLOAT_EQ(
        static_cast<float>(destination_key[index]), expected_key[index]);
    EXPECT_FLOAT_EQ(
        static_cast<float>(destination_value[index]), expected_value[index]);
  }
  static_assert(std::is_same_v<Source, float> || std::is_same_v<Source, Half>);
  static_assert(
      std::is_same_v<Destination, float> || std::is_same_v<Destination, Half>);
}

TEST(EdgeInferCacheBridgeTest, CoversAllFloatHalfSourceDestinationPairs) {
  using executorch::aten::Half;
  expect_layer_cache_bridge_conversion<float, float>();
  expect_layer_cache_bridge_conversion<float, Half>();
  expect_layer_cache_bridge_conversion<Half, float>();
  expect_layer_cache_bridge_conversion<Half, Half>();
}

executorch::extension::TensorPtr make_binding_tensor(
    void* data,
    std::vector<executorch::aten::SizesType> sizes,
    executorch::aten::ScalarType scalar_type,
    std::vector<executorch::aten::DimOrderType> dim_order = {},
    std::vector<executorch::aten::StridesType> strides = {}) {
  return executorch::extension::make_tensor_ptr(
      std::move(sizes),
      data,
      std::move(dim_order),
      std::move(strides),
      scalar_type,
      executorch::aten::TensorShapeDynamism::STATIC,
      [](void*) {});
}

TEST(StageOutputBindingCacheTest, IdenticalFingerprintBindsOnlyOnce) {
  alignas(float) std::array<float, 4> storage{};
  auto tensor = make_binding_tensor(
      storage.data(), {2, 2}, executorch::aten::ScalarType::Float);
  auto fingerprint =
      ::example::edgeinfer::detail::make_stage_output_binding_fingerprint(
          {tensor});
  ASSERT_TRUE(fingerprint.has_value());
  auto* module = reinterpret_cast<executorch::extension::Module*>(uintptr_t{1});
  ::example::edgeinfer::detail::StageOutputBindingCache cache;
  size_t planned_checks = 0;
  size_t bind_calls = 0;
  size_t unload_calls = 0;
  auto prepare = [&]() {
    return ::example::edgeinfer::detail::prepare_stage_output_binding(
        cache,
        module,
        "stage",
        *fingerprint,
        [&]() {
          ++planned_checks;
          return false;
        },
        [&]() {
          ++bind_calls;
          return Error::Ok;
        },
        [&]() { ++unload_calls; });
  };

  EXPECT_EQ(
      prepare(),
      ::example::edgeinfer::detail::StageOutputBindingDisposition::Bound);
  EXPECT_EQ(
      prepare(),
      ::example::edgeinfer::detail::StageOutputBindingDisposition::Bound);
  EXPECT_EQ(planned_checks, size_t{1});
  EXPECT_EQ(bind_calls, size_t{1});
  EXPECT_EQ(unload_calls, size_t{0});
}

TEST(
    StageOutputBindingCacheTest,
    AddressShapeDtypeAndLayoutChangesEachRebindSafely) {
  alignas(8) std::array<std::byte, 64> first_storage{};
  alignas(8) std::array<std::byte, 64> second_storage{};
  std::vector<executorch::extension::TensorPtr> tensors{
      make_binding_tensor(
          first_storage.data(), {2, 2}, executorch::aten::ScalarType::Float),
      make_binding_tensor(
          second_storage.data(), {2, 2}, executorch::aten::ScalarType::Float),
      make_binding_tensor(
          second_storage.data(), {1, 4}, executorch::aten::ScalarType::Float),
      make_binding_tensor(
          second_storage.data(), {1, 4}, executorch::aten::ScalarType::Int),
      make_binding_tensor(
          second_storage.data(),
          {2, 2},
          executorch::aten::ScalarType::Int,
          {1, 0},
          {1, 2})};
  auto* module = reinterpret_cast<executorch::extension::Module*>(uintptr_t{2});
  ::example::edgeinfer::detail::StageOutputBindingCache cache;
  size_t planned_checks = 0;
  size_t bind_calls = 0;
  size_t unload_calls = 0;

  for (const auto& tensor : tensors) {
    auto fingerprint =
        ::example::edgeinfer::detail::make_stage_output_binding_fingerprint(
            {tensor});
    ASSERT_TRUE(fingerprint.has_value());
    EXPECT_EQ(
        ::example::edgeinfer::detail::prepare_stage_output_binding(
            cache,
            module,
            "stage",
            *fingerprint,
            [&]() {
              ++planned_checks;
              return false;
            },
            [&]() {
              ++bind_calls;
              return Error::Ok;
            },
            [&]() { ++unload_calls; }),
        ::example::edgeinfer::detail::StageOutputBindingDisposition::Bound);
  }
  EXPECT_EQ(planned_checks, tensors.size());
  EXPECT_EQ(bind_calls, tensors.size());
  EXPECT_EQ(unload_calls, tensors.size() - 1);
}

TEST(StageOutputBindingCacheTest, ExplicitUnloadInvalidatesFingerprint) {
  alignas(float) std::array<float, 4> storage{};
  auto tensor = make_binding_tensor(
      storage.data(), {4}, executorch::aten::ScalarType::Float);
  auto fingerprint =
      ::example::edgeinfer::detail::make_stage_output_binding_fingerprint(
          {tensor});
  ASSERT_TRUE(fingerprint.has_value());
  auto* module = reinterpret_cast<executorch::extension::Module*>(uintptr_t{3});
  ::example::edgeinfer::detail::StageOutputBindingCache cache;
  size_t bind_calls = 0;
  auto bind = [&]() {
    return ::example::edgeinfer::detail::prepare_stage_output_binding(
        cache,
        module,
        "stage",
        *fingerprint,
        []() { return false; },
        [&]() {
          ++bind_calls;
          return Error::Ok;
        },
        []() {});
  };
  EXPECT_EQ(
      bind(),
      ::example::edgeinfer::detail::StageOutputBindingDisposition::Bound);
  cache.forget_module(module);
  EXPECT_EQ(
      bind(),
      ::example::edgeinfer::detail::StageOutputBindingDisposition::Bound);
  EXPECT_EQ(bind_calls, size_t{2});
}

TEST(
    StageOutputBindingCacheTest,
    PartialBindFailureUnloadsAndUsesCompatibleCopyPath) {
  alignas(float) std::array<float, 4> storage{};
  auto tensor = make_binding_tensor(
      storage.data(), {4}, executorch::aten::ScalarType::Float);
  auto fingerprint =
      ::example::edgeinfer::detail::make_stage_output_binding_fingerprint(
          {tensor});
  ASSERT_TRUE(fingerprint.has_value());
  auto* module = reinterpret_cast<executorch::extension::Module*>(uintptr_t{4});
  ::example::edgeinfer::detail::StageOutputBindingCache cache;
  size_t unload_calls = 0;
  EXPECT_EQ(
      ::example::edgeinfer::detail::prepare_stage_output_binding(
          cache,
          module,
          "stage",
          *fingerprint,
          []() { return false; },
          []() { return Error::InvalidArgument; },
          [&]() { ++unload_calls; }),
      ::example::edgeinfer::detail::StageOutputBindingDisposition::Copy);
  EXPECT_EQ(unload_calls, size_t{1});
  EXPECT_EQ(cache.find(module, "stage"), nullptr);
}

TEST(
    StageOutputBindingCacheTest,
    FirstExecutionAddressMismatchClearsBindingAndMarksCopy) {
  alignas(float) std::array<float, 4> storage{};
  auto tensor = make_binding_tensor(
      storage.data(), {4}, executorch::aten::ScalarType::Float);
  auto fingerprint =
      ::example::edgeinfer::detail::make_stage_output_binding_fingerprint(
          {tensor});
  ASSERT_TRUE(fingerprint.has_value());
  auto* module = reinterpret_cast<executorch::extension::Module*>(uintptr_t{5});
  ::example::edgeinfer::detail::StageOutputBindingCache cache;
  ASSERT_EQ(
      ::example::edgeinfer::detail::prepare_stage_output_binding(
          cache,
          module,
          "stage",
          *fingerprint,
          []() { return false; },
          []() { return Error::Ok; },
          []() {}),
      ::example::edgeinfer::detail::StageOutputBindingDisposition::Bound);
  size_t unload_calls = 0;
  size_t mark_copy_calls = 0;
  ::example::edgeinfer::detail::reject_stage_output_binding(
      cache,
      module,
      "stage",
      [&]() { ++mark_copy_calls; },
      [&]() { ++unload_calls; });
  EXPECT_EQ(unload_calls, size_t{1});
  EXPECT_EQ(mark_copy_calls, size_t{1});
  EXPECT_EQ(cache.find(module, "stage"), nullptr);
}

} // namespace
} // namespace llm
} // namespace extension
} // namespace executorch
