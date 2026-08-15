/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/runner/causal_attention_block.h>
#include <executorch/extension/llm/runner/static_attention_planner.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

using executorch::extension::llm::causal_block_visible_columns;
using executorch::extension::llm::causal_prefill_plan_stats;
using executorch::extension::llm::causal_visible_prefix_end;
using executorch::extension::llm::StaticAttentionGraphCost;
using executorch::extension::llm::StaticAttentionPlan;
using executorch::extension::llm::StaticAttentionPlanner;
using executorch::extension::llm::StaticAttentionPrefillPlanner;
using executorch::extension::llm::StaticAttentionShapeCost;

namespace {

StaticAttentionPlanner make_planner() {
  auto planner = StaticAttentionPlanner::create(
      {{64, 4.0, 3.0}, {128, 6.0, 5.0}, {256, 9.0, 8.0}});
  EXPECT_TRUE(planner.has_value());
  return std::move(*planner);
}

struct PlanObjective {
  bool valid = false;
  double cost = 0.0;
  size_t padding = 0;
  size_t calls = 0;
  size_t blocks = 0;
  size_t coverage = 0;
};

bool is_better_objective(
    const PlanObjective& candidate,
    const PlanObjective& current) {
  if (!current.valid || candidate.cost != current.cost) {
    return !current.valid || candidate.cost < current.cost;
  }
  if (candidate.padding != current.padding) {
    return candidate.padding < current.padding;
  }
  if (candidate.calls != current.calls) {
    return candidate.calls < current.calls;
  }
  if (candidate.blocks != current.blocks) {
    return candidate.blocks < current.blocks;
  }
  return candidate.coverage < current.coverage;
}

const StaticAttentionGraphCost& graph_cost_for_width(
    const std::vector<StaticAttentionGraphCost>& graph_costs,
    size_t width) {
  const auto graph_cost = std::find_if(
      graph_costs.begin(), graph_costs.end(), [width](const auto& entry) {
        return entry.width == width;
      });
  EXPECT_NE(graph_cost, graph_costs.end());
  return *graph_cost;
}

PlanObjective decode_objective(
    const std::vector<StaticAttentionGraphCost>& graph_costs,
    size_t target,
    const std::vector<size_t>& widths) {
  PlanObjective result;
  result.valid = !widths.empty();
  result.blocks = widths.size();
  result.calls = widths.size();
  for (size_t index = 0; index < widths.size(); ++index) {
    const auto& graph_cost = graph_cost_for_width(graph_costs, widths[index]);
    result.cost += index == 0 ? graph_cost.first_cost : graph_cost.merge_cost;
    result.coverage += widths[index];
  }
  result.valid = result.valid && result.coverage >= target;
  result.padding = result.valid ? result.coverage - target : 0;
  return result;
}

PlanObjective prefill_objective(
    const std::vector<StaticAttentionGraphCost>& graph_costs,
    const std::vector<size_t>& visible_prefixes,
    const std::vector<size_t>& widths) {
  PlanObjective result;
  result.valid = !widths.empty() && !visible_prefixes.empty();
  for (const size_t width : widths) {
    result.coverage += width;
  }
  result.blocks = widths.size();
  for (const size_t prefix : visible_prefixes) {
    size_t tile_coverage = 0;
    size_t tile_calls = 0;
    for (const size_t width : widths) {
      if (tile_coverage >= prefix) {
        break;
      }
      const auto& graph_cost = graph_cost_for_width(graph_costs, width);
      result.cost +=
          tile_calls == 0 ? graph_cost.first_cost : graph_cost.merge_cost;
      tile_coverage += width;
      ++tile_calls;
    }
    if (tile_coverage < prefix) {
      result.valid = false;
      return result;
    }
    result.padding += tile_coverage - prefix;
    result.calls += tile_calls;
  }
  return result;
}

template <typename Evaluate>
PlanObjective brute_force_layouts(
    const std::vector<StaticAttentionGraphCost>& graph_costs,
    size_t target,
    Evaluate&& evaluate) {
  PlanObjective best;
  std::vector<size_t> widths;
  auto enumerate = [&](auto&& self, size_t coverage) -> void {
    if (coverage >= target) {
      const PlanObjective candidate = evaluate(widths);
      if (is_better_objective(candidate, best)) {
        best = candidate;
      }
      return;
    }
    for (const auto& graph_cost : graph_costs) {
      widths.push_back(graph_cost.width);
      self(self, coverage + graph_cost.width);
      widths.pop_back();
    }
  };
  enumerate(enumerate, 0);
  return best;
}

TEST(StaticAttentionPlannerTest, SelectsMeasuredMixedWidthPlans) {
  auto planner = make_planner();
  EXPECT_EQ(planner.plan(1)->widths, std::vector<size_t>({64}));
  EXPECT_EQ(planner.plan(192)->widths, std::vector<size_t>({128, 64}));
  EXPECT_EQ(planner.plan(384)->widths, std::vector<size_t>({256, 128}));
}

TEST(StaticAttentionPlannerTest, ComparesOneCoveringGraphWithComposition) {
  auto one_graph =
      StaticAttentionPlanner::create({{64, 1.0, 1.0}, {128, 1.5, 1.5}});
  ASSERT_TRUE(one_graph.has_value());
  EXPECT_EQ(one_graph->plan(100)->widths, std::vector<size_t>({128}));

  auto composed =
      StaticAttentionPlanner::create({{64, 1.0, 0.2}, {128, 2.0, 2.0}});
  ASSERT_TRUE(composed.has_value());
  EXPECT_EQ(composed->plan(100)->widths, std::vector<size_t>({64, 64}));
}

TEST(StaticAttentionPlannerTest, PlansPrefix129As128Plus1WhenMeasuredFastest) {
  auto planner = StaticAttentionPlanner::create(
      {{1, 0.25, 0.20},
       {2, 0.30, 0.25},
       {4, 0.40, 0.35},
       {8, 0.55, 0.50},
       {16, 0.75, 0.70},
       {32, 1.00, 0.95},
       {64, 1.35, 1.30},
       {128, 1.60, 1.55},
       {256, 3.00, 2.90}});
  ASSERT_TRUE(planner.has_value());

  auto plan = planner->plan(129);
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->widths, (std::vector<size_t>{128, 1}));
  EXPECT_EQ(plan->coverage(), size_t{129});
  EXPECT_EQ(plan->padding(), size_t{0});
  EXPECT_DOUBLE_EQ(plan->predicted_cost, 1.80);
}

TEST(StaticAttentionPlannerTest, PlansWithoutACompiledContextLimit) {
  auto planner = make_planner();
  auto plan = planner.plan(100000);
  ASSERT_TRUE(plan.has_value());
  EXPECT_GE(plan->coverage(), size_t{100000});
  EXPECT_LT(plan->padding(), size_t{64});
  EXPECT_EQ(planner.planned_through(), size_t{100000});
  for (const size_t width : plan->widths) {
    EXPECT_NE(
        std::find(
            planner.graph_widths().begin(),
            planner.graph_widths().end(),
            width),
        planner.graph_widths().end());
  }
}

TEST(StaticAttentionPlannerTest, PlansOnlyVisibleCausalPrefix) {
  auto planner = StaticAttentionPlanner::create({{64, 1.0, 1.0}});
  ASSERT_TRUE(planner.has_value());

  auto first = planner->plan_causal_tile(256, 0, 32);
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(first->sequence_length, size_t{32});
  EXPECT_EQ(first->widths, std::vector<size_t>({64}));

  auto second = planner->plan_causal_tile(256, 64, 4);
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(second->sequence_length, size_t{68});
  EXPECT_EQ(second->widths, std::vector<size_t>({64, 64}));
  EXPECT_EQ(second->padding(), size_t{60});

  auto final_padded = planner->plan_causal_tile(256, 128, 4, 2);
  ASSERT_TRUE(final_padded.has_value());
  EXPECT_EQ(final_padded->sequence_length, size_t{130});
  EXPECT_EQ(final_padded->widths, std::vector<size_t>({64, 64, 64}));
}

TEST(StaticAttentionPlannerTest, PlansOnePersistentCausalPrefillLayout) {
  auto planner = StaticAttentionPlanner::create(
      {{4, 1.0, 1.0}, {8, 1.4, 1.4}, {16, 10.0, 10.0}});
  ASSERT_TRUE(planner.has_value());

  auto plan = planner->plan_prefixes({4, 8, 12});
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->sequence_length, size_t{12});
  EXPECT_EQ(plan->widths, (std::vector<size_t>{8, 4}));
  EXPECT_DOUBLE_EQ(plan->predicted_cost, 5.2);
}

TEST(StaticAttentionPlannerTest, LetsOnePaddedGraphCompeteForPrefill) {
  auto planner = StaticAttentionPlanner::create(
      {{4, 1.0, 1.0}, {8, 1.2, 1.2}, {16, 1.5, 1.5}});
  ASSERT_TRUE(planner.has_value());

  auto plan = planner->plan_prefixes({3, 7, 9});
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->widths, (std::vector<size_t>{16}));
  EXPECT_EQ(plan->padding(), size_t{7});
  EXPECT_DOUBLE_EQ(plan->predicted_cost, 4.5);
}

TEST(StaticAttentionPlannerTest, ChargesOnlyTilesThatReachAMergeBlock) {
  auto planner = StaticAttentionPlanner::create({{4, 1.0, 0.5}, {8, 3.0, 3.0}});
  ASSERT_TRUE(planner.has_value());

  auto plan = planner->plan_prefixes({2, 4, 6});
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->widths, (std::vector<size_t>{4, 4}));
  EXPECT_DOUBLE_EQ(plan->predicted_cost, 3.5);
}

TEST(
    StaticAttentionPlannerTest,
    PowerOfTwoDecodeAndPrefillPlansMatchBruteForceObjectives) {
  const std::vector<std::vector<StaticAttentionGraphCost>> profiles{
      {{2, 1.0, 0.5}, {4, 1.75, 1.0}, {8, 2.5, 2.0}},
      {{2, 2.0, 0.25}, {4, 1.25, 1.5}, {8, 3.0, 0.75}},
      {{2, 0.75, 1.5}, {4, 2.0, 0.5}, {8, 2.25, 1.25}},
  };
  const std::vector<std::vector<size_t>> prefix_sets{
      {1}, {1, 3}, {2, 5, 7}, {3, 4, 9}};

  for (const auto& profile : profiles) {
    auto planner = StaticAttentionPlanner::create(profile);
    ASSERT_TRUE(planner.has_value());
    for (size_t target = 1; target <= 10; ++target) {
      SCOPED_TRACE(testing::Message() << "Decode target=" << target);
      auto plan = planner->plan(target);
      ASSERT_TRUE(plan.has_value());
      const PlanObjective actual =
          decode_objective(profile, target, plan->widths);
      const PlanObjective expected = brute_force_layouts(
          profile, target, [&](const std::vector<size_t>& widths) {
            return decode_objective(profile, target, widths);
          });
      ASSERT_TRUE(actual.valid);
      ASSERT_TRUE(expected.valid);
      EXPECT_DOUBLE_EQ(actual.cost, expected.cost);
      EXPECT_DOUBLE_EQ(plan->predicted_cost, expected.cost);
      EXPECT_EQ(actual.padding, expected.padding);
      EXPECT_EQ(actual.calls, expected.calls);
    }

    for (const auto& prefixes : prefix_sets) {
      SCOPED_TRACE(testing::Message() << "Prefill target=" << prefixes.back());
      auto plan = planner->plan_prefixes(prefixes);
      ASSERT_TRUE(plan.has_value());
      const PlanObjective actual =
          prefill_objective(profile, prefixes, plan->widths);
      const PlanObjective expected = brute_force_layouts(
          profile, prefixes.back(), [&](const std::vector<size_t>& widths) {
            return prefill_objective(profile, prefixes, widths);
          });
      ASSERT_TRUE(actual.valid);
      ASSERT_TRUE(expected.valid);
      EXPECT_DOUBLE_EQ(actual.cost, expected.cost);
      EXPECT_DOUBLE_EQ(plan->predicted_cost, expected.cost);
      EXPECT_EQ(actual.padding, expected.padding);
      EXPECT_EQ(actual.calls, expected.calls);
      EXPECT_EQ(actual.blocks, expected.blocks);
      EXPECT_EQ(actual.coverage, expected.coverage);
    }
  }
}

TEST(StaticAttentionPlannerTest, RejectsInvalidCausalPrefillPrefixes) {
  auto planner = make_planner();
  EXPECT_FALSE(planner.plan_prefixes({}).has_value());
  EXPECT_FALSE(planner.plan_prefixes({0, 64}).has_value());
  EXPECT_FALSE(
      planner.plan_prefixes({std::numeric_limits<size_t>::max()}).has_value());
}

TEST(StaticAttentionPlannerTest, CausalStaircaseSkipsUpperRightBlocks) {
  const std::vector<size_t> block_widths{4, 4, 4};
  const std::vector<size_t> prefixes{4, 8, 10};
  const std::vector<size_t> expected_calls{1, 2, 3};
  for (size_t tile = 0; tile < prefixes.size(); ++tile) {
    size_t block_begin = 0;
    size_t calls = 0;
    for (const size_t width : block_widths) {
      if (causal_block_visible_columns(block_begin, width, prefixes[tile]) ==
          0) {
        break;
      }
      ++calls;
      block_begin += width;
    }
    EXPECT_EQ(calls, expected_calls[tile]);
  }
}

TEST(StaticAttentionPlannerTest, CausalBoundaryKeepsOnlyVisibleColumns) {
  auto first_row_end = causal_visible_prefix_end(4, 1);
  auto second_row_end = causal_visible_prefix_end(4, 2);
  ASSERT_TRUE(first_row_end.has_value());
  ASSERT_TRUE(second_row_end.has_value());

  EXPECT_EQ(causal_block_visible_columns(4, 4, *first_row_end), size_t{1});
  EXPECT_EQ(causal_block_visible_columns(4, 4, *second_row_end), size_t{2});
  EXPECT_EQ(causal_block_visible_columns(8, 4, *second_row_end), size_t{0});
  EXPECT_FALSE(causal_visible_prefix_end(std::numeric_limits<size_t>::max(), 1)
                   .has_value());
}

TEST(StaticAttentionPlannerTest, PlansFortyKWithThirteenWidthsWithoutOverflow) {
  std::vector<StaticAttentionGraphCost> graph_costs;
  std::vector<size_t> expected_widths;
  graph_costs.reserve(13);
  expected_widths.reserve(13);
  for (size_t width = 1; width <= 4096; width *= 2) {
    const double merge_cost = 0.20 + static_cast<double>(width) / 10000.0;
    graph_costs.push_back({width, merge_cost + 0.05, merge_cost});
    expected_widths.push_back(width);
  }
  ASSERT_EQ(graph_costs.size(), size_t{13});
  auto planner = StaticAttentionPlanner::create(graph_costs);
  ASSERT_TRUE(planner.has_value());
  EXPECT_EQ(planner->graph_widths(), expected_widths);

  std::vector<size_t> visible_prefixes;
  for (size_t prefix = 128; prefix < 40000; prefix += 128) {
    visible_prefixes.push_back(prefix);
  }
  visible_prefixes.push_back(40000);
  ASSERT_EQ(visible_prefixes.size(), size_t{313});
  EXPECT_EQ(visible_prefixes.front(), size_t{128});
  EXPECT_EQ(visible_prefixes.back(), size_t{40000});
  auto plan = planner->plan_prefixes(visible_prefixes);
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->sequence_length, size_t{40000});
  EXPECT_GE(plan->coverage(), size_t{40000});

  auto stats = causal_prefill_plan_stats(plan->widths, visible_prefixes);
  ASSERT_TRUE(stats.has_value());
  EXPECT_EQ(stats->physical_blocks, plan->graph_calls());
  EXPECT_EQ(stats->tiles.size(), visible_prefixes.size());
  EXPECT_EQ(
      stats->full_graph_calls,
      stats->physical_blocks * visible_prefixes.size());
  EXPECT_EQ(
      stats->visible_graph_calls + stats->skipped_graph_calls,
      stats->full_graph_calls);

  double actual_execution_cost = 0.0;
  size_t actual_graph_calls = 0;
  size_t actual_first_calls = 0;
  size_t actual_merge_calls = 0;
  size_t actual_boundary_padding = 0;
  for (size_t tile = 0; tile < visible_prefixes.size(); ++tile) {
    size_t coverage = 0;
    size_t tile_calls = 0;
    for (const size_t width : plan->widths) {
      if (coverage >= visible_prefixes[tile]) {
        break;
      }
      const auto cost = std::find_if(
          graph_costs.begin(), graph_costs.end(), [width](const auto& entry) {
            return entry.width == width;
          });
      ASSERT_NE(cost, graph_costs.end());
      actual_execution_cost +=
          tile_calls == 0 ? cost->first_cost : cost->merge_cost;
      if (tile_calls == 0) {
        ++actual_first_calls;
      } else {
        ++actual_merge_calls;
      }
      coverage += width;
      ++tile_calls;
    }
    ASSERT_GE(coverage, visible_prefixes[tile]);
    EXPECT_EQ(tile_calls, stats->tiles[tile].visible_graph_calls);
    EXPECT_EQ(
        coverage - visible_prefixes[tile], stats->tiles[tile].boundary_padding);
    actual_graph_calls += tile_calls;
    actual_boundary_padding += coverage - visible_prefixes[tile];
  }
  EXPECT_EQ(actual_graph_calls, stats->visible_graph_calls);
  EXPECT_EQ(actual_first_calls, visible_prefixes.size());
  EXPECT_EQ(
      actual_merge_calls, stats->visible_graph_calls - visible_prefixes.size());
  EXPECT_EQ(actual_boundary_padding, stats->aggregate_boundary_padding);
  EXPECT_NEAR(actual_execution_cost, plan->predicted_cost, 1.0e-9);
}

TEST(StaticAttentionPlannerTest, RejectsInvalidCausalTileMetadata) {
  auto planner = make_planner();
  EXPECT_FALSE(planner.plan_causal_tile(128, 0, 0).has_value());
  EXPECT_FALSE(planner.plan_causal_tile(128, 127, 4).has_value());
  EXPECT_FALSE(planner.plan_causal_tile(128, 0, 4, 5).has_value());
}

TEST(StaticAttentionPlannerTest, RejectsInvalidProfilesAndLengths) {
  EXPECT_FALSE(StaticAttentionPlanner::create({}).has_value());
  EXPECT_FALSE(StaticAttentionPlanner::create({{64, 1.0, 1.0}, {64, 2.0, 2.0}})
                   .has_value());
  EXPECT_FALSE(StaticAttentionPlanner::create(
                   {{64, std::numeric_limits<double>::infinity(), 1.0}})
                   .has_value());
  auto arbitrary_width = StaticAttentionPlanner::create({{3, 1.0, 1.0}});
  ASSERT_TRUE(arbitrary_width.has_value());
  auto arbitrary_plan = arbitrary_width->plan(5);
  ASSERT_TRUE(arbitrary_plan.has_value());
  EXPECT_EQ(arbitrary_plan->widths, (std::vector<size_t>{3, 3}));
  auto planner = make_planner();
  EXPECT_FALSE(planner.plan(0).has_value());
  EXPECT_FALSE(planner.plan(std::numeric_limits<size_t>::max()).has_value());
}

TEST(StaticAttentionPrefillPlannerTest, Plans129As128Plus1WhenMeasuredFastest) {
  std::vector<StaticAttentionShapeCost> costs;
  for (const size_t rows : {size_t{1}, size_t{128}, size_t{256}}) {
    for (const size_t width : {size_t{1}, size_t{128}, size_t{256}}) {
      double cost = 1000.0;
      if ((rows == 128 && width == 128) ||
          (rows == 1 && (width == 1 || width == 128))) {
        cost = 1.0;
      }
      costs.push_back({rows, width, cost, cost});
    }
  }
  auto planner = StaticAttentionPrefillPlanner::create(std::move(costs));
  ASSERT_TRUE(planner.has_value());

  auto plan = planner->plan(129, 0, true);
  ASSERT_TRUE(plan.has_value());
  ASSERT_EQ(plan->query_tiles.size(), size_t{2});
  EXPECT_EQ(plan->query_tiles[0].row_begin, size_t{0});
  EXPECT_EQ(plan->query_tiles[0].valid_query_rows, size_t{128});
  EXPECT_EQ(plan->query_tiles[0].graph_query_rows, size_t{128});
  EXPECT_EQ(plan->query_tiles[0].visible_prefix, size_t{128});
  EXPECT_EQ(
      plan->query_tiles[0].key_plan.widths, (std::vector<size_t>{128}));
  EXPECT_EQ(plan->query_tiles[1].row_begin, size_t{128});
  EXPECT_EQ(plan->query_tiles[1].valid_query_rows, size_t{1});
  EXPECT_EQ(plan->query_tiles[1].graph_query_rows, size_t{1});
  EXPECT_EQ(plan->query_tiles[1].visible_prefix, size_t{129});
  EXPECT_EQ(
      plan->query_tiles[1].key_plan.widths,
      (std::vector<size_t>{128, 1}));
  EXPECT_EQ(plan->query_padding_rows, size_t{0});

  auto persistent = planner->plan_persistent(129, 0, true);
  ASSERT_TRUE(persistent.has_value());
  ASSERT_EQ(persistent->query_tiles.size(), size_t{2});
  EXPECT_EQ(persistent->query_tiles[0].graph_query_rows, size_t{128});
  EXPECT_EQ(persistent->query_tiles[1].graph_query_rows, size_t{1});
  EXPECT_TRUE(persistent->persistent_widths.empty());
  EXPECT_EQ(
      persistent->query_tiles[1].key_plan.widths,
      (std::vector<size_t>{128, 1}));
}

TEST(StaticAttentionPrefillPlannerTest, AllowsPaddingOnlyWhenMeasuredFaster) {
  auto planner = StaticAttentionPrefillPlanner::create(
      {{1, 1, 4.0, 4.0},
       {1, 2, 7.0, 7.0},
       {4, 1, 4.0, 4.0},
       {4, 2, 7.0, 7.0},
       {8, 1, 4.0, 4.0},
       {8, 2, 0.5, 0.5}});
  ASSERT_TRUE(planner.has_value());

  auto padded = planner->plan(6, 0, true);
  ASSERT_TRUE(padded.has_value());
  ASSERT_EQ(padded->query_tiles.size(), size_t{1});
  EXPECT_EQ(padded->query_tiles[0].valid_query_rows, size_t{6});
  EXPECT_EQ(padded->query_tiles[0].graph_query_rows, size_t{8});
  EXPECT_EQ(padded->query_tiles[0].key_plan.widths,
            (std::vector<size_t>{2, 2, 2}));
  EXPECT_EQ(padded->query_padding_rows, size_t{2});

  auto exact = planner->plan(6, 0, false);
  ASSERT_TRUE(exact.has_value());
  EXPECT_EQ(exact->query_padding_rows, size_t{0});
  EXPECT_EQ(exact->query_tiles.size(), size_t{3});
}

TEST(StaticAttentionPrefillPlannerTest, CompatibilityAliasKeepsTileOptima) {
  auto exact = StaticAttentionPrefillPlanner::create(
      {{1, 1, 0.1, 0.1},
       {1, 128, 1.0, 1.0},
       {1, 256, 4.0, 4.0},
       {128, 1, 0.1, 0.1},
       {128, 128, 1.0, 1.0},
       {128, 256, 4.0, 4.0}});
  ASSERT_TRUE(exact.has_value());
  auto exact_plan = exact->plan_persistent(129);
  ASSERT_TRUE(exact_plan.has_value());
  EXPECT_TRUE(exact_plan->persistent_widths.empty());
  ASSERT_EQ(exact_plan->query_tiles.size(), size_t{2});
  EXPECT_EQ(
      exact_plan->query_tiles.back().key_plan.widths,
      (std::vector<size_t>{128, 1}));

  auto padded = StaticAttentionPrefillPlanner::create(
      {{1, 1, 2.0, 2.0},
       {1, 128, 2.0, 2.0},
       {1, 256, 0.1, 0.1},
       {128, 1, 2.0, 2.0},
       {128, 128, 2.0, 2.0},
       {128, 256, 0.1, 0.1}});
  ASSERT_TRUE(padded.has_value());
  auto padded_plan = padded->plan_persistent(129);
  ASSERT_TRUE(padded_plan.has_value());
  EXPECT_TRUE(padded_plan->persistent_widths.empty());
  ASSERT_EQ(padded_plan->query_tiles.size(), size_t{2});
  EXPECT_EQ(
      padded_plan->query_tiles.back().key_plan.widths,
      (std::vector<size_t>{256}));
}

TEST(StaticAttentionPrefillPlannerTest, RejectsDuplicateAndInvalidShapes) {
  EXPECT_FALSE(StaticAttentionPrefillPlanner::create({}).has_value());
  EXPECT_FALSE(
      StaticAttentionPrefillPlanner::create(
          {{1, 1, 1.0, 1.0}, {1, 1, 2.0, 2.0}})
          .has_value());
  EXPECT_FALSE(
      StaticAttentionPrefillPlanner::create({{0, 1, 1.0, 1.0}}).has_value());
  auto planner =
      StaticAttentionPrefillPlanner::create({{1, 1, 1.0, 1.0}});
  ASSERT_TRUE(planner.has_value());
  EXPECT_FALSE(planner->plan(0).has_value());
}

TEST(StaticAttentionPlannerTest, CoverageDoesNotWrap) {
  StaticAttentionPlan malformed{
      1, {std::numeric_limits<size_t>::max(), 1}, 1.0};
  EXPECT_EQ(malformed.coverage(), std::numeric_limits<size_t>::max());
}

} // namespace
