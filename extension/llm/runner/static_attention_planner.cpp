/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/runner/static_attention_planner.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <utility>

#include <c10/util/safe_numerics.h>

namespace executorch {
namespace extension {
namespace llm {
namespace {

struct Candidate {
  double cost;
  size_t coverage;
  size_t calls;
  size_t width;
  size_t remaining;
};

struct PrefixState {
  bool valid = false;
  double cost = 0.0;
  size_t padding = 0;
  size_t calls = 0;
  size_t blocks = 0;
  size_t first_width = 0;
  size_t previous = 0;
  size_t width = 0;
  size_t coverage = 0;
};

struct QueryState {
  bool valid = false;
  double cost = 0.0;
  size_t calls = 0;
  size_t tiles = 0;
  size_t computed = 0;
  size_t valid_elements = 0;
  size_t query_padding = 0;
  size_t previous = 0;
  size_t valid_rows = 0;
  size_t graph_rows = 0;
  StaticAttentionPlan key_plan;
};

bool is_better(
    const Candidate& candidate,
    const Candidate& current,
    size_t required) {
  if (candidate.cost != current.cost) {
    return candidate.cost < current.cost;
  }
  const size_t candidate_padding = candidate.coverage - required;
  const size_t current_padding = current.coverage - required;
  if (candidate_padding != current_padding) {
    return candidate_padding < current_padding;
  }
  if (candidate.calls != current.calls) {
    return candidate.calls < current.calls;
  }
  return candidate.width > current.width;
}

std::optional<Candidate> make_candidate(
    double graph_cost,
    size_t width,
    size_t remaining,
    double remaining_cost,
    size_t remaining_coverage,
    size_t remaining_calls) {
  const double cost = graph_cost + remaining_cost;
  size_t coverage = 0;
  size_t calls = 0;
  if (!std::isfinite(cost) ||
      c10::add_overflows(width, remaining_coverage, &coverage) ||
      c10::add_overflows(size_t{1}, remaining_calls, &calls)) {
    return std::nullopt;
  }
  return Candidate{cost, coverage, calls, width, remaining};
}

bool is_better_prefix_state(
    const PrefixState& candidate,
    const PrefixState& current) {
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
  if (candidate.first_width != current.first_width) {
    return candidate.first_width > current.first_width;
  }
  if (candidate.coverage != current.coverage) {
    return candidate.coverage < current.coverage;
  }
  return candidate.width > current.width;
}

bool is_better_query_state(
    const QueryState& candidate,
    const QueryState& current) {
  if (!current.valid || candidate.cost != current.cost) {
    return !current.valid || candidate.cost < current.cost;
  }
  if (candidate.calls != current.calls) {
    return candidate.calls < current.calls;
  }
  if (candidate.tiles != current.tiles) {
    return candidate.tiles < current.tiles;
  }
  const size_t candidate_waste =
      candidate.computed - candidate.valid_elements;
  const size_t current_waste = current.computed - current.valid_elements;
  if (candidate_waste != current_waste) {
    return candidate_waste < current_waste;
  }
  if (candidate.query_padding != current.query_padding) {
    return candidate.query_padding < current.query_padding;
  }
  if (candidate.valid_rows != current.valid_rows) {
    return candidate.valid_rows > current.valid_rows;
  }
  return candidate.graph_rows > current.graph_rows;
}

std::optional<size_t> causal_valid_elements(
    size_t history_length,
    size_t begin,
    size_t end) {
  if (begin >= end) {
    return std::nullopt;
  }
  size_t first = 0;
  size_t last = 0;
  size_t count = end - begin;
  size_t endpoints = 0;
  size_t product = 0;
  if (c10::add_overflows(history_length, begin, &first) ||
      c10::add_overflows(first, size_t{1}, &first) ||
      c10::add_overflows(history_length, end, &last) ||
      c10::add_overflows(first, last, &endpoints) ||
      c10::mul_overflows(count, endpoints, &product)) {
    return std::nullopt;
  }
  return product / 2;
}

} // namespace

size_t StaticAttentionPlan::coverage() const {
  size_t result = 0;
  for (const size_t width : widths) {
    size_t next = 0;
    if (c10::add_overflows(result, width, &next)) {
      return std::numeric_limits<size_t>::max();
    }
    result = next;
  }
  return result;
}

size_t StaticAttentionPlan::padding() const {
  const size_t covered = coverage();
  return covered >= sequence_length ? covered - sequence_length : 0;
}

size_t StaticAttentionPlan::graph_calls() const {
  return widths.size();
}

size_t StaticAttentionPrefillPlan::wasted_score_elements() const {
  return computed_score_elements >= valid_score_elements
      ? computed_score_elements - valid_score_elements
      : 0;
}

std::optional<StaticAttentionPlanner> StaticAttentionPlanner::create(
    std::vector<StaticAttentionGraphCost> graph_costs) {
  if (graph_costs.empty()) {
    return std::nullopt;
  }
  std::sort(
      graph_costs.begin(),
      graph_costs.end(),
      [](const auto& lhs, const auto& rhs) { return lhs.width < rhs.width; });
  size_t previous_width = 0;
  for (const auto& graph_cost : graph_costs) {
    if (graph_cost.width == 0 || graph_cost.width == previous_width ||
        !std::isfinite(graph_cost.first_cost) || graph_cost.first_cost <= 0.0 ||
        !std::isfinite(graph_cost.merge_cost) || graph_cost.merge_cost <= 0.0) {
      return std::nullopt;
    }
    previous_width = graph_cost.width;
  }
  return StaticAttentionPlanner(std::move(graph_costs));
}

StaticAttentionPlanner::StaticAttentionPlanner(
    std::vector<StaticAttentionGraphCost> graph_costs)
    : graph_costs_(std::move(graph_costs)),
      merge_cost_(1, 0.0),
      merge_coverage_(1, 0),
      merge_calls_(1, 0),
      merge_choice_(1, 0) {
  graph_widths_.reserve(graph_costs_.size());
  for (const auto& graph_cost : graph_costs_) {
    graph_widths_.push_back(graph_cost.width);
  }
}

std::optional<StaticAttentionPlan> StaticAttentionPlanner::plan(
    size_t sequence_length) {
  if (sequence_length == 0) {
    return std::nullopt;
  }
  if (!extend_merge_plans(sequence_length)) {
    return std::nullopt;
  }

  std::optional<Candidate> best;
  for (const auto& graph_cost : graph_costs_) {
    const size_t remaining = sequence_length > graph_cost.width
        ? sequence_length - graph_cost.width
        : 0;
    auto candidate = make_candidate(
        graph_cost.first_cost,
        graph_cost.width,
        remaining,
        merge_cost_[remaining],
        merge_coverage_[remaining],
        merge_calls_[remaining]);
    if (candidate && (!best || is_better(*candidate, *best, sequence_length))) {
      best = *candidate;
    }
  }
  if (!best) {
    return std::nullopt;
  }

  StaticAttentionPlan result{sequence_length, {best->width}, best->cost};
  size_t remaining = best->remaining;
  while (remaining > 0) {
    const size_t width = merge_choice_[remaining];
    if (width == 0) {
      return std::nullopt;
    }
    result.widths.push_back(width);
    remaining = remaining > width ? remaining - width : 0;
  }
  return result;
}

std::optional<StaticAttentionPlan> StaticAttentionPlanner::plan_causal_tile(
    size_t full_sequence_length,
    size_t causal_query_begin,
    size_t query_rows,
    size_t valid_query_rows) {
  if (full_sequence_length == 0 || query_rows == 0) {
    return std::nullopt;
  }
  const size_t valid_rows =
      valid_query_rows == 0 ? query_rows : valid_query_rows;
  size_t visible_prefix_length = 0;
  if (valid_rows == 0 || valid_rows > query_rows ||
      c10::add_overflows(
          causal_query_begin, valid_rows, &visible_prefix_length) ||
      visible_prefix_length > full_sequence_length) {
    return std::nullopt;
  }
  return plan(visible_prefix_length);
}

std::optional<StaticAttentionPlan> StaticAttentionPlanner::plan_prefixes(
    const std::vector<size_t>& visible_prefixes) const {
  if (visible_prefixes.empty() ||
      std::any_of(
          visible_prefixes.begin(), visible_prefixes.end(), [](size_t prefix) {
            return prefix == 0;
          })) {
    return std::nullopt;
  }

  std::vector<size_t> prefixes = visible_prefixes;
  std::sort(prefixes.begin(), prefixes.end());
  const size_t target = prefixes.back();
  const size_t max_width = graph_costs_.back().width;
  size_t state_count = 0;
  if (c10::add_overflows(target, max_width, &state_count) ||
      state_count > std::vector<PrefixState>().max_size()) {
    return std::nullopt;
  }

  std::vector<size_t> prefix_sum(prefixes.size() + 1, 0);
  for (size_t index = 0; index < prefixes.size(); ++index) {
    if (c10::add_overflows(
            prefix_sum[index], prefixes[index], &prefix_sum[index + 1])) {
      return std::nullopt;
    }
  }

  std::vector<PrefixState> states(state_count);
  states[0].valid = true;
  PrefixState best_terminal;
  for (size_t coverage = 0; coverage < target; ++coverage) {
    const PrefixState& state = states[coverage];
    if (!state.valid) {
      continue;
    }
    const size_t first_visible = static_cast<size_t>(
        std::upper_bound(prefixes.begin(), prefixes.end(), coverage) -
        prefixes.begin());
    const size_t executing_tiles = prefixes.size() - first_visible;
    if (executing_tiles == 0) {
      continue;
    }
    for (const auto& graph_cost : graph_costs_) {
      size_t next_coverage = 0;
      if (c10::add_overflows(coverage, graph_cost.width, &next_coverage) ||
          next_coverage >= state_count) {
        return std::nullopt;
      }
      const double invocation_cost =
          (coverage == 0 ? graph_cost.first_cost : graph_cost.merge_cost) *
          static_cast<double>(executing_tiles);
      const double cost = state.cost + invocation_cost;
      if (!std::isfinite(cost)) {
        continue;
      }

      const size_t resolved_end = static_cast<size_t>(
          std::upper_bound(prefixes.begin(), prefixes.end(), next_coverage) -
          prefixes.begin());
      const size_t resolved_count = resolved_end - first_visible;
      size_t covered_positions = 0;
      size_t padding_increment = 0;
      size_t padding = 0;
      size_t calls = 0;
      size_t blocks = 0;
      if (c10::mul_overflows(
              resolved_count, next_coverage, &covered_positions) ||
          covered_positions <
              prefix_sum[resolved_end] - prefix_sum[first_visible] ||
          (padding_increment = covered_positions -
               (prefix_sum[resolved_end] - prefix_sum[first_visible]),
           c10::add_overflows(state.padding, padding_increment, &padding)) ||
          c10::add_overflows(state.calls, executing_tiles, &calls) ||
          c10::add_overflows(state.blocks, size_t{1}, &blocks)) {
        continue;
      }

      PrefixState candidate{
          true,
          cost,
          padding,
          calls,
          blocks,
          coverage == 0 ? graph_cost.width : state.first_width,
          coverage,
          graph_cost.width,
          next_coverage};
      if (next_coverage >= target) {
        if (is_better_prefix_state(candidate, best_terminal)) {
          best_terminal = candidate;
        }
      } else if (is_better_prefix_state(candidate, states[next_coverage])) {
        states[next_coverage] = candidate;
      }
    }
  }
  if (!best_terminal.valid) {
    return std::nullopt;
  }

  std::vector<size_t> widths{best_terminal.width};
  size_t coverage = best_terminal.previous;
  while (coverage > 0) {
    const PrefixState& state = states[coverage];
    if (!state.valid || state.width == 0 || state.previous >= coverage) {
      return std::nullopt;
    }
    widths.push_back(state.width);
    coverage = state.previous;
  }
  std::reverse(widths.begin(), widths.end());
  return StaticAttentionPlan{target, std::move(widths), best_terminal.cost};
}

bool StaticAttentionPlanner::extend_merge_plans(size_t required_length) {
  const size_t max_entries = std::min(
      {merge_cost_.max_size(),
       merge_coverage_.max_size(),
       merge_calls_.max_size(),
       merge_choice_.max_size()});
  if (required_length >= max_entries) {
    return false;
  }
  while (merge_cost_.size() <= required_length) {
    const size_t required = merge_cost_.size();
    std::optional<Candidate> best;
    for (const auto& graph_cost : graph_costs_) {
      const size_t remaining =
          required > graph_cost.width ? required - graph_cost.width : 0;
      auto candidate = make_candidate(
          graph_cost.merge_cost,
          graph_cost.width,
          remaining,
          merge_cost_[remaining],
          merge_coverage_[remaining],
          merge_calls_[remaining]);
      if (candidate && (!best || is_better(*candidate, *best, required))) {
        best = *candidate;
      }
    }
    if (!best) {
      return false;
    }
    merge_cost_.push_back(best->cost);
    merge_coverage_.push_back(best->coverage);
    merge_calls_.push_back(best->calls);
    merge_choice_.push_back(best->width);
  }
  return true;
}

std::optional<StaticAttentionPrefillPlanner>
StaticAttentionPrefillPlanner::create(
    std::vector<StaticAttentionShapeCost> shape_costs) {
  if (shape_costs.empty()) {
    return std::nullopt;
  }
  std::sort(
      shape_costs.begin(),
      shape_costs.end(),
      [](const auto& lhs, const auto& rhs) {
        return std::pair(lhs.query_rows, lhs.width) <
            std::pair(rhs.query_rows, rhs.width);
      });

  std::vector<RowPlanner> row_planners;
  size_t index = 0;
  while (index < shape_costs.size()) {
    const size_t query_rows = shape_costs[index].query_rows;
    if (query_rows == 0) {
      return std::nullopt;
    }
    std::vector<StaticAttentionGraphCost> graph_costs;
    size_t previous_width = 0;
    while (index < shape_costs.size() &&
           shape_costs[index].query_rows == query_rows) {
      const auto& shape = shape_costs[index++];
      if (shape.width == 0 || shape.width == previous_width) {
        return std::nullopt;
      }
      previous_width = shape.width;
      graph_costs.push_back(
          {shape.width, shape.first_cost, shape.merge_cost});
    }
    auto key_planner = StaticAttentionPlanner::create(graph_costs);
    if (!key_planner.has_value()) {
      return std::nullopt;
    }
    row_planners.push_back(
        {query_rows, std::move(graph_costs), std::move(*key_planner)});
  }
  return StaticAttentionPrefillPlanner(std::move(row_planners));
}

StaticAttentionPrefillPlanner::StaticAttentionPrefillPlanner(
    std::vector<RowPlanner> planners)
    : row_planners_(std::move(planners)) {
  query_rows_.reserve(row_planners_.size());
  for (const auto& planner : row_planners_) {
    query_rows_.push_back(planner.query_rows);
  }
}

std::optional<StaticAttentionPrefillPlan> StaticAttentionPrefillPlanner::plan(
    size_t sequence_length,
    size_t history_length,
    bool allow_query_padding) {
  size_t maximum_prefix = 0;
  size_t state_count = 0;
  if (sequence_length == 0 || row_planners_.empty() ||
      c10::add_overflows(history_length, sequence_length, &maximum_prefix) ||
      c10::add_overflows(sequence_length, size_t{1}, &state_count) ||
      state_count > std::vector<QueryState>().max_size()) {
    return std::nullopt;
  }

  std::vector<QueryState> states(state_count);
  states[0].valid = true;
  for (size_t end = 1; end <= sequence_length; ++end) {
    size_t visible_prefix = 0;
    if (c10::add_overflows(history_length, end, &visible_prefix)) {
      return std::nullopt;
    }
    QueryState best;
    for (auto& row_planner : row_planners_) {
      const size_t graph_rows = row_planner.query_rows;
      if (end < graph_rows) {
        continue;
      }
      const size_t begin = end - graph_rows;
      if (!states[begin].valid) {
        continue;
      }
      auto key_plan = row_planner.key_planner.plan(visible_prefix);
      if (!key_plan.has_value()) {
        continue;
      }
      const QueryState& previous = states[begin];
      const auto valid_elements =
          causal_valid_elements(history_length, begin, end);
      size_t calls = 0;
      size_t tiles = 0;
      size_t tile_computed = 0;
      size_t computed = 0;
      size_t total_valid = 0;
      size_t query_padding = 0;
      if (!valid_elements.has_value() ||
          c10::add_overflows(
              previous.calls, key_plan->graph_calls(), &calls) ||
          c10::add_overflows(previous.tiles, size_t{1}, &tiles) ||
          c10::mul_overflows(
              graph_rows, key_plan->coverage(), &tile_computed) ||
          c10::add_overflows(previous.computed, tile_computed, &computed) ||
          c10::add_overflows(
              previous.valid_elements, *valid_elements, &total_valid) ||
          c10::add_overflows(
              previous.query_padding, size_t{0}, &query_padding)) {
        continue;
      }
      const double cost = previous.cost + key_plan->predicted_cost;
      if (!std::isfinite(cost)) {
        continue;
      }
      QueryState candidate{
          true,
          cost,
          calls,
          tiles,
          computed,
          total_valid,
          query_padding,
          begin,
          graph_rows,
          graph_rows,
          std::move(*key_plan)};
      if (is_better_query_state(candidate, best)) {
        best = std::move(candidate);
      }
    }
    if (best.valid) {
      states[end] = std::move(best);
    }
  }

  QueryState terminal = states[sequence_length];
  if (allow_query_padding) {
    const size_t end = sequence_length;
    size_t visible_prefix = 0;
    if (c10::add_overflows(history_length, end, &visible_prefix)) {
      return std::nullopt;
    }
    for (auto& row_planner : row_planners_) {
      const size_t graph_rows = row_planner.query_rows;
      const size_t minimum_begin = end > graph_rows ? end - graph_rows : 0;
      for (size_t begin = minimum_begin; begin < end; ++begin) {
        if (!states[begin].valid) {
          continue;
        }
        const size_t valid_rows = end - begin;
        if (valid_rows == graph_rows && states[end].valid) {
          continue;
        }
        auto key_plan = row_planner.key_planner.plan(visible_prefix);
        const auto valid_elements =
            causal_valid_elements(history_length, begin, end);
        if (!key_plan.has_value() || !valid_elements.has_value()) {
          continue;
        }
        const QueryState& previous = states[begin];
        size_t calls = 0;
        size_t tiles = 0;
        size_t tile_computed = 0;
        size_t computed = 0;
        size_t total_valid = 0;
        size_t query_padding = 0;
        if (c10::add_overflows(
                previous.calls, key_plan->graph_calls(), &calls) ||
            c10::add_overflows(previous.tiles, size_t{1}, &tiles) ||
            c10::mul_overflows(
                graph_rows, key_plan->coverage(), &tile_computed) ||
            c10::add_overflows(previous.computed, tile_computed, &computed) ||
            c10::add_overflows(
                previous.valid_elements, *valid_elements, &total_valid) ||
            c10::add_overflows(
                previous.query_padding,
                graph_rows - valid_rows,
                &query_padding)) {
          continue;
        }
        const double cost = previous.cost + key_plan->predicted_cost;
        QueryState candidate{
            true,
            cost,
            calls,
            tiles,
            computed,
            total_valid,
            query_padding,
            begin,
            valid_rows,
            graph_rows,
            std::move(*key_plan)};
        if (std::isfinite(cost) && is_better_query_state(candidate, terminal)) {
          terminal = std::move(candidate);
        }
      }
    }
  }
  if (!terminal.valid) {
    return std::nullopt;
  }
  states[sequence_length] = std::move(terminal);

  const QueryState& final_state = states[sequence_length];
  std::vector<StaticAttentionQueryTilePlan> reverse_tiles;
  reverse_tiles.reserve(final_state.tiles);
  size_t end = sequence_length;
  while (end > 0) {
    const QueryState& state = states[end];
    if (!state.valid || state.previous >= end || state.valid_rows == 0 ||
        state.graph_rows == 0) {
      return std::nullopt;
    }
    size_t visible_prefix = 0;
    if (c10::add_overflows(history_length, end, &visible_prefix)) {
      return std::nullopt;
    }
    reverse_tiles.push_back(
        {state.previous,
         state.valid_rows,
         state.graph_rows,
         visible_prefix,
         state.key_plan});
    end = state.previous;
  }
  std::reverse(reverse_tiles.begin(), reverse_tiles.end());
  return StaticAttentionPrefillPlan{
      sequence_length,
      history_length,
      std::move(reverse_tiles),
      final_state.cost,
      final_state.calls,
      final_state.computed,
      final_state.valid_elements,
      final_state.query_padding,
      {}};
}

std::optional<StaticAttentionPlan>
StaticAttentionPrefillPlanner::persistent_layout_for_tiles(
    const std::vector<StaticAttentionQueryTilePlan>& tiles) const {
  if (tiles.empty()) {
    return std::nullopt;
  }
  std::vector<std::vector<size_t>> prefixes(row_planners_.size());
  size_t target = 0;
  for (const auto& tile : tiles) {
    const auto row = std::lower_bound(
        row_planners_.begin(),
        row_planners_.end(),
        tile.graph_query_rows,
        [](const RowPlanner& planner, size_t rows) {
          return planner.query_rows < rows;
        });
    if (tile.visible_prefix == 0 || row == row_planners_.end() ||
        row->query_rows != tile.graph_query_rows) {
      return std::nullopt;
    }
    const size_t index = static_cast<size_t>(row - row_planners_.begin());
    prefixes[index].push_back(tile.visible_prefix);
    target = std::max(target, tile.visible_prefix);
  }
  for (auto& values : prefixes) {
    std::sort(values.begin(), values.end());
  }

  std::vector<size_t> candidate_widths;
  for (const auto& row : row_planners_) {
    for (const auto& graph : row.graph_costs) {
      candidate_widths.push_back(graph.width);
    }
  }
  std::sort(candidate_widths.begin(), candidate_widths.end());
  candidate_widths.erase(
      std::unique(candidate_widths.begin(), candidate_widths.end()),
      candidate_widths.end());
  if (candidate_widths.empty()) {
    return std::nullopt;
  }

  size_t state_count = 0;
  if (c10::add_overflows(target, candidate_widths.back(), &state_count) ||
      state_count > std::vector<PrefixState>().max_size()) {
    return std::nullopt;
  }
  std::vector<PrefixState> states(state_count);
  states[0].valid = true;
  PrefixState best_terminal;
  for (size_t coverage = 0; coverage < target; ++coverage) {
    const PrefixState& state = states[coverage];
    if (!state.valid) {
      continue;
    }
    for (const size_t width : candidate_widths) {
      size_t next_coverage = 0;
      if (c10::add_overflows(coverage, width, &next_coverage) ||
          next_coverage >= state_count) {
        return std::nullopt;
      }
      double invocation_cost = 0.0;
      size_t executing_tiles = 0;
      size_t padding_increment = 0;
      bool supported = true;
      for (size_t row_index = 0; row_index < row_planners_.size(); ++row_index) {
        const auto& row_prefixes = prefixes[row_index];
        if (row_prefixes.empty()) {
          continue;
        }
        const auto first_active =
            std::upper_bound(row_prefixes.begin(), row_prefixes.end(), coverage);
        const size_t active =
            static_cast<size_t>(row_prefixes.end() - first_active);
        if (active == 0) {
          continue;
        }
        const auto& graph_costs = row_planners_[row_index].graph_costs;
        const auto graph = std::lower_bound(
            graph_costs.begin(),
            graph_costs.end(),
            width,
            [](const StaticAttentionGraphCost& cost, size_t candidate) {
              return cost.width < candidate;
            });
        if (graph == graph_costs.end() || graph->width != width) {
          supported = false;
          break;
        }
        invocation_cost += static_cast<double>(active) *
            (coverage == 0 ? graph->first_cost : graph->merge_cost);
        if (c10::add_overflows(executing_tiles, active, &executing_tiles)) {
          supported = false;
          break;
        }
        const auto resolved_end =
            std::upper_bound(first_active, row_prefixes.end(), next_coverage);
        for (auto prefix = first_active; prefix != resolved_end; ++prefix) {
          if (c10::add_overflows(
                  padding_increment, next_coverage - *prefix, &padding_increment)) {
            supported = false;
            break;
          }
        }
        if (!supported) {
          break;
        }
      }
      if (!supported || executing_tiles == 0) {
        continue;
      }
      size_t padding = 0;
      size_t calls = 0;
      size_t blocks = 0;
      const double cost = state.cost + invocation_cost;
      if (!std::isfinite(cost) ||
          c10::add_overflows(state.padding, padding_increment, &padding) ||
          c10::add_overflows(state.calls, executing_tiles, &calls) ||
          c10::add_overflows(state.blocks, size_t{1}, &blocks)) {
        continue;
      }
      PrefixState candidate{
          true,
          cost,
          padding,
          calls,
          blocks,
          coverage == 0 ? width : state.first_width,
          coverage,
          width,
          next_coverage};
      if (next_coverage >= target) {
        if (is_better_prefix_state(candidate, best_terminal)) {
          best_terminal = candidate;
        }
      } else if (is_better_prefix_state(candidate, states[next_coverage])) {
        states[next_coverage] = candidate;
      }
    }
  }
  if (!best_terminal.valid) {
    return std::nullopt;
  }
  std::vector<size_t> widths{best_terminal.width};
  size_t coverage = best_terminal.previous;
  while (coverage > 0) {
    const PrefixState& state = states[coverage];
    if (!state.valid || state.width == 0 || state.previous >= coverage) {
      return std::nullopt;
    }
    widths.push_back(state.width);
    coverage = state.previous;
  }
  std::reverse(widths.begin(), widths.end());
  return StaticAttentionPlan{target, std::move(widths), best_terminal.cost};
}

std::optional<StaticAttentionPrefillPlan>
StaticAttentionPrefillPlanner::query_plan_for_layout(
    size_t sequence_length,
    size_t history_length,
    bool allow_query_padding,
    const StaticAttentionPlan& layout) const {
  size_t maximum_prefix = 0;
  size_t state_count = 0;
  if (sequence_length == 0 || layout.widths.empty() ||
      c10::add_overflows(history_length, sequence_length, &maximum_prefix) ||
      layout.coverage() < maximum_prefix ||
      c10::add_overflows(sequence_length, size_t{1}, &state_count) ||
      state_count > std::vector<QueryState>().max_size()) {
    return std::nullopt;
  }

  auto tile_plan = [&](const RowPlanner& row, size_t prefix)
      -> std::optional<StaticAttentionPlan> {
    size_t coverage = 0;
    double cost = 0.0;
    std::vector<size_t> widths;
    for (const size_t width : layout.widths) {
      const auto graph = std::lower_bound(
          row.graph_costs.begin(),
          row.graph_costs.end(),
          width,
          [](const StaticAttentionGraphCost& item, size_t candidate) {
            return item.width < candidate;
          });
      if (graph == row.graph_costs.end() || graph->width != width) {
        return std::nullopt;
      }
      cost += widths.empty() ? graph->first_cost : graph->merge_cost;
      widths.push_back(width);
      if (c10::add_overflows(coverage, width, &coverage)) {
        return std::nullopt;
      }
      if (coverage >= prefix) {
        break;
      }
    }
    if (coverage < prefix || !std::isfinite(cost)) {
      return std::nullopt;
    }
    return StaticAttentionPlan{prefix, std::move(widths), cost};
  };

  std::vector<QueryState> states(state_count);
  states[0].valid = true;
  for (size_t end = 1; end <= sequence_length; ++end) {
    size_t visible_prefix = 0;
    if (c10::add_overflows(history_length, end, &visible_prefix)) {
      return std::nullopt;
    }
    QueryState best;
    for (const auto& row : row_planners_) {
      const size_t graph_rows = row.query_rows;
      if (!allow_query_padding && end < graph_rows) {
        continue;
      }
      const size_t minimum_begin =
          allow_query_padding && end < graph_rows ? 0 : end - graph_rows;
      const size_t maximum_begin = allow_query_padding ? end - 1 : minimum_begin;
      auto key_plan = tile_plan(row, visible_prefix);
      if (!key_plan.has_value()) {
        continue;
      }
      for (size_t begin = minimum_begin; begin <= maximum_begin; ++begin) {
        if (!states[begin].valid) {
          continue;
        }
        const size_t valid_rows = end - begin;
        const QueryState& previous = states[begin];
        const auto valid_elements =
            causal_valid_elements(history_length, begin, end);
        size_t calls = 0;
        size_t tiles = 0;
        size_t tile_computed = 0;
        size_t computed = 0;
        size_t total_valid = 0;
        size_t query_padding = 0;
        if (!valid_elements.has_value() ||
            c10::add_overflows(
                previous.calls, key_plan->graph_calls(), &calls) ||
            c10::add_overflows(previous.tiles, size_t{1}, &tiles) ||
            c10::mul_overflows(
                graph_rows, key_plan->coverage(), &tile_computed) ||
            c10::add_overflows(previous.computed, tile_computed, &computed) ||
            c10::add_overflows(
                previous.valid_elements, *valid_elements, &total_valid) ||
            c10::add_overflows(
                previous.query_padding,
                graph_rows - valid_rows,
                &query_padding)) {
          continue;
        }
        const double cost = previous.cost + key_plan->predicted_cost;
        if (!std::isfinite(cost)) {
          continue;
        }
        QueryState candidate{
            true,
            cost,
            calls,
            tiles,
            computed,
            total_valid,
            query_padding,
            begin,
            valid_rows,
            graph_rows,
            *key_plan};
        if (is_better_query_state(candidate, best)) {
          best = std::move(candidate);
        }
      }
    }
    if (!best.valid) {
      return std::nullopt;
    }
    states[end] = std::move(best);
  }

  const QueryState& terminal = states[sequence_length];
  std::vector<StaticAttentionQueryTilePlan> reverse_tiles;
  reverse_tiles.reserve(terminal.tiles);
  for (size_t end = sequence_length; end > 0;) {
    const QueryState& state = states[end];
    if (!state.valid || state.previous >= end) {
      return std::nullopt;
    }
    size_t prefix = 0;
    if (c10::add_overflows(history_length, end, &prefix)) {
      return std::nullopt;
    }
    reverse_tiles.push_back(
        {state.previous,
         state.valid_rows,
         state.graph_rows,
         prefix,
         state.key_plan});
    end = state.previous;
  }
  std::reverse(reverse_tiles.begin(), reverse_tiles.end());
  return StaticAttentionPrefillPlan{
      sequence_length,
      history_length,
      std::move(reverse_tiles),
      terminal.cost,
      terminal.calls,
      terminal.computed,
      terminal.valid_elements,
      terminal.query_padding,
      layout.widths};
}

std::optional<StaticAttentionPrefillPlan>
StaticAttentionPrefillPlanner::plan_persistent(
    size_t sequence_length,
    size_t history_length,
    bool allow_query_padding) {
  return plan(sequence_length, history_length, allow_query_padding);
}

} // namespace llm
} // namespace extension
} // namespace executorch
