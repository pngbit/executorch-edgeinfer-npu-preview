/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include <executorch/runtime/platform/compiler.h>

namespace executorch {
namespace extension {
namespace llm {

/// EXPERIMENTAL: Measured costs for a fixed-width Attention graph pair.
struct ET_EXPERIMENTAL StaticAttentionGraphCost {
  size_t width;
  double first_cost;
  double merge_cost;
};

/// EXPERIMENTAL: Measured costs for one fixed G(R, C) Attention graph pair.
struct ET_EXPERIMENTAL StaticAttentionShapeCost {
  size_t query_rows;
  size_t width;
  double first_cost;
  double merge_cost;
};

/// EXPERIMENTAL: Graph calls selected for one KV length.
struct ET_EXPERIMENTAL StaticAttentionPlan {
  size_t sequence_length;
  std::vector<size_t> widths;
  double predicted_cost;

  /// Returns total covered K/V positions, saturating on malformed overflow.
  size_t coverage() const;

  /// Returns positions covered beyond sequence_length.
  size_t padding() const;

  /// Returns the number of selected graph calls.
  size_t graph_calls() const;
};

/// EXPERIMENTAL: One physical Query tile and its selected K/V composition.
struct ET_EXPERIMENTAL StaticAttentionQueryTilePlan {
  size_t row_begin;
  size_t valid_query_rows;
  size_t graph_query_rows;
  size_t visible_prefix;
  StaticAttentionPlan key_plan;
};

/// EXPERIMENTAL: A measured-cost Prefill plan over Query and K/V dimensions.
struct ET_EXPERIMENTAL StaticAttentionPrefillPlan {
  size_t sequence_length;
  size_t history_length;
  std::vector<StaticAttentionQueryTilePlan> query_tiles;
  double predicted_cost;
  size_t graph_calls;
  size_t computed_score_elements;
  size_t valid_score_elements;
  size_t query_padding_rows;
  /// One K/V layout shared by all Query tiles. Empty means no persistent
  /// layout was requested and each tile's key_plan is authoritative.
  std::vector<size_t> persistent_widths;

  size_t wasted_score_elements() const;
};

/// EXPERIMENTAL: Cost-aware planner for reusable static graphs.
///
/// The planner has no configured context limit. It extends its dynamic
/// programming cache lazily when a larger sequence length is requested.
class ET_EXPERIMENTAL StaticAttentionPlanner {
 public:
  /// Creates a planner from positive, finite costs with unique widths.
  ///
  /// @param graph_costs Measured first and merge costs from one target device.
  /// @return A planner, or std::nullopt when the profile is invalid.
  static std::optional<StaticAttentionPlanner> create(
      std::vector<StaticAttentionGraphCost> graph_costs);

  /// Selects the lowest-cost graph composition for sequence_length.
  ///
  /// Cost is the primary objective, followed by padding and graph calls.
  /// @return A plan, or std::nullopt when the length is zero or
  /// unrepresentable.
  std::optional<StaticAttentionPlan> plan(size_t sequence_length);

  /// Plans only the K/V prefix visible to one causal Query tile.
  ///
  /// Fully invisible upper-right blocks are omitted. The final block may
  /// cross the causal boundary and is expected to mask its invisible suffix.
  std::optional<StaticAttentionPlan> plan_causal_tile(
      size_t full_sequence_length,
      size_t causal_query_begin,
      size_t query_rows,
      size_t valid_query_rows = 0);

  /// Selects one persistent layout for a sequence of causal Query tiles.
  ///
  /// Each entry is the K/V prefix visible to one tile. A block contributes
  /// its measured first/merge cost only to tiles that execute it. This allows
  /// one padded graph and mixed-width compositions to compete without
  /// repacking the persistent cache between tiles.
  std::optional<StaticAttentionPlan> plan_prefixes(
      const std::vector<size_t>& visible_prefixes) const;

  /// Returns the reusable widths represented in the measured profile.
  const std::vector<size_t>& graph_widths() const {
    return graph_widths_;
  }

  const std::vector<StaticAttentionGraphCost>& graph_costs() const {
    return graph_costs_;
  }

  /// Returns the largest required suffix already cached by the planner.
  size_t planned_through() const {
    return merge_cost_.size() - 1;
  }

 private:
  explicit StaticAttentionPlanner(
      std::vector<StaticAttentionGraphCost> graph_costs);

  bool extend_merge_plans(size_t required_length);

  std::vector<StaticAttentionGraphCost> graph_costs_;
  std::vector<size_t> graph_widths_;
  std::vector<double> merge_cost_;
  std::vector<size_t> merge_coverage_;
  std::vector<size_t> merge_calls_;
  std::vector<size_t> merge_choice_;
};

/// EXPERIMENTAL: Two-level planner for causal Prefill with static G(R, C).
///
/// The inner dynamic program selects one first graph and zero or more merge
/// graphs for every visible K/V prefix. The outer dynamic program selects a
/// nonuniform Query-row partition. Both exact compositions and a padded final
/// graph remain candidates; measured cost is always the primary objective.
class ET_EXPERIMENTAL StaticAttentionPrefillPlanner {
 public:
  /// Creates a planner from unique, positive G(R, C) profiles.
  static std::optional<StaticAttentionPrefillPlanner> create(
      std::vector<StaticAttentionShapeCost> shape_costs);

  /// Selects the lowest-cost Query/KV composition for causal Prefill.
  ///
  /// `history_length` is the already materialized prefix preceding these
  /// Query rows. When `allow_query_padding` is true, a physical R graph may
  /// execute fewer valid rows. A measured R=1 graph is therefore sufficient
  /// to represent every positive length exactly, but is not required when
  /// Query padding is enabled.
  std::optional<StaticAttentionPrefillPlan> plan(
      size_t sequence_length,
      size_t history_length = 0,
      bool allow_query_padding = true);

  /// Backward-compatible alias for plan(). Query tiles keep independent K/V
  /// compositions so measured-cost planning is not constrained by one shared
  /// physical layout.
  std::optional<StaticAttentionPrefillPlan> plan_persistent(
      size_t sequence_length,
      size_t history_length = 0,
      bool allow_query_padding = true);

  const std::vector<size_t>& query_rows() const {
    return query_rows_;
  }

 private:
  struct RowPlanner {
    size_t query_rows;
    std::vector<StaticAttentionGraphCost> graph_costs;
    StaticAttentionPlanner key_planner;
  };

  explicit StaticAttentionPrefillPlanner(std::vector<RowPlanner> planners);

  std::optional<StaticAttentionPlan> persistent_layout_for_tiles(
      const std::vector<StaticAttentionQueryTilePlan>& tiles) const;
  std::optional<StaticAttentionPrefillPlan> query_plan_for_layout(
      size_t sequence_length,
      size_t history_length,
      bool allow_query_padding,
      const StaticAttentionPlan& layout) const;

  std::vector<RowPlanner> row_planners_;
  std::vector<size_t> query_rows_;
};

} // namespace llm
} // namespace extension
} // namespace executorch
