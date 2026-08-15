/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <optional>
#include <vector>

#include <c10/util/safe_numerics.h>

namespace executorch {
namespace extension {
namespace llm {

/// Returns the exclusive K/V prefix visible to a causal Query tile.
inline std::optional<size_t> causal_visible_prefix_end(
    size_t causal_query_begin,
    size_t valid_query_rows) {
  size_t visible_end = 0;
  if (valid_query_rows == 0 ||
      c10::add_overflows(causal_query_begin, valid_query_rows, &visible_end)) {
    return std::nullopt;
  }
  return visible_end;
}

/// Returns valid columns visible in one physical K/V block.
///
/// Zero means the entire block is in the causally invalid upper-right region.
/// A result below the physical width leaves a masked padding suffix.
inline size_t causal_block_visible_columns(
    size_t block_begin,
    size_t block_valid_width,
    size_t visible_prefix_end) {
  if (block_begin >= visible_prefix_end) {
    return 0;
  }
  return std::min(block_valid_width, visible_prefix_end - block_begin);
}

struct CausalPrefillTileStats {
  size_t visible_graph_calls;
  size_t boundary_padding;
};

struct CausalPrefillPlanStats {
  size_t physical_blocks;
  size_t full_graph_calls;
  size_t visible_graph_calls;
  size_t skipped_graph_calls;
  size_t aggregate_boundary_padding;
  std::vector<CausalPrefillTileStats> tiles;
};

/// Summarizes the physical calls made by a causal Prefill staircase.
inline std::optional<CausalPrefillPlanStats> causal_prefill_plan_stats(
    const std::vector<size_t>& block_widths,
    const std::vector<size_t>& visible_prefixes) {
  if (block_widths.empty() || visible_prefixes.empty()) {
    return std::nullopt;
  }
  size_t total_coverage = 0;
  for (const size_t width : block_widths) {
    if (width == 0 ||
        c10::add_overflows(total_coverage, width, &total_coverage)) {
      return std::nullopt;
    }
  }

  size_t full_graph_calls = 0;
  if (c10::mul_overflows(
          block_widths.size(), visible_prefixes.size(), &full_graph_calls)) {
    return std::nullopt;
  }
  CausalPrefillPlanStats result{
      block_widths.size(), full_graph_calls, 0, 0, 0, {}};
  result.tiles.reserve(visible_prefixes.size());
  for (const size_t prefix : visible_prefixes) {
    if (prefix == 0 || prefix > total_coverage) {
      return std::nullopt;
    }
    size_t coverage = 0;
    size_t calls = 0;
    for (const size_t width : block_widths) {
      if (coverage >= prefix) {
        break;
      }
      if (c10::add_overflows(coverage, width, &coverage) ||
          c10::add_overflows(calls, size_t{1}, &calls)) {
        return std::nullopt;
      }
    }
    size_t visible_graph_calls = 0;
    size_t aggregate_padding = 0;
    if (coverage < prefix ||
        c10::add_overflows(
            result.visible_graph_calls, calls, &visible_graph_calls) ||
        c10::add_overflows(
            result.aggregate_boundary_padding,
            coverage - prefix,
            &aggregate_padding)) {
      return std::nullopt;
    }
    result.visible_graph_calls = visible_graph_calls;
    result.aggregate_boundary_padding = aggregate_padding;
    result.tiles.push_back({calls, coverage - prefix});
  }
  if (result.visible_graph_calls > result.full_graph_calls) {
    return std::nullopt;
  }
  result.skipped_graph_calls =
      result.full_graph_calls - result.visible_graph_calls;
  return result;
}

} // namespace llm
} // namespace extension
} // namespace executorch
