# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exact attention composed from reusable, fixed-shape graph calls.

The modules in this file keep every delegated graph static. A host runtime can
cover an arbitrary key/value length by invoking a finite graph library more
than once and carrying the online-softmax state between invocations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple

import torch
import torch.nn.functional as F
from executorch.exir._warnings import experimental
from torch import nn


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


@experimental(
    "Composable static Attention APIs are experimental and may change without notice."
)
@dataclass(frozen=True)
class StaticAttentionGraphCost:
    """Measured cost of one first or merge graph with a fixed KV width.

    .. warning::
        This API is experimental and may change or be removed without notice.
    """

    width: int
    first_cost: float
    merge_cost: float

    def __post_init__(self) -> None:
        if not _is_power_of_two(self.width):
            raise ValueError("graph width must be a positive power of two")
        if not math.isfinite(self.first_cost) or self.first_cost <= 0.0:
            raise ValueError("first graph cost must be finite and positive")
        if not math.isfinite(self.merge_cost) or self.merge_cost <= 0.0:
            raise ValueError("merge graph cost must be finite and positive")


@experimental(
    "Composable static Attention APIs are experimental and may change without notice."
)
@dataclass(frozen=True)
class StaticAttentionPlan:
    """An ordered first-plus-merge graph composition for one KV length.

    .. warning::
        This API is experimental and may change or be removed without notice.
    """

    sequence_length: int
    widths: Tuple[int, ...]
    predicted_cost: float

    @property
    def coverage(self) -> int:
        return sum(self.widths)

    @property
    def padding(self) -> int:
        return self.coverage - self.sequence_length

    @property
    def graph_calls(self) -> int:
        return len(self.widths)


@experimental(
    "Composable static Attention APIs are experimental and may change without notice."
)
@dataclass(frozen=True)
class StaticAttentionPortfolio:
    """Fixed-shape modules and example inputs for one query shape.

    Every KV width contributes a first method and a merge method. The exported
    method set is finite, but a runtime can invoke merge methods repeatedly to
    cover an unbounded KV length.

    .. warning::
        This API is experimental and may change or be removed without notice.
    """

    modules: Mapping[str, nn.Module]
    example_inputs: Mapping[str, Tuple[torch.Tensor, ...]]

    @property
    def method_names(self) -> Tuple[str, ...]:
        return tuple(self.modules)


@experimental(
    "Composable static Attention APIs are experimental and may change without notice."
)
def static_attention_method_name(
    kind: str,
    query_rows: int,
    width: int,
    assume_nonempty: bool = False,
) -> str:
    """Return the stable PTE method name for one fixed attention graph.

    .. warning::
        This API is experimental and may change or be removed without notice.
    """

    if kind not in ("first", "merge"):
        raise ValueError("attention method kind must be first or merge")
    if query_rows <= 0 or not _is_power_of_two(width):
        raise ValueError(
            "query rows must be positive and graph width must be a power of two"
        )
    contract = "_nonempty" if assume_nonempty else ""
    return f"attn_{kind}{contract}_r{query_rows}_c{width}"


@experimental(
    "Composable static Attention APIs are experimental and may change without notice."
)
class StaticAttentionPlanner:
    """Lazily plan any KV length from a finite measured graph library.

    The planner has no configured maximum sequence length. Dynamic programming
    state grows only to the largest length requested by the application.

    .. warning::
        This API is experimental and may change or be removed without notice.
    """

    def __init__(
        self,
        graph_costs: (
            Iterable[StaticAttentionGraphCost] | Mapping[int, Tuple[float, float]]
        ),
    ) -> None:
        if isinstance(graph_costs, Mapping):
            profiles = tuple(
                StaticAttentionGraphCost(width, costs[0], costs[1])
                for width, costs in graph_costs.items()
            )
        else:
            profiles = tuple(graph_costs)
        if not profiles:
            raise ValueError("at least one static attention graph is required")
        if len({profile.width for profile in profiles}) != len(profiles):
            raise ValueError("static attention graph widths must be unique")

        self._profiles = tuple(sorted(profiles, key=lambda item: item.width))
        self._merge_cost = [0.0]
        self._merge_coverage = [0]
        self._merge_calls = [0]
        self._merge_choice = [0]

    @property
    def graph_widths(self) -> Tuple[int, ...]:
        return tuple(profile.width for profile in self._profiles)

    @property
    def planned_through(self) -> int:
        return len(self._merge_cost) - 1

    def plan(self, sequence_length: int) -> StaticAttentionPlan:
        if sequence_length <= 0:
            raise ValueError("sequence length must be positive")
        self._extend_merge_plans(sequence_length)

        best = None
        for profile in self._profiles:
            remaining = max(0, sequence_length - profile.width)
            coverage = profile.width + self._merge_coverage[remaining]
            calls = 1 + self._merge_calls[remaining]
            cost = profile.first_cost + self._merge_cost[remaining]
            key = (cost, coverage - sequence_length, calls, -profile.width)
            if best is None or key < best[0]:
                best = (key, profile.width, remaining, cost)

        assert best is not None
        _, first_width, remaining, cost = best
        widths = [first_width]
        while remaining > 0:
            width = self._merge_choice[remaining]
            if width <= 0:
                raise RuntimeError("invalid static attention plan backpointer")
            widths.append(width)
            remaining = max(0, remaining - width)
        return StaticAttentionPlan(sequence_length, tuple(widths), cost)

    def plan_causal_tile(
        self,
        full_sequence_length: int,
        causal_query_begin: int,
        query_rows: int,
        valid_query_rows: int = 0,
    ) -> StaticAttentionPlan:
        """Plan only the K/V prefix visible to one causal Query tile."""

        if full_sequence_length <= 0 or causal_query_begin < 0 or query_rows <= 0:
            raise ValueError("causal tile dimensions must be valid and positive")
        valid_rows = query_rows if valid_query_rows == 0 else valid_query_rows
        if valid_rows <= 0 or valid_rows > query_rows:
            raise ValueError("valid Query rows must be within the physical tile")
        visible_prefix = causal_query_begin + valid_rows
        if visible_prefix > full_sequence_length:
            raise ValueError("causal Query tile exceeds the full K/V length")
        return self.plan(visible_prefix)

    def _extend_merge_plans(self, required_length: int) -> None:
        for required in range(len(self._merge_cost), required_length + 1):
            best = None
            for profile in self._profiles:
                remaining = max(0, required - profile.width)
                coverage = profile.width + self._merge_coverage[remaining]
                calls = 1 + self._merge_calls[remaining]
                cost = profile.merge_cost + self._merge_cost[remaining]
                key = (cost, coverage - required, calls, -profile.width)
                if best is None or key < best[0]:
                    best = (key, profile.width, coverage, calls, cost)

            assert best is not None
            _, width, coverage, calls, cost = best
            self._merge_choice.append(width)
            self._merge_coverage.append(coverage)
            self._merge_calls.append(calls)
            self._merge_cost.append(cost)


def _repeat_kv_heads(value: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return value
    expanded = value.unsqueeze(2)
    repeats = [1] * expanded.dim()
    repeats[2] = groups
    return expanded.repeat(*repeats).flatten(1, 2)


def _prepare_attention_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    visibility: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4 or visibility.dim() != 4:
        raise ValueError("Q, K, V, and visibility must be rank-four tensors")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("Q, K, and V batch dimensions differ")
    if visibility.shape[0] != q.shape[0] or visibility.shape[1] not in (
        1,
        q.shape[1],
    ):
        raise ValueError("visibility batch or head dimension is not broadcastable")
    if k.shape[1] != v.shape[1] or k.shape[-1] != v.shape[-2]:
        raise ValueError("K and V head counts or block widths differ")
    if q.shape[-1] != k.shape[-2] or q.shape[-1] != v.shape[-1]:
        raise ValueError("Q, K, and V head dimensions differ")
    if k.shape[1] <= 0 or q.shape[1] % k.shape[1] != 0:
        raise ValueError("query heads must be divisible by KV heads")
    if visibility.shape[-2:] != (q.shape[-2], k.shape[-1]):
        raise ValueError("visibility shape does not match Q and K")

    groups = q.shape[1] // k.shape[1]
    return _repeat_kv_heads(k, groups), _repeat_kv_heads(v, groups)


def _masked_block_weights(
    scores: torch.Tensor,
    visibility: torch.Tensor,
    reference_max: torch.Tensor | None = None,
    exponent_floor: float | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    visible = visibility > 0
    row_min = torch.amin(scores, dim=-1, keepdim=True)
    masked_scores = torch.where(visible, scores, row_min)
    block_max = torch.amax(masked_scores, dim=-1, keepdim=True)
    normalizer = (
        block_max if reference_max is None else torch.maximum(reference_max, block_max)
    )
    # Delegates may evaluate both branches of a later where. Give masked
    # positions a zero exponent instead of asking the backend to evaluate an
    # irrelevant extreme value and then discard it.
    exponent_input = torch.where(visible, scores - normalizer, torch.zeros_like(scores))
    if exponent_floor is not None:
        exponent_input = torch.clamp(exponent_input, min=exponent_floor)
    weights = torch.exp(exponent_input)
    weights = torch.where(visible, weights, torch.zeros_like(weights))
    return weights, block_max, normalizer


def _validate_attention_options(
    softmax_scale: float,
    state_scale: float,
    exponent_floor: float | None,
    internal_kv_tile_width: int,
    internal_query_tile_rows: int,
) -> None:
    if (
        not math.isfinite(softmax_scale)
        or softmax_scale <= 0.0
        or not math.isfinite(state_scale)
        or state_scale <= 0.0
    ):
        raise ValueError("attention and state scales must be finite and positive")
    if exponent_floor is not None and (
        not math.isfinite(exponent_floor) or exponent_floor > 0.0
    ):
        raise ValueError("exponent floor must be finite and non-positive")
    if internal_kv_tile_width < 0:
        raise ValueError("internal KV tile width must be non-negative")
    if internal_query_tile_rows < 0:
        raise ValueError("internal Query tile rows must be non-negative")


def _first_attention_state(
    q: torch.Tensor,
    k_block: torch.Tensor,
    v_block: torch.Tensor,
    visibility_block: torch.Tensor,
    softmax_scale: float,
    state_scale: float,
    exponent_floor: float | None,
    assume_nonempty: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = (q @ k_block) * softmax_scale
    weights, block_max, _ = _masked_block_weights(
        scores, visibility_block, exponent_floor=exponent_floor
    )
    block_sum = torch.sum(weights, dim=-1, keepdim=True)
    block_numerator = weights @ v_block
    if assume_nonempty:
        return (
            block_numerator / block_sum,
            block_max,
            block_sum / state_scale,
            block_numerator / state_scale,
        )
    has_visible = block_sum > 0.0
    safe_sum = torch.where(has_visible, block_sum, torch.ones_like(block_sum))
    output = torch.where(
        has_visible, block_numerator / safe_sum, torch.zeros_like(block_numerator)
    )
    running_max = torch.where(
        has_visible, block_max, torch.full_like(block_max, -10000.0)
    )
    return (
        output,
        running_max,
        block_sum / state_scale,
        block_numerator / state_scale,
    )


def _merge_attention_state(
    q: torch.Tensor,
    k_block: torch.Tensor,
    v_block: torch.Tensor,
    visibility_block: torch.Tensor,
    running_max: torch.Tensor,
    running_sum: torch.Tensor,
    running_numerator: torch.Tensor,
    softmax_scale: float,
    state_scale: float,
    exponent_floor: float | None,
    assume_nonempty: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = (q @ k_block) * softmax_scale
    weights, block_max, candidate_max = _masked_block_weights(
        scores,
        visibility_block,
        running_max,
        exponent_floor=exponent_floor,
    )
    unscaled_block_sum = torch.sum(weights, dim=-1, keepdim=True)
    if assume_nonempty:
        merged_max = candidate_max
        previous_delta = running_max - candidate_max
    else:
        has_visible = unscaled_block_sum > 0.0
        merged_max = torch.where(has_visible, candidate_max, running_max)
        rescale_previous = torch.logical_and(has_visible, running_sum > 0.0)
        previous_delta = torch.where(
            rescale_previous,
            running_max - candidate_max,
            torch.zeros_like(running_max),
        )
    if exponent_floor is not None:
        previous_delta = torch.clamp(previous_delta, min=exponent_floor)
    previous_scale = torch.exp(previous_delta)
    block_sum = unscaled_block_sum / state_scale
    block_numerator = (weights @ v_block) / state_scale
    merged_sum = running_sum * previous_scale + block_sum
    merged_numerator = running_numerator * previous_scale + block_numerator
    if assume_nonempty:
        output = merged_numerator / merged_sum
    else:
        safe_sum = torch.where(
            merged_sum > 0.0, merged_sum, torch.ones_like(merged_sum)
        )
        output = torch.where(
            merged_sum > 0.0,
            merged_numerator / safe_sum,
            torch.zeros_like(merged_numerator),
        )
    return output, merged_max, merged_sum, merged_numerator


def _first_attention_with_optional_internal_tiling(
    q: torch.Tensor,
    k_block: torch.Tensor,
    v_block: torch.Tensor,
    visibility_block: torch.Tensor,
    softmax_scale: float,
    state_scale: float,
    exponent_floor: float | None,
    assume_nonempty: bool,
    internal_kv_tile_width: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Statically expand one fixed-shape method into narrower QK/PV tiles."""

    width = int(k_block.shape[-1])
    if internal_kv_tile_width <= 0 or internal_kv_tile_width >= width:
        return _first_attention_state(
            q,
            k_block,
            v_block,
            visibility_block,
            softmax_scale,
            state_scale,
            exponent_floor,
            assume_nonempty,
        )

    state = None
    for begin in range(0, width, internal_kv_tile_width):
        end = min(begin + internal_kv_tile_width, width)
        tile_inputs = (
            q,
            k_block[..., begin:end],
            v_block[..., begin:end, :],
            visibility_block[..., begin:end],
        )
        if state is None:
            state = _first_attention_state(
                *tile_inputs,
                softmax_scale,
                state_scale,
                exponent_floor,
                False,
            )
        else:
            state = _merge_attention_state(
                *tile_inputs,
                *state[1:],
                softmax_scale,
                state_scale,
                exponent_floor,
                False,
            )
    assert state is not None
    return state


def _merge_attention_with_optional_internal_tiling(
    q: torch.Tensor,
    k_block: torch.Tensor,
    v_block: torch.Tensor,
    visibility_block: torch.Tensor,
    running_max: torch.Tensor,
    running_sum: torch.Tensor,
    running_numerator: torch.Tensor,
    softmax_scale: float,
    state_scale: float,
    exponent_floor: float | None,
    assume_nonempty: bool,
    internal_kv_tile_width: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Statically expand one merge method without adding runtime graph calls."""

    width = int(k_block.shape[-1])
    if internal_kv_tile_width <= 0 or internal_kv_tile_width >= width:
        return _merge_attention_state(
            q,
            k_block,
            v_block,
            visibility_block,
            running_max,
            running_sum,
            running_numerator,
            softmax_scale,
            state_scale,
            exponent_floor,
            assume_nonempty,
        )

    state = (None, running_max, running_sum, running_numerator)
    for begin in range(0, width, internal_kv_tile_width):
        end = min(begin + internal_kv_tile_width, width)
        state = _merge_attention_state(
            q,
            k_block[..., begin:end],
            v_block[..., begin:end, :],
            visibility_block[..., begin:end],
            *state[1:],
            softmax_scale,
            state_scale,
            exponent_floor,
            False,
        )
    return state


def _first_attention_with_optional_query_tiling(
    q: torch.Tensor,
    k_block: torch.Tensor,
    v_block: torch.Tensor,
    visibility_block: torch.Tensor,
    softmax_scale: float,
    state_scale: float,
    exponent_floor: float | None,
    assume_nonempty: bool,
    internal_kv_tile_width: int,
    internal_query_tile_rows: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Statically unroll Query-row tiles inside one fixed-shape method."""

    query_rows = int(q.shape[-2])
    if internal_query_tile_rows <= 0 or internal_query_tile_rows >= query_rows:
        return _first_attention_with_optional_internal_tiling(
            q,
            k_block,
            v_block,
            visibility_block,
            softmax_scale,
            state_scale,
            exponent_floor,
            assume_nonempty,
            internal_kv_tile_width,
        )

    outputs = ([], [], [], [])
    for begin in range(0, query_rows, internal_query_tile_rows):
        end = min(begin + internal_query_tile_rows, query_rows)
        tile_outputs = _first_attention_with_optional_internal_tiling(
            q[..., begin:end, :],
            k_block,
            v_block,
            visibility_block[..., begin:end, :],
            softmax_scale,
            state_scale,
            exponent_floor,
            assume_nonempty,
            internal_kv_tile_width,
        )
        for values, value in zip(outputs, tile_outputs):
            values.append(value)
    return tuple(torch.cat(values, dim=-2) for values in outputs)


def _merge_attention_with_optional_query_tiling(
    q: torch.Tensor,
    k_block: torch.Tensor,
    v_block: torch.Tensor,
    visibility_block: torch.Tensor,
    running_max: torch.Tensor,
    running_sum: torch.Tensor,
    running_numerator: torch.Tensor,
    softmax_scale: float,
    state_scale: float,
    exponent_floor: float | None,
    assume_nonempty: bool,
    internal_kv_tile_width: int,
    internal_query_tile_rows: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Statically unroll Query-row tiles for one online-softmax merge."""

    query_rows = int(q.shape[-2])
    if internal_query_tile_rows <= 0 or internal_query_tile_rows >= query_rows:
        return _merge_attention_with_optional_internal_tiling(
            q,
            k_block,
            v_block,
            visibility_block,
            running_max,
            running_sum,
            running_numerator,
            softmax_scale,
            state_scale,
            exponent_floor,
            assume_nonempty,
            internal_kv_tile_width,
        )

    outputs = ([], [], [], [])
    for begin in range(0, query_rows, internal_query_tile_rows):
        end = min(begin + internal_query_tile_rows, query_rows)
        tile_outputs = _merge_attention_with_optional_internal_tiling(
            q[..., begin:end, :],
            k_block,
            v_block,
            visibility_block[..., begin:end, :],
            running_max[..., begin:end, :],
            running_sum[..., begin:end, :],
            running_numerator[..., begin:end, :],
            softmax_scale,
            state_scale,
            exponent_floor,
            assume_nonempty,
            internal_kv_tile_width,
        )
        for values, value in zip(outputs, tile_outputs):
            values.append(value)
    return tuple(torch.cat(values, dim=-2) for values in outputs)


@experimental(
    "Composable static Attention APIs are experimental and may change without notice."
)
class StaticAttentionFirstBlock(nn.Module):
    """Initialize exact online-softmax state in one fixed-shape graph.

    .. warning::
        This API is experimental and may change or be removed without notice.
    """

    def __init__(
        self,
        softmax_scale: float,
        state_scale: float = 1.0,
        exponent_floor: float | None = None,
        assume_nonempty: bool = False,
        internal_kv_tile_width: int = 0,
        internal_query_tile_rows: int = 0,
    ) -> None:
        super().__init__()
        _validate_attention_options(
            softmax_scale,
            state_scale,
            exponent_floor,
            internal_kv_tile_width,
            internal_query_tile_rows,
        )
        self.softmax_scale = float(softmax_scale)
        self.state_scale = float(state_scale)
        self.exponent_floor = exponent_floor
        self.assume_nonempty = bool(assume_nonempty)
        self.internal_kv_tile_width = int(internal_kv_tile_width)
        self.internal_query_tile_rows = int(internal_query_tile_rows)

    def forward(
        self,
        q: torch.Tensor,
        k_block: torch.Tensor,
        v_block: torch.Tensor,
        visibility_block: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        k_block, v_block = _prepare_attention_inputs(
            q, k_block, v_block, visibility_block
        )
        return _first_attention_with_optional_query_tiling(
            q,
            k_block,
            v_block,
            visibility_block,
            self.softmax_scale,
            self.state_scale,
            self.exponent_floor,
            self.assume_nonempty,
            self.internal_kv_tile_width,
            self.internal_query_tile_rows,
        )


@experimental(
    "Composable static Attention APIs are experimental and may change without notice."
)
class StaticAttentionMergeBlock(nn.Module):
    """Merge one fixed-shape graph into an existing online-softmax state.

    .. warning::
        This API is experimental and may change or be removed without notice.
    """

    def __init__(
        self,
        softmax_scale: float,
        state_scale: float = 1.0,
        exponent_floor: float | None = None,
        assume_nonempty: bool = False,
        internal_kv_tile_width: int = 0,
        internal_query_tile_rows: int = 0,
    ) -> None:
        super().__init__()
        _validate_attention_options(
            softmax_scale,
            state_scale,
            exponent_floor,
            internal_kv_tile_width,
            internal_query_tile_rows,
        )
        self.softmax_scale = float(softmax_scale)
        self.state_scale = float(state_scale)
        self.exponent_floor = exponent_floor
        self.assume_nonempty = bool(assume_nonempty)
        self.internal_kv_tile_width = int(internal_kv_tile_width)
        self.internal_query_tile_rows = int(internal_query_tile_rows)

    def forward(
        self,
        q: torch.Tensor,
        k_block: torch.Tensor,
        v_block: torch.Tensor,
        visibility_block: torch.Tensor,
        running_max: torch.Tensor,
        running_sum: torch.Tensor,
        running_numerator: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        k_block, v_block = _prepare_attention_inputs(
            q, k_block, v_block, visibility_block
        )
        return _merge_attention_with_optional_query_tiling(
            q,
            k_block,
            v_block,
            visibility_block,
            running_max,
            running_sum,
            running_numerator,
            self.softmax_scale,
            self.state_scale,
            self.exponent_floor,
            self.assume_nonempty,
            self.internal_kv_tile_width,
            self.internal_query_tile_rows,
        )


@experimental(
    "Composable static Attention APIs are experimental and may change without notice."
)
def build_static_attention_portfolio(
    *,
    batch_size: int,
    query_heads: int,
    kv_heads: int,
    query_rows: int,
    head_dim: int,
    widths: Iterable[int],
    softmax_scale: float,
    state_scale: float = 1.0,
    exponent_floor: float | None = None,
    assume_nonempty: bool = False,
    internal_kv_tile_width: int = 0,
    internal_query_tile_rows: int = 0,
    dtype: torch.dtype = torch.float32,
) -> StaticAttentionPortfolio:
    """Build a multi-method library of reusable static Attention graphs.

    The returned tensors are examples for ``torch.export``. They intentionally
    contain no sequence-length symbol: only the host-side invocation count is
    dynamic at runtime. ``internal_kv_tile_width`` and
    ``internal_query_tile_rows`` optionally expand narrower QK/PV operations
    inside each exported method. This remains one static graph execute and is
    independent of the runtime multi-method composition plan.

    .. warning::
        This API is experimental and may change or be removed without notice.
    """

    dimensions = (batch_size, query_heads, kv_heads, query_rows, head_dim)
    if any(dimension <= 0 for dimension in dimensions):
        raise ValueError("attention dimensions must be positive")
    if query_heads % kv_heads != 0:
        raise ValueError("query heads must be divisible by KV heads")
    if assume_nonempty and query_rows != 1:
        raise ValueError("nonempty static attention is restricted to Decode (R=1)")
    if internal_query_tile_rows > query_rows:
        raise ValueError("internal Query tile rows cannot exceed Query rows")
    if not dtype.is_floating_point:
        raise ValueError("static attention graphs require a floating-point dtype")

    normalized_widths = tuple(sorted({int(width) for width in widths}))
    if not normalized_widths or any(
        not _is_power_of_two(width) for width in normalized_widths
    ):
        raise ValueError(
            "at least one graph width is required and every width must be a "
            "positive power of two"
        )

    q = torch.zeros(batch_size, query_heads, query_rows, head_dim, dtype=dtype)
    state = (
        torch.full((batch_size, query_heads, query_rows, 1), -10000.0, dtype=dtype),
        torch.zeros(batch_size, query_heads, query_rows, 1, dtype=dtype),
        torch.zeros(batch_size, query_heads, query_rows, head_dim, dtype=dtype),
    )
    modules: Dict[str, nn.Module] = {}
    example_inputs: Dict[str, Tuple[torch.Tensor, ...]] = {}
    for width in normalized_widths:
        k = torch.zeros(batch_size, kv_heads, head_dim, width, dtype=dtype)
        v = torch.zeros(batch_size, kv_heads, width, head_dim, dtype=dtype)
        visibility = torch.ones(batch_size, 1, query_rows, width, dtype=dtype)

        first_name = static_attention_method_name(
            "first", query_rows, width, assume_nonempty
        )
        modules[first_name] = StaticAttentionFirstBlock(
            softmax_scale,
            state_scale,
            exponent_floor,
            assume_nonempty,
            internal_kv_tile_width,
            internal_query_tile_rows,
        ).eval()
        example_inputs[first_name] = (q, k, v, visibility)

        merge_name = static_attention_method_name(
            "merge", query_rows, width, assume_nonempty
        )
        modules[merge_name] = StaticAttentionMergeBlock(
            softmax_scale,
            state_scale,
            exponent_floor,
            assume_nonempty,
            internal_kv_tile_width,
            internal_query_tile_rows,
        ).eval()
        example_inputs[merge_name] = (q, k, v, visibility, *state)

    return StaticAttentionPortfolio(modules, example_inputs)


@experimental(
    "Composable static Attention APIs are experimental and may change without notice."
)
def execute_attention_plan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    visibility: torch.Tensor,
    plan: StaticAttentionPlan,
    softmax_scale: float,
    state_scale: float = 1.0,
    exponent_floor: float | None = None,
    assume_nonempty: bool = False,
) -> torch.Tensor:
    """Eager reference for host orchestration of compiled static methods.

    .. warning::
        This API is experimental and may change or be removed without notice.
    """

    if k.shape[-1] != plan.sequence_length or v.shape[-2] != plan.sequence_length:
        raise ValueError("plan sequence length does not match K and V")
    if visibility.shape[-1] != plan.sequence_length:
        raise ValueError("plan sequence length does not match visibility")

    if assume_nonempty and q.shape[-2] != 1:
        raise ValueError("nonempty static attention is restricted to Decode (R=1)")
    first = StaticAttentionFirstBlock(
        softmax_scale, state_scale, exponent_floor, assume_nonempty
    )
    merge = StaticAttentionMergeBlock(
        softmax_scale, state_scale, exponent_floor, assume_nonempty
    )
    offset = 0
    state = None
    for width in plan.widths:
        valid_width = min(width, plan.sequence_length - offset)
        k_block = k[..., offset : offset + valid_width]
        v_block = v[..., offset : offset + valid_width, :]
        mask_block = visibility[..., offset : offset + valid_width]
        padding = width - valid_width
        if padding:
            k_block = F.pad(k_block, (0, padding))
            v_block = F.pad(v_block, (0, 0, 0, padding))
            mask_block = F.pad(mask_block, (0, padding))

        if state is None:
            state = first(q, k_block, v_block, mask_block)
        else:
            state = merge(q, k_block, v_block, mask_block, *state[1:])
        offset += valid_width

    if offset != plan.sequence_length or state is None:
        raise RuntimeError("static attention plan did not cover the sequence")
    return state[0]
