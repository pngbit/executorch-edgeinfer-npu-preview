# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fixed-shape Llama stages for host-composed, arbitrary-context inference."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple

import torch
from executorch.examples.models.llama.attention import AttentionMHA
from executorch.examples.models.llama.feed_forward import FeedForward
from executorch.examples.models.llama.llama_transformer import (
    Transformer,
    TransformerBlock,
)
from executorch.examples.models.llama.rope import apply_rotary_emb, hf_apply_rotary_emb
from executorch.extension.llm.modules.composable_attention import (
    build_static_attention_portfolio,
    execute_attention_plan,
    StaticAttentionGraphCost,
    StaticAttentionPlanner,
)
from torch import nn


ExampleInputs = Dict[str, Tuple[torch.Tensor, ...]]


def composable_llama_method_name(
    kind: str,
    layer: int | None = None,
    *,
    query_rows: int | None = None,
) -> str:
    """Return a stable method name for one fixed-shape Llama stage.

    The primary query-row shape keeps the original unsuffixed ABI. Additional
    Prefill shapes use ``_r<R>`` so they can coexist in the same PTE.
    """

    if query_rows is not None and query_rows <= 0:
        raise ValueError("query rows must be positive")
    suffix = "" if query_rows is None else f"_r{query_rows}"

    if kind in ("embedding", "output"):
        if layer is not None:
            raise ValueError(f"{kind} does not take a layer index")
        return f"llama_{kind}{suffix}"
    if kind not in ("qkv", "pre", "post") or layer is None or layer < 0:
        raise ValueError("kind must be embedding, output, qkv, pre, or post")
    return f"llama_layer_{layer}_{kind}{suffix}"


def composable_llama_full_layer_method_name(layer: int, capacity: int) -> str:
    """Return the method name for a fixed-capacity complete-layer control."""

    if layer < 0 or capacity <= 1:
        raise ValueError("layer must be non-negative and capacity must exceed one")
    return f"llama_layer_{layer}_full_c{capacity}"


def _validate_supported_layer(layer: TransformerBlock) -> AttentionMHA:
    attention = layer.attention
    if not isinstance(attention, AttentionMHA):
        raise ValueError("composable Llama currently requires MHA Attention")
    feature_checks = (
        (
            attention.use_qk_norm and not attention.qk_norm_before_rope,
            "Q/K normalization after RoPE",
        ),
        (attention.scale_query_by != 1.0, "query scaling"),
        (attention.use_q_gate, "Query gating"),
        (attention.use_attn_o_gate, "Attention-output gating"),
        (attention.use_attn_o_norm, "Attention-output normalization"),
        (getattr(attention, "is_kv_shared_layer", False), "shared K/V projections"),
        (layer.use_residual_gate, "residual gating"),
        (layer.mlp_type != "default", f"MLP type {layer.mlp_type!r}"),
        (
            not isinstance(getattr(layer, "feed_forward", None), FeedForward),
            "non-standard feed-forward network",
        ),
        (hasattr(layer, "post_ffn_norm"), "post-FFN normalization"),
    )
    unsupported = [name for enabled, name in feature_checks if enabled]
    if unsupported:
        raise ValueError(
            "composable Llama does not support this layer configuration: "
            + ", ".join(unsupported)
        )
    if attention.wk is None or attention.wv is None:
        raise ValueError("composable Llama requires per-layer K/V projections")
    return attention


def validate_composable_llama(model: Transformer) -> None:
    """Validate the model semantics implemented by the current host runner.

    Position-dependent RoPE is intentionally outside the exported graphs. The
    host supports unscaled, full-dimension Llama and Hugging Face layouts; all
    other variants must be rejected before export.
    """

    if not isinstance(model, Transformer):
        raise ValueError("composable Llama requires a Transformer model")
    params = model.params

    rope_checks = (
        (params.use_scaled_rope, "scaled RoPE"),
        (params.partial_rotary_factor != 1.0, "partial-dimension RoPE"),
        (params.no_rope_layer_interval is not None, "layers without RoPE"),
        (params.local_rope_theta is not None, "per-layer local RoPE"),
        (
            params.rope_scaling_short_factor is not None,
            "LongRoPE short factors",
        ),
        (params.rope_scaling_long_factor is not None, "LongRoPE long factors"),
        (
            params.rope_scaling_attention_factor is not None,
            "LongRoPE attention scaling",
        ),
    )
    unsupported_rope = [name for enabled, name in rope_checks if enabled]
    if unsupported_rope:
        raise ValueError(
            "composable Llama currently requires unscaled full-dimension RoPE; "
            "unsupported settings: " + ", ".join(unsupported_rope)
        )
    if (
        params.head_dim <= 0
        or params.head_dim % 2 != 0
        or not math.isfinite(params.rope_freq_base)
        or params.rope_freq_base <= 0.0
    ):
        raise ValueError(
            "composable Llama requires an even head dimension and positive "
            "finite RoPE frequency base"
        )

    model_checks = (
        (
            params.attention_type != "mha",
            f"Attention type {params.attention_type!r}",
        ),
        (params.layer_types is not None, "hybrid layer types"),
        (params.num_kv_shared_layers != 0, "cross-layer K/V sharing"),
        (params.sliding_window is not None, "sliding-window Attention"),
        (
            params.attn_logit_softcapping is not None,
            "Attention-logit soft capping",
        ),
        (
            params.post_attention_norm or params.post_ffn_norm,
            "post-norm Transformer blocks",
        ),
        (params.target_modules is not None, "runtime LoRA adapters"),
        (
            params.input_prune_map is not None or params.output_prune_map is not None,
            "pruned vocabulary maps",
        ),
    )
    unsupported_model = [name for enabled, name in model_checks if enabled]
    if unsupported_model:
        raise ValueError(
            "composable Llama currently requires standard dense Llama blocks; "
            "unsupported settings: " + ", ".join(unsupported_model)
        )

    if params.n_heads % params.n_kv_heads != 0:
        raise ValueError("query heads must be divisible by K/V heads")
    if len(model.layers) != params.n_layers:
        raise ValueError("model layer count does not match its configuration")
    for layer in model.layers:
        if not isinstance(layer, TransformerBlock):
            raise ValueError("composable Llama requires TransformerBlock layers")
        _validate_supported_layer(layer)


class ComposableLlamaEmbedding(nn.Module):
    """Run token embedding as one reusable fixed-row graph."""

    def __init__(self, model: Transformer) -> None:
        super().__init__()
        if model.tok_embeddings is None:
            raise ValueError("the Llama model does not own token embeddings")
        self.embedding = model.tok_embeddings

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedding(tokens)


class ComposableLlamaQKV(nn.Module):
    """Normalize and project Q/K/V into one backend-stable output."""

    def __init__(self, layer: TransformerBlock) -> None:
        super().__init__()
        attention = _validate_supported_layer(layer)
        self.attention_norm = layer.attention_norm
        self.wq = attention.wq
        self.wk = attention.wk
        self.wv = attention.wv
        self.query_heads = attention.n_local_heads
        self.kv_heads = attention.n_local_kv_heads
        self.head_dim = attention.head_dim
        self.q_norm = attention.q_norm_fn if attention.use_qk_norm else None
        self.k_norm = attention.k_norm_fn if attention.use_qk_norm else None

    def forward(
        self,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.attention_norm(hidden)
        batch_size, rows, _ = normalized.shape
        query = self.wq(normalized).view(
            batch_size, rows, self.query_heads, self.head_dim
        )
        key = self.wk(normalized).view(batch_size, rows, self.kv_heads, self.head_dim)
        value = self.wv(normalized)
        if self.q_norm is not None:
            query = self.q_norm(query)
            key = self.k_norm(key)
        # Some delegates do not preserve tuple-output buffer ordering. A
        # single packed output keeps one graph launch and gives the host an
        # unambiguous method ABI: [..., Q | K | V].
        return torch.cat(
            (
                query.flatten(-2),
                key.flatten(-2),
                value,
            ),
            dim=-1,
        )


class ComposableLlamaPreAttention(nn.Module):
    """Project, normalize, and rotate Q/K/V inside one static graph.

    This ABI matches the proven xllm-edge graph boundary and emits tensors in
    the layouts consumed directly by the composable Attention methods:
    Q=[B,Hq,R,D], K=[B,Hkv,D,R], and V=[B,Hkv,R,D].
    """

    def __init__(self, layer: TransformerBlock) -> None:
        super().__init__()
        attention = _validate_supported_layer(layer)
        self.attention_norm = layer.attention_norm
        self.wq = attention.wq
        self.wk = attention.wk
        self.wv = attention.wv
        self.query_heads = attention.n_local_heads
        self.kv_heads = attention.n_local_kv_heads
        self.head_dim = attention.head_dim
        self.q_norm = attention.q_norm_fn if attention.use_qk_norm else None
        self.k_norm = attention.k_norm_fn if attention.use_qk_norm else None

    def forward(
        self,
        hidden: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = self.attention_norm(hidden)
        batch_size, rows, _ = normalized.shape
        query = self.wq(normalized).view(
            batch_size, rows, self.query_heads, self.head_dim
        )
        key = self.wk(normalized).view(batch_size, rows, self.kv_heads, self.head_dim)
        value = self.wv(normalized).view(batch_size, rows, self.kv_heads, self.head_dim)
        if self.q_norm is not None:
            query = self.q_norm(query)
            key = self.k_norm(key)

        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).transpose(2, 3).contiguous()
        query, key = apply_composable_rotary_emb(query, key, freqs_cos, freqs_sin)
        return query, key, value.transpose(1, 2).contiguous()


def unpack_composable_qkv(
    packed: torch.Tensor,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert packed Q/K/V projections to static Attention layouts."""

    if packed.dim() != 3:
        raise ValueError("packed Q/K/V must have shape [B,R,F]")
    query_features = query_heads * head_dim
    kv_features = kv_heads * head_dim
    if packed.shape[-1] != query_features + 2 * kv_features:
        raise ValueError("packed Q/K/V feature size does not match model heads")
    batch_size, rows, _ = packed.shape
    q, k, v = torch.split(packed, (query_features, kv_features, kv_features), dim=-1)
    return (
        q.view(batch_size, rows, query_heads, head_dim).transpose(1, 2).contiguous(),
        k.view(batch_size, rows, kv_heads, head_dim)
        .transpose(1, 2)
        .transpose(2, 3)
        .contiguous(),
        v.view(batch_size, rows, kv_heads, head_dim).transpose(1, 2).contiguous(),
    )


def apply_composable_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to Attention-layout Q/K tensors outside a static graph."""

    sequence_q = q.transpose(1, 2)
    sequence_k = k.transpose(2, 3).transpose(1, 2)
    if freqs_cos.shape[-1] == q.shape[-1]:
        sequence_q, sequence_k = hf_apply_rotary_emb(
            sequence_q, sequence_k, freqs_cos, freqs_sin
        )
    else:
        sequence_q, sequence_k = apply_rotary_emb(
            sequence_q, sequence_k, freqs_cos, freqs_sin
        )
    return (
        sequence_q.transpose(1, 2).contiguous(),
        sequence_k.transpose(1, 2).transpose(2, 3).contiguous(),
    )


class ComposableLlamaPostAttention(nn.Module):
    """Apply the Attention output projection, residual, and feed-forward path."""

    def __init__(self, layer: TransformerBlock) -> None:
        super().__init__()
        attention = _validate_supported_layer(layer)
        self.output_projection = attention.wo
        self.ffn_norm = layer.ffn_norm
        self.feed_forward = layer.feed_forward
        self.attention_dim = attention.n_local_heads * attention.head_dim

    def forward(
        self, hidden: torch.Tensor, attention_output: torch.Tensor
    ) -> torch.Tensor:
        batch_size, _, rows, _ = attention_output.shape
        attention_output = attention_output.transpose(1, 2).reshape(
            batch_size, rows, self.attention_dim
        )
        residual = hidden + self.output_projection(attention_output)
        return residual + self.feed_forward(self.ffn_norm(residual))


class ComposableLlamaFullLayer(nn.Module):
    """Fixed-capacity complete decoder layer used as a same-weight control.

    The K/V history and current token remain separate. Only their score vectors
    are concatenated, matching the proven long-context graph boundary used by
    the original EdgeInfer evaluation while avoiding a large K/V concatenate.
    """

    def __init__(self, layer: TransformerBlock, capacity: int) -> None:
        super().__init__()
        if capacity <= 1:
            raise ValueError("complete-layer capacity must exceed one")
        attention = _validate_supported_layer(layer)
        self.pre = ComposableLlamaPreAttention(layer)
        self.post = ComposableLlamaPostAttention(layer)
        self.capacity = int(capacity)
        self.kv_groups = attention.n_local_heads // attention.n_local_kv_heads
        self.softmax_scale = 1.0 / math.sqrt(attention.head_dim)

    def forward(
        self,
        hidden: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        additive_mask: torch.Tensor,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query, new_key, new_value = self.pre(hidden, freqs_cos, freqs_sin)
        attention_past_key = _repeat_kv_heads_for_full_layer(past_key, self.kv_groups)
        attention_past_value = _repeat_kv_heads_for_full_layer(
            past_value, self.kv_groups
        )
        attention_new_key = _repeat_kv_heads_for_full_layer(new_key, self.kv_groups)
        attention_new_value = _repeat_kv_heads_for_full_layer(new_value, self.kv_groups)

        past_scores = query @ attention_past_key
        current_scores = query @ attention_new_key
        scores = torch.cat((past_scores, current_scores), dim=-1)
        probabilities = torch.softmax(
            scores * self.softmax_scale + additive_mask, dim=-1
        )
        past_probabilities, current_probabilities = torch.split(
            probabilities, (self.capacity - 1, 1), dim=-1
        )
        attention_output = (
            past_probabilities @ attention_past_value
            + current_probabilities @ attention_new_value
        )
        return self.post(hidden, attention_output), new_key, new_value


def _repeat_kv_heads_for_full_layer(value: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return value
    expanded = value.unsqueeze(2)
    repeats = [1] * expanded.dim()
    repeats[2] = groups
    return expanded.repeat(*repeats).flatten(1, 2)


class ComposableLlamaOutput(nn.Module):
    """Normalize hidden states and project them to logits."""

    def __init__(self, model: Transformer) -> None:
        super().__init__()
        if model.output is None:
            raise ValueError("the Llama model does not own an output projection")
        self.norm = model.norm
        self.output = model.output

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.output(self.norm(hidden))


@dataclass(frozen=True)
class ComposableLlamaPortfolio:
    """All fixed-shape methods needed for one host-composed Llama model."""

    modules: Mapping[str, nn.Module]
    example_inputs: Mapping[str, Tuple[torch.Tensor, ...]]

    @property
    def method_names(self) -> Tuple[str, ...]:
        return tuple(self.modules)


def build_composable_llama_portfolio(
    model: Transformer,
    *,
    query_rows: int,
    prefill_query_rows: Iterable[int] | None = None,
    widths: Iterable[int],
    widths_by_query_rows: Mapping[int, Iterable[int]] | None = None,
    internal_query_tile_rows_by_query_rows: Mapping[int, int] | None = None,
    dtype: torch.dtype,
    state_scale: float = 1.0,
    exponent_floor: float | None = None,
    pre_attention_rope: bool = False,
) -> ComposableLlamaPortfolio:
    """Build finite fixed-shape methods for arbitrary-context Llama execution."""

    if query_rows <= 0 or not dtype.is_floating_point:
        raise ValueError("query rows and floating-point dtype are required")
    requested_rows = (query_rows,) if prefill_query_rows is None else tuple(
        int(rows) for rows in prefill_query_rows
    )
    if not requested_rows:
        raise ValueError("at least one Prefill query-row shape is required")
    if query_rows not in requested_rows:
        requested_rows = (query_rows, *requested_rows)
    prefill_rows = tuple(sorted(set(requested_rows)))
    for rows in prefill_rows:
        if rows <= 0:
            raise ValueError("Prefill query rows must be positive")
        if rows != query_rows and rows & (rows - 1):
            raise ValueError("additional Prefill query rows must be powers of two")
    validate_composable_llama(model)
    if not model.apply_embedding or not model.apply_output:
        raise ValueError("the model must include embedding and output projections")
    params = model.params
    common_widths = tuple(sorted({int(width) for width in widths}))
    if not common_widths or common_widths[0] <= 0:
        raise ValueError("at least one positive Attention width is required")
    row_widths = {rows: common_widths for rows in prefill_rows}
    if widths_by_query_rows is not None:
        unknown_rows = set(widths_by_query_rows) - set(prefill_rows)
        if unknown_rows:
            raise ValueError(
                f"Attention widths were provided for unknown Query rows: "
                f"{sorted(unknown_rows)}"
            )
        for rows, requested_widths in widths_by_query_rows.items():
            normalized = tuple(sorted({int(width) for width in requested_widths}))
            if not normalized or normalized[0] <= 0:
                raise ValueError(
                    f"Query rows {rows} requires at least one positive Attention width"
                )
            row_widths[rows] = normalized
    row_query_tiles = {rows: 0 for rows in prefill_rows}
    if internal_query_tile_rows_by_query_rows is not None:
        unknown_rows = set(internal_query_tile_rows_by_query_rows) - set(
            prefill_rows
        )
        if unknown_rows:
            raise ValueError(
                "internal Query tiles were provided for unknown Query rows: "
                f"{sorted(unknown_rows)}"
            )
        for rows, requested_tile in internal_query_tile_rows_by_query_rows.items():
            tile = int(requested_tile)
            if tile < 0 or tile > rows:
                raise ValueError(
                    f"Query rows {rows} requires an internal Query tile in "
                    f"[0, {rows}]"
                )
            row_query_tiles[rows] = tile

    modules: Dict[str, nn.Module] = {}
    example_inputs: ExampleInputs = {}
    rope_width = params.head_dim if params.use_hf_rope else params.head_dim // 2
    for rows in prefill_rows:
        # The primary shape retains the original ABI. Every additional shape
        # is explicitly named and is built from these same model weights.
        stage_rows = None if rows == query_rows else rows
        tokens = torch.zeros(1, rows, dtype=torch.long)
        hidden = torch.zeros(1, rows, params.dim, dtype=dtype)
        attention_output = torch.zeros(
            1, params.n_heads, rows, params.head_dim, dtype=dtype
        )
        freqs_cos = torch.ones(rows, rope_width, dtype=dtype)
        freqs_sin = torch.zeros(rows, rope_width, dtype=dtype)

        embedding_name = composable_llama_method_name(
            "embedding", query_rows=stage_rows
        )
        modules[embedding_name] = ComposableLlamaEmbedding(model).eval()
        example_inputs[embedding_name] = (tokens,)

        for layer_index, layer in enumerate(model.layers):
            if pre_attention_rope:
                pre_name = composable_llama_method_name(
                    "pre", layer_index, query_rows=stage_rows
                )
                modules[pre_name] = ComposableLlamaPreAttention(layer).eval()
                example_inputs[pre_name] = (hidden, freqs_cos, freqs_sin)
            else:
                qkv_name = composable_llama_method_name(
                    "qkv", layer_index, query_rows=stage_rows
                )
                modules[qkv_name] = ComposableLlamaQKV(layer).eval()
                example_inputs[qkv_name] = (hidden,)

            post_name = composable_llama_method_name(
                "post", layer_index, query_rows=stage_rows
            )
            modules[post_name] = ComposableLlamaPostAttention(layer).eval()
            example_inputs[post_name] = (hidden, attention_output)

        output_name = composable_llama_method_name("output", query_rows=stage_rows)
        modules[output_name] = ComposableLlamaOutput(model).eval()
        example_inputs[output_name] = (hidden,)

        attention = build_static_attention_portfolio(
            batch_size=1,
            query_heads=params.n_heads,
            kv_heads=params.n_kv_heads,
            query_rows=rows,
            head_dim=params.head_dim,
            widths=row_widths[rows],
            softmax_scale=1.0 / math.sqrt(params.head_dim),
            state_scale=state_scale,
            exponent_floor=exponent_floor,
            internal_query_tile_rows=row_query_tiles[rows],
            dtype=dtype,
        )
        modules.update(attention.modules)
        example_inputs.update(attention.example_inputs)
    return ComposableLlamaPortfolio(modules, example_inputs)


def execute_composable_llama(
    model: Transformer,
    tokens: torch.Tensor,
    graph_costs: Iterable[StaticAttentionGraphCost],
    *,
    state_scale: float = 1.0,
) -> torch.Tensor:
    """Eager reference for token-by-token host orchestration with growing K/V."""

    if tokens.dim() != 2 or tokens.shape[0] != 1 or tokens.shape[1] <= 0:
        raise ValueError("tokens must have shape [1, sequence_length]")
    validate_composable_llama(model)
    params = model.params
    planner = StaticAttentionPlanner(graph_costs)
    embedding = ComposableLlamaEmbedding(model)
    qkv_stages = [ComposableLlamaQKV(layer) for layer in model.layers]
    post_stages = [ComposableLlamaPostAttention(layer) for layer in model.layers]
    output_stage = ComposableLlamaOutput(model)
    key_caches: list[torch.Tensor | None] = [None] * len(model.layers)
    value_caches: list[torch.Tensor | None] = [None] * len(model.layers)
    freqs_cos, freqs_sin = model.rope.precompute_freqs_cis(
        params.head_dim,
        tokens.shape[1],
        params.rope_freq_base,
    )
    logits = []

    for position in range(tokens.shape[1]):
        hidden = embedding(tokens[:, position : position + 1])
        position_cos = freqs_cos[position : position + 1]
        position_sin = freqs_sin[position : position + 1]
        for layer_index, (qkv_stage, post_stage) in enumerate(
            zip(qkv_stages, post_stages)
        ):
            packed_qkv = qkv_stage(hidden)
            q, new_k, new_v = unpack_composable_qkv(
                packed_qkv,
                params.n_heads,
                params.n_kv_heads,
                params.head_dim,
            )
            q, new_k = apply_composable_rotary_emb(q, new_k, position_cos, position_sin)
            key_caches[layer_index] = (
                new_k
                if key_caches[layer_index] is None
                else torch.cat((key_caches[layer_index], new_k), dim=-1)
            )
            value_caches[layer_index] = (
                new_v
                if value_caches[layer_index] is None
                else torch.cat((value_caches[layer_index], new_v), dim=-2)
            )
            plan = planner.plan(position + 1)
            visibility = torch.ones(
                1,
                1,
                1,
                position + 1,
                dtype=hidden.dtype,
                device=hidden.device,
            )
            attention_output = execute_attention_plan(
                q,
                key_caches[layer_index],
                value_caches[layer_index],
                visibility,
                plan,
                1.0 / math.sqrt(params.head_dim),
                state_scale,
            )
            hidden = post_stage(hidden, attention_output)
        logits.append(output_stage(hidden))
    return torch.cat(logits, dim=1)
