# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export Llama for host-composed, runtime-length Attention execution."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from executorch.examples.models.llama.composable_transformer import (
    build_composable_llama_portfolio,
    composable_llama_method_name,
)
from executorch.examples.models.llama.llama_transformer import construct_transformer
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.extension.llm.export.export_composable_attention import (
    export_portable,
    export_qnn,
    normalize_row_widths,
    parse_row_width_specs,
    QNN_EXPONENT_FLOOR,
    resolve_state_scale,
)
from torch import nn


ExampleInputs = Mapping[str, tuple[torch.Tensor, ...]]


@dataclass(frozen=True)
class ComposableLlamaLayerShard:
    """One contiguous group of QKV/post methods compiled into one PTE."""

    first_layer: int
    last_layer_exclusive: int
    modules: Mapping[str, nn.Module]
    example_inputs: ExampleInputs


def validate_prefill_query_rows(
    query_rows: int, prefill_query_rows: Sequence[int] | None
) -> tuple[int, ...]:
    """Normalize the static Prefill row portfolio without changing legacy R."""

    if query_rows <= 0:
        raise ValueError("--query-rows must be positive")
    rows = (query_rows,) if prefill_query_rows is None else tuple(prefill_query_rows)
    if not rows:
        raise ValueError("--prefill-query-rows must not be empty")
    normalized = tuple(sorted(set((query_rows, *rows))))
    for row_count in normalized:
        if row_count <= 0:
            raise ValueError("--prefill-query-rows values must be positive")
        if row_count != query_rows and row_count & (row_count - 1):
            raise ValueError(
                "additional --prefill-query-rows values must be powers of two"
            )
    return normalized


def parse_row_query_tile_specs(
    values: Sequence[str] | None,
) -> dict[int, int]:
    """Parse ``R:Q`` entries for per-row internal Query tiling."""

    result: dict[int, int] = {}
    for value in values or ():
        row_text, separator, tile_text = value.partition(":")
        try:
            row = int(row_text)
            tile = int(tile_text)
        except ValueError as error:
            raise ValueError(f"invalid row-query-tile entry: {value}") from error
        if (
            not separator
            or row <= 0
            or tile <= 0
            or tile > row
            or row in result
        ):
            raise ValueError(f"invalid row-query-tile entry: {value}")
        result[row] = tile
    return result


def partition_composable_llama_portfolio(
    portfolio,
    *,
    layers: int,
    layers_per_shard: int,
) -> tuple[
    Mapping[str, nn.Module],
    ExampleInputs,
    list[ComposableLlamaLayerShard],
]:
    """Separate shared methods from bounded groups of layer stage methods."""

    if layers <= 0 or layers_per_shard <= 0:
        raise ValueError("layers and layers_per_shard must be positive")
    layer_stage_names: dict[int, tuple[str, ...]] = {}
    for layer in range(layers):
        names = tuple(
            name
            for name in portfolio.modules
            if name.startswith(f"llama_layer_{layer}_")
            and ("_qkv" in name or "_pre" in name or "_post" in name)
        )
        if not names:
            raise ValueError(f"layer {layer} must contain stage methods")
        # Every row variant has exactly one projection stage and one post stage.
        row_variants: dict[str, set[str]] = {}
        for name in names:
            if "_qkv" in name:
                stage, suffix = "qkv", name.split("_qkv", 1)[1]
            elif "_pre" in name:
                stage, suffix = "pre", name.split("_pre", 1)[1]
            else:
                stage, suffix = "post", name.split("_post", 1)[1]
            row_variants.setdefault(suffix, set()).add(stage)
        if any(stages not in ({"qkv", "post"}, {"pre", "post"}) for stages in row_variants.values()):
            raise ValueError(f"layer {layer} contains incomplete or mixed stage variants")
        layer_stage_names[layer] = names
    layer_methods = {name for names in layer_stage_names.values() for name in names}
    core_modules = {
        name: module
        for name, module in portfolio.modules.items()
        if name not in layer_methods
    }
    core_inputs = {name: portfolio.example_inputs[name] for name in core_modules}
    shards = []
    for first_layer in range(0, layers, layers_per_shard):
        last_layer_exclusive = min(first_layer + layers_per_shard, layers)
        names = [
            name
            for layer in range(first_layer, last_layer_exclusive)
            for name in layer_stage_names[layer]
        ]
        shards.append(
            ComposableLlamaLayerShard(
                first_layer=first_layer,
                last_layer_exclusive=last_layer_exclusive,
                modules={name: portfolio.modules[name] for name in names},
                example_inputs={name: portfolio.example_inputs[name] for name in names},
            )
        )
    return core_modules, core_inputs, shards


def normalize_checkpoint_keys(state_dict):
    """Remove the uniform prefix produced by torch.compile checkpoints."""

    prefix = "_orig_mod."
    if state_dict and all(key.startswith(prefix) for key in state_dict):
        return {key.removeprefix(prefix): value for key, value in state_dict.items()}
    return state_dict


def load_model(
    checkpoint_path: Path,
    params_path: Path,
    *,
    query_rows: int,
    dtype: torch.dtype,
):
    """Load a dense Llama checkpoint without embedding a context bound."""

    params = json.loads(params_path.read_text(encoding="utf-8"))
    params.pop("max_seq_len", None)
    params.pop("max_context_len", None)
    # ModelArgs requires these construction values, but no exported stage reads
    # them: RoPE, K/V storage, and Attention invocation count are host-owned.
    args = ModelArgs(
        max_seq_len=query_rows,
        max_context_len=query_rows,
        max_batch_size=1,
        use_kv_cache=False,
        generate_full_logits=True,
        **params,
    )
    model = construct_transformer(args).eval()
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=False
    )
    state_dict = normalize_checkpoint_keys(checkpoint.get("model", checkpoint))
    model.load_state_dict(state_dict, strict=True)
    return model.to(dtype=dtype)


def export(args: argparse.Namespace) -> Path:
    """Write a multi-method PTE or a bounded-context bundle and manifest."""

    export_started = time.perf_counter()
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    if args.backend == "qnn" and dtype == torch.float16 and not args.pre_attention_rope:
        raise ValueError(
            "QNN FP16 export requires --pre-attention-rope so Q/K layout and "
            "RoPE stay inside the delegated graph"
        )
    if args.layers_per_shard < 0:
        raise ValueError("layers_per_shard must be non-negative")
    state_scale = resolve_state_scale(args.backend, args.state_scale)
    prefill_query_rows = validate_prefill_query_rows(
        args.query_rows, args.prefill_query_rows
    )
    widths_by_query_rows = normalize_row_widths(
        prefill_query_rows, args.widths, args.widths_by_query_rows
    )
    model = load_model(
        Path(args.checkpoint),
        Path(args.params),
        query_rows=max(prefill_query_rows),
        dtype=dtype,
    )
    portfolio = build_composable_llama_portfolio(
        model,
        query_rows=args.query_rows,
        prefill_query_rows=prefill_query_rows,
        widths=args.widths,
        widths_by_query_rows=widths_by_query_rows,
        internal_query_tile_rows_by_query_rows=args.query_tiles_by_query_rows,
        dtype=dtype,
        state_scale=state_scale,
        exponent_floor=(QNN_EXPONENT_FLOOR if args.backend == "qnn" else None),
        pre_attention_rope=args.pre_attention_rope,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def lower(modules, example_inputs):
        return (
            export_qnn(
                dict(modules),
                dict(example_inputs),
                args.soc_model,
                caller_owned_io=True,
            )
            if args.backend == "qnn"
            else export_portable(dict(modules), dict(example_inputs))
        )

    def write_program(path: Path, modules, example_inputs) -> dict[str, object]:
        program_started = time.perf_counter()
        program = lower(modules, example_inputs)
        with path.open("wb") as file:
            program.write_to_file(file)
        return {
            "path": path.name,
            "methods": sorted(modules),
            "static_graph_count": len(modules),
            "pte_bytes": path.stat().st_size,
            "export_wall_seconds": round(time.perf_counter() - program_started, 6),
        }

    params = model.params
    layer_shard_manifest = []
    pte_files = []
    if args.layers_per_shard:
        core_modules, core_inputs, layer_shards = partition_composable_llama_portfolio(
            portfolio,
            layers=params.n_layers,
            layers_per_shard=args.layers_per_shard,
        )
        core_record = write_program(output, core_modules, core_inputs)
        pte_files.append({"role": "core", **core_record})
        for shard in layer_shards:
            shard_path = output.with_name(
                f"{output.stem}.layers_{shard.first_layer:03d}_"
                f"{shard.last_layer_exclusive - 1:03d}{output.suffix}"
            )
            shard_record = write_program(
                shard_path, shard.modules, shard.example_inputs
            )
            shard_metadata = {
                "first_layer": shard.first_layer,
                "last_layer_exclusive": shard.last_layer_exclusive,
                **shard_record,
            }
            layer_shard_manifest.append(shard_metadata)
            pte_files.append({"role": "layer_shard", **shard_metadata})
        core_methods = sorted(core_modules)
        pte_layout = "layer_sharded"
    else:
        core_record = write_program(output, portfolio.modules, portfolio.example_inputs)
        pte_files.append({"role": "core", **core_record})
        core_methods = sorted(portfolio.modules)
        pte_layout = "single"

    manifest = {
        "schema_version": 1,
        "backend": args.backend,
        "soc_model": args.soc_model if args.backend == "qnn" else None,
        "dtype": args.dtype,
        "method_io_dtype": args.dtype,
        "delegate_compute_dtype": "fp16" if args.backend == "qnn" else args.dtype,
        "query_rows": args.query_rows,
        # Keep the legacy field bound to the primary, unsuffixed R shape.
        "widths": list(widths_by_query_rows[args.query_rows]),
        "prefill_query_rows": list(prefill_query_rows),
        "attention_widths_by_query_rows": {
            str(rows): list(widths_by_query_rows[rows])
            for rows in prefill_query_rows
        },
        "internal_query_tile_rows_by_query_rows": {
            str(rows): args.query_tiles_by_query_rows.get(rows, 0)
            for rows in prefill_query_rows
        },
        "stage_method_shape_suffix": "_r<R>",
        "state_scale": state_scale,
        "exponent_floor": (QNN_EXPONENT_FLOOR if args.backend == "qnn" else None),
        "dim": params.dim,
        "layers": params.n_layers,
        "query_heads": params.n_heads,
        "kv_heads": params.n_kv_heads,
        "head_dim": params.head_dim,
        "hidden_dim": params.hidden_dim,
        "vocab_size": params.vocab_size,
        "rope_theta": params.rope_freq_base,
        "rope_style": "hf" if params.use_hf_rope else "llama",
        "pre_attention_abi": (
            "separate_q_k_v_with_rope"
            if args.pre_attention_rope
            else "packed_q_k_v_host_rope"
        ),
        "qkv_output_layout": (
            "q_bhqd_k_bhdc_v_bhcd" if args.pre_attention_rope else "packed_q_k_v"
        ),
        "methods": sorted(portfolio.modules),
        "static_graph_count": len(portfolio.modules),
        "pte_file_count": len(pte_files),
        "total_pte_bytes": sum(record["pte_bytes"] for record in pte_files),
        "export_wall_seconds": round(time.perf_counter() - export_started, 6),
        "pte_files": pte_files,
        "pte_layout": pte_layout,
        "core_path": output.name,
        "core_methods": core_methods,
        "layers_per_shard": args.layers_per_shard or None,
        "layer_shards": layer_shard_manifest,
        # The host supplies position-dependent RoPE and repeatedly invokes the
        # finite Attention methods, so no method embeds a context-length bound.
        "max_context_len": None,
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=("portable", "qnn"), default="portable")
    parser.add_argument("--soc-model", default="SM8550")
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp32")
    parser.add_argument("--query-rows", type=int, default=1)
    parser.add_argument(
        "--prefill-query-rows",
        type=int,
        nargs="+",
        help=(
            "Optional static Prefill row shapes. The primary --query-rows shape "
            "keeps legacy method names; additional power-of-two shapes use _r<R>."
        ),
    )
    parser.add_argument("--widths", type=int, nargs="+", required=True)
    parser.add_argument(
        "--row-widths",
        nargs="+",
        help=(
            "Optional R:C1,C2,... Attention-width overrides. Unspecified "
            "Prefill Query-row shapes use --widths."
        ),
    )
    parser.add_argument(
        "--row-query-tiles",
        nargs="+",
        help=(
            "Optional R:Q internal Query-tile overrides. For example, "
            "128:32 keeps the external R=128 ABI while statically unrolling "
            "Q32 Attention operations inside each graph."
        ),
    )
    parser.add_argument("--state-scale", type=float)
    parser.add_argument(
        "--pre-attention-rope",
        action="store_true",
        help=(
            "Export per-layer pre(hidden, cos, sin)->Q,K,V methods so Q/K "
            "normalization, layout conversion, and RoPE remain in the graph."
        ),
    )
    parser.add_argument(
        "--layers-per-shard",
        type=int,
        default=0,
        help=(
            "Compile each contiguous QKV/post layer group into a separate PTE. "
            "Zero preserves the single-PTE layout."
        ),
    )
    args = parser.parse_args(argv)
    try:
        args.prefill_query_rows = validate_prefill_query_rows(
            args.query_rows, args.prefill_query_rows
        )
        args.widths_by_query_rows = parse_row_width_specs(args.row_widths)
        normalize_row_widths(
            args.prefill_query_rows,
            args.widths,
            args.widths_by_query_rows,
        )
        args.query_tiles_by_query_rows = parse_row_query_tile_specs(
            args.row_query_tiles
        )
        unknown_rows = set(args.query_tiles_by_query_rows) - set(
            args.prefill_query_rows
        )
        if unknown_rows:
            raise ValueError(
                "internal Query tiles were provided for unknown Query rows: "
                f"{sorted(unknown_rows)}"
            )
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv: Sequence[str] | None = None) -> None:
    output = export(parse_args(argv))
    print(f"Saved composable Llama program to {output}")


if __name__ == "__main__":
    main()
