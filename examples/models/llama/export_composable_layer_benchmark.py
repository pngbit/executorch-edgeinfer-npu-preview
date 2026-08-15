# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export one same-weight decoder layer for long-context benchmarking."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from executorch.examples.models.llama.composable_transformer import (
    ComposableLlamaFullLayer,
    ComposableLlamaPostAttention,
    ComposableLlamaPreAttention,
    composable_llama_full_layer_method_name,
    composable_llama_method_name,
)
from executorch.examples.models.llama.export_composable_llama import load_model
from executorch.examples.models.llama.norm import RMSNorm, ScalelessRMSNorm
from executorch.extension.llm.export.export_composable_attention import export_qnn


class _NativeFp16RMSNorm(nn.Module):
    """QNN-friendly RMSNorm that never promotes its input to FP32."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        if isinstance(source, RMSNorm) and source.add_unit_offset:
            raise ValueError("unit-offset RMSNorm is not supported by this benchmark")
        dim = getattr(source, "dim", None)
        if dim is None:
            dim = source.normalized_shape[-1]
        self.eps = float(source.eps)
        weight = getattr(source, "weight", None)
        if weight is None:
            self.register_parameter("weight", None)
        else:
            if weight.dtype != torch.float16:
                raise ValueError("the QNN FP16 benchmark requires FP16 norm weights")
            self.weight = weight

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = value * torch.rsqrt(
            (value * value).mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized if self.weight is None else normalized * self.weight


def _replace_norms_with_native_fp16(module: nn.Module) -> int:
    """Replace only this benchmark's norms; the stock model remains unchanged."""

    replacements = 0
    for name, child in list(module.named_children()):
        if isinstance(child, (RMSNorm, ScalelessRMSNorm, torch.nn.RMSNorm)):
            setattr(module, name, _NativeFp16RMSNorm(child))
            replacements += 1
        else:
            replacements += _replace_norms_with_native_fp16(child)
    return replacements


def _delegation_record(program, method: str) -> dict[str, object]:
    plans = program.executorch_program.execution_plan
    if len(plans) != 1 or plans[0].name != method:
        raise RuntimeError(
            f"expected exactly one execution plan named {method}, got "
            f"{[plan.name for plan in plans]}"
        )
    plan = plans[0]
    delegate_ids = [delegate.id for delegate in plan.delegates]
    record = {
        "execution_plan": plan.name,
        "delegate_ids": delegate_ids,
        "delegate_count": len(delegate_ids),
        "remaining_operator_count": len(plan.operators),
        "fully_delegated_qnn": (
            delegate_ids == ["QnnBackend"] and len(plan.operators) == 0
        ),
    }
    if not record["fully_delegated_qnn"]:
        raise RuntimeError(f"method {method} is not fully delegated to QNN: {record}")
    return record


def _component_path(output: Path, component: str, export_all: bool) -> Path:
    if not export_all:
        return output
    return output.with_name(f"{output.stem}.{component}{output.suffix}")


def export(args: argparse.Namespace) -> list[Path]:
    overall_started = time.perf_counter()
    model = load_model(
        Path(args.checkpoint),
        Path(args.params),
        query_rows=args.query_rows,
        dtype=torch.float16,
    )
    params = model.params
    layer = model.layers[args.layer]
    norm_replacements = _replace_norms_with_native_fp16(layer)
    if norm_replacements != 4:
        raise RuntimeError(
            "Qwen3 benchmark expected four RMSNorm modules in one layer, "
            f"replaced {norm_replacements}"
        )

    hidden = torch.zeros(1, args.query_rows, params.dim, dtype=torch.float16)
    rope_width = params.head_dim if params.use_hf_rope else params.head_dim // 2
    freqs_cos = torch.ones(args.query_rows, rope_width, dtype=torch.float16)
    freqs_sin = torch.zeros(args.query_rows, rope_width, dtype=torch.float16)
    attention_output = torch.zeros(
        1, params.n_heads, args.query_rows, params.head_dim, dtype=torch.float16
    )
    pre_name = composable_llama_method_name("pre", args.layer)
    post_name = composable_llama_method_name("post", args.layer)
    full_name = composable_llama_full_layer_method_name(args.layer, args.capacity)
    components: dict[str, tuple[str, nn.Module, tuple[torch.Tensor, ...]]] = {
        "pre": (
            pre_name,
            ComposableLlamaPreAttention(layer).eval(),
            (hidden, freqs_cos, freqs_sin),
        ),
        "post": (
            post_name,
            ComposableLlamaPostAttention(layer).eval(),
            (hidden, attention_output),
        ),
        "full": (
            full_name,
            ComposableLlamaFullLayer(layer, args.capacity).eval(),
            (
                hidden,
                freqs_cos,
                freqs_sin,
                torch.zeros(1, 1, 1, args.capacity, dtype=torch.float16),
                torch.zeros(
                    1,
                    params.n_kv_heads,
                    params.head_dim,
                    args.capacity - 1,
                    dtype=torch.float16,
                ),
                torch.zeros(
                    1,
                    params.n_kv_heads,
                    args.capacity - 1,
                    params.head_dim,
                    dtype=torch.float16,
                ),
            ),
        ),
    }
    selected = tuple(components) if args.component == "all" else (args.component,)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    paths = []
    for component in selected:
        started = time.perf_counter()
        method, module, inputs = components[component]
        program = export_qnn(
            {method: module},
            {method: inputs},
            args.soc_model,
            caller_owned_io=True,
            use_weight_sharing=False,
        )
        delegation = _delegation_record(program, method)
        path = _component_path(output, component, args.component == "all")
        with path.open("wb") as file:
            program.write_to_file(file)
        record = {
            "component": component,
            "path": path.name,
            "method": method,
            "static_graph_count": 1,
            "pte_bytes": path.stat().st_size,
            "export_wall_seconds": round(time.perf_counter() - started, 6),
            **delegation,
        }
        path.with_suffix(path.suffix + ".json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        records.append(record)
        paths.append(path)

    manifest = {
        "schema_version": 2,
        "purpose": "same_weight_long_context_layer_benchmark",
        "backend": "qnn_htp",
        "soc_model": args.soc_model,
        "dtype": "fp16",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "params": str(Path(args.params).resolve()),
        "layer": args.layer,
        "logical_layer_weights": "one_exported_layer_reused",
        "independent_kv_per_logical_layer": True,
        "capacity": args.capacity,
        "block_width": args.block_width,
        "query_rows": args.query_rows,
        "dim": params.dim,
        "hidden_dim": params.hidden_dim,
        "query_heads": params.n_heads,
        "kv_heads": params.n_kv_heads,
        "head_dim": params.head_dim,
        "norm_implementation": "native_fp16_no_fp32_promotion",
        "norm_replacements": norm_replacements,
        "caller_owned_io": True,
        "qnn_weight_sharing": False,
        "components": records,
        "export_wall_seconds": round(time.perf_counter() - overall_started, 6),
    }
    output.with_suffix(output.suffix + ".bundle.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--soc-model", default="SM8650")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--capacity", type=int, default=40960)
    parser.add_argument("--block-width", type=int, default=1024)
    parser.add_argument("--query-rows", type=int, default=1)
    parser.add_argument(
        "--component", choices=("all", "pre", "post", "full"), default="all"
    )
    args = parser.parse_args(argv)
    if (
        args.layer < 0
        or args.capacity <= 1
        or args.block_width <= 0
        or args.query_rows <= 0
    ):
        parser.error("layer, capacity, block width, and query rows are invalid")
    if args.capacity % args.block_width:
        parser.error("capacity must be divisible by block width")
    if args.query_rows != 1 and args.component in ("all", "full"):
        parser.error("the fixed-capacity full graph currently requires query rows = 1")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    outputs = export(parse_args(argv))
    print("Saved same-weight layer benchmark programs: " + ", ".join(map(str, outputs)))


if __name__ == "__main__":
    main()
