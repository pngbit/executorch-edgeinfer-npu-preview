# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export arbitrary-context Attention as a finite multi-method PTE."""

from __future__ import annotations

import argparse
import json
import math
import time

from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import torch
from executorch.exir import ExecutorchBackendConfig, to_edge
from executorch.exir._warnings import experimental
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from executorch.extension.llm.modules.composable_attention import (
    build_static_attention_portfolio,
    static_attention_method_name,
)
from torch import nn


ExampleInputs = Dict[str, Tuple[torch.Tensor, ...]]
QNN_EXPONENT_FLOOR = -32.0
QNN_STATE_SCALE = 1024.0


def parse_row_width_specs(values: Sequence[str] | None) -> Dict[int, Tuple[int, ...]]:
    """Parse ``R:C1,C2,...`` entries used by non-rectangular portfolios."""

    result: Dict[int, Tuple[int, ...]] = {}
    for value in values or ():
        row_text, separator, width_text = value.partition(":")
        try:
            row = int(row_text)
            widths = tuple(sorted({int(item) for item in width_text.split(",")}))
        except ValueError as error:
            raise ValueError(f"invalid row-width entry: {value}") from error
        if not separator or row <= 0 or not widths or widths[0] <= 0:
            raise ValueError(f"invalid row-width entry: {value}")
        if row in result:
            raise ValueError(f"duplicate row-width entry for R={row}")
        result[row] = widths
    return result


def normalize_row_widths(
    query_rows: Iterable[int],
    widths: Iterable[int],
    widths_by_query_rows: Mapping[int, Iterable[int]] | None = None,
) -> Dict[int, Tuple[int, ...]]:
    """Return one explicit width portfolio for each Query-row shape."""

    rows = tuple(sorted({int(row) for row in query_rows}))
    common_widths = tuple(sorted({int(width) for width in widths}))
    if not rows or rows[0] <= 0:
        raise ValueError("at least one positive query-row shape is required")
    if not common_widths or common_widths[0] <= 0:
        raise ValueError("at least one positive Attention width is required")

    result = {row: common_widths for row in rows}
    if widths_by_query_rows is None:
        return result
    unknown_rows = set(widths_by_query_rows) - set(rows)
    if unknown_rows:
        raise ValueError(
            f"Attention widths were provided for unknown Query rows: "
            f"{sorted(unknown_rows)}"
        )
    for row, requested_widths in widths_by_query_rows.items():
        normalized = tuple(sorted({int(width) for width in requested_widths}))
        if not normalized or normalized[0] <= 0:
            raise ValueError(
                f"Query rows {row} requires at least one positive Attention width"
            )
        result[row] = normalized
    return result


def resolve_state_scale(backend: str, state_scale: float | None) -> float:
    """Choose a numerically safe default for the delegated state dtype."""

    if state_scale is not None:
        return state_scale
    return QNN_STATE_SCALE if backend == "qnn" else 1.0


@experimental(
    "Composable static Attention export is experimental and may change without notice."
)
def build_portfolios(
    *,
    batch_size: int,
    query_heads: int,
    kv_heads: int,
    query_rows: Iterable[int],
    head_dim: int,
    widths: Iterable[int],
    widths_by_query_rows: Mapping[int, Iterable[int]] | None = None,
    softmax_scale: float,
    state_scale: float,
    exponent_floor: float | None,
    assume_nonempty: bool = False,
    internal_query_tile_rows: int = 0,
    dtype: torch.dtype,
) -> Tuple[Dict[str, nn.Module], ExampleInputs]:
    """Combine fixed-width portfolios for multiple query-row shapes.

    .. warning::
        This API is experimental and may change or be removed without notice.
    """

    modules: Dict[str, nn.Module] = {}
    example_inputs: ExampleInputs = {}
    row_widths = normalize_row_widths(
        query_rows, widths, widths_by_query_rows
    )
    for row, normalized_widths in row_widths.items():
        portfolio = build_static_attention_portfolio(
            batch_size=batch_size,
            query_heads=query_heads,
            kv_heads=kv_heads,
            query_rows=row,
            head_dim=head_dim,
            widths=normalized_widths,
            softmax_scale=softmax_scale,
            state_scale=state_scale,
            exponent_floor=exponent_floor,
            assume_nonempty=assume_nonempty,
            internal_query_tile_rows=internal_query_tile_rows,
            dtype=dtype,
        )
        modules.update(portfolio.modules)
        example_inputs.update(portfolio.example_inputs)
    return modules, example_inputs


@experimental(
    "Composable static Attention export is experimental and may change without notice."
)
def export_portable(modules: Dict[str, nn.Module], example_inputs: ExampleInputs):
    """Export every static method without delegating to a backend.

    .. warning::
        This API is experimental and may change or be removed without notice.
    """

    exported_programs = {
        name: torch.export.export(modules[name], example_inputs[name], strict=True)
        for name in modules
    }
    return to_edge(exported_programs).to_executorch(ExecutorchBackendConfig())


@experimental(
    "Composable static Attention export is experimental and may change without notice."
)
def export_qnn(
    modules: Dict[str, nn.Module],
    example_inputs: ExampleInputs,
    soc_model: str,
    *,
    use_fp16: bool | Mapping[str, bool] = True,
    caller_owned_io: bool = True,
    use_weight_sharing: bool = False,
):
    """Lower every static method to the Qualcomm QNN backend.

    .. warning::
        This API is experimental and may change or be removed without notice.
    """

    from executorch.backends.qualcomm.utils.utils import (
        generate_htp_compiler_spec,
        generate_qnn_executorch_compiler_spec,
        get_soc_to_chipset_map,
        to_edge_transform_and_lower_to_qnn,
    )

    soc_to_chipset = get_soc_to_chipset_map()
    if soc_model not in soc_to_chipset:
        raise ValueError(f"unsupported Qualcomm SoC: {soc_model}")

    def compiler_spec(fp16: bool):
        return generate_qnn_executorch_compiler_spec(
            soc_model=soc_to_chipset[soc_model],
            backend_options=generate_htp_compiler_spec(
                use_fp16=fp16, use_weight_sharing=use_weight_sharing
            ),
            shared_buffer=caller_owned_io,
        )

    if isinstance(use_fp16, bool):
        compiler_specs = compiler_spec(use_fp16)
    else:
        if set(use_fp16) != set(modules):
            raise ValueError("per-method precision keys must match module names")
        compiler_specs = {name: compiler_spec(bool(use_fp16[name])) for name in modules}
    edge_program = to_edge_transform_and_lower_to_qnn(
        module=modules,
        inputs=example_inputs,
        compiler_specs=compiler_specs,
    )
    return edge_program.to_executorch(
        ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=not caller_owned_io,
                alloc_graph_output=not caller_owned_io,
            )
        )
    )


@experimental(
    "Composable static Attention export is experimental and may change without notice."
)
def export(args: argparse.Namespace) -> Path:
    """Build and save a PTE plus its host-runtime manifest.

    .. warning::
        This API is experimental and may change or be removed without notice.
    """

    export_started = time.perf_counter()
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    if args.backend == "qnn" and dtype != torch.float16:
        raise ValueError("the QNN portfolio path requires --dtype fp16")
    softmax_scale = (
        args.softmax_scale
        if args.softmax_scale is not None
        else 1.0 / math.sqrt(args.head_dim)
    )
    state_scale = resolve_state_scale(args.backend, args.state_scale)
    modules, example_inputs = build_portfolios(
        batch_size=args.batch_size,
        query_heads=args.query_heads,
        kv_heads=args.kv_heads,
        query_rows=args.query_rows,
        head_dim=args.head_dim,
        widths=args.widths,
        widths_by_query_rows=args.widths_by_query_rows,
        softmax_scale=softmax_scale,
        state_scale=state_scale,
        exponent_floor=(QNN_EXPONENT_FLOOR if args.backend == "qnn" else None),
        assume_nonempty=args.assume_nonempty,
        internal_query_tile_rows=args.internal_query_tile_rows,
        dtype=dtype,
    )
    program = (
        export_qnn(modules, example_inputs, args.soc_model)
        if args.backend == "qnn"
        else export_portable(modules, example_inputs)
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file:
        program.write_to_file(file)

    row_widths = normalize_row_widths(
        args.query_rows, args.widths, args.widths_by_query_rows
    )
    query_rows = list(row_widths)
    widths = sorted({width for values in row_widths.values() for width in values})
    graph_portfolio = [
        {
            "method": static_attention_method_name(
                kind, row, width, args.assume_nonempty
            ),
            "kind": kind,
            "query_rows": row,
            "kv_width": width,
            "internal_query_tile_rows": args.internal_query_tile_rows,
        }
        for kind in ("first", "merge")
        for row in query_rows
        for width in row_widths[row]
    ]
    manifest = {
        "schema_version": 1,
        "backend": args.backend,
        "soc_model": args.soc_model if args.backend == "qnn" else None,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "query_heads": args.query_heads,
        "kv_heads": args.kv_heads,
        "query_rows": query_rows,
        "head_dim": args.head_dim,
        "widths": widths,
        "widths_by_query_rows": {
            str(row): list(row_widths[row]) for row in query_rows
        },
        "softmax_scale": softmax_scale,
        "state_scale": state_scale,
        "exponent_floor": (QNN_EXPONENT_FLOOR if args.backend == "qnn" else None),
        "assume_nonempty": args.assume_nonempty,
        "internal_query_tile_rows": args.internal_query_tile_rows,
        "mask_contract": (
            "every_query_row_visible_in_every_block"
            if args.assume_nonempty
            else "fully_masked_rows_supported"
        ),
        "methods": sorted(modules),
        "static_graph_count": len(modules),
        "pte_file_count": 1,
        "pte_bytes": output.stat().st_size,
        "export_wall_seconds": round(time.perf_counter() - export_started, 6),
        "graph_portfolio": graph_portfolio,
        # The runtime repeats finite-width merge methods; this is intentionally
        # not a compiled-context limit.
        "max_context_len": None,
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=("portable", "qnn"), default="portable")
    parser.add_argument("--soc-model", default="SM8550")
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--query-heads", type=int, required=True)
    parser.add_argument("--kv-heads", type=int, required=True)
    parser.add_argument("--query-rows", type=int, nargs="+", default=(1,))
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--widths", type=int, nargs="+", required=True)
    parser.add_argument(
        "--row-widths",
        nargs="+",
        help=(
            "Optional R:C1,C2,... overrides. Unspecified Query-row shapes use "
            "--widths."
        ),
    )
    parser.add_argument("--softmax-scale", type=float)
    parser.add_argument("--state-scale", type=float)
    parser.add_argument(
        "--internal-query-tile-rows",
        type=int,
        default=0,
        help=(
            "Statically split Query rows inside each method while preserving "
            "one backend graph invocation; zero disables internal tiling."
        ),
    )
    parser.add_argument(
        "--assume-nonempty",
        action="store_true",
        help=(
            "Export a Decode-only graph that requires every executed block "
            "to contain a visible key for its query row."
        ),
    )
    args = parser.parse_args(argv)
    try:
        args.widths_by_query_rows = parse_row_width_specs(args.row_widths)
        normalize_row_widths(
            args.query_rows, args.widths, args.widths_by_query_rows
        )
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv: Sequence[str] | None = None) -> None:
    output_path = export(parse_args(argv))
    print(f"Saved composable Attention portfolio to {output_path}")


if __name__ == "__main__":
    main()
