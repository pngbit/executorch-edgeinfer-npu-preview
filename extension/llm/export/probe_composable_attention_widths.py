# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Probe power-of-two composable Attention widths independently.

Each width is exported to its own PTE so one unsupported or resource-limited
shape cannot discard the successful prefix. An optional device command may
then validate loading and execution. All commands, timings, sizes, and failure
provenance are persisted in one JSON manifest.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence


RESOURCE_FAILURE_PATTERNS = (
    "vtcm allocation failed",
    "vtcm allocation failure",
    "insufficient vtcm",
    "out of memory",
    "out-of-memory",
    "memory allocation",
    "resource exhausted",
    "failed to allocate",
    "allocation failed",
    "context binary size",
)

SPILL_FILL_PATTERN = re.compile(
    r"\b(?P<kind>spill|fill)_bytes\s*=\s*(?P<bytes>[0-9]+)\b",
    re.IGNORECASE,
)


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def power_of_two_widths(start_width: int, max_width: int) -> tuple[int, ...]:
    if not is_power_of_two(start_width):
        raise ValueError("start width must be a positive power of two")
    if max_width < start_width:
        raise ValueError("max width must be no smaller than start width")
    result = []
    width = start_width
    while width <= max_width:
        result.append(width)
        width *= 2
    return tuple(result)


def classify_failure(text: str) -> str:
    lowered = text.lower()
    if any(int(match.group("bytes")) > 0 for match in SPILL_FILL_PATTERN.finditer(text)):
        return "resource_limit"
    if any(pattern in lowered for pattern in RESOURCE_FAILURE_PATTERNS):
        return "resource_limit"
    return "unsupported_or_other"


def spill_fill_bytes(text: str) -> dict[str, int]:
    """Return maximum QNN spill/fill bytes; zero-valued records are normal."""

    result = {"spill_bytes": 0, "fill_bytes": 0}
    for match in SPILL_FILL_PATTERN.finditer(text):
        key = f'{match.group("kind").lower()}_bytes'
        result[key] = max(result[key], int(match.group("bytes")))
    return result


def run_command(command: Sequence[str], stdout_path: Path, stderr_path: Path) -> dict:
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    wall_seconds = time.perf_counter() - started
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "wall_seconds": round(wall_seconds, 6),
        "stdout_file": stdout_path.name,
        "stderr_file": stderr_path.name,
        "combined_output": completed.stdout + "\n" + completed.stderr,
    }


def export_command(
    args: argparse.Namespace, query_rows: int, width: int, output: Path
) -> list[str]:
    command = [
        args.python,
        "-m",
        "executorch.extension.llm.export.export_composable_attention",
        "--output",
        str(output),
        "--backend",
        args.backend,
        "--soc-model",
        args.soc_model,
        "--dtype",
        args.dtype,
        "--batch-size",
        str(args.batch_size),
        "--query-heads",
        str(args.query_heads),
        "--kv-heads",
        str(args.kv_heads),
        "--query-rows",
        str(query_rows),
        "--head-dim",
        str(args.head_dim),
        "--widths",
        str(width),
    ]
    if args.softmax_scale is not None:
        command.extend(("--softmax-scale", str(args.softmax_scale)))
    if args.state_scale is not None:
        command.extend(("--state-scale", str(args.state_scale)))
    if args.exponent_floor is not None:
        command.extend(("--exponent-floor", str(args.exponent_floor)))
    if args.vulkan_force_fp16:
        command.append("--vulkan-force-fp16")
    return command


def probe(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    successful_widths_by_query_rows = {}
    stop_reason_by_query_rows = {}

    for query_rows in args.query_rows:
        successful_widths = []
        stop_reason = "max_width_reached"
        row_prefix = f"r{query_rows}"
        for width in power_of_two_widths(args.start_width, args.max_width):
            pte = (
                output_dir
                / f"attention_{args.backend}_{row_prefix}_c{width}_{args.dtype}.pte"
            )
            export_result = run_command(
                export_command(args, query_rows, width, pte),
                output_dir / f"{row_prefix}_c{width}.export.stdout",
                output_dir / f"{row_prefix}_c{width}.export.stderr",
            )
            combined_output = export_result.pop("combined_output")
            spill_fill = spill_fill_bytes(combined_output)
            record = {
                "query_rows": query_rows,
                "width": width,
                "export": export_result,
                **spill_fill,
            }
            if any(spill_fill.values()):
                record.update(
                    {
                        "status": "resource_limited",
                        "failure_class": "resource_limit",
                        "failure_excerpt": combined_output[-2000:],
                    }
                )
                results.append(record)
                stop_reason = f"nonzero_spill_fill_at_width_{width}"
                break
            if export_result["returncode"] != 0 or not pte.is_file():
                failure_class = classify_failure(combined_output)
                record.update(
                    {
                        "status": "export_failed",
                        "failure_class": failure_class,
                        "failure_excerpt": combined_output[-2000:],
                    }
                )
                results.append(record)
                if failure_class == "resource_limit":
                    stop_reason = f"resource_limit_at_width_{width}"
                    break
                continue

            record["pte_file"] = pte.name
            record["pte_bytes"] = pte.stat().st_size
            manifest_path = pte.with_suffix(pte.suffix + ".json")
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                record["export_manifest_file"] = manifest_path.name
                record["static_graph_count"] = manifest.get("static_graph_count")
                record["delegate_segment_count"] = manifest.get(
                    "delegate_segment_count"
                )

            if args.device_probe_command:
                rendered = args.device_probe_command.format(
                    pte=shlex.quote(str(pte)),
                    manifest=shlex.quote(str(manifest_path)),
                    query_rows=query_rows,
                    width=width,
                )
                device_result = run_command(
                    ["bash", "-lc", rendered],
                    output_dir / f"{row_prefix}_c{width}.device.stdout",
                    output_dir / f"{row_prefix}_c{width}.device.stderr",
                )
                device_output = device_result.pop("combined_output")
                record["device_probe"] = device_result
                if device_result["returncode"] != 0:
                    failure_class = classify_failure(device_output)
                    record.update(
                        {
                            "status": "device_failed",
                            "failure_class": failure_class,
                            "failure_excerpt": device_output[-2000:],
                        }
                    )
                    results.append(record)
                    if failure_class == "resource_limit":
                        stop_reason = f"resource_limit_at_width_{width}"
                        break
                    continue

            record["status"] = "passed"
            successful_widths.append(width)
            results.append(record)
        successful_widths_by_query_rows[str(query_rows)] = successful_widths
        stop_reason_by_query_rows[str(query_rows)] = stop_reason

    manifest = {
        "schema_version": 2,
        "backend": args.backend,
        "soc_model": args.soc_model,
        "dtype": args.dtype,
        "query_rows": args.query_rows,
        "start_width": args.start_width,
        "max_width": args.max_width,
        "successful_widths_by_query_rows": successful_widths_by_query_rows,
        "stop_reason_by_query_rows": stop_reason_by_query_rows,
        "results": results,
    }
    if len(args.query_rows) == 1:
        row_key = str(args.query_rows[0])
        manifest["successful_widths"] = successful_widths_by_query_rows[row_key]
        manifest["stop_reason"] = stop_reason_by_query_rows[row_key]
    manifest_path = output_dir / "width_probe_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--backend",
        choices=("portable", "qnn", "qnn-gpu", "xnnpack", "vulkan"),
        required=True,
    )
    parser.add_argument("--soc-model", default="SM8550")
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--query-heads", type=int, required=True)
    parser.add_argument("--kv-heads", type=int, required=True)
    parser.add_argument("--query-rows", type=int, nargs="+", default=(1,))
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--start-width", type=int, default=1)
    parser.add_argument("--max-width", type=int, required=True)
    parser.add_argument("--softmax-scale", type=float)
    parser.add_argument("--state-scale", type=float)
    parser.add_argument("--exponent-floor", type=float)
    parser.add_argument("--vulkan-force-fp16", action="store_true")
    parser.add_argument(
        "--device-probe-command",
        help="Optional shell template using {pte}, {manifest}, and {width}.",
    )
    args = parser.parse_args(argv)
    power_of_two_widths(args.start_width, args.max_width)
    if any(
        value <= 0
        for value in (args.batch_size, args.query_heads, args.kv_heads, args.head_dim)
    ):
        parser.error("attention dimensions must be positive")
    if any(not is_power_of_two(value) for value in args.query_rows):
        parser.error("query-row shapes must be positive powers of two")
    args.query_rows = tuple(sorted(set(args.query_rows)))
    return args


def main(argv: Sequence[str] | None = None) -> None:
    print(probe(parse_args(argv)))


if __name__ == "__main__":
    main()
