# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import tempfile
import unittest

from pathlib import Path

import torch
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.extension.llm.export.export_composable_attention import (
    build_portfolios,
    export,
    normalize_row_widths,
    parse_args,
    parse_row_width_specs,
    resolve_state_scale,
)


class ExportComposableAttentionTest(unittest.TestCase):
    def test_builds_deduplicated_fixed_shape_methods(self) -> None:
        modules, inputs = build_portfolios(
            batch_size=1,
            query_heads=4,
            kv_heads=2,
            query_rows=(4, 1, 4),
            head_dim=8,
            widths=(width for width in (16, 8, 16)),
            softmax_scale=0.25,
            state_scale=64.0,
            exponent_floor=None,
            dtype=torch.float32,
        )

        expected = {
            f"attn_{kind}_r{rows}_c{width}"
            for kind in ("first", "merge")
            for rows in (1, 4)
            for width in (8, 16)
        }
        self.assertEqual(set(modules), expected)
        self.assertEqual(set(inputs), expected)
        self.assertEqual(inputs["attn_first_r4_c16"][0].shape, (1, 4, 4, 8))
        self.assertEqual(inputs["attn_first_r4_c16"][1].shape, (1, 2, 8, 16))

    def test_builds_nonrectangular_row_width_portfolio(self) -> None:
        modules, _ = build_portfolios(
            batch_size=1,
            query_heads=4,
            kv_heads=2,
            query_rows=(1, 4),
            head_dim=8,
            widths=(8, 16),
            widths_by_query_rows={4: (8,)},
            softmax_scale=0.25,
            state_scale=64.0,
            exponent_floor=None,
            dtype=torch.float32,
        )
        self.assertIn("attn_first_r1_c16", modules)
        self.assertIn("attn_first_r4_c8", modules)
        self.assertNotIn("attn_first_r4_c16", modules)

    def test_parses_and_normalizes_row_width_overrides(self) -> None:
        overrides = parse_row_width_specs(("4:8,4,16", "1:2,1"))
        self.assertEqual(overrides, {4: (4, 8, 16), 1: (1, 2)})
        self.assertEqual(
            normalize_row_widths((1, 4), (32,), overrides),
            {1: (1, 2), 4: (4, 8, 16)},
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_row_width_specs(("1:1", "1:2"))
        with self.assertRaisesRegex(ValueError, "unknown Query rows"):
            normalize_row_widths((1,), (1,), {2: (1,)})

    def test_exports_portable_pte_without_a_maximum_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "attention.pte"
            args = parse_args(
                [
                    "--output",
                    str(output),
                    "--backend",
                    "portable",
                    "--dtype",
                    "fp32",
                    "--query-heads",
                    "4",
                    "--kv-heads",
                    "2",
                    "--query-rows",
                    "1",
                    "4",
                    "--head-dim",
                    "8",
                    "--widths",
                    "8",
                    "16",
                ]
            )
            self.assertEqual(export(args), output)

            program = deserialize_pte_binary(output.read_bytes()).program
            method_names = [plan.name for plan in program.execution_plan]
            expected_names = sorted(
                f"attn_{kind}_r{rows}_c{width}"
                for kind in ("first", "merge")
                for rows in (1, 4)
                for width in (8, 16)
            )
            self.assertEqual(method_names, expected_names)

            manifest = json.loads(
                output.with_suffix(".pte.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["schema_version"], 1)
            self.assertIsNone(manifest["max_context_len"])
            self.assertEqual(manifest["methods"], expected_names)
            self.assertEqual(manifest["query_rows"], [1, 4])
            self.assertEqual(manifest["widths"], [8, 16])
            self.assertEqual(
                manifest["widths_by_query_rows"],
                {"1": [8, 16], "4": [8, 16]},
            )
            self.assertEqual(manifest["static_graph_count"], 8)
            self.assertEqual(manifest["pte_file_count"], 1)
            self.assertEqual(manifest["pte_bytes"], output.stat().st_size)
            self.assertGreater(manifest["export_wall_seconds"], 0.0)
            self.assertEqual(
                {entry["method"] for entry in manifest["graph_portfolio"]},
                set(expected_names),
            )
            self.assertIsNone(manifest["exponent_floor"])
            self.assertEqual(manifest["state_scale"], 1.0)

    def test_uses_backend_appropriate_default_state_scale(self) -> None:
        self.assertEqual(resolve_state_scale("portable", None), 1.0)
        self.assertEqual(resolve_state_scale("qnn", None), 1024.0)
        self.assertEqual(resolve_state_scale("qnn", 64.0), 64.0)

    def test_manifest_records_nonrectangular_row_widths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "attention.pte"
            args = parse_args(
                [
                    "--output",
                    str(output),
                    "--backend",
                    "portable",
                    "--dtype",
                    "fp32",
                    "--query-heads",
                    "4",
                    "--kv-heads",
                    "2",
                    "--query-rows",
                    "1",
                    "4",
                    "--head-dim",
                    "8",
                    "--widths",
                    "8",
                    "16",
                    "--row-widths",
                    "4:8",
                ]
            )
            self.assertEqual(export(args), output)
            manifest = json.loads(
                output.with_suffix(".pte.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["widths_by_query_rows"], {"1": [8, 16], "4": [8]}
            )
            self.assertEqual(manifest["static_graph_count"], 6)

    def test_records_internal_query_tiling(self) -> None:
        modules, _ = build_portfolios(
            batch_size=1,
            query_heads=4,
            kv_heads=2,
            query_rows=(8,),
            head_dim=8,
            widths=(16,),
            softmax_scale=0.25,
            state_scale=64.0,
            exponent_floor=None,
            internal_query_tile_rows=2,
            dtype=torch.float32,
        )
        for module in modules.values():
            self.assertEqual(module.internal_query_tile_rows, 2)

    def test_nonempty_export_contract_is_explicit(self) -> None:
        modules, _ = build_portfolios(
            batch_size=1,
            query_heads=4,
            kv_heads=2,
            query_rows=(1,),
            head_dim=8,
            widths=(8,),
            softmax_scale=0.25,
            state_scale=64.0,
            exponent_floor=None,
            assume_nonempty=True,
            dtype=torch.float32,
        )
        self.assertEqual(
            set(modules),
            {"attn_first_nonempty_r1_c8", "attn_merge_nonempty_r1_c8"},
        )

    def test_rejects_non_fp16_qnn_before_loading_backend(self) -> None:
        args = parse_args(
            [
                "--output",
                "unused.pte",
                "--backend",
                "qnn",
                "--dtype",
                "fp32",
                "--query-heads",
                "4",
                "--kv-heads",
                "2",
                "--head-dim",
                "8",
                "--widths",
                "8",
            ]
        )
        with self.assertRaisesRegex(ValueError, "requires --dtype fp16"):
            export(args)


if __name__ == "__main__":
    unittest.main()
