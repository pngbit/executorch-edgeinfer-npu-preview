# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from executorch.examples.models.llama.composable_transformer import (
    apply_composable_rotary_emb,
    build_composable_llama_portfolio,
    composable_llama_method_name,
    ComposableLlamaPreAttention,
    ComposableLlamaPostAttention,
    ComposableLlamaQKV,
    execute_composable_llama,
    unpack_composable_qkv,
)
from executorch.examples.models.llama.export_composable_llama import (
    export as export_composable_llama,
    normalize_checkpoint_keys,
    parse_args as parse_composable_llama_args,
    partition_composable_llama_portfolio,
)
from executorch.examples.models.llama.llama_transformer import construct_transformer
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.rope import apply_rotary_emb
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.extension.llm.export.export_composable_attention import export_portable
from executorch.extension.llm.modules.composable_attention import (
    StaticAttentionGraphCost,
)


class TestComposableTransformer(unittest.TestCase):
    def test_normalizes_uniform_torch_compile_checkpoint_prefix(self) -> None:
        tensor = torch.tensor([1.0])
        prefixed = {"_orig_mod.layer.weight": tensor}
        normalized = normalize_checkpoint_keys(prefixed)
        self.assertEqual(list(normalized), ["layer.weight"])
        self.assertIs(normalized["layer.weight"], tensor)

        mixed = {"_orig_mod.layer.weight": tensor, "other": tensor}
        self.assertIs(normalize_checkpoint_keys(mixed), mixed)

    def _model_args(self, max_seq_len: int) -> ModelArgs:
        return ModelArgs(
            dim=32,
            hidden_dim=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            vocab_size=64,
            multiple_of=8,
            max_seq_len=max_seq_len,
            max_context_len=max_seq_len,
            max_batch_size=1,
            use_kv_cache=False,
            generate_full_logits=True,
        )

    def _qwen_like_args(self, max_seq_len: int) -> ModelArgs:
        return ModelArgs(
            dim=32,
            hidden_dim=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            head_dim=12,
            vocab_size=64,
            multiple_of=8,
            max_seq_len=max_seq_len,
            max_context_len=max_seq_len,
            max_batch_size=1,
            use_kv_cache=False,
            generate_full_logits=True,
            use_qk_norm=True,
            qk_norm_before_rope=True,
            use_hf_rope=True,
            rope_theta=1_000_000.0,
        )

    def _biased_odd_gqa_args(self, max_seq_len: int) -> ModelArgs:
        return ModelArgs(
            dim=72,
            hidden_dim=144,
            n_layers=2,
            n_heads=9,
            n_kv_heads=3,
            head_dim=8,
            vocab_size=64,
            multiple_of=8,
            max_seq_len=max_seq_len,
            max_context_len=max_seq_len,
            max_batch_size=1,
            use_kv_cache=False,
            generate_full_logits=True,
            attention_qkv_bias=True,
            rope_theta=100_000.0,
        )

    def test_packed_qkv_and_host_rope_match_layer_math(self) -> None:
        torch.manual_seed(17)
        model = construct_transformer(self._model_args(3)).eval()
        layer = model.layers[0]
        hidden = torch.randn(1, 3, model.params.dim)

        packed = ComposableLlamaQKV(layer)(hidden)
        q, k, v = unpack_composable_qkv(
            packed,
            model.params.n_heads,
            model.params.n_kv_heads,
            model.params.head_dim,
        )
        normalized = layer.attention_norm(hidden)
        expected_q = layer.attention.wq(normalized).view(1, 3, 4, 8)
        expected_k = layer.attention.wk(normalized).view(1, 3, 2, 8)
        expected_v = layer.attention.wv(normalized).view(1, 3, 2, 8)
        torch.testing.assert_close(q, expected_q.transpose(1, 2))
        torch.testing.assert_close(k, expected_k.transpose(1, 2).transpose(2, 3))
        torch.testing.assert_close(v, expected_v.transpose(1, 2))

        freqs_cos, freqs_sin = model.rope.precompute_freqs_cis(
            model.params.head_dim,
            3,
            model.params.rope_freq_base,
        )
        rotated_q, rotated_k = apply_composable_rotary_emb(q, k, freqs_cos, freqs_sin)
        expected_q, expected_k = apply_rotary_emb(
            expected_q, expected_k, freqs_cos, freqs_sin
        )
        torch.testing.assert_close(rotated_q, expected_q.transpose(1, 2))
        torch.testing.assert_close(
            rotated_k, expected_k.transpose(1, 2).transpose(2, 3)
        )

    def test_pre_attention_graph_matches_qwen_qk_norm_and_hf_rope(self) -> None:
        torch.manual_seed(19)
        model = construct_transformer(self._qwen_like_args(3)).eval()
        layer = model.layers[0]
        hidden = torch.randn(1, 3, model.params.dim)
        freqs_cos, freqs_sin = model.rope.precompute_freqs_cis(
            model.params.head_dim,
            3,
            model.params.rope_freq_base,
        )

        query, key, value = ComposableLlamaPreAttention(layer)(
            hidden, freqs_cos, freqs_sin
        )
        packed = ComposableLlamaQKV(layer)(hidden)
        expected_q, expected_k, expected_v = unpack_composable_qkv(
            packed,
            model.params.n_heads,
            model.params.n_kv_heads,
            model.params.head_dim,
        )
        expected_q, expected_k = apply_composable_rotary_emb(
            expected_q, expected_k, freqs_cos, freqs_sin
        )
        torch.testing.assert_close(query, expected_q)
        torch.testing.assert_close(key, expected_k)
        torch.testing.assert_close(value, expected_v)

    def test_pre_attention_portfolio_preserves_legacy_qkv_default(self) -> None:
        model = construct_transformer(self._qwen_like_args(1)).eval()
        legacy = build_composable_llama_portfolio(
            model,
            query_rows=1,
            widths=(4, 8),
            dtype=torch.float32,
        )
        graph_rope = build_composable_llama_portfolio(
            model,
            query_rows=1,
            widths=(4, 8),
            dtype=torch.float32,
            pre_attention_rope=True,
        )
        self.assertIn("llama_layer_0_qkv", legacy.modules)
        self.assertNotIn("llama_layer_0_pre", legacy.modules)
        self.assertIn("llama_layer_0_pre", graph_rope.modules)
        self.assertNotIn("llama_layer_0_qkv", graph_rope.modules)
        self.assertEqual(len(graph_rope.example_inputs["llama_layer_0_pre"]), 3)

    def test_multi_r_portfolio_preserves_primary_names_and_suffixes(self) -> None:
        model = construct_transformer(self._model_args(4)).eval()
        portfolio = build_composable_llama_portfolio(
            model,
            query_rows=4,
            prefill_query_rows=(1, 2, 4),
            widths=(4,),
            dtype=torch.float32,
        )
        self.assertIn("llama_embedding", portfolio.modules)
        self.assertIn("llama_layer_0_qkv", portfolio.modules)
        self.assertIn("llama_layer_0_post", portfolio.modules)
        self.assertIn("llama_output", portfolio.modules)
        self.assertIn("attn_first_r4_c4", portfolio.modules)
        for rows in (1, 2):
            self.assertIn(f"llama_embedding_r{rows}", portfolio.modules)
            self.assertIn(f"llama_layer_0_qkv_r{rows}", portfolio.modules)
            self.assertIn(f"llama_layer_0_post_r{rows}", portfolio.modules)
            self.assertIn(f"llama_output_r{rows}", portfolio.modules)
            self.assertIn(f"attn_first_r{rows}_c4", portfolio.modules)
        self.assertEqual(portfolio.example_inputs["llama_embedding_r1"][0].shape, (1, 1))
        self.assertEqual(portfolio.example_inputs["llama_embedding"][0].shape, (1, 4))

    def test_multi_r_portfolio_applies_row_specific_internal_query_tiles(self) -> None:
        model = construct_transformer(self._model_args(4)).eval()
        portfolio = build_composable_llama_portfolio(
            model,
            query_rows=4,
            prefill_query_rows=(2, 4),
            widths=(4,),
            internal_query_tile_rows_by_query_rows={4: 2},
            dtype=torch.float32,
        )
        self.assertEqual(
            portfolio.modules["attn_first_r4_c4"].internal_query_tile_rows,
            2,
        )
        self.assertEqual(
            portfolio.modules["attn_first_r2_c4"].internal_query_tile_rows,
            0,
        )
        with self.assertRaisesRegex(ValueError, "unknown Query rows"):
            build_composable_llama_portfolio(
                model,
                query_rows=4,
                prefill_query_rows=(2, 4),
                widths=(4,),
                internal_query_tile_rows_by_query_rows={8: 2},
                dtype=torch.float32,
            )

    def test_stage_method_names_support_explicit_row_suffixes(self) -> None:
        self.assertEqual(
            composable_llama_method_name("embedding", query_rows=8),
            "llama_embedding_r8",
        )
        self.assertEqual(
            composable_llama_method_name("pre", 3, query_rows=8),
            "llama_layer_3_pre_r8",
        )
        with self.assertRaisesRegex(ValueError, "query rows must be positive"):
            composable_llama_method_name("output", query_rows=0)

    def test_execution_exceeds_export_context_and_static_widths(self) -> None:
        torch.manual_seed(29)
        sequence_length = 21
        # The composable model is configured for one row only. Its fixed
        # methods are repeatedly invoked beyond both that value and width 8.
        composable_model = construct_transformer(self._model_args(1)).eval()
        reference_model = construct_transformer(
            self._model_args(sequence_length)
        ).eval()
        reference_model.load_state_dict(composable_model.state_dict())
        tokens = torch.randint(0, composable_model.params.vocab_size, (1, 21))
        costs = [
            StaticAttentionGraphCost(4, first_cost=1.5, merge_cost=0.5),
            StaticAttentionGraphCost(8, first_cost=1.0, merge_cost=2.0),
        ]

        with torch.no_grad():
            expected = reference_model(tokens)
            actual = execute_composable_llama(composable_model, tokens, costs)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-5)
        self.assertTrue(torch.equal(actual.argmax(-1), expected.argmax(-1)))

    def test_qwen_qk_norm_hf_rope_and_wide_attention(self) -> None:
        torch.manual_seed(31)
        model = construct_transformer(self._qwen_like_args(3)).eval()
        layer = model.layers[0]
        hidden = torch.randn(1, 3, model.params.dim)

        packed = ComposableLlamaQKV(layer)(hidden)
        q, k, v = unpack_composable_qkv(
            packed,
            model.params.n_heads,
            model.params.n_kv_heads,
            model.params.head_dim,
        )
        normalized = layer.attention_norm(hidden)
        expected_q = layer.attention.wq(normalized).view(1, 3, 4, 12)
        expected_k = layer.attention.wk(normalized).view(1, 3, 2, 12)
        expected_v = layer.attention.wv(normalized).view(1, 3, 2, 12)
        expected_q = layer.attention.q_norm_fn(expected_q)
        expected_k = layer.attention.k_norm_fn(expected_k)
        torch.testing.assert_close(q, expected_q.transpose(1, 2))
        torch.testing.assert_close(k, expected_k.transpose(1, 2).transpose(2, 3))
        torch.testing.assert_close(v, expected_v.transpose(1, 2))

        freqs_cos, freqs_sin = model.rope.precompute_freqs_cis(
            model.params.head_dim,
            3,
            model.params.rope_freq_base,
        )
        rotated_q, rotated_k = apply_composable_rotary_emb(q, k, freqs_cos, freqs_sin)
        expected_q, expected_k = model.rope.forward(
            expected_q, expected_k, freqs_cos, freqs_sin
        )
        torch.testing.assert_close(rotated_q, expected_q.transpose(1, 2))
        torch.testing.assert_close(
            rotated_k, expected_k.transpose(1, 2).transpose(2, 3)
        )

        attention_output = torch.randn(1, 4, 3, 12)
        actual_post = ComposableLlamaPostAttention(layer)(hidden, attention_output)
        flattened = attention_output.transpose(1, 2).reshape(1, 3, 48)
        residual = hidden + layer.attention.wo(flattened)
        expected_post = residual + layer.feed_forward(layer.ffn_norm(residual))
        torch.testing.assert_close(actual_post, expected_post)

    def test_qwen_like_execution_matches_reference(self) -> None:
        torch.manual_seed(37)
        sequence_length = 7
        composable_model = construct_transformer(self._qwen_like_args(1)).eval()
        reference_model = construct_transformer(
            self._qwen_like_args(sequence_length)
        ).eval()
        reference_model.load_state_dict(composable_model.state_dict())
        tokens = torch.randint(0, composable_model.params.vocab_size, (1, 7))
        costs = [
            StaticAttentionGraphCost(4, first_cost=1.5, merge_cost=0.5),
            StaticAttentionGraphCost(8, first_cost=1.0, merge_cost=2.0),
        ]

        with torch.no_grad():
            expected = reference_model(tokens)
            actual = execute_composable_llama(composable_model, tokens, costs)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-5)
        self.assertTrue(torch.equal(actual.argmax(-1), expected.argmax(-1)))

    def test_biased_odd_head_gqa_execution_matches_reference(self) -> None:
        torch.manual_seed(41)
        sequence_length = 5
        composable_model = construct_transformer(self._biased_odd_gqa_args(1)).eval()
        reference_model = construct_transformer(
            self._biased_odd_gqa_args(sequence_length)
        ).eval()
        reference_model.load_state_dict(composable_model.state_dict())
        tokens = torch.randint(0, composable_model.params.vocab_size, (1, 5))
        costs = [
            StaticAttentionGraphCost(2, first_cost=1.0, merge_cost=1.0),
            StaticAttentionGraphCost(4, first_cost=1.5, merge_cost=1.2),
        ]

        with torch.no_grad():
            expected = reference_model(tokens)
            actual = execute_composable_llama(composable_model, tokens, costs)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-5)
        self.assertTrue(torch.equal(actual.argmax(-1), expected.argmax(-1)))

    def test_rejects_rope_variants_not_implemented_by_host(self) -> None:
        variants = (
            ("use_scaled_rope", True),
            ("partial_rotary_factor", 0.5),
            ("no_rope_layer_interval", 2),
            ("local_rope_theta", 10000.0),
            ("rope_scaling_short_factor", [1.0, 1.0, 1.0, 1.0]),
        )
        for name, value in variants:
            with self.subTest(name=name):
                args = self._model_args(1)
                setattr(args, name, value)
                model = construct_transformer(args).eval()
                with self.assertRaisesRegex(
                    ValueError, "requires unscaled full-dimension RoPE"
                ):
                    build_composable_llama_portfolio(
                        model,
                        query_rows=1,
                        widths=(4, 8),
                        dtype=torch.float32,
                    )

    def test_rejects_qk_norm_after_host_rope(self) -> None:
        args = self._model_args(1)
        args.use_qk_norm = True
        model = construct_transformer(args).eval()
        with self.assertRaisesRegex(ValueError, "Q/K normalization after RoPE"):
            build_composable_llama_portfolio(
                model,
                query_rows=1,
                widths=(4, 8),
                dtype=torch.float32,
            )

    def test_rejects_non_standard_transformer_features(self) -> None:
        args = self._model_args(1)
        args.use_ffn_learnable_scales = True
        model = construct_transformer(args).eval()
        with self.assertRaisesRegex(ValueError, "post-FFN normalization"):
            build_composable_llama_portfolio(
                model,
                query_rows=1,
                widths=(4, 8),
                dtype=torch.float32,
            )

    def test_exports_complete_portable_multimethod_model(self) -> None:
        model = construct_transformer(self._model_args(1)).eval()
        portfolio = build_composable_llama_portfolio(
            model,
            query_rows=1,
            widths=(4, 8),
            dtype=torch.float32,
        )
        program = export_portable(
            dict(portfolio.modules), dict(portfolio.example_inputs)
        )
        serialized = deserialize_pte_binary(bytes(program.buffer)).program
        actual_methods = {plan.name for plan in serialized.execution_plan}
        expected_methods = {
            "llama_embedding",
            "llama_output",
            *(
                f"llama_layer_{layer}_{stage}"
                for layer in range(2)
                for stage in ("qkv", "post")
            ),
            *(
                f"attn_{kind}_r1_c{width}"
                for kind in ("first", "merge")
                for width in (4, 8)
            ),
        }
        self.assertEqual(actual_methods, expected_methods)

    def test_partitions_layer_methods_without_loss_or_overlap(self) -> None:
        model = construct_transformer(self._model_args(1)).eval()
        portfolio = build_composable_llama_portfolio(
            model,
            query_rows=1,
            widths=(4, 8),
            dtype=torch.float32,
        )
        core_modules, core_inputs, shards = partition_composable_llama_portfolio(
            portfolio,
            layers=2,
            layers_per_shard=1,
        )

        self.assertEqual(set(core_modules), set(core_inputs))
        self.assertEqual(
            set(core_modules),
            {
                "llama_embedding",
                "llama_output",
                "attn_first_r1_c4",
                "attn_first_r1_c8",
                "attn_merge_r1_c4",
                "attn_merge_r1_c8",
            },
        )
        self.assertEqual(
            [(shard.first_layer, shard.last_layer_exclusive) for shard in shards],
            [(0, 1), (1, 2)],
        )
        shard_methods = [set(shard.modules) for shard in shards]
        self.assertEqual(shard_methods[0], {"llama_layer_0_qkv", "llama_layer_0_post"})
        self.assertEqual(shard_methods[1], {"llama_layer_1_qkv", "llama_layer_1_post"})
        self.assertFalse(shard_methods[0].intersection(shard_methods[1]))
        self.assertEqual(
            set(core_modules).union(*shard_methods), set(portfolio.modules)
        )

    def test_partitions_all_query_row_stage_variants(self) -> None:
        model = construct_transformer(self._model_args(2)).eval()
        portfolio = build_composable_llama_portfolio(
            model,
            query_rows=2,
            prefill_query_rows=(1, 2),
            widths=(4,),
            dtype=torch.float32,
        )
        core_modules, _, shards = partition_composable_llama_portfolio(
            portfolio,
            layers=2,
            layers_per_shard=1,
        )
        self.assertNotIn("llama_layer_0_qkv", core_modules)
        self.assertIn("llama_embedding_r1", core_modules)
        self.assertIn("llama_output_r1", core_modules)
        self.assertIn("attn_first_r1_c4", core_modules)
        self.assertEqual(
            set(shards[0].modules),
            {
                "llama_layer_0_qkv",
                "llama_layer_0_post",
                "llama_layer_0_qkv_r1",
                "llama_layer_0_post_r1",
            },
        )

    def test_export_manifest_records_graphs_pte_files_and_time(self) -> None:
        methods = {
            "llama_embedding": object(),
            "llama_output": object(),
            "attn_first_r1_c8": object(),
            "attn_merge_r1_c8": object(),
            "llama_layer_0_qkv": object(),
            "llama_layer_0_post": object(),
            "llama_layer_1_qkv": object(),
            "llama_layer_1_post": object(),
        }
        portfolio = SimpleNamespace(
            modules=methods,
            example_inputs={name: () for name in methods},
        )
        model = SimpleNamespace(
            params=SimpleNamespace(
                dim=32,
                n_layers=2,
                n_heads=4,
                n_kv_heads=2,
                head_dim=8,
                hidden_dim=64,
                vocab_size=64,
                rope_freq_base=10_000.0,
                use_hf_rope=False,
            )
        )

        class FakeProgram:
            def write_to_file(self, file) -> None:
                file.write(b"pte")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "llama.pte"
            args = parse_composable_llama_args(
                [
                    "--checkpoint",
                    "unused.pt",
                    "--params",
                    "unused.json",
                    "--output",
                    str(output),
                    "--widths",
                    "8",
                    "--layers-per-shard",
                    "1",
                ]
            )
            module = "executorch.examples.models.llama.export_composable_llama"
            with (
                mock.patch(f"{module}.load_model", return_value=model),
                mock.patch(
                    f"{module}.build_composable_llama_portfolio",
                    return_value=portfolio,
                ),
                mock.patch(f"{module}.export_portable", return_value=FakeProgram()),
            ):
                self.assertEqual(export_composable_llama(args), output)

            manifest = json.loads(
                output.with_suffix(".pte.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["static_graph_count"], 8)
            self.assertEqual(manifest["pte_file_count"], 3)
            self.assertEqual(manifest["total_pte_bytes"], 9)
            self.assertGreater(manifest["export_wall_seconds"], 0.0)
            self.assertEqual(
                [record["static_graph_count"] for record in manifest["pte_files"]],
                [4, 2, 2],
            )
            self.assertEqual(
                [record["pte_bytes"] for record in manifest["pte_files"]],
                [3, 3, 3],
            )
            self.assertEqual(
                [record["first_layer"] for record in manifest["layer_shards"]],
                [0, 1],
            )
            self.assertEqual(manifest["prefill_query_rows"], [1])
            self.assertEqual(manifest["stage_method_shape_suffix"], "_r<R>")

    def test_export_loads_max_prefill_rows_and_passes_portfolio(self) -> None:
        methods = {"llama_embedding": object()}
        portfolio = SimpleNamespace(
            modules=methods,
            example_inputs={"llama_embedding": ()},
        )
        model = SimpleNamespace(
            params=SimpleNamespace(
                dim=32,
                n_layers=1,
                n_heads=4,
                n_kv_heads=2,
                head_dim=8,
                hidden_dim=64,
                vocab_size=64,
                rope_freq_base=10_000.0,
                use_hf_rope=False,
            )
        )

        class FakeProgram:
            def write_to_file(self, file) -> None:
                file.write(b"pte")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "llama.pte"
            args = parse_composable_llama_args(
                [
                    "--checkpoint",
                    "unused.pt",
                    "--params",
                    "unused.json",
                    "--output",
                    str(output),
                    "--query-rows",
                    "2",
                    "--prefill-query-rows",
                    "1",
                    "2",
                    "4",
                    "--widths",
                    "8",
                    "--row-query-tiles",
                    "2:1",
                    "4:2",
                ]
            )
            module = "executorch.examples.models.llama.export_composable_llama"
            with (
                mock.patch(f"{module}.load_model", return_value=model) as load,
                mock.patch(
                    f"{module}.build_composable_llama_portfolio",
                    return_value=portfolio,
                ) as build,
                mock.patch(f"{module}.export_portable", return_value=FakeProgram()),
            ):
                self.assertEqual(export_composable_llama(args), output)

            self.assertEqual(load.call_args.kwargs["query_rows"], 4)
            self.assertEqual(
                build.call_args.kwargs["prefill_query_rows"], (1, 2, 4)
            )
            self.assertEqual(
                build.call_args.kwargs[
                    "internal_query_tile_rows_by_query_rows"
                ],
                {2: 1, 4: 2},
            )
            manifest = json.loads(
                output.with_suffix(".pte.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["query_rows"], 2)
            self.assertEqual(manifest["prefill_query_rows"], [1, 2, 4])
            self.assertEqual(
                manifest["internal_query_tile_rows_by_query_rows"],
                {"1": 0, "2": 1, "4": 2},
            )

    def test_prefill_query_rows_parser_default_and_validation(self) -> None:
        base = [
            "--checkpoint",
            "unused.pt",
            "--params",
            "unused.json",
            "--output",
            "unused.pte",
            "--query-rows",
            "4",
            "--widths",
            "8",
        ]
        self.assertEqual(
            parse_composable_llama_args(base).prefill_query_rows,
            (4,),
        )
        self.assertEqual(
            parse_composable_llama_args(
                [*base, "--prefill-query-rows", "1", "2", "8"]
            ).prefill_query_rows,
            (1, 2, 4, 8),
        )
        with self.assertRaises(SystemExit):
            parse_composable_llama_args(
                [*base, "--prefill-query-rows", "3"]
            )
        self.assertEqual(
            parse_composable_llama_args(
                [
                    *base,
                    "--prefill-query-rows",
                    "2",
                    "4",
                    "--row-query-tiles",
                    "2:1",
                    "4:2",
                ]
            ).query_tiles_by_query_rows,
            {2: 1, 4: 2},
        )
        with self.assertRaises(SystemExit):
            parse_composable_llama_args(
                [*base, "--row-query-tiles", "8:2"]
            )

    def test_qnn_fp16_export_requires_graph_native_rope(self) -> None:
        args = parse_composable_llama_args(
            [
                "--checkpoint",
                "unused.pt",
                "--params",
                "unused.json",
                "--output",
                "unused.pte",
                "--backend",
                "qnn",
                "--dtype",
                "fp16",
                "--widths",
                "8",
            ]
        )
        with self.assertRaisesRegex(ValueError, "requires --pre-attention-rope"):
            export_composable_llama(args)


if __name__ == "__main__":
    unittest.main()
