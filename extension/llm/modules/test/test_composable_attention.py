# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import unittest

import torch
from executorch.extension.llm.modules.composable_attention import (
    build_static_attention_portfolio,
    execute_attention_plan,
    static_attention_method_name,
    StaticAttentionFirstBlock,
    StaticAttentionGraphCost,
    StaticAttentionMergeBlock,
    StaticAttentionPlanner,
)
from torch.testing import assert_close


class StaticAttentionPlannerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = StaticAttentionPlanner(
            (
                StaticAttentionGraphCost(64, 4.0, 3.0),
                StaticAttentionGraphCost(128, 6.0, 5.0),
                StaticAttentionGraphCost(256, 9.0, 8.0),
            )
        )

    def test_cost_aware_mixed_width_plan(self) -> None:
        self.assertEqual(self.planner.plan(1).widths, (64,))
        self.assertEqual(self.planner.plan(192).widths, (128, 64))
        self.assertEqual(self.planner.plan(384).widths, (256, 128))

    def test_compares_one_covering_graph_with_composition(self) -> None:
        one_graph = StaticAttentionPlanner(
            (
                StaticAttentionGraphCost(64, 1.0, 1.0),
                StaticAttentionGraphCost(128, 1.5, 1.5),
            )
        )
        self.assertEqual(one_graph.plan(100).widths, (128,))

        composed = StaticAttentionPlanner(
            (
                StaticAttentionGraphCost(64, 1.0, 0.2),
                StaticAttentionGraphCost(128, 2.0, 2.0),
            )
        )
        self.assertEqual(composed.plan(100).widths, (64, 64))

    def test_plans_beyond_any_compiled_context_limit(self) -> None:
        plan = self.planner.plan(100_000)
        self.assertGreaterEqual(plan.coverage, 100_000)
        self.assertLess(plan.padding, min(self.planner.graph_widths))
        self.assertEqual(self.planner.planned_through, 100_000)
        self.assertTrue(set(plan.widths).issubset(self.planner.graph_widths))

    def test_plans_only_visible_causal_prefix(self) -> None:
        planner = StaticAttentionPlanner((StaticAttentionGraphCost(64, 1.0, 1.0),))
        plan = planner.plan_causal_tile(256, 64, 4)
        self.assertEqual(plan.sequence_length, 68)
        self.assertEqual(plan.widths, (64, 64))
        self.assertEqual(plan.padding, 60)

        final_padded = planner.plan_causal_tile(256, 128, 4, 2)
        self.assertEqual(final_padded.sequence_length, 130)
        self.assertEqual(final_padded.widths, (64, 64, 64))

    def test_rejects_invalid_causal_tile_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "physical tile"):
            self.planner.plan_causal_tile(128, 0, 4, 5)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            self.planner.plan_causal_tile(128, 127, 4)

    def test_rejects_invalid_profiles(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            StaticAttentionPlanner(
                (
                    StaticAttentionGraphCost(64, 1.0, 1.0),
                    StaticAttentionGraphCost(64, 2.0, 2.0),
                )
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            StaticAttentionGraphCost(64, 0.0, 1.0)
        with self.assertRaisesRegex(ValueError, "power of two"):
            StaticAttentionGraphCost(3, 1.0, 1.0)


class ComposableStaticAttentionTest(unittest.TestCase):
    def _check_attention(self, query_rows: int, sequence_length: int) -> None:
        torch.manual_seed(20260730 + query_rows + sequence_length)
        batch = 1
        query_heads = 4
        kv_heads = 2
        head_dim = 16
        q = torch.randn(batch, query_heads, query_rows, head_dim)
        k = torch.randn(batch, kv_heads, head_dim, sequence_length)
        v = torch.randn(batch, kv_heads, sequence_length, head_dim)

        first_query_position = sequence_length - query_rows
        query_positions = torch.arange(first_query_position, sequence_length).view(
            1, 1, query_rows, 1
        )
        key_positions = torch.arange(sequence_length).view(1, 1, 1, -1)
        visibility = (key_positions <= query_positions).to(q.dtype)

        repeated_k = torch.repeat_interleave(k, query_heads // kv_heads, dim=1)
        repeated_v = torch.repeat_interleave(v, query_heads // kv_heads, dim=1)
        scale = 1.0 / math.sqrt(head_dim)
        scores = (q @ repeated_k) * scale
        scores = scores.masked_fill(visibility == 0, float("-inf"))
        expected = torch.softmax(scores, dim=-1) @ repeated_v

        planner = StaticAttentionPlanner(
            {
                8: (1.0, 0.9),
                16: (1.2, 1.1),
                32: (3.0, 2.8),
            }
        )
        plan = planner.plan(sequence_length)
        actual = execute_attention_plan(
            q,
            k,
            v,
            visibility,
            plan,
            scale,
            state_scale=1024.0,
        )
        assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    def test_exact_decode_and_prefill_composition(self) -> None:
        for query_rows, sequence_length in (
            (1, 7),
            (1, 23),
            (4, 23),
            (8, 47),
        ):
            with self.subTest(query_rows=query_rows, sequence_length=sequence_length):
                self._check_attention(query_rows, sequence_length)

    def test_fully_masked_merge_preserves_state(self) -> None:
        torch.manual_seed(7)
        q = torch.randn(1, 4, 3, 16)
        k = torch.randn(1, 2, 16, 8)
        v = torch.randn(1, 2, 8, 16)
        visible = torch.ones(1, 1, 3, 8)
        hidden = torch.zeros_like(visible)
        first = StaticAttentionFirstBlock(0.25, state_scale=64.0)
        merge = StaticAttentionMergeBlock(0.25, state_scale=64.0)

        state = first(q, k, v, visible)
        unchanged = merge(q, k * 100.0, v, hidden, *state[1:])
        for actual, expected in zip(unchanged, state):
            assert_close(actual, expected, rtol=0.0, atol=0.0)

        empty_state = first(q, k, v, hidden)
        initialized_by_merge = merge(q, k, v, visible, *empty_state[1:])
        for actual, expected in zip(initialized_by_merge, state):
            assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_composition_preserves_small_softmax_contributions(self) -> None:
        q = torch.ones(1, 1, 1, 1)
        first_k = torch.zeros(1, 1, 1, 1)
        first_v = torch.ones(1, 1, 1, 1)
        second_k = torch.full((1, 1, 1, 1), 40.0)
        second_v = torch.zeros(1, 1, 1, 1)
        visibility = torch.ones(1, 1, 1, 1)
        first = StaticAttentionFirstBlock(1.0)
        merge = StaticAttentionMergeBlock(1.0)

        state = first(q, first_k, first_v, visibility)
        actual = merge(q, second_k, second_v, visibility, *state[1:])[0]
        expected = torch.softmax(torch.tensor([0.0, 40.0]), dim=0)[0]
        assert_close(actual.flatten()[0], expected, rtol=1e-5, atol=0.0)

    def test_rejects_non_broadcastable_inputs_and_nonfinite_scales(self) -> None:
        q = torch.zeros(1, 4, 1, 8)
        k = torch.zeros(2, 2, 8, 4)
        v = torch.zeros(2, 2, 4, 8)
        visibility = torch.ones(1, 1, 1, 4)
        first = StaticAttentionFirstBlock(1.0)
        with self.assertRaisesRegex(ValueError, "batch dimensions differ"):
            first(q, k, v, visibility)

        k = k[:1]
        v = v[:1]
        invalid_visibility = torch.ones(1, 2, 1, 4)
        with self.assertRaisesRegex(ValueError, "not broadcastable"):
            first(q, k, v, invalid_visibility)
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            StaticAttentionFirstBlock(float("nan"))

    def test_fixed_shape_blocks_are_exportable(self) -> None:
        torch.manual_seed(11)
        q = torch.randn(1, 4, 4, 16)
        k = torch.randn(1, 2, 16, 16)
        v = torch.randn(1, 2, 16, 16)
        visibility = torch.ones(1, 1, 4, 16)
        first = StaticAttentionFirstBlock(0.25, state_scale=64.0)
        first_ep = torch.export.export(first, (q, k, v, visibility), strict=True)

        expected = first(q, k, v, visibility)
        actual = first_ep.module()(q, k, v, visibility)
        for actual_value, expected_value in zip(actual, expected):
            assert_close(actual_value, expected_value)

        merge = StaticAttentionMergeBlock(0.25, state_scale=64.0)
        merge_inputs = (q, k, v, visibility, *expected[1:])
        merge_ep = torch.export.export(merge, merge_inputs, strict=True)
        merge_expected = merge(*merge_inputs)
        merge_actual = merge_ep.module()(*merge_inputs)
        for actual_value, expected_value in zip(merge_actual, merge_expected):
            assert_close(actual_value, expected_value)

    def test_internal_kv_tiling_matches_full_width_with_masked_tiles(self) -> None:
        for dtype, rtol, atol in (
            (torch.float16, 2e-3, 2e-3),
            (torch.float32, 2e-5, 2e-6),
        ):
            with self.subTest(dtype=dtype):
                torch.manual_seed(23)
                q = torch.randn(1, 4, 3, 16, dtype=dtype)
                k = torch.randn(1, 2, 16, 16, dtype=dtype)
                v = torch.randn(1, 2, 16, 16, dtype=dtype)
                visibility = torch.zeros(1, 1, 3, 16, dtype=dtype)
                visibility[..., 0, :2] = 1.0
                visibility[..., 1, :7] = 1.0
                visibility[..., 2, :] = 1.0

                full_first = StaticAttentionFirstBlock(
                    0.25, state_scale=64.0, exponent_floor=-32.0
                )
                tiled_first = StaticAttentionFirstBlock(
                    0.25,
                    state_scale=64.0,
                    exponent_floor=-32.0,
                    internal_kv_tile_width=5,
                )
                full_state = full_first(q, k, v, visibility)
                tiled_state = tiled_first(q, k, v, visibility)
                for actual, expected in zip(tiled_state, full_state):
                    assert_close(actual, expected, rtol=rtol, atol=atol)

                full_merge = StaticAttentionMergeBlock(
                    0.25, state_scale=64.0, exponent_floor=-32.0
                )
                tiled_merge = StaticAttentionMergeBlock(
                    0.25,
                    state_scale=64.0,
                    exponent_floor=-32.0,
                    internal_kv_tile_width=5,
                )
                full_merged = full_merge(q, k, v, visibility, *full_state[1:])
                tiled_merged = tiled_merge(q, k, v, visibility, *full_state[1:])
                for actual, expected in zip(tiled_merged, full_merged):
                    assert_close(actual, expected, rtol=rtol, atol=atol)

    def test_internal_kv_tiling_is_statically_unrolled_during_export(self) -> None:
        torch.manual_seed(31)
        q = torch.randn(1, 4, 2, 8, dtype=torch.float16)
        k = torch.randn(1, 2, 8, 16, dtype=torch.float16)
        v = torch.randn(1, 2, 16, 8, dtype=torch.float16)
        visibility = torch.ones(1, 1, 2, 16, dtype=torch.float16)
        default = StaticAttentionFirstBlock(0.25)
        tiled = StaticAttentionFirstBlock(0.25, internal_kv_tile_width=4)

        default_ep = torch.export.export(default, (q, k, v, visibility), strict=True)
        tiled_ep = torch.export.export(tiled, (q, k, v, visibility), strict=True)

        def matmul_count(exported: torch.export.ExportedProgram) -> int:
            return sum(
                node.op == "call_function"
                and node.target == torch.ops.aten.matmul.default
                for node in exported.graph.nodes
            )

        self.assertEqual(matmul_count(default_ep), 2)
        self.assertEqual(matmul_count(tiled_ep), 8)
        self.assertEqual(
            tiled_ep.graph_signature.user_inputs,
            default_ep.graph_signature.user_inputs,
        )
        self.assertEqual(
            len(tiled_ep.graph_signature.user_outputs),
            len(default_ep.graph_signature.user_outputs),
        )
        self.assertEqual(tiled_ep.call_spec.out_spec, default_ep.call_spec.out_spec)
        for actual, expected in zip(
            tiled_ep.module()(q, k, v, visibility),
            default_ep.module()(q, k, v, visibility),
        ):
            self.assertEqual(actual.shape, expected.shape)

        tiled_merge = StaticAttentionMergeBlock(0.25, internal_kv_tile_width=4)
        state = default(q, k, v, visibility)
        merge_inputs = (q, k, v, visibility, *state[1:])
        merge_ep = torch.export.export(tiled_merge, merge_inputs, strict=True)
        self.assertEqual(matmul_count(merge_ep), 8)

    def test_portfolio_enables_internal_tiling_without_new_methods(self) -> None:
        portfolio = build_static_attention_portfolio(
            batch_size=1,
            query_heads=4,
            kv_heads=2,
            query_rows=1,
            head_dim=8,
            widths=(8, 16),
            softmax_scale=0.25,
            internal_kv_tile_width=4,
        )
        self.assertEqual(
            portfolio.method_names,
            (
                "attn_first_r1_c8",
                "attn_merge_r1_c8",
                "attn_first_r1_c16",
                "attn_merge_r1_c16",
            ),
        )
        for module in portfolio.modules.values():
            self.assertEqual(module.internal_kv_tile_width, 4)

    def test_rejects_negative_internal_kv_tile_width(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            StaticAttentionFirstBlock(0.25, internal_kv_tile_width=-1)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            StaticAttentionMergeBlock(0.25, internal_kv_tile_width=-1)

    def test_internal_query_tiling_matches_dense_attention(self) -> None:
        torch.manual_seed(37)
        q = torch.randn(1, 4, 8, 16)
        k = torch.randn(1, 2, 16, 16)
        v = torch.randn(1, 2, 16, 16)
        visibility = torch.randint(0, 2, (1, 1, 8, 16), dtype=torch.float32)
        visibility[..., 0] = 1.0

        dense_first = StaticAttentionFirstBlock(0.25, state_scale=64.0)
        tiled_first = StaticAttentionFirstBlock(
            0.25, state_scale=64.0, internal_query_tile_rows=2
        )
        dense_state = dense_first(q, k, v, visibility)
        tiled_state = tiled_first(q, k, v, visibility)
        for actual, expected in zip(tiled_state, dense_state):
            assert_close(actual, expected, rtol=0.0, atol=0.0)

        dense_merge = StaticAttentionMergeBlock(0.25, state_scale=64.0)
        tiled_merge = StaticAttentionMergeBlock(
            0.25, state_scale=64.0, internal_query_tile_rows=2
        )
        dense_merged = dense_merge(q, k, v, visibility, *dense_state[1:])
        tiled_merged = tiled_merge(q, k, v, visibility, *tiled_state[1:])
        for actual, expected in zip(tiled_merged, dense_merged):
            assert_close(actual, expected, rtol=0.0, atol=0.0)

        for module, inputs in (
            (tiled_first, (q, k, v, visibility)),
            (tiled_merge, (q, k, v, visibility, *tiled_state[1:])),
        ):
            self.assertIsNotNone(torch.export.export(module, inputs, strict=True))

    def test_rejects_invalid_internal_query_tile_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            StaticAttentionFirstBlock(0.25, internal_query_tile_rows=-1)
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            build_static_attention_portfolio(
                batch_size=1,
                query_heads=4,
                kv_heads=2,
                query_rows=4,
                head_dim=16,
                widths=(8,),
                softmax_scale=0.25,
                internal_query_tile_rows=8,
            )

    def test_decode_nonempty_contract_matches_safe_path(self) -> None:
        torch.manual_seed(29)
        q = torch.randn(1, 4, 1, 16)
        k = torch.randn(1, 2, 16, 8)
        v = torch.randn(1, 2, 8, 16)
        visibility = torch.tensor([[[[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]])
        safe_first = StaticAttentionFirstBlock(0.25, state_scale=64.0)
        fast_first = StaticAttentionFirstBlock(
            0.25, state_scale=64.0, assume_nonempty=True
        )
        safe_state = safe_first(q, k, v, visibility)
        fast_state = fast_first(q, k, v, visibility)
        for actual, expected in zip(fast_state, safe_state):
            assert_close(actual, expected, rtol=0.0, atol=0.0)

        tiled_fast_first = StaticAttentionFirstBlock(
            0.25,
            state_scale=64.0,
            assume_nonempty=True,
            internal_kv_tile_width=2,
        )
        tiled_fast_state = tiled_fast_first(q, k, v, visibility)
        for actual, expected in zip(tiled_fast_state, safe_state):
            assert_close(actual, expected, rtol=2e-5, atol=2e-6)

        safe_merge = StaticAttentionMergeBlock(0.25, state_scale=64.0)
        fast_merge = StaticAttentionMergeBlock(
            0.25, state_scale=64.0, assume_nonempty=True
        )
        safe_merged = safe_merge(q, k, v, visibility, *safe_state[1:])
        fast_merged = fast_merge(q, k, v, visibility, *fast_state[1:])
        for actual, expected in zip(fast_merged, safe_merged):
            assert_close(actual, expected, rtol=0.0, atol=0.0)

        tiled_fast_merge = StaticAttentionMergeBlock(
            0.25,
            state_scale=64.0,
            assume_nonempty=True,
            internal_kv_tile_width=2,
        )
        tiled_fast_merged = tiled_fast_merge(q, k, v, visibility, *tiled_fast_state[1:])
        for actual, expected in zip(tiled_fast_merged, safe_merged):
            assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    def test_nonempty_contract_is_decode_only(self) -> None:
        with self.assertRaisesRegex(ValueError, "Decode"):
            build_static_attention_portfolio(
                batch_size=1,
                query_heads=4,
                kv_heads=2,
                query_rows=2,
                head_dim=16,
                widths=(8,),
                softmax_scale=0.25,
                assume_nonempty=True,
            )
        self.assertEqual(
            static_attention_method_name("first", 1, 8, True),
            "attn_first_nonempty_r1_c8",
        )

    def test_portfolio_exports_fixed_shape_multimethods(self) -> None:
        portfolio = build_static_attention_portfolio(
            batch_size=1,
            query_heads=4,
            kv_heads=2,
            query_rows=3,
            head_dim=16,
            widths=(8, 16, 8),
            softmax_scale=0.25,
            state_scale=64.0,
        )
        self.assertEqual(
            portfolio.method_names,
            (
                "attn_first_r3_c8",
                "attn_merge_r3_c8",
                "attn_first_r3_c16",
                "attn_merge_r3_c16",
            ),
        )
        for name in portfolio.method_names:
            exported = torch.export.export(
                portfolio.modules[name], portfolio.example_inputs[name], strict=True
            )
            self.assertIsNotNone(exported)

        self.assertEqual(
            static_attention_method_name("merge", 3, 16),
            "attn_merge_r3_c16",
        )
        with self.assertRaisesRegex(ValueError, "kind"):
            static_attention_method_name("invalid", 3, 16)
        with self.assertRaisesRegex(ValueError, "power of two"):
            build_static_attention_portfolio(
                batch_size=1,
                query_heads=4,
                kv_heads=2,
                query_rows=3,
                head_dim=16,
                widths=(8, 12),
                softmax_scale=0.25,
            )


if __name__ == "__main__":
    unittest.main()
