# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from executorch.extension.llm.export.probe_composable_attention_widths import (
    classify_failure,
    power_of_two_widths,
    spill_fill_bytes,
)


class ProbeComposableAttentionWidthsTest(unittest.TestCase):
    def test_generates_power_of_two_sequence_from_one(self) -> None:
        self.assertEqual(power_of_two_widths(1, 17), (1, 2, 4, 8, 16))

    def test_rejects_non_power_of_two_start(self) -> None:
        with self.assertRaisesRegex(ValueError, "power of two"):
            power_of_two_widths(3, 16)

    def test_classifies_resource_failures(self) -> None:
        self.assertEqual(classify_failure("VTCM allocation failed"), "resource_limit")
        self.assertEqual(
            classify_failure("VTCM Allocation complete\nspill_bytes=0\nfill_bytes=0"),
            "unsupported_or_other",
        )
        self.assertEqual(
            classify_failure("spill_bytes=0\nfill_bytes=4096"),
            "resource_limit",
        )
        self.assertEqual(
            classify_failure("unsupported reshape dimension"),
            "unsupported_or_other",
        )

    def test_parses_maximum_spill_fill_bytes(self) -> None:
        self.assertEqual(
            spill_fill_bytes(
                "spill_bytes=0\nfill_bytes=128\nspill_bytes=64\nfill_bytes=32"
            ),
            {"spill_bytes": 64, "fill_bytes": 128},
        )


if __name__ == "__main__":
    unittest.main()
