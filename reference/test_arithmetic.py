"""Test the shared primitives against hand-computed values; cross-checking the
two references cannot detect defects in what they share."""

from __future__ import annotations

import unittest

from reference.arithmetic import (
    clog2,
    matrix_multiply,
    saturating_lsb_update,
    ternary_sign,
    to_signed,
    weighted_vector_reduction,
)


class TestPrimitives(unittest.TestCase):
    def test_clog2_matches_systemverilog(self) -> None:
        self.assertEqual([clog2(v) for v in (1, 2, 3, 4, 5, 8, 9)], [0, 1, 2, 2, 3, 3, 4])
        for bad in (0, -1, True, 1.0):
            with self.assertRaises(ValueError):
                clog2(bad)  # type: ignore[arg-type]

    def test_to_signed_sign_extends_and_wraps(self) -> None:
        self.assertEqual([to_signed(v, 8) for v in (255, 128, 127, 0, 200, -129)], [-1, -128, 127, 0, -56, 127])
        with self.assertRaises(ValueError):
            to_signed(1, 0)

    def test_ternary_sign(self) -> None:
        self.assertEqual([ternary_sign(v) for v in (5, 0, -5)], [1, 0, -1])

    def test_saturating_lsb_update(self) -> None:
        self.assertEqual([saturating_lsb_update(3, d, 4) for d in (1, 0, -1)], [4, 3, 2])
        self.assertEqual(saturating_lsb_update(7, 1, 4), 7)
        self.assertEqual(saturating_lsb_update(-8, -1, 4), -8)
        self.assertEqual(saturating_lsb_update(7, -1, 4), 6)
        with self.assertRaises(ValueError):
            saturating_lsb_update(0, 2, 4)


class TestSingleLayerOracle(unittest.TestCase):
    def test_orientation_is_row_indexed_by_input_lane(self) -> None:
        self.assertEqual(matrix_multiply((1, 2, 3), ((1, 2, 3), (4, 5, 6), (7, 8, 9)), 8), [30, 36, 42])
        # A transposed model would return [10, 0, 0] for this stimulus.
        self.assertEqual(matrix_multiply((1, 0, 0), ((10, 20, 30), (0, 0, 0), (0, 0, 0)), 8), [10, 20, 30])

    def test_inputs_are_signed_at_the_given_width(self) -> None:
        self.assertEqual(matrix_multiply((-2, 3), ((2, -1), (1, -2)), 4), [-1, -4])
        self.assertEqual(matrix_multiply((255,), ((7,),), 8), [-7])

    def test_readout_shifts_the_complete_sum_once(self) -> None:
        self.assertEqual(weighted_vector_reduction((30, 36, 42), (16, 24, 32), 8, 8, 0), 21)
        self.assertEqual(weighted_vector_reduction((30, 36, 42), (16, 24, 32), 8, 8, 4), 1)
        # Per-term rescaling would give 0; one final shift of 192 gives 1.
        self.assertEqual(weighted_vector_reduction((1, 1, 1), (64, 64, 64), 8, 8, 0), 1)
        # The arithmetic shift floors negative sums: -480 >> 7 == -4.
        self.assertEqual(weighted_vector_reduction((30, 0, 0), (-16, 0, 0), 8, 8, 0), -4)

    def test_rejects_size_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            matrix_multiply((1, 2), ((1, 2, 3), (4, 5, 6), (7, 8, 9)), 8)
        with self.assertRaises(ValueError):
            weighted_vector_reduction((1, 2), (1, 2, 3), 8, 8, 0)


if __name__ == "__main__":
    unittest.main()
