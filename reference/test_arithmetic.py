"""Test shared arithmetic against hand-computed expectations; cross-checking
the two references cannot detect defects in their shared primitives."""

from __future__ import annotations

import unittest

from reference.arithmetic import (
    activate,
    activation_gates,
    apply_matrix_update,
    apply_reduction_update,
    clog2,
    fixed_point_rescale,
    learning_direction,
    matrix_multiply,
    matrix_result_width,
    matrix_update_directions,
    prediction_width,
    reduction_fraction_bits,
    reduction_update_directions,
    rescale_shift,
    saturating_lsb_update,
    signed_max,
    signed_min,
    ternary_product,
    ternary_sign,
    to_signed,
    to_unsigned,
    weighted_reduction_accumulator,
    weighted_vector_reduction,
)


class TestWidthHelpers(unittest.TestCase):
    def test_clog2_matches_systemverilog(self) -> None:
        self.assertEqual([clog2(v) for v in (1, 2, 3, 4, 5, 8, 9)], [0, 1, 2, 2, 3, 3, 4])

    def test_clog2_rejects_non_positive_and_bool(self) -> None:
        for bad in (0, -1, True, 1.0):
            with self.assertRaises(ValueError):
                clog2(bad)  # type: ignore[arg-type]

    def test_signed_bounds(self) -> None:
        self.assertEqual((signed_min(8), signed_max(8)), (-128, 127))
        self.assertEqual((signed_min(4), signed_max(4)), (-8, 7))
        self.assertEqual((signed_min(1), signed_max(1)), (-1, 0))

    def test_result_widths(self) -> None:
        # 2*WIDTH + $clog2(N), then + $clog2(N) again for the reduction.
        self.assertEqual(matrix_result_width(8, 3), 18)
        self.assertEqual(prediction_width(8, 3), 20)
        self.assertEqual(matrix_result_width(16, 4), 34)
        self.assertEqual(prediction_width(16, 4), 36)

    def test_reduction_format_is_one_sign_bit_and_the_rest_fractional(self) -> None:
        self.assertEqual(reduction_fraction_bits(8), 7)
        self.assertEqual(rescale_shift(0, 8), 7)
        self.assertEqual(rescale_shift(4, 8), 11)

    def test_width_helpers_reject_invalid_widths(self) -> None:
        for bad in (0, -3, True):
            with self.assertRaises(ValueError):
                to_signed(1, bad)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            rescale_shift(-1, 8)


class TestTwosComplement(unittest.TestCase):
    def test_to_unsigned_keeps_low_bits(self) -> None:
        self.assertEqual(to_unsigned(-1, 8), 255)
        self.assertEqual(to_unsigned(256, 8), 0)
        self.assertEqual(to_unsigned(5, 3), 5)

    def test_to_signed_sign_extends_the_top_bit(self) -> None:
        self.assertEqual(to_signed(255, 8), -1)
        self.assertEqual(to_signed(128, 8), -128)
        self.assertEqual(to_signed(127, 8), 127)
        self.assertEqual(to_signed(0, 8), 0)

    def test_to_signed_wraps_out_of_range_values(self) -> None:
        self.assertEqual(to_signed(200, 8), -56)
        self.assertEqual(to_signed(-129, 8), 127)


class TestTernary(unittest.TestCase):
    def test_ternary_sign(self) -> None:
        self.assertEqual([ternary_sign(v) for v in (5, 0, -5)], [1, 0, -1])

    def test_ternary_product_preserves_zero(self) -> None:
        self.assertEqual(ternary_product(1, -1), -1)
        self.assertEqual(ternary_product(-1, -1), 1)
        self.assertEqual(ternary_product(0, -1), 0)

    def test_ternary_product_rejects_non_ternary_inputs(self) -> None:
        for bad in (2, -2, 3):
            with self.assertRaises(ValueError):
                ternary_product(bad, 1)
            with self.assertRaises(ValueError):
                ternary_product(1, bad)


class TestMatrixMultiply(unittest.TestCase):
    def test_signed_two_by_two_case(self) -> None:
        self.assertEqual(matrix_multiply((-2, 3), ((2, -1), (1, -2)), 4), [-1, -4])

    def test_orientation_is_row_indexed_by_input_lane(self) -> None:
        # result[column] = sum over row of input[row] * W[row][column]
        # col0 = 1+8+21 = 30, col1 = 2+10+24 = 36, col2 = 3+12+27 = 42
        result = matrix_multiply((1, 2, 3), ((1, 2, 3), (4, 5, 6), (7, 8, 9)), 8)
        self.assertEqual(result, [30, 36, 42])

    def test_asymmetric_matrix_distinguishes_transposed_orientation(self) -> None:
        # A transposed model would return [10, 0, 0] for this stimulus.
        result = matrix_multiply((1, 0, 0), ((10, 20, 30), (0, 0, 0), (0, 0, 0)), 8)
        self.assertEqual(result, [10, 20, 30])

    def test_signed_inputs_and_weights(self) -> None:
        # col0 = (-1)*2 + 3*(-4) = -14
        result = matrix_multiply((-1, 3), ((2, 0), (-4, 0)), 8)
        self.assertEqual(result[0], -14)

    def test_inputs_are_interpreted_at_the_given_width(self) -> None:
        # 255 at WIDTH=8 is -1, so the product is -7, not 1785.
        self.assertEqual(matrix_multiply((255,), ((7,),), 8), [-7])

    def test_rejects_non_square_or_mismatched_sizes(self) -> None:
        with self.assertRaises(ValueError):
            matrix_multiply((1, 2), ((1, 2, 3), (4, 5, 6), (7, 8, 9)), 8)
        with self.assertRaises(ValueError):
            matrix_multiply((1, 2), ((1, 2), (3,)), 8)


class TestActivation(unittest.TestCase):
    def test_pass_through_is_the_identity(self) -> None:
        self.assertEqual(activate((5, 0, -5), True, 18), [5, 0, -5])

    def test_relu_is_closed_at_zero(self) -> None:
        self.assertEqual(activate((5, 0, -5), False, 18), [5, 0, 0])

    def test_gates_follow_pass_through_or_nonzero(self) -> None:
        # pass_through opens every gate, including a zero lane.
        self.assertEqual(activation_gates((5, 0, -5), True), [True, True, True])
        # Without pass-through only a non-zero activated lane opens.
        self.assertEqual(activation_gates((5, 0, 0), False), [True, False, False])


class TestFixedPointRescale(unittest.TestCase):
    def test_shift_by_four(self) -> None:
        self.assertEqual(fixed_point_rescale(2048, 4, 16), 128)

    def test_positive_shift(self) -> None:
        self.assertEqual(fixed_point_rescale(2688, 7, 28), 21)

    def test_negative_values_floor_rather_than_truncate_toward_zero(self) -> None:
        # An arithmetic right shift floors: -7 >> 1 is -4, not -3.
        self.assertEqual(fixed_point_rescale(-7, 1, 8), -4)
        self.assertEqual(fixed_point_rescale(-1, 4, 8), -1)

    def test_optional_output_narrowing_wraps(self) -> None:
        self.assertEqual(fixed_point_rescale(1 << 9, 0, 16, 8), 0)
        self.assertEqual(fixed_point_rescale(200, 0, 16, 8), -56)

    def test_rejects_negative_shift(self) -> None:
        with self.assertRaises(ValueError):
            fixed_point_rescale(1, -1, 8)


class TestWeightedReduction(unittest.TestCase):
    def test_q_format_and_cancelling_terms(self) -> None:
        # 256*64 >> (4+7) = 8; equal opposite lanes cancel before shifting.
        self.assertEqual(weighted_vector_reduction((256, 0), (64, 0), 8, 8, 4), 8)
        self.assertEqual(weighted_vector_reduction((16, -16), (64, 64), 8, 8, 4), 0)

    def test_accumulator_is_the_plain_weighted_sum_when_it_fits(self) -> None:
        # 30*16 + 36*24 + 42*32 = 480 + 864 + 1344 = 2688
        self.assertEqual(weighted_reduction_accumulator((30, 36, 42), (16, 24, 32), 8, 8), 2688)

    def test_prediction_applies_the_q_format_shift(self) -> None:
        # 2688 >> rescale_shift(0, 8) == 2688 >> 7 == 21
        self.assertEqual(weighted_vector_reduction((30, 36, 42), (16, 24, 32), 8, 8, 0), 21)

    def test_fraction_bits_add_to_the_shift(self) -> None:
        # 2688 >> (4 + 7) == 2688 >> 11 == 1
        self.assertEqual(weighted_vector_reduction((30, 36, 42), (16, 24, 32), 8, 8, 4), 1)

    def test_no_term_is_individually_rescaled(self) -> None:
        # Sum 192 shifts once at the end to 1; per-term rescaling gives 0.
        self.assertEqual(weighted_vector_reduction((1, 1, 1), (64, 64, 64), 8, 8, 0), 1)

    def test_negative_coefficients(self) -> None:
        # 30*(-16) = -480; -480 >> 7 == -4 after flooring.
        self.assertEqual(weighted_vector_reduction((30, 0, 0), (-16, 0, 0), 8, 8, 0), -4)

    def test_rejects_size_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            weighted_reduction_accumulator((1, 2), (1, 2, 3), 8, 8)


class TestLearningDirection(unittest.TestCase):
    def test_sign_of_target_minus_prediction(self) -> None:
        self.assertEqual(learning_direction(127, 21, 8, 20), 1)
        self.assertEqual(learning_direction(21, 21, 8, 20), 0)
        self.assertEqual(learning_direction(-40, 21, 8, 20), -1)

    def test_narrow_target_is_sign_extended_before_comparison(self) -> None:
        # target 255 at TARGET_WIDTH=8 is -1, which is below a prediction of 0.
        self.assertEqual(learning_direction(255, 0, 8, 20), -1)

    def test_comparison_uses_the_wider_of_the_two_widths(self) -> None:
        # A 20-bit prediction of all-ones is -1, not a large positive value.
        self.assertEqual(learning_direction(0, (1 << 20) - 1, 8, 20), 1)


class TestUpdateDirections(unittest.TestCase):
    def test_zero_reduction_coefficient_and_input_signs(self) -> None:
        _, columns, update = matrix_update_directions(
            (1, -1, 0), (2, -2, 0), (5, -5, 0), 1, 8, 8, True
        )
        self.assertEqual(columns, [1, -1, 0])
        self.assertEqual(update, [[1, -1, 0], [-1, 1, 0], [0, 0, 0]])
        self.assertEqual(reduction_update_directions((2, -2, 0), -1), [-1, 1, 0])

    def test_outer_product_of_row_and_column_directions(self) -> None:
        row, column, update = matrix_update_directions(
            (1, -2, 0), (5, 5, 5), (16, -24, 32), 1, 8, 8, True
        )
        self.assertEqual(row, [1, -1, 0])
        self.assertEqual(column, [1, -1, 1])
        self.assertEqual(update, [[1, -1, 1], [-1, 1, -1], [0, 0, 0]])

    def test_learning_direction_flips_every_column(self) -> None:
        _, column, _ = matrix_update_directions(
            (1, 1, 1), (5, 5, 5), (16, -24, 32), -1, 8, 8, True
        )
        self.assertEqual(column, [-1, 1, -1])

    def test_zero_learning_direction_zeroes_the_update(self) -> None:
        _, column, update = matrix_update_directions(
            (1, 1, 1), (5, 5, 5), (16, 24, 32), 0, 8, 8, True
        )
        self.assertEqual(column, [0, 0, 0])
        self.assertEqual(update, [[0, 0, 0]] * 3)

    def test_closed_relu_gate_zeroes_only_its_own_column(self) -> None:
        _, column, _ = matrix_update_directions(
            (1, 1, 1), (5, 0, 5), (16, 24, 32), 1, 8, 8, False
        )
        self.assertEqual(column, [1, 0, 1])

    def test_pass_through_keeps_a_zero_lane_open(self) -> None:
        _, column, _ = matrix_update_directions(
            (1, 1, 1), (5, 0, 5), (16, 24, 32), 1, 8, 8, True
        )
        self.assertEqual(column, [1, 1, 1])

    def test_reduction_directions_are_direction_times_activated_sign(self) -> None:
        self.assertEqual(reduction_update_directions((5, 0, -5), 1), [1, 0, -1])
        self.assertEqual(reduction_update_directions((5, 0, -5), -1), [-1, 0, 1])
        self.assertEqual(reduction_update_directions((5, 0, -5), 0), [0, 0, 0])

    def test_rejects_non_ternary_learning_direction(self) -> None:
        with self.assertRaises(ValueError):
            reduction_update_directions((1, 2, 3), 2)
        with self.assertRaises(ValueError):
            matrix_update_directions((1, 1, 1), (1, 1, 1), (1, 1, 1), 2, 8, 8, True)


class TestSaturatingUpdate(unittest.TestCase):
    def test_eight_bit_reduction_saturates_at_both_endpoints(self) -> None:
        self.assertEqual(apply_reduction_update((127, -128), (1, -1), 8), [127, -128])

    def test_single_lsb_step(self) -> None:
        self.assertEqual(saturating_lsb_update(3, 1, 4), 4)
        self.assertEqual(saturating_lsb_update(3, -1, 4), 2)
        self.assertEqual(saturating_lsb_update(3, 0, 4), 3)

    def test_saturates_at_both_signed_limits(self) -> None:
        self.assertEqual(saturating_lsb_update(7, 1, 4), 7)
        self.assertEqual(saturating_lsb_update(-8, -1, 4), -8)
        # The opposite direction still moves away from the limit.
        self.assertEqual(saturating_lsb_update(7, -1, 4), 6)
        self.assertEqual(saturating_lsb_update(-8, 1, 4), -7)

    def test_rejects_non_ternary_direction(self) -> None:
        with self.assertRaises(ValueError):
            saturating_lsb_update(0, 2, 4)

    def test_apply_matrix_update_is_elementwise(self) -> None:
        updated = apply_matrix_update(((7, 0), (-8, 3)), ((1, -1), (-1, 1)), 4)
        self.assertEqual(updated, [[7, -1], [-8, 4]])

    def test_apply_reduction_update_is_elementwise(self) -> None:
        self.assertEqual(apply_reduction_update((7, 0, -8), (1, -1, -1), 4), [7, -1, -8])

    def test_update_helpers_reject_shape_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            apply_matrix_update(((1, 2), (3, 4)), ((1, 0),), 4)
        with self.assertRaises(ValueError):
            apply_reduction_update((1, 2), (1,), 4)


if __name__ == "__main__":
    unittest.main()
