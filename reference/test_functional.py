"""Compact unit tests for the independent accelerator reference model."""

from __future__ import annotations

import unittest

from reference.arithmetic import (
    activate,
    apply_matrix_update,
    apply_reduction_update,
    fixed_point_rescale,
    learning_direction,
    matrix_multiply,
    matrix_update_directions,
    reduction_update_directions,
    saturating_lsb_update,
    ternary_sign,
    weighted_vector_reduction,
)
from reference.functional import FunctionalReference, ReferenceConfig


class ArithmeticTests(unittest.TestCase):
    def test_signed_matrix_and_direction_signs(self) -> None:
        self.assertEqual(ternary_sign(7), 1)
        self.assertEqual(ternary_sign(-7), -1)
        self.assertEqual(ternary_sign(0), 0)
        self.assertEqual(
            matrix_multiply([-2, 3], [[2, -1], [1, -2]], width=4),
            [-1, -4],
        )

    def test_learning_directions_include_positive_zero_negative(self) -> None:
        self.assertEqual(learning_direction(5, 2, 8, 8), 1)
        self.assertEqual(learning_direction(2, 2, 8, 8), 0)
        self.assertEqual(learning_direction(-3, 2, 8, 8), -1)

    def test_relu_pass_through_and_closed_gate(self) -> None:
        raw = [-3, 0, 4]
        self.assertEqual(activate(raw, pass_through=False, width=8), [0, 0, 4])
        self.assertEqual(activate(raw, pass_through=True, width=8), raw)

        _, relu_columns, _ = matrix_update_directions(
            [1, 1],
            [0, 4],
            [3, 3],
            1,
            width=8,
            reduction_weight_width=8,
            pass_through=False,
        )
        self.assertEqual(relu_columns, [0, 1])

        _, pass_columns, _ = matrix_update_directions(
            [1, 1],
            [-3, 0],
            [3, 3],
            1,
            width=8,
            reduction_weight_width=8,
            pass_through=True,
        )
        self.assertEqual(pass_columns, [1, 1])

    def test_reduction_coefficient_signs_zero_and_update_signs(self) -> None:
        _, columns, matrix_direction = matrix_update_directions(
            [1, -1, 0],
            [2, -2, 0],
            [5, -5, 0],
            1,
            width=8,
            reduction_weight_width=8,
            pass_through=True,
        )
        self.assertEqual(columns, [1, -1, 0])
        self.assertEqual(matrix_direction, [[1, -1, 0], [-1, 1, 0], [0, 0, 0]])
        self.assertEqual(reduction_update_directions([2, -2, 0], 1), [1, -1, 0])
        self.assertEqual(reduction_update_directions([2, -2, 0], -1), [-1, 1, 0])

    def test_training_disabled_does_not_change_state(self) -> None:
        config = ReferenceConfig(
            n=2,
            width=8,
            fraction_bits=4,
            target_width=8,
            reduction_weight_width=8,
            pass_through=True,
        )
        W = [[2, -1], [1, 3]]
        R = [64, -32]
        model = FunctionalReference(config, W, R)
        records = model.run([([4, -2], 100, False)])
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0].update_generated)
        self.assertEqual(records[0].matrix_update_directions, ((0, 0), (0, 0)))
        self.assertEqual(records[0].reduction_update_directions, (0, 0))
        self.assertEqual(model.final_W, W)
        self.assertEqual(model.final_R, R)

    def test_saturating_matrix_and_reduction_updates_at_both_endpoints(self) -> None:
        matrix_max = 7
        matrix_min = -8
        reduction_max = 127
        reduction_min = -128
        self.assertEqual(saturating_lsb_update(matrix_max, 1, 4), matrix_max)
        self.assertEqual(saturating_lsb_update(matrix_min, -1, 4), matrix_min)
        self.assertEqual(
            apply_matrix_update(
                [[matrix_max, matrix_min], [0, 0]],
                [[1, -1], [0, 0]],
                4,
            ),
            [[matrix_max, matrix_min], [0, 0]],
        )
        self.assertEqual(saturating_lsb_update(reduction_max, 1, 8), reduction_max)
        self.assertEqual(saturating_lsb_update(reduction_min, -1, 8), reduction_min)
        self.assertEqual(
            apply_reduction_update([reduction_max, reduction_min], [1, -1], 8),
            [reduction_max, reduction_min],
        )

    def test_fixed_point_rescaling_and_cancellation(self) -> None:
        self.assertEqual(fixed_point_rescale(2048, 4, 16), 128)
        # A matrix result carries 2*4 fractional bits.  The full product is
        # shifted once by 4+7, returning the target-scale stored value 8
        # (a matrix value of 1.0 multiplied by a Q1.7 coefficient of 0.5).
        self.assertEqual(
            weighted_vector_reduction([256, 0], [64, 0], 8, 8, 4),
            8,
        )
        self.assertEqual(
            weighted_vector_reduction([16, -16], [64, 64], 8, 8, 4),
            0,
        )


class FunctionalVisibilityTests(unittest.TestCase):
    def test_n3_continuous_training_visibility_is_s7_s8_s9(self) -> None:
        config = ReferenceConfig(
            n=3,
            width=8,
            fraction_bits=0,
            target_width=8,
            reduction_weight_width=8,
            pass_through=True,
        )
        initial_W = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        initial_R = [16, 24, 32]
        model = FunctionalReference(config, initial_W, initial_R)
        samples = [([1, 2, 3], 127, True) for _ in range(10)]

        records = model.run(samples)

        self.assertEqual([record.W_used for record in records[:7]], [
            tuple(tuple(row) for row in initial_W)
        ] * 7)
        self.assertEqual([record.R_used for record in records[:7]], [
            tuple(initial_R)
        ] * 7)

        for sample_index in (7, 8, 9):
            generation = sample_index - 6
            expected_W = tuple(
                tuple(value + generation for value in row)
                for row in initial_W
            )
            expected_R = tuple(value + generation for value in initial_R)
            self.assertEqual(records[sample_index].W_used, expected_W)
            self.assertEqual(records[sample_index].R_used, expected_R)
            self.assertEqual(records[sample_index].learning_direction, 1)
            self.assertTrue(records[sample_index].update_generated)

        self.assertEqual(
            [record.update_visible_at for record in records[:3]],
            [7, 8, 9],
        )
        self.assertEqual(model.final_W, [
            [value + 10 for value in row] for row in initial_W
        ])
        self.assertEqual(model.final_R, [value + 10 for value in initial_R])


if __name__ == "__main__":
    unittest.main()
