"""Compact unit tests for the independent accelerator reference model."""

from __future__ import annotations

import unittest

from reference.functional import FunctionalReference, ReferenceConfig, Sample


class FunctionalApiTests(unittest.TestCase):
    def test_config_and_sample_types_are_required(self) -> None:
        W = [[1, 0], [0, 1]]
        R = [1, 1]
        config = ReferenceConfig(n=2, width=8, target_width=8)
        model = FunctionalReference(config, W, R)

        with self.assertRaises(TypeError):
            FunctionalReference({"n": 2}, W, R)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            model.step(([1, 0], 0, False))  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            model.run([([1, 0], 0, False)])  # type: ignore[list-item]

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
        records = model.run([Sample([4, -2], 100, False)])
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0].update_generated)
        self.assertEqual(records[0].matrix_update_directions, ((0, 0), (0, 0)))
        self.assertEqual(records[0].reduction_update_directions, (0, 0))
        self.assertEqual(model.final_W, W)
        self.assertEqual(model.final_R, R)


class FunctionalVisibilityTests(unittest.TestCase):
    def test_update_visibility_is_derived_from_n(self) -> None:
        for n, expected_delay in ((2, 5), (3, 7), (4, 9)):
            with self.subTest(n=n):
                config = ReferenceConfig(n=n)
                self.assertEqual(config.update_visibility_delay, expected_delay)

    def test_update_visibility_matches_w_and_r_generation_for_n2_n3_n4(self) -> None:
        for n, expected_delay in ((2, 5), (3, 7), (4, 9)):
            with self.subTest(n=n):
                config = ReferenceConfig(
                    n=n,
                    width=8,
                    fraction_bits=0,
                    target_width=8,
                    reduction_weight_width=8,
                    pass_through=True,
                )
                initial_W = tuple(tuple(1 for _ in range(n)) for _ in range(n))
                initial_R = tuple(1 for _ in range(n))
                samples = [
                    Sample((1,) * n, 127, True)
                    for _ in range(expected_delay + 1)
                ]

                records = FunctionalReference(config, initial_W, initial_R).run(
                    samples,
                    drain_updates=False,
                )

                self.assertEqual(config.update_visibility_delay, expected_delay)
                self.assertEqual(records[0].update_visible_at, expected_delay)
                self.assertEqual(records[expected_delay - 1].W_used, initial_W)
                self.assertEqual(records[expected_delay - 1].R_used, initial_R)
                self.assertEqual(
                    records[expected_delay].W_used,
                    tuple(tuple(2 for _ in range(n)) for _ in range(n)),
                )
                self.assertEqual(
                    records[expected_delay].R_used,
                    tuple(2 for _ in range(n)),
                )

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
        samples = [Sample((1, 2, 3), 127, True) for _ in range(10)]

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
