"""Focused cycle-level checks for the Phase 6C architectural model."""

from __future__ import annotations

import unittest

from reference.arithmetic import apply_matrix_update, apply_reduction_update
from reference.cycle import CycleConfig, CycleReference, format_trace
from reference.functional import FunctionalReference


class CycleReferenceTests(unittest.TestCase):
    def make_config(
        self,
        *,
        n: int = 3,
        width: int = 8,
        fraction_bits: int = 0,
        target_width: int = 8,
        reduction_weight_width: int = 8,
        input_fifo_depth: int | None = None,
        output_fifo_depth: int | None = None,
        pass_through: bool = True,
    ) -> CycleConfig:
        return CycleConfig(
            n=n,
            width=width,
            fraction_bits=fraction_bits,
            target_width=target_width,
            reduction_weight_width=reduction_weight_width,
            pass_through=pass_through,
            input_fifo_depth=input_fifo_depth,
            output_fifo_depth=output_fifo_depth,
        )

    @staticmethod
    def initial_matrix(n: int = 3) -> list[list[int]]:
        return [
            [row * n + column + 1 for column in range(n)]
            for row in range(n)
        ]

    @staticmethod
    def sample_cycle(
        model: CycleReference,
        *,
        x: list[int] | tuple[int, ...] = (1, 2, 3),
        target: int = 127,
        training: bool = True,
        valid: bool = True,
        ready: bool = True,
        reduce_output: bool = True,
    ):
        return model.step(
            activation_valid=valid,
            activation_data=x,
            target_data=target,
            training_enable=training,
            result_ready=ready,
            reduce_output=reduce_output,
        )

    def test_empty_pipeline_startup(self) -> None:
        config = self.make_config()
        W = self.initial_matrix()
        R = [16, 24, 32]
        model = CycleReference(config, W, R)

        snapshot = model.step(result_ready=True)

        self.assertEqual(snapshot.cycle, 0)
        self.assertTrue(snapshot.datapath_advance)
        self.assertFalse(snapshot.result_valid)
        self.assertIsNone(snapshot.accepted_sample)
        self.assertEqual(snapshot.activation_fifo_contents, ())
        self.assertEqual(snapshot.sample_context_fifo_contents, ())
        self.assertEqual(snapshot.output_fifo_contents, ())
        self.assertEqual(snapshot.W, tuple(tuple(row) for row in W))
        self.assertEqual(snapshot.R, tuple(R))

    def test_continuous_n3_accepts_one_sample_per_cycle(self) -> None:
        model = CycleReference(
            self.make_config(),
            self.initial_matrix(),
            [16, 24, 32],
        )

        snapshots = [self.sample_cycle(model) for _ in range(10)]
        model.flush()

        self.assertEqual(
            [snapshot.accepted_sample for snapshot in snapshots],
            list(range(10)),
        )
        self.assertTrue(all(snapshot.datapath_advance for snapshot in snapshots))
        self.assertEqual(
            [record.sample_index for record in model.records],
            list(range(10)),
        )
        self.assertTrue(all(record.result_enqueue_cycle is not None for record in model.records))
        self.assertEqual(
            [snapshot.retired_sample for snapshot in model.snapshots if snapshot.result_last],
            [2, 5, 8],
        )

    def test_exact_n3_feedback_timing_s0_to_s7(self) -> None:
        config = self.make_config()
        W = self.initial_matrix()
        R = [16, 24, 32]
        model = CycleReference(config, W, R)

        for _ in range(10):
            self.sample_cycle(model)
        model.flush()
        records = model.records

        old_W = tuple(tuple(row) for row in W)
        old_R = tuple(R)
        self.assertEqual([record.W_used for record in records[:7]], [old_W] * 7)
        self.assertEqual([record.R_used for record in records[:7]], [old_R] * 7)
        self.assertEqual(records[7].W_used, tuple(tuple(value + 1 for value in row) for row in W))
        self.assertEqual(records[7].R_used, tuple(value + 1 for value in R))
        self.assertEqual(records[8].W_used, tuple(tuple(value + 2 for value in row) for row in W))
        self.assertEqual(records[8].R_used, tuple(value + 2 for value in R))

    def test_input_bubbles_advance_learning_wave(self) -> None:
        model = CycleReference(
            self.make_config(),
            self.initial_matrix(),
            [16, 24, 32],
        )

        snapshots = [self.sample_cycle(model)]
        snapshots.extend(
            self.sample_cycle(model, valid=False) for _ in range(14)
        )

        generated = next(snapshot for snapshot in snapshots if snapshot.matrix_update_valid)
        update_id = generated.matrix_update_packages[0].update_id
        applied = [
            (snapshot.cycle, diagonal)
            for snapshot in snapshots
            for package_id, diagonal in snapshot.matrix_updates_applied
            if package_id == update_id
        ]

        self.assertEqual([diagonal for _, diagonal in applied], [0, 1, 2, 3, 4])
        self.assertEqual([cycle for cycle, _ in applied], list(range(generated.cycle, generated.cycle + 5)))
        self.assertTrue(all(snapshot.datapath_bubble for snapshot in snapshots[generated.cycle:generated.cycle + 5]))

    def test_output_backpressure_freezes_shared_state(self) -> None:
        config = self.make_config(output_fifo_depth=1)
        model = CycleReference(config, self.initial_matrix(), [16, 24, 32])

        for _ in range(9):
            self.sample_cycle(model, ready=True)
        held = self.sample_cycle(model, valid=False, ready=False)
        frozen = self.sample_cycle(model, valid=False, ready=False)

        self.assertFalse(held.datapath_advance)
        self.assertFalse(frozen.datapath_advance)
        self.assertEqual(held.W, frozen.W)
        self.assertEqual(held.R, frozen.R)
        self.assertEqual(held.skew_slots, frozen.skew_slots)
        self.assertEqual(held.sample_positions, frozen.sample_positions)
        self.assertEqual(held.result_alignment_contents, frozen.result_alignment_contents)
        self.assertEqual(held.matrix_update_packages, frozen.matrix_update_packages)
        self.assertEqual(held.reduction_update_packages, frozen.reduction_update_packages)

    def test_backpressure_release_preserves_result_order(self) -> None:
        config = self.make_config(output_fifo_depth=1)
        model = CycleReference(config, self.initial_matrix(), [16, 24, 32])
        for _ in range(9):
            self.sample_cycle(model, ready=True)
        for _ in range(3):
            self.sample_cycle(model, valid=False, ready=False)

        released = [
            self.sample_cycle(model, valid=False, ready=True)
            for _ in range(30)
        ]
        retired = [
            snapshot.retired_sample
            for snapshot in model.snapshots
            if snapshot.result_retired and snapshot.retired_sample is not None
        ]

        self.assertEqual(retired, list(range(9)))
        self.assertTrue(any(snapshot.datapath_advance for snapshot in released))
        self.assertEqual(len(set(retired)), len(retired))

    def test_consecutive_training_updates_overlap(self) -> None:
        model = CycleReference(
            self.make_config(),
            self.initial_matrix(),
            [16, 24, 32],
        )
        for _ in range(14):
            self.sample_cycle(model)

        self.assertTrue(
            any(len(snapshot.matrix_update_packages) >= 2 for snapshot in model.snapshots)
        )
        self.assertTrue(
            any(len(snapshot.reduction_update_packages) >= 2 for snapshot in model.snapshots)
        )
        model.flush()
        self.assertEqual(len(model.generated_matrix_updates), 14)
        self.assertEqual(len(model.generated_reduction_updates), 14)

    def test_mixed_training_and_inference_samples(self) -> None:
        model = CycleReference(
            self.make_config(),
            self.initial_matrix(),
            [16, 24, 32],
        )
        training = [False, True, False, True, True, False, True]
        for enabled in training:
            self.sample_cycle(model, training=enabled)
        model.flush()

        self.assertEqual(
            [record.update_generated for record in model.records],
            training,
        )
        self.assertEqual(
            [package.source_sample for package in model.generated_matrix_updates],
            [index for index, enabled in enumerate(training) if enabled],
        )
        self.assertEqual(
            [package.source_sample for package in model.generated_reduction_updates],
            [index for index, enabled in enumerate(training) if enabled],
        )

    def test_output_fifo_full_allows_simultaneous_pop_and_push(self) -> None:
        model = CycleReference(
            self.make_config(output_fifo_depth=1),
            self.initial_matrix(),
            [16, 24, 32],
        )
        snapshots = [self.sample_cycle(model, ready=True) for _ in range(10)]
        simultaneous = [
            snapshot
            for snapshot in snapshots
            if snapshot.result_retired and snapshot.result_enqueued
        ]

        self.assertTrue(simultaneous)
        snapshot = simultaneous[0]
        self.assertTrue(snapshot.datapath_advance)
        self.assertEqual(len(snapshot.output_fifo_contents), 1)
        self.assertEqual(snapshot.output_fifo_contents[0].sample_index, snapshot.retired_sample + 1)

    def test_saturation_remains_sequential_under_overlap(self) -> None:
        config = self.make_config(
            width=4,
            target_width=8,
            reduction_weight_width=4,
        )
        max_value = 7
        W = [[max_value for _ in range(3)] for _ in range(3)]
        R = [8, 8, 8]
        model = CycleReference(config, W, R)
        targets = [127, -128, 127, -128, 127, -128, 127, -128]
        for target in targets:
            self.sample_cycle(model, target=target)
        model.flush()

        expected_W = [row[:] for row in W]
        expected_R = R[:]
        for record in model.records:
            if record.update_generated:
                expected_W = apply_matrix_update(
                    expected_W,
                    record.matrix_update_directions,
                    config.width,
                )
                expected_R = apply_reduction_update(
                    expected_R,
                    record.reduction_update_directions,
                    config.reduction_weight_width,
                )
        self.assertEqual(model.final_W, expected_W)
        self.assertEqual(model.final_R, expected_R)
        self.assertTrue(any(len(snapshot.matrix_update_packages) >= 2 for snapshot in model.snapshots))
        self.assertTrue(all(-8 <= value <= 7 for row in model.final_W for value in row))

    def test_streamed_weight_loading_matches_rtl_row_order(self) -> None:
        config = self.make_config()
        model = CycleReference(config, R=[16, 24, 32])
        host_rows = [(7, 8, 9), (4, 5, 6), (1, 2, 3)]
        for row in host_rows:
            model.step(weight_valid=True, weight_data=row, result_ready=True)
        # The first cycle fills the weight FIFO; the next three cycles pop the
        # reverse-order host frame through the vertical loader.
        for _ in range(3):
            model.step(result_ready=True)

        self.assertTrue(model.weights_loaded if hasattr(model, "weights_loaded") else model.snapshots[-1].weights_loaded)
        self.assertEqual(model.final_W, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def test_cycle_and_functional_models_cross_check_no_stall(self) -> None:
        config = self.make_config()
        W = self.initial_matrix()
        R = [16, 24, 32]
        samples = [
            ([1, 2, 3], 127, True),
            ([-2, 3, 1], -20, False),
            ([3, -1, 2], 40, True),
            ([0, 2, -3], 0, True),
            ([-1, -2, -3], -60, True),
            ([4, 1, 0], 12, False),
            ([2, 2, 1], 100, True),
            ([-3, 0, 2], -40, True),
            ([1, -4, 3], 25, True),
            ([2, -2, -1], 7, False),
        ]
        functional = FunctionalReference(config, W, R)
        functional_records = functional.run(samples)
        cycle = CycleReference(config, W, R)
        for x, target, training in samples:
            self.sample_cycle(cycle, x=x, target=target, training=training)
        cycle.flush()

        self.assertEqual(len(cycle.records), len(functional_records))
        for expected, actual in zip(functional_records, cycle.records):
            self.assertEqual(actual.raw_matrix_result, expected.raw_matrix_result)
            self.assertEqual(actual.activated_result, expected.activated_result)
            self.assertEqual(actual.prediction, expected.prediction)
            self.assertEqual(actual.learning_direction, expected.learning_direction)
            self.assertEqual(actual.W_used, expected.W_used)
            self.assertEqual(actual.R_used, expected.R_used)
            self.assertEqual(actual.matrix_update_directions, expected.matrix_update_directions)
            self.assertEqual(actual.reduction_update_directions, expected.reduction_update_directions)
        self.assertEqual(cycle.final_W, functional.final_W)
        self.assertEqual(cycle.final_R, functional.final_R)
        self.assertEqual(
            [package.source_sample for package in cycle.generated_matrix_updates],
            [record.sample_index for record in functional_records if record.update_generated],
        )
        self.assertEqual(
            [package.source_sample for package in cycle.generated_reduction_updates],
            [record.sample_index for record in functional_records if record.update_generated],
        )

    def test_trace_printer_is_compact_but_snapshot_is_full(self) -> None:
        model = CycleReference(self.make_config(), self.initial_matrix(), [16, 24, 32])
        snapshot = self.sample_cycle(model)
        trace = format_trace(snapshot)
        self.assertIn("C0:", trace)
        self.assertIn("advance=1", trace)
        self.assertIsInstance(snapshot.W, tuple)
        self.assertIsInstance(snapshot.activation_fifo_contents, tuple)


if __name__ == "__main__":
    unittest.main()
