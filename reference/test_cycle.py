"""Architectural-state checks for the cycle-indexed Phase 6 reference."""

from __future__ import annotations

import unittest
from dataclasses import fields

from reference.arithmetic import (
    apply_matrix_update,
    apply_reduction_update,
    matrix_multiply,
)
from reference.cycle import CycleConfig, CycleInputs, CycleReference, CycleSnapshot
from reference.functional import FunctionalReference, ReferenceConfig, Sample


class CycleReferenceTests(unittest.TestCase):
    @staticmethod
    def config(**overrides: object) -> CycleConfig:
        values: dict[str, object] = {
            "n": 3,
            "width": 8,
            "fraction_bits": 0,
            "target_width": 8,
            "reduction_weight_width": 8,
            "pass_through": True,
            "reduce_output": False,
        }
        values.update(overrides)
        return CycleConfig(**values)  # type: ignore[arg-type]

    @staticmethod
    def initial_W() -> list[list[int]]:
        return [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    @staticmethod
    def drive(
        *,
        x: tuple[int, ...] = (1, 2, 3),
        target: int = 127,
        training: bool = True,
        valid: bool = True,
        ready: bool = True,
    ) -> CycleInputs:
        return CycleInputs(
            input_valid=valid,
            input_data=x,
            target_data=target,
            training_enable=training,
            result_ready=ready,
        )

    @staticmethod
    def architectural_state(snapshot: CycleSnapshot) -> tuple[object, ...]:
        return (
            snapshot.W,
            snapshot.R,
            snapshot.pending_weight_row,
            snapshot.input_fifo,
            snapshot.sample_context_fifo,
            snapshot.result_fifo,
        )

    @staticmethod
    def attempt_matrix_reload(model: CycleReference) -> bool:
        was_loaded = model.weights_loaded
        model.step(CycleInputs(reload_weights=True, result_ready=True))
        return was_loaded and not model.weights_loaded

    def test_snapshot_is_small_and_post_edge(self) -> None:
        self.assertEqual(
            [field.name for field in fields(CycleSnapshot)],
            [
                "cycle",
                "W",
                "R",
                "pending_weight_row",
                "input_fifo",
                "sample_context_fifo",
                "result_fifo",
            ],
        )
        model = CycleReference(self.config(), self.initial_W(), [16, 24, 32])
        snapshot = model.step(self.drive(training=False))
        self.assertEqual(snapshot.cycle, 0)
        self.assertEqual(snapshot.input_fifo, ((1, 2, 3),))
        self.assertEqual(snapshot.sample_context_fifo[0].input_signs, (1, 1, 1))
        self.assertEqual(snapshot.sample_context_fifo[0].target, 127)
        self.assertEqual(snapshot.result_fifo, ())

    def test_reduction_weights_are_resident_independently_of_w(self) -> None:
        # R loads over its own port, so an unloaded W must not zero it.
        model = CycleReference(self.config(), W=None, R=[16, 24, 32])
        self.assertFalse(model.weights_loaded)
        self.assertEqual(model.R, [16, 24, 32])
        model.reset()
        self.assertEqual(model.R, [16, 24, 32])
        # A hardware reset clears both.
        model.step(CycleInputs(reset_n=False))
        self.assertEqual(model.R, [0, 0, 0])
        self.assertEqual(model.W, [[0] * 3 for _ in range(3)])

    def test_configuration_change_requires_quiescence(self) -> None:
        self.assertNotIn("pass_through", {field.name for field in fields(CycleInputs)})
        self.assertNotIn("reduce_output", {field.name for field in fields(CycleInputs)})
        model = CycleReference(self.config(), self.initial_W(), [16, 24, 32])
        model.step(self.drive(training=False))

        with self.assertRaisesRegex(RuntimeError, "only when the accelerator is quiescent"):
            model.reconfigure(pass_through=False)
        with self.assertRaisesRegex(RuntimeError, "only when the accelerator is quiescent"):
            model.reconfigure(reduce_output=True)

        model.flush()
        model.reconfigure(pass_through=False, reduce_output=True)
        self.assertFalse(model.config.pass_through)
        self.assertTrue(model.config.reduce_output)

    def test_any_drained_sample_count_allows_reload(self) -> None:
        for sample_count in (1, 2, self.config().n, self.config().n + 1):
            with self.subTest(sample_count=sample_count):
                model = CycleReference(self.config(), self.initial_W(), [16, 24, 32])
                for index in range(sample_count):
                    model.step(self.drive(x=(index + 1, 0, 0), training=False))
                self.assertFalse(self.attempt_matrix_reload(model))
                model.flush()

                self.assertEqual(
                    model._retired_sample_indices,
                    list(range(sample_count)),
                )
                self.assertTrue(model.stream_quiescent)
                self.assertTrue(self.attempt_matrix_reload(model))

    def test_no_loss_no_duplication_tracks_outstanding_sample_contexts(self) -> None:
        model = CycleReference(
            self.config(output_fifo_depth=1),
            self.initial_W(),
            [16, 24, 32],
        )

        def check_accounting() -> None:
            accepted = model._next_sample_index
            retired = len(model._retired_sample_indices)
            self.assertEqual(
                accepted - retired,
                len(model._sample_context_fifo),
            )

        for index in range(9):
            model.step(self.drive(x=(index + 1, 1, -1), ready=True))
            check_accounting()
        for _ in range(3):
            model.step(self.drive(valid=False, ready=False))
            check_accounting()
        for _ in range(40):
            model.step(self.drive(valid=False, ready=True))
            check_accounting()
            if not model.in_flight:
                break

        self.assertEqual(model._enqueued_sample_indices, list(range(9)))
        self.assertEqual(model._retired_sample_indices, list(range(9)))

    def test_in_flight_depth_is_admission_capacity_and_skid_is_fixed(self) -> None:
        model = CycleReference(
            self.config(in_flight_depth=3),
            self.initial_W(),
            [16, 24, 32],
        )

        snapshots = [
            model.step(self.drive(x=(index + 1, 0, 0), training=False, ready=False))
            for index in range(12)
        ]

        self.assertEqual(model.config.sample_context_depth, 3)
        self.assertEqual(model.config.activation_skid_depth, 1)
        self.assertEqual(len(model._accepted_cycles), 3)
        self.assertLessEqual(max(len(snapshot.input_fifo) for snapshot in snapshots), 1)
        self.assertLessEqual(
            max(len(snapshot.sample_context_fifo) for snapshot in snapshots),
            model.config.sample_context_depth,
        )

    def test_absolute_n3_latency_is_e0_e7_e8(self) -> None:
        model = CycleReference(self.config(), self.initial_W(), [16, 24, 32])
        samples = [(1, 2, 3), (2, 0, -1), (-1, 1, 2)]
        snapshots = [
            model.step(self.drive(x=x, target=0, training=False))
            for x in samples
        ]
        for _ in range(4):
            snapshots.append(model.step(self.drive(valid=False, training=False)))

        # The first input is accepted at E0 and no complete result exists in
        # the architectural output storage through E6.
        self.assertEqual(model._accepted_cycles[:1], [0])
        self.assertTrue(all(snapshot.result_fifo == () for snapshot in snapshots[:7]))

        e7 = model.step(self.drive(x=(0, -2, 1), target=0, training=False))
        self.assertEqual(e7.cycle, 7)
        self.assertEqual(len(e7.result_fifo), 1)
        self.assertEqual(
            e7.result_fifo[0].activated_result,
            tuple(matrix_multiply(samples[0], self.initial_W(), self.config().width)),
        )

        e8 = model.step(self.drive(x=(3, 1, 0), target=0, training=False))
        self.assertEqual(e8.cycle, 8)
        self.assertEqual(model._enqueue_cycles[:2], [7, 8])
        self.assertEqual(model._retirement_cycles[:1], [8])
        self.assertEqual(len(e8.result_fifo), 1)
        self.assertEqual(
            e8.result_fifo[0].activated_result,
            tuple(matrix_multiply(samples[1], self.initial_W(), self.config().width)),
        )

    def test_incomplete_observation_fails_at_result_boundary(self) -> None:
        model = CycleReference(self.config(), self.initial_W(), [16, 24, 32])
        model.step(self.drive(training=False))
        model.step(self.drive(valid=False, training=False))
        token = model._data_tokens[0]
        self.assertTrue(
            all(value is None for row in token.observed_weights for value in row)
        )

        # Let the first anti-diagonal be observed, then model a missing PE
        # observation before the token reaches its final anti-diagonal.
        model.step(self.drive(valid=False, training=False))
        token.observed_weights[0][0] = None

        with self.assertRaisesRegex(
            AssertionError,
            "sample reached result boundary without all PE weights",
        ):
            for _ in range(2 * model.config.n - 2):
                model.step(self.drive(valid=False, training=False))

    def test_continuous_throughput_is_one_result_per_cycle_after_fill(self) -> None:
        model = CycleReference(self.config(), self.initial_W(), [16, 24, 32])
        for index in range(12):
            model.step(self.drive(x=(index + 1, 0, 0), target=0, training=False))

        self.assertEqual(model._accepted_cycles, list(range(12)))
        self.assertEqual(model._enqueued_sample_indices[:5], list(range(5)))
        self.assertEqual(model._retired_sample_indices[:4], list(range(4)))
        self.assertEqual(model._enqueue_cycles[:5], [7, 8, 9, 10, 11])
        self.assertEqual(model._retirement_cycles[:4], [8, 9, 10, 11])

    def test_feedback_visibility_is_s0_s7_s1_s8_s2_s9(self) -> None:
        initial_W = self.initial_W()
        initial_R = [16, 24, 32]
        model = CycleReference(self.config(), initial_W, initial_R)
        for _ in range(10):
            model.step(self.drive())
        model.flush()

        records = model._sample_results
        old_W = tuple(tuple(row) for row in initial_W)
        old_R = tuple(initial_R)
        self.assertEqual([record.W_used for record in records[:7]], [old_W] * 7)
        self.assertEqual([record.R_used for record in records[:7]], [old_R] * 7)
        for sample_index in (7, 8, 9):
            generation = sample_index - 6
            self.assertEqual(
                records[sample_index].W_used,
                tuple(tuple(value + generation for value in row) for row in initial_W),
            )
            self.assertEqual(records[sample_index].R_used, tuple(value + generation for value in initial_R))

    def test_buffer_snapshots_cover_startup_fill_and_drain(self) -> None:
        model = CycleReference(self.config(), self.initial_W(), [16, 24, 32])
        first = model.step(self.drive(x=(1, -2, 0), target=17, training=False))
        self.assertIsNone(first.pending_weight_row)
        self.assertEqual(first.input_fifo, ((1, -2, 0),))
        self.assertEqual(
            first.sample_context_fifo[0].input_signs,
            (1, -1, 0),
        )

        for x in ((2, 3, 0), (-1, 0, 4), (3, -3, 1)):
            model.step(self.drive(x=x, target=0, training=False))
        fill = model.snapshots[-1]
        self.assertEqual(len(fill.input_fifo), 1)
        self.assertEqual(len(fill.sample_context_fifo), 4)

        for x in ((4, 1, 0), (0, 2, 2), (-2, 1, 3), (1, 1, -1)):
            model.step(self.drive(x=x, target=0, training=False))
        result_storage = model.snapshots[7]
        self.assertEqual(
            tuple(entry.activated_result for entry in result_storage.result_fifo),
            (tuple(matrix_multiply((1, -2, 0), self.initial_W(), self.config().width)),),
        )
        self.assertEqual(
            result_storage.result_fifo[0].reduction_weight_signs,
            (1, 1, 1),
        )

        model.flush()
        drained = model.snapshots[-1]
        self.assertEqual(drained.input_fifo, ())
        self.assertEqual(drained.sample_context_fifo, ())
        self.assertEqual(drained.result_fifo, ())

    def test_pending_weight_snapshot_is_logical_host_order(self) -> None:
        model = CycleReference(self.config(), R=[16, 24, 32])
        e0 = model.step(CycleInputs(weight_valid=True, weight_data=(7, 8, 9)))
        e1 = model.step(CycleInputs(weight_valid=True, weight_data=(4, 5, 6)))
        e2 = model.step(CycleInputs(weight_valid=True, weight_data=(1, 2, 3)))
        self.assertEqual(e0.pending_weight_row, (7, 8, 9))
        self.assertEqual(e1.pending_weight_row, (4, 5, 6))
        self.assertEqual(e2.pending_weight_row, (1, 2, 3))
        model.step(CycleInputs(result_ready=True))
        self.assertEqual(model.W, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def test_pending_weight_stage_supports_simultaneous_refill(self) -> None:
        for n in (2, 3, 4):
            with self.subTest(n=n):
                rows = tuple(
                    tuple(row * n + lane + 1 for lane in range(n))
                    for row in range(n)
                )
                model = CycleReference(self.config(n=n), R=[1] * n)

                snapshots = [
                    model.step(CycleInputs(weight_valid=True, weight_data=row))
                    for row in rows
                ]
                self.assertEqual(
                    [snapshot.pending_weight_row for snapshot in snapshots],
                    list(rows),
                )
                zero_row = [0] * n
                for index, snapshot in enumerate(snapshots):
                    expected_rows = [list(row) for row in reversed(rows[:index])]
                    expected_rows.extend([zero_row] * (n - len(expected_rows)))
                    self.assertEqual(snapshot.W, tuple(tuple(row) for row in expected_rows))

    def test_pending_weight_stage_rejects_extra_row_on_final_consumption(self) -> None:
        for n in (2, 3, 4):
            with self.subTest(n=n):
                rows = tuple(
                    tuple(row * n + lane + 1 for lane in range(n))
                    for row in range(n)
                )
                extra = tuple(90 + lane for lane in range(n))
                model = CycleReference(self.config(n=n), R=[1] * n)

                for row in rows:
                    model.step(CycleInputs(weight_valid=True, weight_data=row))
                final = model.step(CycleInputs(weight_valid=True, weight_data=extra))
                self.assertTrue(model.weights_loaded)
                self.assertIsNone(final.pending_weight_row)
                self.assertEqual(model.W, [list(row) for row in reversed(rows)])

                reloaded = model.step(CycleInputs(reload_weights=True, result_ready=True))
                self.assertFalse(model.weights_loaded)
                self.assertIsNone(reloaded.pending_weight_row)

                first_reloaded = model.step(
                    CycleInputs(weight_valid=True, weight_data=extra)
                )
                self.assertEqual(first_reloaded.pending_weight_row, extra)

    def test_input_bubbles_do_not_stop_learning(self) -> None:
        initial_W = self.initial_W()
        initial_R = [16, 24, 32]
        model = CycleReference(self.config(), initial_W, initial_R)
        model.step(self.drive())
        bubble_snapshots = [
            model.step(self.drive(valid=False))
            for _ in range(14)
        ]

        self.assertEqual(bubble_snapshots[7].W[0][0], initial_W[0][0] + 1)
        self.assertEqual(bubble_snapshots[11].W, tuple(tuple(v + 1 for v in row) for row in initial_W))
        self.assertEqual(bubble_snapshots[12].R, tuple(value + 1 for value in initial_R))

    def test_output_backpressure_holds_architectural_state(self) -> None:
        model = CycleReference(
            self.config(output_fifo_depth=1),
            self.initial_W(),
            [16, 24, 32],
        )
        for _ in range(9):
            model.step(self.drive(training=False))
        held_snapshots = [
            model.step(self.drive(valid=False, ready=False, training=False))
            for _ in range(3)
        ]
        self.assertTrue(
            all(
                self.architectural_state(snapshot) == self.architectural_state(held_snapshots[0])
                for snapshot in held_snapshots[1:]
            )
        )

    def test_backpressure_release_preserves_order_and_updates(self) -> None:
        model = CycleReference(
            self.config(output_fifo_depth=1),
            self.initial_W(),
            [16, 24, 32],
        )
        for _ in range(9):
            model.step(self.drive())
        for _ in range(3):
            model.step(self.drive(valid=False, ready=False))
        for _ in range(40):
            model.step(self.drive(valid=False, ready=True))
            if not model.in_flight:
                break

        self.assertEqual(model._retired_sample_indices, list(range(9)))
        self.assertEqual(model._enqueued_sample_indices, list(range(9)))
        self.assertEqual(len(model._sample_results), 9)
        expected_W = [row[:] for row in self.initial_W()]
        expected_R = [16, 24, 32]
        for record in model._sample_results:
            if record.update_generated:
                expected_W = apply_matrix_update(expected_W, record.matrix_update_directions, model.config.width)
                expected_R = apply_reduction_update(expected_R, record.reduction_update_directions, model.config.reduction_weight_width)
        self.assertEqual(model.W, expected_W)
        self.assertEqual(model.R, expected_R)

    def test_overlapping_updates_change_post_edge_W_on_the_wave_clocks(self) -> None:
        model = CycleReference(self.config(), [[0] * 3 for _ in range(3)], [64, 64, 64])
        snapshots = [model.step(self.drive()) for _ in range(10)]

        self.assertEqual(snapshots[8].W[0][0], 1)
        self.assertEqual(snapshots[9].W[0][0], 2)
        self.assertEqual(snapshots[9].W[0][1], 1)
        self.assertEqual(snapshots[9].W[1][0], 1)

    def test_saturation_is_sequential_under_overlapping_updates(self) -> None:
        config = self.config(width=4, target_width=8, reduction_weight_width=4)
        initial_W = [[7, 7, 7], [-8, -8, -8], [0, 0, 0]]
        initial_R = [7, -8, 0]
        targets = [127, -128, 127, -128, 127, -128, 127, -128]
        model = CycleReference(config, initial_W, initial_R)
        for target in targets:
            model.step(self.drive(target=target))
        model.flush()

        expected_W = [row[:] for row in initial_W]
        expected_R = initial_R[:]
        for record in model._sample_results:
            if record.update_generated:
                expected_W = apply_matrix_update(expected_W, record.matrix_update_directions, config.width)
                expected_R = apply_reduction_update(expected_R, record.reduction_update_directions, config.reduction_weight_width)
        self.assertEqual(model.W, expected_W)
        self.assertEqual(model.R, expected_R)
        self.assertTrue(all(-8 <= value <= 7 for row in model.W for value in row))
        self.assertTrue(all(-8 <= value <= 7 for value in model.R))

    def test_closed_form_update_visibility_matches_wave_propagation(self) -> None:
        """Cross-check weight visibility: functional derives the 2*N+1 sample delay
        in closed form; cycle derives it independently through wave propagation."""

        for n in (2, 3, 4):
            with self.subTest(n=n):
                config = self.config(n=n)
                functional_config = ReferenceConfig(
                    n=n,
                    width=config.width,
                    fraction_bits=config.fraction_bits,
                    target_width=config.target_width,
                    reduction_weight_width=config.reduction_weight_width,
                    pass_through=config.pass_through,
                )
                W = [[1 + row * n + column for column in range(n)] for row in range(n)]
                R = [16 + 8 * lane for lane in range(n)]
                # Inference samples emit no package, so the stride varies.
                raw_samples = [
                    ((1, 2, 3), 127, True),
                    ((-2, 3, 1), -20, False),
                    ((3, -1, 2), 40, True),
                    ((0, 2, -3), 0, True),
                    ((-1, -2, -3), -60, True),
                    ((4, 1, 0), 12, False),
                    ((2, 2, 1), 100, True),
                    ((-3, 0, 2), -40, True),
                    ((1, -4, 3), 25, True),
                    ((2, -2, -1), 7, False),
                ]
                samples = [
                    Sample(
                        tuple(x[lane % len(x)] for lane in range(n)),
                        target,
                        training,
                    )
                    for x, target, training in raw_samples
                ]

                functional = FunctionalReference(functional_config, W, R)
                functional_records = functional.run(samples)
                cycle = CycleReference(config, W, R)
                for sample in samples:
                    cycle.step(
                        self.drive(
                            x=sample.x,
                            target=sample.target,
                            training=sample.training_enable,
                        )
                    )
                cycle.flush()

                self.assertEqual(len(cycle._sample_results), len(functional_records))

                # The load-bearing comparison.
                self.assertEqual(
                    [record.W_used for record in cycle._sample_results],
                    [record.W_used for record in functional_records],
                )
                self.assertEqual(
                    [record.R_used for record in cycle._sample_results],
                    [record.R_used for record in functional_records],
                )
                self.assertEqual(cycle.W, functional.final_W)
                self.assertEqual(cycle.R, functional.final_R)

                # Guard against a vacuous check.
                observed_generations = {
                    record.W_used for record in cycle._sample_results
                }
                self.assertGreater(len(observed_generations), 1)

                # The emergent delay must equal the closed form.
                delay = functional_config.update_visibility_delay
                self.assertEqual(delay, 2 * n + 1)
                self.assertEqual(functional_records[0].update_visible_at, delay)
                self.assertEqual(
                    cycle._sample_results[delay - 1].W_used,
                    cycle._sample_results[0].W_used,
                )
                self.assertNotEqual(
                    cycle._sample_results[delay].W_used,
                    cycle._sample_results[0].W_used,
                )


if __name__ == "__main__":
    unittest.main()
