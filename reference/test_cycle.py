"""Architectural-state checks for the cycle-indexed Phase 6 reference."""

from __future__ import annotations

import random
import unittest
from dataclasses import fields, replace

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
            snapshot.weight_buffer,
            snapshot.weights_loaded,
            snapshot.loaded_weight_rows,
            snapshot.loading_weights,
            snapshot.load_step,
            snapshot.capture_weights,
            snapshot.input_stage,
            snapshot.sample_context_fifo,
            snapshot.result_fifo,
        )

    @staticmethod
    def attempt_matrix_reload(model: CycleReference) -> bool:
        was_loaded = model.weights_loaded
        model.step(CycleInputs(reload_weights=True, result_ready=True))
        return was_loaded and not model.weights_loaded

    def test_snapshot_is_semantic_and_post_edge(self) -> None:
        self.assertEqual(
            [field.name for field in fields(CycleSnapshot)],
            [
                "cycle",
                "W",
                "R",
                "weight_buffer",
                "weights_loaded",
                "loaded_weight_rows",
                "loading_weights",
                "load_step",
                "capture_weights",
                "input_stage",
                "sample_context_fifo",
                "result_fifo",
            ],
        )
        model = CycleReference(self.config(), self.initial_W(), [16, 24, 32])
        snapshot = model.step(self.drive(training=False))
        self.assertEqual(snapshot.cycle, 0)
        self.assertEqual(snapshot.input_stage, (1, 2, 3))
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
                    [index for index, timing in enumerate(model.timings) if timing.retired_at is not None],
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
            accepted = len(model.timings)
            completed = sum(timing.enqueued_at is not None for timing in model.timings)
            retired = sum(timing.retired_at is not None for timing in model.timings)
            self.assertEqual(
                accepted - completed,
                len(model._sample_context_fifo),
            )
            self.assertEqual(completed - retired, len(model._result_fifo))

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

        self.assertEqual([record.sample_index for record in model.records], list(range(9)))
        self.assertTrue(all(timing.retired_at is not None for timing in model.timings))

    def test_in_flight_depth_is_admission_capacity(self) -> None:
        model = CycleReference(
            self.config(in_flight_depth=3),
            self.initial_W(),
            [16, 24, 32],
        )

        snapshots = [
            model.step(self.drive(x=(index + 1, 0, 0), training=False, ready=False))
            for index in range(24)
        ]

        self.assertEqual(model.config.sample_context_depth, 3)
        self.assertEqual(len(model.timings), 3 + model.config.output_fifo_depth)
        self.assertEqual(len(snapshots[-1].result_fifo), model.config.output_fifo_depth)
        self.assertLessEqual(
            max(len(snapshot.sample_context_fifo) for snapshot in snapshots),
            model.config.sample_context_depth,
        )

    def test_absolute_latency_and_single_vector_all_sizes(self) -> None:
        for n in (1, 2, 3, 4, 5, 8):
            with self.subTest(n=n):
                W = [[(-1 if (r + c) % 2 else 1) * (r * n + c + 1)
                      for c in range(n)] for r in range(n)]
                x = tuple(2 * r - n for r in range(n))
                model = CycleReference(self.config(n=n), W, [1] * n)
                model.step(self.drive(x=x, training=False))
                for _ in range(n + 1):
                    snapshot = model.step(self.drive(valid=False, training=False))
                    self.assertEqual(snapshot.result_fifo, ())
                enqueued = model.step(self.drive(valid=False, training=False))
                self.assertEqual(enqueued.cycle, n + 2)
                self.assertEqual(enqueued.result_fifo[0].activated_result,
                                 tuple(matrix_multiply(x, W, model.config.width)))
                model.step(self.drive(valid=False, training=False))
                self.assertEqual(model.timings[0].accepted_at, 0)
                self.assertEqual(model.timings[0].enqueued_at, n + 2)
                self.assertEqual(model.timings[0].retired_at, n + 3)

    def test_training_finishes_with_consumer_held_and_empty_context_all_sizes(self) -> None:
        for n in (2, 3, 4, 5, 8):
            with self.subTest(n=n):
                W = [[1] * n for _ in range(n)]
                R = [1] * n
                model = CycleReference(self.config(n=n, in_flight_depth=1, output_fifo_depth=1), W, R)
                snapshots = [model.step(self.drive(x=(1,) * n, ready=False))]
                snapshots.extend(model.step(self.drive(valid=False, ready=False))
                                 for _ in range(2*n + 5))
                self.assertEqual(snapshots[n+1].W[0][0], 1)
                self.assertEqual(snapshots[n+2].W[0][0], 2)
                self.assertFalse(snapshots[n+2].sample_context_fifo)
                self.assertEqual(snapshots[2*n+1].R, (1,) * n)
                self.assertEqual(snapshots[2*n+2].R, (2,) * n)
                self.assertEqual(model.W, [[2] * n for _ in range(n)])
                self.assertIsNone(model.timings[0].retired_at)
                with self.assertRaisesRegex(RuntimeError, "quiescent"):
                    model.reconfigure(reduce_output=True)
                held = model.step(CycleInputs(reload_weights=True, result_ready=False))
                self.assertTrue(held.weights_loaded)
                self.assertEqual(len(held.result_fifo), 1)
                model.flush()
                self.assertEqual(model.W, [[2] * n for _ in range(n)])
                self.assertEqual(model.R, [2] * n)
                self.assertTrue(self.attempt_matrix_reload(model))

    def test_incomplete_observation_fails_at_result_boundary(self) -> None:
        model = CycleReference(self.config(), self.initial_W(), [16, 24, 32])
        model.step(self.drive(training=False))
        model.step(self.drive(valid=False, training=False))
        token = model._data_tokens[0]
        self.assertTrue(
            all(value is None for row in token.observed_weights for value in row)
        )

        # Let the first inward shell be observed, then model a missing PE
        # observation before the token reaches its final anti-diagonal.
        model.step(self.drive(valid=False, training=False))
        token.observed_weights[0][0] = None

        with self.assertRaisesRegex(
            AssertionError,
            "sample reached result boundary without all PE weights",
        ):
            for _ in range(model.config.n - 1):
                model.step(self.drive(valid=False, training=False))

    def test_continuous_throughput_is_one_result_per_cycle_after_fill(self) -> None:
        model = CycleReference(self.config(), self.initial_W(), [16, 24, 32])
        for index in range(12):
            model.step(self.drive(x=(index + 1, 0, 0), target=0, training=False))

        self.assertEqual([timing.accepted_at for timing in model.timings], list(range(12)))
        self.assertEqual([record.sample_index for record in model.records[:5]], list(range(5)))
        self.assertTrue(all(timing.retired_at is not None for timing in model.timings[:4]))
        self.assertEqual([timing.enqueued_at for timing in model.timings[:5]], [5, 6, 7, 8, 9])
        self.assertEqual([timing.retired_at for timing in model.timings[:4]], [6, 7, 8, 9])

    def test_feedback_visibility_is_s0_s4_s1_s5_s2_s6(self) -> None:
        initial_W = self.initial_W()
        initial_R = [16, 24, 32]
        model = CycleReference(self.config(), initial_W, initial_R)
        for _ in range(10):
            model.step(self.drive())
        model.flush()

        records = model.records
        old_W = tuple(tuple(row) for row in initial_W)
        old_R = tuple(initial_R)
        self.assertEqual([record.W_used for record in records[:4]], [old_W] * 4)
        self.assertEqual([record.R_used for record in records[:4]], [old_R] * 4)
        for sample_index in (4, 5, 6):
            generation = sample_index - 3
            self.assertEqual(
                records[sample_index].W_used,
                tuple(tuple(value + generation for value in row) for row in initial_W),
            )
            self.assertEqual(records[sample_index].R_used, tuple(value + generation for value in initial_R))

    def test_buffer_snapshots_cover_startup_fill_and_drain(self) -> None:
        model = CycleReference(self.config(), self.initial_W(), [16, 24, 32])
        first = model.step(self.drive(x=(1, -2, 0), target=17, training=False))
        self.assertEqual(first.loaded_weight_rows, 0)
        self.assertEqual(first.input_stage, (1, -2, 0))
        self.assertEqual(
            first.sample_context_fifo[0].input_signs,
            (1, -1, 0),
        )

        for x in ((2, 3, 0), (-1, 0, 4), (3, -3, 1)):
            model.step(self.drive(x=x, target=0, training=False))
        fill = model.snapshots[-1]
        self.assertIsNotNone(fill.input_stage)
        self.assertEqual(len(fill.sample_context_fifo), 4)

        for x in ((4, 1, 0), (0, 2, 2), (-2, 1, 3), (1, 1, -1)):
            model.step(self.drive(x=x, target=0, training=False))
        result_storage = model.snapshots[5]
        self.assertEqual(
            tuple(entry.activated_result for entry in result_storage.result_fifo),
            (tuple(matrix_multiply((1, -2, 0), self.initial_W(), self.config().width)),),
        )
        self.assertEqual(len(result_storage.sample_context_fifo), 5)

        model.flush()
        drained = model.snapshots[-1]
        self.assertIsNone(drained.input_stage)
        self.assertEqual(drained.sample_context_fifo, ())
        self.assertEqual(drained.result_fifo, ())

    def test_two_ended_loading_is_atomic_and_keeps_host_order(self) -> None:
        for n in (1, 2, 3, 4, 5, 8):
            with self.subTest(n=n):
                rows = [[(-1 if (r + c) % 2 else 1) * (r * n + c + 1)
                         for c in range(n)] for r in range(n)]
                model = CycleReference(self.config(n=n), R=[1] * n)
                expected_buffer = [[0] * n for _ in range(n)]
                for index, row in enumerate(reversed(rows)):
                    snap = model.step(CycleInputs(weight_valid=True, weight_data=tuple(row)))
                    expected_buffer[n - 1 - index] = row
                    self.assertEqual(snap.weight_buffer, tuple(map(tuple, expected_buffer)))
                    self.assertEqual(snap.loaded_weight_rows, index + 1)
                    self.assertEqual(model.W, [[0] * n for _ in range(n)])
                    if index < n - 1:
                        model.step(CycleInputs())  # Host gaps do not shift or capture.
                        self.assertFalse(model._loading_weights)
                self.assertTrue(snap.loading_weights)
                for q in range(n):
                    snap = model.step(CycleInputs(weight_valid=True, weight_data=(99,) * n))
                    self.assertEqual(model.W, [[0] * n for _ in range(n)])
                    self.assertFalse(snap.weights_loaded)
                self.assertTrue(snap.capture_weights)
                self.assertFalse(snap.loading_weights)
                # Independent shift-chain simulation must produce every resident
                # value before W can change on the subsequent capture edge.
                self.assertEqual(model._load_psums, rows)
                snap = model.step(CycleInputs(weight_valid=True, weight_data=(99,) * n))
                self.assertEqual(model.W, rows)
                self.assertTrue(snap.weights_loaded)
                self.assertFalse(snap.capture_weights)
                self.assertEqual(snap.loaded_weight_rows, 0)
                reloaded = model.step(CycleInputs(reload_weights=True))
                self.assertFalse(reloaded.weights_loaded)
                self.assertEqual(reloaded.weight_buffer, tuple(tuple(0 for _ in range(n)) for _ in range(n)))
                self.assertEqual(model.W, rows)  # Reload does not destroy resident W.
                extra = (-3,) * n
                model.step(CycleInputs(weight_valid=True, weight_data=extra))
                self.assertEqual(model._weight_buffer[-1], list(extra))

    def test_accepted_reload_blocks_simultaneous_input(self) -> None:
        model = CycleReference(self.config(), self.initial_W(), [1, 1, 1])
        snap = model.step(CycleInputs(reload_weights=True, input_valid=True,
                                     input_data=(3, -2, 1), result_ready=True))
        self.assertFalse(snap.weights_loaded)
        self.assertIsNone(snap.input_stage)
        self.assertEqual(snap.sample_context_fifo, ())
        self.assertEqual(model.timings, [])
        replacement = [[-1, 2, 0], [3, 0, -2], [4, 1, 5]]
        for row in reversed(replacement):
            model.step(CycleInputs(weight_valid=True, weight_data=tuple(row)))
        while not model.weights_loaded:
            model.step(CycleInputs())
        model.step(self.drive(x=(3, -2, 1), training=False))
        model.flush()
        self.assertEqual(len(model.records), 1)
        self.assertEqual(model.records[0].raw_matrix_result,
                         tuple(matrix_multiply((3, -2, 1), replacement, model.config.width)))

    def test_reset_clears_partial_loading_and_capture(self) -> None:
        for n in (2, 3, 4, 5, 8):
            for reset_age in range(2 * n + 1):
                with self.subTest(n=n, reset_age=reset_age):
                    model = CycleReference(self.config(n=n))
                    for edge in range(reset_age):
                        model.step(CycleInputs(weight_valid=edge < n, weight_data=(edge + 1,) * n))
                    snap = model.step(CycleInputs(reset_n=False))
                    self.assertEqual(model.W, [[0] * n for _ in range(n)])
                    self.assertEqual(snap.loaded_weight_rows, 0)
                    self.assertFalse(snap.loading_weights or snap.capture_weights or snap.weights_loaded)
                    self.assertFalse(model.in_flight)

    def test_input_bubbles_do_not_stop_learning(self) -> None:
        initial_W = self.initial_W()
        initial_R = [16, 24, 32]
        model = CycleReference(self.config(), initial_W, initial_R)
        model.step(self.drive())
        bubble_snapshots = [
            model.step(self.drive(valid=False))
            for _ in range(14)
        ]

        self.assertEqual(bubble_snapshots[4].W[0][0], initial_W[0][0] + 1)
        self.assertEqual(bubble_snapshots[6].W, tuple(tuple(v + 1 for v in row) for row in initial_W))
        self.assertEqual(bubble_snapshots[7].R, tuple(value + 1 for value in initial_R))

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

        self.assertTrue(all(timing.retired_at is not None for timing in model.timings))
        self.assertEqual([record.sample_index for record in model.records], list(range(9)))
        self.assertEqual(len(model.records), 9)
        expected_W = [row[:] for row in self.initial_W()]
        expected_R = [16, 24, 32]
        for record in model.records:
            if record.update_generated:
                expected_W = apply_matrix_update(expected_W, record.matrix_update_directions, model.config.width)
                expected_R = apply_reduction_update(expected_R, record.reduction_update_directions, model.config.reduction_weight_width)
        self.assertEqual(model.W, expected_W)
        self.assertEqual(model.R, expected_R)

    def test_overlapping_updates_change_post_edge_W_on_the_wave_clocks(self) -> None:
        model = CycleReference(self.config(), [[0] * 3 for _ in range(3)], [64, 64, 64])
        snapshots = [model.step(self.drive()) for _ in range(10)]

        self.assertEqual(snapshots[5].W[0][0], 1)
        self.assertEqual(snapshots[6].W[0][0], 2)
        self.assertEqual(snapshots[6].W[0][1], 1)
        self.assertEqual(snapshots[6].W[1][0], 1)

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
        for record in model.records:
            if record.update_generated:
                expected_W = apply_matrix_update(expected_W, record.matrix_update_directions, config.width)
                expected_R = apply_reduction_update(expected_R, record.reduction_update_directions, config.reduction_weight_width)
        self.assertEqual(model.W, expected_W)
        self.assertEqual(model.R, expected_R)
        self.assertTrue(all(-8 <= value <= 7 for row in model.W for value in row))
        self.assertTrue(all(-8 <= value <= 7 for value in model.R))

    def test_streaming_bubbles_stalls_learning_and_reload_all_sizes(self) -> None:
        for n in (2, 3, 4, 5, 8):
            with self.subTest(n=n):
                rng = random.Random(8300 + n)
                config = self.config(n=n, output_fifo_depth=2)
                W = [[rng.randint(-8, 8) for _ in range(n)] for _ in range(n)]
                R = [rng.randint(-8, 8) for _ in range(n)]
                model = CycleReference(config, W, R)
                samples = [Sample(tuple(rng.randint(-4, 4) for _ in range(n)),
                                  rng.choice((-100, 0, 100)), i % 4 != 0)
                           for i in range(4 * n + 12)]
                source = 0
                offered = False
                stalls = 0
                for edge in range(2000):
                    if source < len(samples) and not offered:
                        offered = rng.random() > 0.28
                    ready = edge % 23 > 10  # Long held intervals force a true stall.
                    sample = samples[source] if source < len(samples) else None
                    before_count = len(model.timings)
                    blocked = bool(len(model._result_fifo) == 2 and model._alignment and not ready)
                    frozen = (tuple(t.age for t in model._data_tokens),
                              tuple(model._matrix_waves), tuple(model._reduction_pipe))
                    snap = model.step(CycleInputs(
                        input_valid=offered and sample is not None,
                        input_data=sample.x if sample else (),
                        target_data=sample.target if sample else 0,
                        training_enable=sample.training_enable if sample else False,
                        result_ready=ready,
                    ))
                    if blocked:
                        stalls += 1
                        self.assertEqual(frozen, (tuple(t.age for t in model._data_tokens),
                                                 tuple(model._matrix_waves), tuple(model._reduction_pipe)))
                    if len(model.timings) > before_count:
                        source += 1
                        offered = False
                    if source == len(samples) and not model.in_flight:
                        break
                self.assertGreater(stalls, 0)
                self.assertEqual(source, len(samples))
                self.assertEqual([r.input_vector for r in model.records], [s.x for s in samples])
                self.assertTrue(all(t.retired_at is not None for t in model.timings))

                # Independently replay issued arithmetic in completion order.
                # Every sample must observe a whole W/R generation despite gaps
                # and held edges; mixed forward/update phases fail this check.
                expected_W, expected_R = W, R
                coherent = {(tuple(map(tuple, W)), tuple(R))}
                for record in model.records:
                    if record.update_generated:
                        expected_W = apply_matrix_update(expected_W, record.matrix_update_directions, config.width)
                        expected_R = apply_reduction_update(expected_R, record.reduction_update_directions,
                                                             config.reduction_weight_width)
                        coherent.add((tuple(map(tuple, expected_W)), tuple(expected_R)))
                for record in model.records:
                    self.assertIn((record.W_used, record.R_used), coherent)
                self.assertEqual(model.W, expected_W)
                self.assertEqual(model.R, expected_R)

                self.assertTrue(self.attempt_matrix_reload(model))
                replacement = [[rng.randint(-10, 10) for _ in range(n)] for _ in range(n)]
                for row in reversed(replacement):
                    model.step(CycleInputs(weight_valid=True, weight_data=tuple(row)))
                    model.step(CycleInputs())
                while not model.weights_loaded:
                    model.step(CycleInputs())
                self.assertEqual(model.W, replacement)
                probe = tuple(rng.randint(-3, 3) for _ in range(n))
                model.step(self.drive(x=probe, training=False))
                model.flush()
                self.assertEqual(model.records[-1].raw_matrix_result,
                                 tuple(matrix_multiply(probe, replacement, config.width)))
                self.assertEqual(model.R, expected_R)

    def test_closed_form_update_visibility_matches_wave_propagation(self) -> None:
        """Cross-check weight visibility: functional derives the N+2 sample delay
        in closed form; cycle derives it independently through wave propagation."""

        for n in (2, 3, 4, 5, 8):
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
                raw_samples = raw_samples * 2
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

                # Functional uses a closed-form visibility delay; cycle observes
                # each PE and propagates update waves independently, so it makes
                # no visibility claim and the delay must emerge in W_used/R_used.
                self.assertEqual(
                    cycle.records,
                    [replace(record, update_visible_at=None) for record in functional_records],
                )
                self.assertEqual(
                    [t.enqueued_at - t.accepted_at for t in cycle.timings],
                    [n + 2] * len(samples),
                )
                self.assertEqual(
                    [t.retired_at - t.accepted_at for t in cycle.timings],
                    [n + 3] * len(samples),
                )
                self.assertEqual(cycle.W, functional.final_W)
                self.assertEqual(cycle.R, functional.final_R)

                # Guard against a vacuous check.
                observed_generations = {
                    record.W_used for record in cycle.records
                }
                self.assertGreater(len(observed_generations), 1)

                # The emergent delay must equal the closed form.
                delay = functional_config.update_visibility_delay
                self.assertEqual(delay, n + 1)
                self.assertEqual(functional_records[0].update_visible_at, delay)
                self.assertEqual(
                    cycle.records[delay - 1].W_used,
                    cycle.records[0].W_used,
                )
                self.assertNotEqual(
                    cycle.records[delay].W_used,
                    cycle.records[0].W_used,
                )


if __name__ == "__main__":
    unittest.main()
