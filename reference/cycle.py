"""Cycle-indexed architectural reference for the Phase 6 accelerator.

``CycleReference.step`` applies the external values present before one rising
edge.  Its return value is a photograph of the persistent architectural state
immediately after that edge.  Only the resident weights and the five
architectural FIFO payloads are part of that photograph; timing positions and
handshake signals remain private implementation details of this model.

The numerical operations are deliberately delegated to ``arithmetic.py``.
This module decides when those operations happen and how samples, results, and
learning updates move through the accelerator-specific schedule.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Sequence

from .arithmetic import (
    activate,
    apply_matrix_update,
    apply_reduction_update,
    learning_direction,
    matrix_multiply,
    matrix_result_width,
    matrix_update_directions,
    prediction_width,
    reduction_update_directions,
    ternary_product,
    ternary_sign,
    to_signed,
    weighted_vector_reduction,
)


def _matrix_copy(values: Sequence[Sequence[int]], n: int, width: int) -> list[list[int]]:
    if len(values) != n or any(len(row) != n for row in values):
        raise ValueError("W must be an N by N matrix")
    return [[to_signed(value, width) for value in row] for row in values]


def _vector_copy(values: Sequence[int], n: int, width: int, name: str) -> list[int]:
    if len(values) != n:
        raise ValueError(f"{name} must have length n")
    return [to_signed(value, width) for value in values]


def _zero_matrix(n: int) -> list[list[int]]:
    return [[0 for _ in range(n)] for _ in range(n)]


def _matrix_tuple(values: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in values)


@dataclass(frozen=True)
class CycleConfig:
    """Parameters and quiescent-lifetime configuration for the cycle model.

    ``pass_through`` and ``reduce_output`` describe a stream configuration,
    not an accepted sample.  Use :meth:`CycleReference.reconfigure` to change
    either value, and only after all work from the preceding stream drains.
    """

    n: int = 3
    width: int = 16
    fraction_bits: int = 4
    target_width: int | None = None
    reduction_weight_width: int = 8
    input_fifo_depth: int | None = None
    output_fifo_depth: int | None = None
    pass_through: bool = True
    reduce_output: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.n, int) or isinstance(self.n, bool) or self.n < 1:
            raise ValueError("n must be a positive integer")
        if not isinstance(self.width, int) or isinstance(self.width, bool) or self.width < 1:
            raise ValueError("width must be a positive integer")
        if self.target_width is None:
            object.__setattr__(self, "target_width", self.width)
        for name in ("target_width", "reduction_weight_width"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(self.fraction_bits, int)
            or isinstance(self.fraction_bits, bool)
            or self.fraction_bits < 0
        ):
            raise ValueError("fraction_bits must be a non-negative integer")
        for name in ("input_fifo_depth", "output_fifo_depth"):
            value = getattr(self, name)
            if value is None:
                object.__setattr__(self, name, 2 * self.n)
            elif not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def matrix_result_width(self) -> int:
        return matrix_result_width(self.width, self.n)

    @property
    def prediction_width(self) -> int:
        return prediction_width(self.width, self.n)

    @property
    def sample_context_depth(self) -> int:
        return max(self.input_fifo_depth, 2 * self.n + 2)  # type: ignore[arg-type]


@dataclass(frozen=True)
class CycleInputs:
    """Per-cycle transaction and flow-control values before a rising edge.

    The stream configuration is intentionally absent.  In particular,
    ``passThrough`` and ``reduceOutput`` are not sample metadata.
    """

    activation_valid: bool = False
    activation_data: tuple[int, ...] = ()
    target_data: int = 0
    training_enable: bool = True
    result_ready: bool = False
    weight_valid: bool = False
    weight_data: tuple[int, ...] = ()
    reduction_weight: tuple[int, ...] = ()
    load_reduction_weights: bool = False
    reload_weights: bool = False
    reset_n: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "activation_valid", bool(self.activation_valid))
        object.__setattr__(self, "activation_data", tuple(int(v) for v in self.activation_data))
        object.__setattr__(self, "target_data", int(self.target_data))
        object.__setattr__(self, "training_enable", bool(self.training_enable))
        object.__setattr__(self, "result_ready", bool(self.result_ready))
        object.__setattr__(self, "weight_valid", bool(self.weight_valid))
        object.__setattr__(self, "weight_data", tuple(int(v) for v in self.weight_data))
        object.__setattr__(
            self,
            "reduction_weight",
            tuple(int(v) for v in self.reduction_weight),
        )
        object.__setattr__(self, "load_reduction_weights", bool(self.load_reduction_weights))
        object.__setattr__(self, "reload_weights", bool(self.reload_weights))
        object.__setattr__(self, "reset_n", bool(self.reset_n))


@dataclass(frozen=True)
class SampleContext:
    """Payload of the architectural sample-context FIFO.

    The RTL stores the target, ternary input signs, and training-enable bit;
    a sample index is model bookkeeping and is intentionally not exposed.
    """

    target: int
    input_signs: tuple[int, ...]
    training_enable: bool


@dataclass(frozen=True)
class ResultReadout:
    """Prediction and resident reduction-weight signs for one result."""

    prediction: int
    reduction_weight_signs: tuple[int, ...]


@dataclass(frozen=True)
class CycleSnapshot:
    """Persistent architectural state after one simulated rising edge."""

    cycle: int
    W: tuple[tuple[int, ...], ...]
    R: tuple[int, ...]
    weight_fifo: tuple[tuple[int, ...], ...]
    activation_fifo: tuple[tuple[int, ...], ...]
    sample_context_fifo: tuple[SampleContext, ...]
    output_fifo: tuple[tuple[int, ...], ...]
    result_readout_fifo: tuple[ResultReadout, ...]


@dataclass
class _Sample:
    index: int
    input_vector: tuple[int, ...]
    target: int
    training_enable: bool

    @property
    def input_signs(self) -> tuple[int, ...]:
        return tuple(ternary_sign(value) for value in self.input_vector)


@dataclass
class _DataToken:
    sample: _Sample
    age: int
    observed_weights: list[list[int | None]]


@dataclass(frozen=True)
class _AlignedResult:
    sample_index: int
    raw_matrix_result: tuple[int, ...]


@dataclass(frozen=True)
class _OutputEntry:
    sample_index: int
    raw_matrix_result: tuple[int, ...]


@dataclass(frozen=True)
class _ReadoutEntry:
    sample_index: int
    prediction: int
    reduction_weight_signs: tuple[int, ...]


@dataclass(frozen=True)
class _MatrixWave:
    row_direction: tuple[int, ...]
    column_direction: tuple[int, ...]
    next_diagonal: int


@dataclass(frozen=True)
class _ReductionWave:
    direction: tuple[int, ...]


@dataclass(frozen=True)
class _SampleResult:
    """Minimal private numerical bookkeeping for functional cross-checks."""

    sample_index: int
    input_vector: tuple[int, ...]
    target: int
    training_enable: bool
    W_used: tuple[tuple[int, ...], ...]
    R_used: tuple[int, ...]
    raw_matrix_result: tuple[int, ...]
    activated_result: tuple[int, ...]
    prediction: int
    learning_direction: int
    matrix_update_directions: tuple[tuple[int, ...], ...]
    reduction_update_directions: tuple[int, ...]
    update_generated: bool


class CycleReference:
    """Cycle-indexed reference of resident W/R state and architectural FIFOs."""

    def __init__(
        self,
        config: CycleConfig | None = None,
        W: Sequence[Sequence[int]] | None = None,
        R: Sequence[int] | None = None,
        *,
        weights_loaded: bool | None = None,
    ) -> None:
        self.config = config or CycleConfig()
        self._initial_W = (
            _matrix_copy(W, self.config.n, self.config.width)
            if W is not None
            else _zero_matrix(self.config.n)
        )
        self._initial_R = (
            _vector_copy(R, self.config.n, self.config.reduction_weight_width, "R")
            if R is not None
            else [0 for _ in range(self.config.n)]
        )
        self._initial_weights_loaded = bool(W is not None) if weights_loaded is None else bool(weights_loaded)
        if self._initial_weights_loaded and W is None:
            raise ValueError("weights_loaded=True requires an initial W")
        self.snapshots: list[CycleSnapshot] = []
        self.reset()

    @property
    def W(self) -> list[list[int]]:
        return [row[:] for row in self._W]

    @property
    def R(self) -> list[int]:
        return self._R[:]

    @property
    def cycle(self) -> int:
        return self._cycle

    @property
    def weights_loaded(self) -> bool:
        return self._weights_loaded

    @property
    def in_flight(self) -> bool:
        return bool(
            self._weight_fifo
            or (not self._weights_loaded and self._loaded_weight_count)
            or self._activation_fifo
            or self._sample_context_fifo
            or self._data_tokens
            or self._completed_tokens
            or self._alignment
            or self._output_fifo
            or self._matrix_waves
            or any(wave is not None for wave in self._reduction_pipe)
        )

    @property
    def stream_quiescent(self) -> bool:
        """Whether accepted sample/result/update work has completely drained.

        This is the verification configuration boundary.  It deliberately
        excludes weight-loading state and the output frame position: neither
        represents outstanding work whose interpretation depends on the
        stream mode.
        """

        return not bool(
            self._activation_fifo
            or self._data_tokens
            or self._completed_tokens
            or self._matrix_waves
            or self._alignment
            or self._output_fifo
            or self._sample_context_fifo
            or self._result_readout_fifo
            or any(wave is not None for wave in self._reduction_pipe)
        )

    def reset(self) -> None:
        """Return to the constructor's initial state."""

        self.snapshots = []
        self._cycle = 0
        self._next_sample_index = 0
        self._W = [row[:] for row in self._initial_W] if self._initial_weights_loaded else _zero_matrix(self.config.n)
        self._R = self._initial_R[:]
        self._weights_loaded = self._initial_weights_loaded
        self._loaded_weight_count = self.config.n if self._weights_loaded else 0
        self._output_row_index = 0
        self._weight_fifo: deque[tuple[int, ...]] = deque()
        self._activation_fifo: deque[_Sample] = deque()
        self._sample_context_fifo: deque[_Sample] = deque()
        self._output_fifo: deque[_OutputEntry] = deque()
        self._result_readout_fifo: deque[_ReadoutEntry] = deque()
        self._weight_load_pipe: list[tuple[int, ...] | None] = [None] * self.config.n
        self._data_tokens: list[_DataToken] = []
        self._completed_tokens: dict[int, _DataToken] = {}
        self._alignment: deque[_AlignedResult] = deque()
        self._matrix_waves: list[_MatrixWave] = []
        self._reduction_pipe: list[_ReductionWave | None] = [None] * (2 * self.config.n - 1)
        self._sample_results: list[_SampleResult] = []
        # Private timing-test aids.  They are deliberately absent from
        # CycleSnapshot and do not form a public event-oriented API.
        self._accepted_cycles: list[int] = []
        self._enqueue_cycles: list[int] = []
        self._retirement_cycles: list[int] = []
        self._enqueued_sample_indices: list[int] = []
        self._retired_sample_indices: list[int] = []

    def _hardware_reset(self) -> None:
        self._next_sample_index = 0
        self._W = _zero_matrix(self.config.n)
        self._R = [0 for _ in range(self.config.n)]
        self._weights_loaded = False
        self._loaded_weight_count = 0
        self._output_row_index = 0
        self._weight_fifo.clear()
        self._activation_fifo.clear()
        self._sample_context_fifo.clear()
        self._output_fifo.clear()
        self._result_readout_fifo.clear()
        self._weight_load_pipe = [None] * self.config.n
        self._data_tokens.clear()
        self._completed_tokens.clear()
        self._alignment.clear()
        self._matrix_waves.clear()
        self._reduction_pipe = [None] * (2 * self.config.n - 1)
        self._sample_results.clear()
        self._accepted_cycles.clear()
        self._enqueue_cycles.clear()
        self._retirement_cycles.clear()
        self._enqueued_sample_indices.clear()
        self._retired_sample_indices.clear()

    def flush(self, max_cycles: int = 10000) -> list[CycleSnapshot]:
        """Advance ready idle edges until all private and FIFO state drains."""

        if not isinstance(max_cycles, int) or isinstance(max_cycles, bool) or max_cycles < 0:
            raise ValueError("max_cycles must be a non-negative integer")
        added: list[CycleSnapshot] = []
        for _ in range(max_cycles):
            if not self.in_flight:
                return added
            added.append(self.step(CycleInputs(result_ready=True)))
        if self.in_flight:
            raise RuntimeError("cycle model did not drain within max_cycles")
        return added

    def reconfigure(
        self,
        *,
        pass_through: bool | None = None,
        reduce_output: bool | None = None,
    ) -> None:
        """Change stream configuration at a quiescent boundary.

        An accepted sample, buffered result, or pending learning update keeps
        the current configuration live.  The model rejects a change until all
        such work has drained; no mode value is copied into a sample or FIFO.
        """

        next_pass_through = (
            self.config.pass_through if pass_through is None else bool(pass_through)
        )
        next_reduce_output = (
            self.config.reduce_output if reduce_output is None else bool(reduce_output)
        )
        if (
            next_pass_through != self.config.pass_through
            or next_reduce_output != self.config.reduce_output
        ) and not self.stream_quiescent:
            raise RuntimeError(
                "passThrough/reduceOutput may change only when the accelerator is quiescent"
            )
        self.config = replace(
            self.config,
            pass_through=next_pass_through,
            reduce_output=next_reduce_output,
        )

    def run(self, cycles: Sequence[CycleInputs]) -> list[CycleSnapshot]:
        return [self.step(cycle_inputs) for cycle_inputs in cycles]

    def _normalize_input(self, values: Sequence[int]) -> tuple[int, ...]:
        if len(values) != self.config.n:
            raise ValueError("activation_data must have length n")
        return tuple(to_signed(value, self.config.width) for value in values)

    def _normalize_weight(self, values: Sequence[int]) -> tuple[int, ...]:
        if len(values) != self.config.n:
            raise ValueError("weight_data must have length n")
        return tuple(to_signed(value, self.config.width) for value in values)

    def _normalize_reduction(self, values: Sequence[int]) -> tuple[int, ...]:
        return tuple(_vector_copy(values, self.config.n, self.config.reduction_weight_width, "reduction_weight"))

    def _output_head(self) -> tuple[_OutputEntry | None, _ReadoutEntry | None, _Sample | None]:
        return (
            self._output_fifo[0] if self._output_fifo else None,
            self._result_readout_fifo[0] if self._result_readout_fifo else None,
            self._sample_context_fifo[0] if self._sample_context_fifo else None,
        )

    def _matrix_reload_ready(self, reduction_update_busy: bool) -> bool:
        """Whether matrix weights may reload at a complete output frame."""

        # Stream quiescence intentionally omits frame position.  Reload adds
        # the surviving output position so a partial frame cannot reload.
        return bool(
            self._weights_loaded
            and not self._activation_fifo
            and not self._data_tokens
            and not self._completed_tokens
            and not self._alignment
            and not self._output_fifo
            and self._output_row_index == 0
            and not self._matrix_waves
            and not reduction_update_busy
        )

    def _assert_result_queues_paired(self) -> None:
        if len(self._output_fifo) > self.config.output_fifo_depth:
            raise AssertionError("output FIFO exceeded its configured depth")
        if len(self._output_fifo) != len(self._result_readout_fifo):
            raise AssertionError("output and readout FIFO occupancies diverged")
        for output, readout in zip(self._output_fifo, self._result_readout_fifo):
            if output.sample_index != readout.sample_index:
                raise AssertionError("output and readout FIFO order diverged")

    def _apply_matrix_diagonal(self, wave: _MatrixWave, diagonal: int) -> None:
        direction = _zero_matrix(self.config.n)
        for row in range(self.config.n):
            for column in range(self.config.n):
                if row + column == diagonal:
                    direction[row][column] = ternary_product(
                        wave.row_direction[row], wave.column_direction[column]
                    )
        self._W = apply_matrix_update(self._W, direction, self.config.width)

    def _advance_matrix_waves(self, injected: _MatrixWave | None) -> None:
        last_diagonal = 2 * self.config.n - 2
        next_waves: list[_MatrixWave] = []
        for wave in self._matrix_waves:
            self._apply_matrix_diagonal(wave, wave.next_diagonal)
            if wave.next_diagonal < last_diagonal:
                next_waves.append(
                    _MatrixWave(
                        wave.row_direction,
                        wave.column_direction,
                        wave.next_diagonal + 1,
                    )
                )
        if injected is not None:
            self._apply_matrix_diagonal(injected, 0)
            if last_diagonal > 0:
                next_waves.append(
                    _MatrixWave(
                        injected.row_direction,
                        injected.column_direction,
                        1,
                    )
                )
        self._matrix_waves = next_waves

    def _advance_reduction_waves(
        self,
        injected: _ReductionWave | None,
        load_reduction_weights: bool,
    ) -> None:
        if load_reduction_weights:
            return
        tail = self._reduction_pipe[-1]
        if tail is not None:
            self._R = apply_reduction_update(
                self._R,
                tail.direction,
                self.config.reduction_weight_width,
            )
        self._reduction_pipe = self._reduction_pipe[:-1]
        self._reduction_pipe.insert(0, injected)

    def _shift_data_and_alignment(
        self,
        accepted: _Sample | None,
        activation_pop: bool,
        output_enqueue: bool,
    ) -> None:
        """Advance the data array and fixed result alignment one slot."""

        W_before = [row[:] for row in self._W]
        final_age = 2 * self.config.n - 1
        active: list[_DataToken] = []
        for token in self._data_tokens:
            token.age += 1
            if token.age <= final_age:
                diagonal = token.age - 1
                for row in range(self.config.n):
                    for column in range(self.config.n):
                        if row + column == diagonal:
                            token.observed_weights[row][column] = W_before[row][column]
            if token.age == final_age:
                observed = token.observed_weights
                if any(value is None for row in observed for value in row):
                    raise AssertionError("sample reached result boundary without all PE weights")
                observed_matrix = [
                    [int(value) for value in row]  # type: ignore[arg-type]
                    for row in observed
                ]
                raw = tuple(
                    matrix_multiply(
                        token.sample.input_vector,
                        observed_matrix,
                        self.config.width,
                    )
                )
                self._completed_tokens[token.sample.index] = token
                self._alignment.append(_AlignedResult(token.sample.index, raw))
            else:
                active.append(token)
        self._data_tokens = active

        if activation_pop:
            if accepted is None:
                raise AssertionError("activation pop without a sample")
            self._data_tokens.append(
                _DataToken(
                    sample=accepted,
                    age=0,
                    observed_weights=_zero_matrix(self.config.n),
                )
            )

        shifted: deque[_AlignedResult] = deque()
        skipped_ready = output_enqueue
        for entry in self._alignment:
            if skipped_ready:
                skipped_ready = False
                continue
            shifted.append(entry)
        self._alignment = shifted

    def _make_result(
        self,
        token: _DataToken,
        raw_result: tuple[int, ...],
        pass_through: bool,
        resident_R: Sequence[int],
    ) -> _SampleResult:
        observed = token.observed_weights
        if any(value is None for row in observed for value in row):
            raise AssertionError("completed sample has an unobserved PE weight")
        W_used = tuple(
            tuple(int(value) for value in row)  # type: ignore[arg-type]
            for row in observed
        )
        R_used = tuple(int(value) for value in resident_R)
        activated = tuple(activate(raw_result, pass_through, self.config.matrix_result_width))
        prediction = weighted_vector_reduction(
            activated,
            R_used,
            self.config.width,
            self.config.reduction_weight_width,
            self.config.fraction_bits,
        )
        direction = learning_direction(
            token.sample.target,
            prediction,
            self.config.target_width,  # type: ignore[arg-type]
            self.config.prediction_width,
        )
        _, _, matrix_direction = matrix_update_directions(
            token.sample.input_vector,
            activated,
            R_used,
            direction,
            self.config.width,
            self.config.reduction_weight_width,
            pass_through,
        )
        reduction_direction = reduction_update_directions(activated, direction)
        zeros = tuple(tuple(0 for _ in range(self.config.n)) for _ in range(self.config.n))
        result = _SampleResult(
            sample_index=token.sample.index,
            input_vector=token.sample.input_vector,
            target=token.sample.target,
            training_enable=token.sample.training_enable,
            W_used=W_used,
            R_used=R_used,
            raw_matrix_result=raw_result,
            activated_result=activated,
            prediction=prediction,
            learning_direction=direction,
            matrix_update_directions=_matrix_tuple(matrix_direction)
            if token.sample.training_enable
            else zeros,
            reduction_update_directions=tuple(reduction_direction)
            if token.sample.training_enable
            else (0,) * self.config.n,
            update_generated=token.sample.training_enable,
        )
        self._sample_results.append(result)
        return result

    def _snapshot(self) -> CycleSnapshot:
        self._assert_result_queues_paired()
        return CycleSnapshot(
            cycle=self._cycle,
            W=_matrix_tuple(self._W),
            R=tuple(self._R),
            weight_fifo=tuple(self._weight_fifo),
            activation_fifo=tuple(sample.input_vector for sample in self._activation_fifo),
            sample_context_fifo=tuple(
                SampleContext(sample.target, sample.input_signs, sample.training_enable)
                for sample in self._sample_context_fifo
            ),
            output_fifo=tuple(entry.raw_matrix_result for entry in self._output_fifo),
            result_readout_fifo=tuple(
                ResultReadout(entry.prediction, entry.reduction_weight_signs)
                for entry in self._result_readout_fifo
            ),
        )

    def step(self, inputs: CycleInputs | None = None) -> CycleSnapshot:
        """Apply one pre-edge input bundle and return the post-edge snapshot."""

        cycle_inputs = inputs if inputs is not None else CycleInputs()
        if not isinstance(cycle_inputs, CycleInputs):
            raise TypeError("step expects CycleInputs")
        cycle_number = self._cycle
        pass_through = self.config.pass_through

        if not cycle_inputs.reset_n:
            self._hardware_reset()
            snapshot = self._snapshot()
            self._cycle = cycle_number + 1
            self.snapshots.append(snapshot)
            return snapshot

        output_head, readout_head, context_head = self._output_head()
        result_valid = output_head is not None and readout_head is not None and context_head is not None
        current_activated: tuple[int, ...] | None = None
        prediction = 0
        direction = 0
        if result_valid:
            assert output_head is not None and readout_head is not None and context_head is not None
            current_activated = tuple(activate(output_head.raw_matrix_result, pass_through, self.config.matrix_result_width))
            prediction = readout_head.prediction
            direction = learning_direction(
                context_head.target,
                prediction,
                self.config.target_width,  # type: ignore[arg-type]
                self.config.prediction_width,
            )
        result_retired = bool(result_valid and cycle_inputs.result_ready)

        context_full = len(self._sample_context_fifo) >= self.config.sample_context_depth
        sample_can_accept = not context_full or result_retired
        activation_full = len(self._activation_fifo) >= self.config.input_fifo_depth
        activation_ready = self._weights_loaded and not activation_full and sample_can_accept
        activation_accepted = bool(cycle_inputs.activation_valid and activation_ready)

        weight_full = len(self._weight_fifo) >= self.config.n
        weight_ready = not self._weights_loaded and not weight_full
        weight_accepted = bool(cycle_inputs.weight_valid and weight_ready)
        weight_popped = bool(not self._weights_loaded and self._weight_fifo)

        reduction_update_entering = bool(
            result_retired and context_head is not None and context_head.training_enable
        )
        reduction_update_busy = bool(
            reduction_update_entering or any(wave is not None for wave in self._reduction_pipe)
        )
        reload_ready = self._matrix_reload_ready(reduction_update_busy)
        reload_accepted = bool(cycle_inputs.reload_weights and reload_ready)

        output_full = len(self._output_fifo) >= self.config.output_fifo_depth
        aligned_head_ready = bool(self._alignment)
        output_blocked = bool(output_full and not result_retired and aligned_head_ready)
        datapath_advance = weight_popped if not self._weights_loaded else not output_blocked

        activation_pop = bool(self._weights_loaded and self._activation_fifo and datapath_advance)
        accepted_sample: _Sample | None = None
        if activation_accepted:
            accepted_sample = _Sample(
                index=self._next_sample_index,
                input_vector=self._normalize_input(cycle_inputs.activation_data),
                target=to_signed(cycle_inputs.target_data, self.config.target_width),  # type: ignore[arg-type]
                training_enable=cycle_inputs.training_enable,
            )
            self._next_sample_index += 1
            self._accepted_cycles.append(cycle_number)

        activation_for_array = self._activation_fifo[0] if activation_pop else None
        result_enqueue = bool(datapath_advance and aligned_head_ready)
        alignment_to_enqueue = self._alignment[0] if result_enqueue else None

        injected_matrix: _MatrixWave | None = None
        injected_reduction: _ReductionWave | None = None
        matrix_update_accepted = bool(
            result_retired
            and context_head is not None
            and context_head.training_enable
            and datapath_advance
        )
        if matrix_update_accepted:
            assert readout_head is not None and current_activated is not None and context_head is not None
            row_direction, column_direction, matrix_direction = matrix_update_directions(
                context_head.input_vector,
                current_activated,
                readout_head.reduction_weight_signs,
                direction,
                self.config.width,
                self.config.reduction_weight_width,
                pass_through,
            )
            reduction_direction = reduction_update_directions(current_activated, direction)
            injected_matrix = _MatrixWave(
                tuple(row_direction),
                tuple(column_direction),
                0,
            )
            injected_reduction = _ReductionWave(tuple(reduction_direction))

        resident_R_before = self._R[:]

        if datapath_advance and self._weights_loaded:
            self._shift_data_and_alignment(activation_for_array, activation_pop, result_enqueue)
        if datapath_advance:
            self._advance_matrix_waves(injected_matrix)
            self._advance_reduction_waves(injected_reduction, cycle_inputs.load_reduction_weights)

        pushed_weight = self._normalize_weight(cycle_inputs.weight_data) if weight_accepted else None
        if weight_popped:
            loaded_row = self._weight_fifo[0]
            self._weight_load_pipe = [loaded_row] + self._weight_load_pipe[:-1]
            for row in range(self.config.n):
                self._W[row] = list(self._weight_load_pipe[row]) if self._weight_load_pipe[row] is not None else [0] * self.config.n
            self._loaded_weight_count += 1
            if self._loaded_weight_count >= self.config.n:
                self._weights_loaded = True
                self._loaded_weight_count = 0

        if cycle_inputs.load_reduction_weights:
            self._R = list(self._normalize_reduction(cycle_inputs.reduction_weight))

        if reload_accepted:
            self._weights_loaded = False
            self._loaded_weight_count = 0
            self._weight_load_pipe = [None] * self.config.n

        if weight_popped:
            self._weight_fifo.popleft()
        if pushed_weight is not None:
            self._weight_fifo.append(pushed_weight)

        if activation_pop:
            self._activation_fifo.popleft()
        if accepted_sample is not None:
            self._activation_fifo.append(accepted_sample)
            self._sample_context_fifo.append(accepted_sample)

        if result_enqueue:
            assert alignment_to_enqueue is not None
            token = self._completed_tokens.pop(alignment_to_enqueue.sample_index, None)
            context_for_result = next(
                (sample for sample in self._sample_context_fifo if sample.index == alignment_to_enqueue.sample_index),
                None,
            )
            if token is None or context_for_result is None:
                raise AssertionError("aligned result is missing its sample state")
            result = self._make_result(
                token,
                alignment_to_enqueue.raw_matrix_result,
                pass_through,
                resident_R_before,
            )
            self._output_fifo.append(_OutputEntry(alignment_to_enqueue.sample_index, alignment_to_enqueue.raw_matrix_result))
            self._result_readout_fifo.append(
                _ReadoutEntry(
                    alignment_to_enqueue.sample_index,
                    result.prediction,
                    tuple(ternary_sign(value) for value in result.R_used),
                )
            )
            self._enqueue_cycles.append(cycle_number)
            self._enqueued_sample_indices.append(alignment_to_enqueue.sample_index)

        if result_retired:
            if not self._output_fifo or not self._result_readout_fifo or not self._sample_context_fifo:
                raise AssertionError("retired result transaction is not fully paired")
            retired_output = self._output_fifo.popleft()
            retired_readout = self._result_readout_fifo.popleft()
            retired_context = self._sample_context_fifo.popleft()
            if not retired_output.sample_index == retired_readout.sample_index == retired_context.index:
                raise AssertionError("result, readout, and context order diverged")
            self._output_row_index = (self._output_row_index + 1) % self.config.n
            self._retirement_cycles.append(cycle_number)
            self._retired_sample_indices.append(retired_context.index)

        snapshot = self._snapshot()
        self._cycle = cycle_number + 1
        self.snapshots.append(snapshot)
        return snapshot


__all__ = [
    "CycleConfig",
    "CycleInputs",
    "CycleReference",
    "CycleSnapshot",
    "ResultReadout",
    "SampleContext",
]
