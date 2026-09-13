"""Cycle-accurate architectural reference for the Phase 6 accelerator.

This module intentionally models the *architectural* schedule rather than
the RTL hierarchy.  A sample is a token moving through the activation FIFO,
skew/data timing slots, fixed result alignment, and the result FIFO.  Learning
packages are separate tokens moving through compact anti-diagonal and
reduction-boundary pipelines.  The pure numerical operations all come from
``reference.arithmetic``; no functional-model state is consulted.

Snapshots returned by :meth:`CycleReference.step` describe the state *after*
the simulated clock edge, while handshake and output fields describe the
signals that were presented during that cycle and caused that edge.  This
convention makes a trace useful for both questions: what was accepted or
retired on cycle C, and what state exists immediately afterwards.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .arithmetic import (
    activate,
    apply_matrix_update,
    apply_reduction_update,
    activation_gates,
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


def _read_config_value(config: Any, names: Sequence[str], default: Any = None) -> Any:
    if isinstance(config, Mapping):
        for name in names:
            if name in config:
                return config[name]
        return default
    if config is None:
        return default
    for name in names:
        if hasattr(config, name):
            return getattr(config, name)
    return default


def _copy_matrix(
    values: Sequence[Sequence[int]],
    n: int,
    width: int,
) -> list[list[int]]:
    if len(values) != n or any(len(row) != n for row in values):
        raise ValueError("W must be an N by N matrix")
    return [[to_signed(value, width) for value in row] for row in values]


def _copy_vector(values: Sequence[int], n: int, width: int) -> list[int]:
    if len(values) != n:
        raise ValueError("vector must have length N")
    return [to_signed(value, width) for value in values]


def _zero_matrix(n: int) -> list[list[int]]:
    return [[0 for _ in range(n)] for _ in range(n)]


def _zero_matrix_tuple(n: int) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(0 for _ in range(n)) for _ in range(n))


def _tuple_matrix(values: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in values)


def _tuple_vector(values: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


@dataclass(frozen=True)
class CycleConfig:
    """Parameters used by the cycle model.

    The names and defaults follow ``nnAccelerator``.  ``input_fifo_depth``
    and ``output_fifo_depth`` are included because they determine when a
    shared datapath freeze occurs.
    """

    n: int = 3
    width: int = 16
    fraction_bits: int = 4
    target_width: int | None = None
    reduction_weight_width: int = 8
    input_fifo_depth: int | None = None
    output_fifo_depth: int | None = None
    pass_through: bool = True
    # Kept for Phase 6B configuration compatibility.  The cycle scheduler
    # derives visibility from actual edges and does not use this as a delay
    # shortcut.
    update_visibility_delay: int | None = None

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
        if self.update_visibility_delay is None:
            object.__setattr__(self, "update_visibility_delay", 2 * self.n + 1)
        elif (
            not isinstance(self.update_visibility_delay, int)
            or isinstance(self.update_visibility_delay, bool)
            or self.update_visibility_delay < 1
        ):
            raise ValueError("update_visibility_delay must be positive")

    @property
    def matrix_result_width(self) -> int:
        return matrix_result_width(self.width, self.n)

    @property
    def prediction_width(self) -> int:
        return prediction_width(self.width, self.n)

    @property
    def sample_context_depth(self) -> int:
        """Depth of the top-level context FIFO (Phase 6B contract)."""

        return max(self.input_fifo_depth, 2 * self.n + 2)  # type: ignore[arg-type]

    @property
    def result_alignment_delays(self) -> tuple[int, ...]:
        """Registered suffix delay for each result column."""

        return tuple(max(0, self.n - 1 - column) for column in range(self.n))

    # RTL-style aliases make parameterized testbench setup less error-prone.
    @property
    def N(self) -> int:
        return self.n

    @property
    def WIDTH(self) -> int:
        return self.width

    @property
    def FRACTION_BITS(self) -> int:
        return self.fraction_bits

    @property
    def fractionBits(self) -> int:
        return self.fraction_bits

    @property
    def TARGET_WIDTH(self) -> int:
        return self.target_width  # type: ignore[return-value]

    @property
    def targetWidth(self) -> int:
        return self.target_width  # type: ignore[return-value]

    @property
    def REDUCTION_WEIGHT_WIDTH(self) -> int:
        return self.reduction_weight_width

    @property
    def reductionWeightWidth(self) -> int:
        return self.reduction_weight_width

    @property
    def INPUT_FIFO_DEPTH(self) -> int:
        return self.input_fifo_depth  # type: ignore[return-value]

    @property
    def OUTPUT_FIFO_DEPTH(self) -> int:
        return self.output_fifo_depth  # type: ignore[return-value]

    @property
    def inputFifoDepth(self) -> int:
        return self.input_fifo_depth  # type: ignore[return-value]

    @property
    def outputFifoDepth(self) -> int:
        return self.output_fifo_depth  # type: ignore[return-value]

    @property
    def SAMPLE_CONTEXT_DEPTH(self) -> int:
        return self.sample_context_depth

    @property
    def passThrough(self) -> bool:
        return self.pass_through

    @property
    def updateVisibilityDelay(self) -> int:
        return self.update_visibility_delay  # type: ignore[return-value]

    @property
    def UPDATE_VISIBILITY_DELAY(self) -> int:
        return self.update_visibility_delay  # type: ignore[return-value]


def _coerce_config(config: CycleConfig | Mapping[str, Any] | Any) -> CycleConfig:
    if isinstance(config, CycleConfig):
        return config
    return CycleConfig(
        n=_read_config_value(config, ("n", "N"), 3),
        width=_read_config_value(config, ("width", "WIDTH"), 16),
        fraction_bits=_read_config_value(
            config,
            ("fraction_bits", "fractionBits", "FRACTION_BITS"),
            4,
        ),
        target_width=_read_config_value(
            config,
            ("target_width", "targetWidth", "TARGET_WIDTH"),
            None,
        ),
        reduction_weight_width=_read_config_value(
            config,
            (
                "reduction_weight_width",
                "reductionWeightWidth",
                "REDUCTION_WEIGHT_WIDTH",
            ),
            8,
        ),
        input_fifo_depth=_read_config_value(
            config,
            ("input_fifo_depth", "inputFifoDepth", "INPUT_FIFO_DEPTH"),
            None,
        ),
        output_fifo_depth=_read_config_value(
            config,
            ("output_fifo_depth", "outputFifoDepth", "OUTPUT_FIFO_DEPTH"),
            None,
        ),
        pass_through=_read_config_value(
            config,
            ("pass_through", "passThrough"),
            True,
        ),
        update_visibility_delay=_read_config_value(
            config,
            (
                "update_visibility_delay",
                "updateVisibilityDelay",
                "UPDATE_VISIBILITY_DELAY",
            ),
            None,
        ),
    )


@dataclass(frozen=True)
class CycleInputs:
    """External conditions for one simulated clock cycle."""

    activation_valid: bool = False
    activation_data: tuple[int, ...] = ()
    target_data: int = 0
    training_enable: bool = True
    result_ready: bool = False
    weight_valid: bool = False
    weight_data: tuple[int, ...] = ()
    reduction_weight: tuple[int, ...] = ()
    load_reduction_weights: bool = False
    reduce_output: bool = False
    pass_through: bool | None = None
    reload_weights: bool = False
    reset_n: bool = True

    def __init__(
        self,
        activation_valid: bool = False,
        activation_data: Sequence[int] = (),
        target_data: int = 0,
        training_enable: bool = True,
        result_ready: bool = False,
        weight_valid: bool = False,
        weight_data: Sequence[int] = (),
        reduction_weight: Sequence[int] = (),
        load_reduction_weights: bool = False,
        reduce_output: bool = False,
        pass_through: bool | None = None,
        reload_weights: bool = False,
        reset_n: bool = True,
        **aliases: Any,
    ) -> None:
        alias_map = {
            "activationValid": "activation_valid",
            "activationData": "activation_data",
            "targetData": "target_data",
            "trainingEnable": "training_enable",
            "resultReady": "result_ready",
            "weightValid": "weight_valid",
            "weightData": "weight_data",
            "reductionWeight": "reduction_weight",
            "loadReductionWeights": "load_reduction_weights",
            "reduceOutput": "reduce_output",
            "passThrough": "pass_through",
            "reloadWeights": "reload_weights",
            "rst_n": "reset_n",
        }
        for alias, canonical in alias_map.items():
            if alias in aliases:
                value = aliases.pop(alias)
                if canonical == "activation_valid":
                    activation_valid = value
                elif canonical == "activation_data":
                    activation_data = value
                elif canonical == "target_data":
                    target_data = value
                elif canonical == "training_enable":
                    training_enable = value
                elif canonical == "result_ready":
                    result_ready = value
                elif canonical == "weight_valid":
                    weight_valid = value
                elif canonical == "weight_data":
                    weight_data = value
                elif canonical == "reduction_weight":
                    reduction_weight = value
                elif canonical == "load_reduction_weights":
                    load_reduction_weights = value
                elif canonical == "reduce_output":
                    reduce_output = value
                elif canonical == "pass_through":
                    pass_through = value
                elif canonical == "reload_weights":
                    reload_weights = value
                elif canonical == "reset_n":
                    reset_n = value
        if aliases:
            unknown = next(iter(aliases))
            raise TypeError(f"unexpected cycle input {unknown!r}")
        object.__setattr__(self, "activation_valid", bool(activation_valid))
        object.__setattr__(self, "activation_data", tuple(int(v) for v in activation_data))
        object.__setattr__(self, "target_data", int(target_data))
        object.__setattr__(self, "training_enable", bool(training_enable))
        object.__setattr__(self, "result_ready", bool(result_ready))
        object.__setattr__(self, "weight_valid", bool(weight_valid))
        object.__setattr__(self, "weight_data", tuple(int(v) for v in weight_data))
        object.__setattr__(
            self,
            "reduction_weight",
            tuple(int(v) for v in reduction_weight),
        )
        object.__setattr__(self, "load_reduction_weights", bool(load_reduction_weights))
        object.__setattr__(self, "reduce_output", bool(reduce_output))
        object.__setattr__(
            self,
            "pass_through",
            None if pass_through is None else bool(pass_through),
        )
        object.__setattr__(self, "reload_weights", bool(reload_weights))
        object.__setattr__(self, "reset_n", bool(reset_n))

    @property
    def activationValid(self) -> bool:
        return self.activation_valid

    @property
    def resultReady(self) -> bool:
        return self.result_ready

    @property
    def weightValid(self) -> bool:
        return self.weight_valid

    @property
    def loadReductionWeights(self) -> bool:
        return self.load_reduction_weights

    @property
    def reduceOutput(self) -> bool:
        return self.reduce_output

    @property
    def passThrough(self) -> bool | None:
        return self.pass_through


def _coerce_inputs(
    inputs: CycleInputs | Mapping[str, Any] | Any | None,
    overrides: Mapping[str, Any],
) -> CycleInputs:
    values: dict[str, Any] = {}
    if isinstance(inputs, CycleInputs):
        values.update(
            {
                "activation_valid": inputs.activation_valid,
                "activation_data": inputs.activation_data,
                "target_data": inputs.target_data,
                "training_enable": inputs.training_enable,
                "result_ready": inputs.result_ready,
                "weight_valid": inputs.weight_valid,
                "weight_data": inputs.weight_data,
                "reduction_weight": inputs.reduction_weight,
                "load_reduction_weights": inputs.load_reduction_weights,
                "reduce_output": inputs.reduce_output,
                "pass_through": inputs.pass_through,
                "reload_weights": inputs.reload_weights,
                "reset_n": inputs.reset_n,
            }
        )
    elif isinstance(inputs, Mapping):
        values.update(inputs)
    elif inputs is not None:
        for name in (
            "activation_valid",
            "activationValid",
            "activation_data",
            "activationData",
            "target_data",
            "targetData",
            "training_enable",
            "trainingEnable",
            "result_ready",
            "resultReady",
            "weight_valid",
            "weightValid",
            "weight_data",
            "weightData",
            "reduction_weight",
            "reductionWeight",
            "load_reduction_weights",
            "loadReductionWeights",
            "reduce_output",
            "reduceOutput",
            "pass_through",
            "passThrough",
            "reload_weights",
            "reloadWeights",
            "reset_n",
            "rst_n",
        ):
            if hasattr(inputs, name):
                values[name] = getattr(inputs, name)
    values.update(overrides)

    def pick(names: Sequence[str], default: Any) -> Any:
        for name in names:
            if name in values:
                return values[name]
        return default

    return CycleInputs(
        activation_valid=pick(("activation_valid", "activationValid"), False),
        activation_data=pick(("activation_data", "activationData"), ()),
        target_data=pick(("target_data", "targetData"), 0),
        training_enable=pick(("training_enable", "trainingEnable"), True),
        result_ready=pick(("result_ready", "resultReady"), False),
        weight_valid=pick(("weight_valid", "weightValid"), False),
        weight_data=pick(("weight_data", "weightData"), ()),
        reduction_weight=pick(("reduction_weight", "reductionWeight"), ()),
        load_reduction_weights=pick(
            ("load_reduction_weights", "loadReductionWeights"),
            False,
        ),
        reduce_output=pick(("reduce_output", "reduceOutput"), False),
        pass_through=pick(("pass_through", "passThrough"), None),
        reload_weights=pick(("reload_weights", "reloadWeights"), False),
        reset_n=pick(("reset_n", "rst_n"), True),
    )


@dataclass(frozen=True)
class SampleContext:
    """The transaction sideband retained until result retirement."""

    sample_index: int
    input_vector: tuple[int, ...]
    target: int
    training_enable: bool
    accepted_cycle: int

    @property
    def x(self) -> tuple[int, ...]:
        return self.input_vector

    @property
    def trainingEnable(self) -> bool:
        return self.training_enable


@dataclass(frozen=True)
class OutputEntry:
    """One complete raw result held by the architectural output FIFO."""

    sample_index: int
    raw_matrix_result: tuple[int, ...]
    accepted_cycle: int
    result_enqueue_cycle: int

    @property
    def raw(self) -> tuple[int, ...]:
        return self.raw_matrix_result


@dataclass(frozen=True)
class ResultMetadata:
    """Metadata FIFO contents paired with an :class:`OutputEntry`."""

    sample_index: int
    prediction: int
    reduction_weights_used: tuple[int, ...]
    reduction_weight_signs: tuple[int, ...]
    result_enqueue_cycle: int

    @property
    def R_used(self) -> tuple[int, ...]:
        return self.reduction_weights_used


@dataclass(frozen=True)
class MatrixUpdatePackage:
    """A live matrix package and its next anti-diagonal position."""

    update_id: int
    source_sample: int
    row_direction: tuple[int, ...]
    column_direction: tuple[int, ...]
    position: int
    generated_cycle: int

    @property
    def anti_diagonal(self) -> int:
        return self.position

    @property
    def rowDirection(self) -> tuple[int, ...]:
        return self.row_direction

    @property
    def columnDirection(self) -> tuple[int, ...]:
        return self.column_direction


@dataclass(frozen=True)
class ReductionUpdatePackage:
    """A live ternary reduction package and its delay-pipeline position."""

    update_id: int
    source_sample: int
    direction: tuple[int, ...]
    position: int
    generated_cycle: int

    @property
    def reduction_direction(self) -> tuple[int, ...]:
        return self.direction


@dataclass(frozen=True)
class AlignmentState:
    """A complete vector in fixed column alignment.

    ``remaining`` counts advancing edges before the vector is eligible for
    output enqueue.  Zero means the vector is currently aligned and waiting
    for an advancing edge, which is also when output backpressure freezes the
    shared datapath.
    """

    sample_index: int
    remaining: int
    raw_matrix_result: tuple[int, ...]


@dataclass(frozen=True)
class SamplePosition:
    """Compact timing position for an in-flight sample."""

    sample_index: int
    age: int
    next_cell_anti_diagonal: int
    state: str = "systolic"


@dataclass
class CycleSampleRecord:
    """Architectural result record produced for one accepted sample.

    ``W_used`` is the effective matrix observed at the individual PE edges.
    It is normally one full matrix generation, but remains a matrix of
    per-cell observations when an update wave overlaps a sample.  The
    ``update_visible_at`` field is the cycle on which that sample's live
    anti-diagonal-zero update is launched; the later matrix/reduction fields
    report their respective completion cycles.
    """

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
    row_direction: tuple[int, ...]
    column_direction: tuple[int, ...]
    matrix_update_directions: tuple[tuple[int, ...], ...]
    reduction_update_directions: tuple[int, ...]
    activation_gate: tuple[bool, ...]
    update_generated: bool
    accepted_cycle: int
    result_enqueue_cycle: int
    update_visible_at: int | None = None
    result_retire_cycle: int | None = None
    matrix_update_cycle: int | None = None
    reduction_update_cycle: int | None = None
    matrix_update_id: int | None = None
    retirement_activated_result: tuple[int, ...] | None = None

    @property
    def x(self) -> tuple[int, ...]:
        return self.input_vector

    @property
    def w_used(self) -> tuple[tuple[int, ...], ...]:
        return self.W_used

    @property
    def r_used(self) -> tuple[int, ...]:
        return self.R_used

    @property
    def trainingEnable(self) -> bool:
        return self.training_enable

    @property
    def learningDirection(self) -> int:
        return self.learning_direction

    @property
    def rowDirection(self) -> tuple[int, ...]:
        return self.row_direction

    @property
    def columnDirection(self) -> tuple[int, ...]:
        return self.column_direction

    @property
    def matrixUpdateDirections(self) -> tuple[tuple[int, ...], ...]:
        return self.matrix_update_directions

    @property
    def reductionUpdateDirections(self) -> tuple[int, ...]:
        return self.reduction_update_directions

    @property
    def candidate_matrix_update_directions(self) -> tuple[tuple[int, ...], ...]:
        if self.update_generated:
            return self.matrix_update_directions
        return tuple(
            tuple(
                ternary_product(row_value, column_value)
                for column_value in self.column_direction
            )
            for row_value in self.row_direction
        )

    @property
    def candidate_reduction_update_directions(self) -> tuple[int, ...]:
        if self.update_generated:
            return self.reduction_update_directions
        return tuple(
            self.learning_direction * ternary_sign(value)
            for value in self.activated_result
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "input_vector": list(self.input_vector),
            "target": self.target,
            "training_enable": self.training_enable,
            "W_used": [list(row) for row in self.W_used],
            "R_used": list(self.R_used),
            "raw_matrix_result": list(self.raw_matrix_result),
            "activated_result": list(self.activated_result),
            "prediction": self.prediction,
            "learning_direction": self.learning_direction,
            "row_direction": list(self.row_direction),
            "column_direction": list(self.column_direction),
            "matrix_update_directions": [
                list(row) for row in self.matrix_update_directions
            ],
            "reduction_update_directions": list(self.reduction_update_directions),
            "activation_gate": list(self.activation_gate),
            "update_generated": self.update_generated,
            "accepted_cycle": self.accepted_cycle,
            "result_enqueue_cycle": self.result_enqueue_cycle,
            "update_visible_at": self.update_visible_at,
            "result_retire_cycle": self.result_retire_cycle,
            "matrix_update_cycle": self.matrix_update_cycle,
            "reduction_update_cycle": self.reduction_update_cycle,
            "matrix_update_id": self.matrix_update_id,
        }


@dataclass(frozen=True)
class CycleSnapshot:
    """Meaningful architectural state and edge events for one cycle."""

    cycle: int
    activation_valid: bool
    activation_ready: bool
    activation_accepted: bool
    accepted_sample: int | None
    weight_valid: bool
    weight_ready: bool
    weight_accepted: bool
    weight_popped: bool
    result_valid: bool
    result_ready: bool
    result_retired: bool
    result_enqueued: bool
    retired_sample: int | None
    result_last: bool
    current_output_sample: int | None
    current_output_result: tuple[int, ...] | None
    current_output_raw: tuple[int, ...] | None
    prediction: int
    learning_direction: int
    datapath_advance: bool
    matrix_update_valid: bool
    matrix_update_accepted: bool
    matrix_update_complete: bool
    completed_matrix_update_id: int | None
    matrix_updates_applied: tuple[tuple[int, int], ...]
    reduction_boundary_apply: bool
    reduction_update_applied: bool
    completed_reduction_update_id: int | None
    reduction_update_generated: bool
    weights_loaded: bool
    reload_ready: bool
    reload_accepted: bool
    w_generation: int
    r_generation: int
    W: tuple[tuple[int, ...], ...]
    R: tuple[int, ...]
    weight_fifo_contents: tuple[tuple[int, ...], ...]
    weight_load_positions: tuple[tuple[int, ...] | None, ...]
    activation_fifo_contents: tuple[SampleContext, ...]
    sample_context_fifo_contents: tuple[SampleContext, ...]
    skew_slots: tuple[int | None, ...]
    sample_positions: tuple[SamplePosition, ...]
    result_alignment_contents: tuple[AlignmentState, ...]
    result_alignment_delays: tuple[int, ...]
    output_fifo_contents: tuple[OutputEntry, ...]
    result_metadata_fifo_contents: tuple[ResultMetadata, ...]
    matrix_update_packages: tuple[MatrixUpdatePackage, ...]
    reduction_update_packages: tuple[ReductionUpdatePackage, ...]
    input_bubble: bool
    activation_popped: bool
    datapath_bubble: bool
    pass_through: bool
    reduce_output: bool

    @property
    def w(self) -> tuple[tuple[int, ...], ...]:
        return self.W

    @property
    def r(self) -> tuple[int, ...]:
        return self.R

    @property
    def result_data(self) -> tuple[int, ...] | None:
        return self.current_output_result

    @property
    def current_output(self) -> tuple[int, ...] | None:
        return self.current_output_result

    @property
    def inputQ(self) -> tuple[int, ...]:
        return tuple(item.sample_index for item in self.activation_fifo_contents)

    @property
    def input_fifo_contents(self) -> tuple[tuple[int, ...], ...]:
        return tuple(item.input_vector for item in self.activation_fifo_contents)

    @property
    def activation_vectors(self) -> tuple[tuple[int, ...], ...]:
        return self.input_fifo_contents

    @property
    def outputQ(self) -> tuple[int, ...]:
        return tuple(item.sample_index for item in self.output_fifo_contents)

    @property
    def activationPop(self) -> bool:
        return self.activation_popped

    @property
    def resultEnqueue(self) -> bool:
        return self.result_enqueued

    @property
    def resultAlignmentDelays(self) -> tuple[int, ...]:
        return self.result_alignment_delays

    @property
    def datapathAdvance(self) -> bool:
        return self.datapath_advance

    @property
    def activationReady(self) -> bool:
        return self.activation_ready

    @property
    def weightReady(self) -> bool:
        return self.weight_ready

    @property
    def resultReady(self) -> bool:
        return self.result_ready

    @property
    def resultValid(self) -> bool:
        return self.result_valid

    @property
    def resultRetired(self) -> bool:
        return self.result_retired

    @property
    def matrixUpdateValid(self) -> bool:
        return self.matrix_update_valid

    @property
    def resultLast(self) -> bool:
        return self.result_last

    @property
    def learningDirection(self) -> int:
        return self.learning_direction

    @property
    def weightsLoaded(self) -> bool:
        return self.weights_loaded

    @property
    def activation_fifo(self) -> tuple[SampleContext, ...]:
        return self.activation_fifo_contents

    @property
    def sample_context_fifo(self) -> tuple[SampleContext, ...]:
        return self.sample_context_fifo_contents

    @property
    def output_fifo(self) -> tuple[OutputEntry, ...]:
        return self.output_fifo_contents

    @property
    def result_metadata_fifo(self) -> tuple[ResultMetadata, ...]:
        return self.result_metadata_fifo_contents

    @property
    def updates(self) -> tuple[MatrixUpdatePackage, ...]:
        return self.matrix_update_packages


@dataclass
class _DataToken:
    context: SampleContext
    age: int
    observed_weights: list[list[int | None]]


@dataclass
class _LiveMatrixPackage:
    update_id: int
    source_sample: int
    row_direction: tuple[int, ...]
    column_direction: tuple[int, ...]
    next_diagonal: int
    generated_cycle: int


@dataclass
class _LiveReductionPackage:
    update_id: int
    source_sample: int
    direction: tuple[int, ...]
    position: int
    generated_cycle: int


class CycleReference:
    """Cycle-by-cycle architectural model of ``nnAccelerator``.

    ``W`` and ``R`` may be supplied to start in the already-loaded state,
    which is convenient for arithmetic cross-checks.  If ``W`` is omitted,
    the model starts after reset with ``weights_loaded == False`` and accepts
    the RTL's streamed weight vectors.  Pass ``weights_loaded=False`` to
    force the loading mode even when an initial matrix is provided.
    """

    def __init__(
        self,
        config: CycleConfig | Mapping[str, Any] | Any = None,
        W: Sequence[Sequence[int]] | None = None,
        R: Sequence[int] | None = None,
        *,
        weights_loaded: bool | None = None,
    ) -> None:
        self.config = _coerce_config(config)
        self._initial_W = (
            _copy_matrix(W, self.config.n, self.config.width)
            if W is not None
            else _zero_matrix(self.config.n)
        )
        self._initial_R = (
            _copy_vector(R, self.config.n, self.config.reduction_weight_width)
            if R is not None
            else [0 for _ in range(self.config.n)]
        )
        self._initial_weights_loaded = bool(W is not None) if weights_loaded is None else bool(weights_loaded)
        if self._initial_weights_loaded and W is None:
            raise ValueError("weights_loaded=True requires an initial W")
        self.snapshots: list[CycleSnapshot] = []
        self.reset()

    # ------------------------------------------------------------------
    # Public state and lifecycle API
    # ------------------------------------------------------------------

    @property
    def W(self) -> list[list[int]]:
        return [row[:] for row in self._W]

    @property
    def R(self) -> list[int]:
        return self._R[:]

    @property
    def final_W(self) -> list[list[int]]:
        return self.W

    @property
    def final_R(self) -> list[int]:
        return self.R

    @property
    def cycle(self) -> int:
        return self._cycle

    @property
    def weights_loaded(self) -> bool:
        return self._weights_loaded

    @property
    def activation_fifo_contents(self) -> tuple[SampleContext, ...]:
        return tuple(self._activation_fifo)

    @property
    def sample_context_fifo_contents(self) -> tuple[SampleContext, ...]:
        return tuple(self._sample_context_fifo)

    @property
    def output_fifo_contents(self) -> tuple[OutputEntry, ...]:
        return tuple(self._output_fifo)

    @property
    def result_metadata_fifo_contents(self) -> tuple[ResultMetadata, ...]:
        return tuple(self._result_metadata_fifo)

    @property
    def next_sample_index(self) -> int:
        return self._next_sample_index

    @property
    def records(self) -> list[CycleSampleRecord]:
        return self._records[:]

    @property
    def generated_matrix_updates(self) -> tuple[MatrixUpdatePackage, ...]:
        return tuple(self._generated_matrix_updates)

    @property
    def generated_reduction_updates(self) -> tuple[ReductionUpdatePackage, ...]:
        return tuple(self._generated_reduction_updates)

    @property
    def matrix_update_count(self) -> int:
        return len(self._generated_matrix_updates)

    @property
    def reduction_update_count(self) -> int:
        return len(self._generated_reduction_updates)

    @property
    def in_flight(self) -> bool:
        return bool(
            self._activation_fifo
            or self._sample_context_fifo
            or self._data_tokens
            or self._capture_pending
            or self._alignment
            or self._output_fifo
            or self._matrix_packages
            or any(package is not None for package in self._reduction_pipe)
        )

    def reset(self) -> None:
        """Return to the constructor's state, including loaded W when supplied."""

        self.snapshots = []
        self._cycle = 0
        self._next_sample_index = 0
        self._next_update_id = 0
        self._W = [row[:] for row in self._initial_W] if self._initial_weights_loaded else _zero_matrix(self.config.n)
        self._R = self._initial_R[:]
        self._weights_loaded = self._initial_weights_loaded
        self._loaded_weight_count = self.config.n if self._weights_loaded else 0
        self._accepted_activation_row = 0
        self._transmitted_result_row = 0
        self._weight_fifo: deque[tuple[int, ...]] = deque()
        self._activation_fifo: deque[SampleContext] = deque()
        self._sample_context_fifo: deque[SampleContext] = deque()
        self._output_fifo: deque[OutputEntry] = deque()
        self._result_metadata_fifo: deque[ResultMetadata] = deque()
        self._weight_load_pipe: list[tuple[int, ...] | None] = [None] * self.config.n
        self._skew_slots: list[int | None] = [None] * self.config.n
        self._data_tokens: list[_DataToken] = []
        self._capture_pending: deque[int] = deque()
        self._completed_tokens: dict[int, _DataToken] = {}
        self._alignment: deque[AlignmentState] = deque()
        self._matrix_packages: list[_LiveMatrixPackage] = []
        self._reduction_pipe: list[_LiveReductionPackage | None] = [
            None
        ] * (2 * self.config.n - 1)
        self._records: list[CycleSampleRecord] = []
        self._record_by_sample: dict[int, CycleSampleRecord] = {}
        self._generated_matrix_updates: list[MatrixUpdatePackage] = []
        self._generated_reduction_updates: list[ReductionUpdatePackage] = []
        self._w_generation = 0
        self._r_generation = 0

    def _hardware_reset(self) -> None:
        """Apply synchronous RTL reset state, not constructor initial state."""

        self._next_sample_index = 0
        self._next_update_id = 0
        self._W = _zero_matrix(self.config.n)
        self._R = [0 for _ in range(self.config.n)]
        self._weights_loaded = False
        self._loaded_weight_count = 0
        self._accepted_activation_row = 0
        self._transmitted_result_row = 0
        self._weight_fifo.clear()
        self._activation_fifo.clear()
        self._sample_context_fifo.clear()
        self._output_fifo.clear()
        self._result_metadata_fifo.clear()
        self._weight_load_pipe = [None] * self.config.n
        self._skew_slots = [None] * self.config.n
        self._data_tokens.clear()
        self._capture_pending.clear()
        self._completed_tokens.clear()
        self._alignment.clear()
        self._matrix_packages.clear()
        self._reduction_pipe = [None] * (2 * self.config.n - 1)
        self._records.clear()
        self._record_by_sample.clear()
        self._generated_matrix_updates.clear()
        self._generated_reduction_updates.clear()
        self._w_generation = 0
        self._r_generation = 0

    def flush(self, max_cycles: int = 10000) -> list[CycleSnapshot]:
        """Advance idle cycles with ready output until the model drains.

        This is a convenience for end-of-stream tests.  It never introduces a
        sample; it only supplies the external ``resultReady`` condition.
        """

        if not isinstance(max_cycles, int) or isinstance(max_cycles, bool) or max_cycles < 0:
            raise ValueError("max_cycles must be a non-negative integer")
        added: list[CycleSnapshot] = []
        for _ in range(max_cycles):
            if not self.in_flight:
                return added
            added.append(self.step(result_ready=True))
        if self.in_flight:
            raise RuntimeError("cycle model did not drain within max_cycles")
        return added

    def run(
        self,
        cycles: Sequence[CycleInputs | Mapping[str, Any] | Any],
    ) -> list[CycleSnapshot]:
        """Simulate a sequence of externally supplied cycles."""

        return [self.step(cycle_input) for cycle_input in cycles]

    # ------------------------------------------------------------------
    # Small architectural helpers
    # ------------------------------------------------------------------

    def _effective_pass_through(self, inputs: CycleInputs) -> bool:
        return self.config.pass_through if inputs.pass_through is None else inputs.pass_through

    def _normalize_input_vector(self, values: Sequence[int]) -> tuple[int, ...]:
        if len(values) != self.config.n:
            raise ValueError("activation_data must have length N")
        return tuple(to_signed(value, self.config.width) for value in values)

    def _normalize_weight_vector(self, values: Sequence[int]) -> tuple[int, ...]:
        if len(values) != self.config.n:
            raise ValueError("weight_data must have length N")
        return tuple(to_signed(value, self.config.width) for value in values)

    def _normalize_reduction_vector(self, values: Sequence[int]) -> tuple[int, ...]:
        if len(values) != self.config.n:
            raise ValueError("reduction_weight must have length N")
        return tuple(
            to_signed(value, self.config.reduction_weight_width) for value in values
        )

    def _pipeline_busy(self) -> bool:
        return bool(
            self._data_tokens
            or self._capture_pending
            or any(slot is not None for slot in self._skew_slots)
            or self._matrix_packages
        )

    def _skew_busy(self) -> bool:
        return any(slot is not None for slot in self._skew_slots)

    def _result_align_busy(self) -> bool:
        return any(entry.remaining > 0 for entry in self._alignment)

    def _matrix_reload_ready(self, reduction_boundary_busy: bool) -> bool:
        return bool(
            self._weights_loaded
            and not self._activation_fifo
            and not self._skew_busy()
            and not self._pipeline_busy()
            and not self._result_align_busy()
            and not self._output_fifo
            and self._accepted_activation_row == 0
            and not reduction_boundary_busy
        )

    def _head_state(self) -> tuple[
        OutputEntry | None,
        ResultMetadata | None,
        SampleContext | None,
    ]:
        output = self._output_fifo[0] if self._output_fifo else None
        metadata = self._result_metadata_fifo[0] if self._result_metadata_fifo else None
        context = self._sample_context_fifo[0] if self._sample_context_fifo else None
        return output, metadata, context

    def _output_view(
        self,
        output: OutputEntry | None,
        metadata: ResultMetadata | None,
        context: SampleContext | None,
        pass_through: bool,
        reduce_output: bool,
    ) -> tuple[bool, int, int, tuple[int, ...] | None, tuple[int, ...] | None, int | None]:
        valid = output is not None and metadata is not None and context is not None
        if not valid:
            return False, 0, 0, None, None, None
        assert output is not None and metadata is not None and context is not None
        activated = tuple(
            activate(output.raw_matrix_result, pass_through, self.config.matrix_result_width)
        )
        if reduce_output:
            result = (metadata.prediction,) + (0,) * (self.config.n - 1)
        else:
            result = tuple(
                to_signed(value, self.config.prediction_width) for value in activated
            )
        direction = learning_direction(
            context.target,
            metadata.prediction,
            self.config.target_width,
            self.config.prediction_width,
        )
        return (
            True,
            metadata.prediction,
            direction,
            result,
            activated,
            output.sample_index,
        )

    def _record_for_enqueue(
        self,
        token: _DataToken,
        raw_result: tuple[int, ...],
        pass_through: bool,
        R_used: Sequence[int],
    ) -> CycleSampleRecord:
        observed = token.observed_weights
        if any(value is None for row in observed for value in row):
            raise AssertionError("completed sample has an unobserved PE weight")
        observed_matrix = tuple(
            tuple(int(value) for value in row)  # type: ignore[arg-type]
            for row in observed
        )
        activated = tuple(
            activate(raw_result, pass_through, self.config.matrix_result_width)
        )
        R_used = tuple(int(value) for value in R_used)
        prediction = weighted_vector_reduction(
            activated,
            R_used,
            self.config.width,
            self.config.reduction_weight_width,
            self.config.fraction_bits,
        )
        direction = learning_direction(
            token.context.target,
            prediction,
            self.config.target_width,
            self.config.prediction_width,
        )
        row_direction, column_direction, candidate_matrix = matrix_update_directions(
            token.context.input_vector,
            activated,
            R_used,
            direction,
            self.config.width,
            self.config.reduction_weight_width,
            pass_through,
        )
        candidate_reduction = reduction_update_directions(activated, direction)
        generated = token.context.training_enable
        matrix_direction = (
            _tuple_matrix(candidate_matrix) if generated else _zero_matrix_tuple(self.config.n)
        )
        reduction_direction = (
            tuple(candidate_reduction) if generated else (0,) * self.config.n
        )
        record = CycleSampleRecord(
            sample_index=token.context.sample_index,
            input_vector=token.context.input_vector,
            target=token.context.target,
            training_enable=token.context.training_enable,
            W_used=observed_matrix,
            R_used=R_used,
            raw_matrix_result=raw_result,
            activated_result=activated,
            prediction=prediction,
            learning_direction=direction,
            row_direction=tuple(row_direction),
            column_direction=tuple(column_direction),
            matrix_update_directions=matrix_direction,
            reduction_update_directions=reduction_direction,
            activation_gate=tuple(activation_gates(activated, pass_through)),
            update_generated=generated,
            accepted_cycle=token.context.accepted_cycle,
            result_enqueue_cycle=self._cycle,
        )
        self._records.append(record)
        self._record_by_sample[record.sample_index] = record
        return record

    def _update_record_at_retirement(
        self,
        record: CycleSampleRecord,
        activated: tuple[int, ...],
        row_direction: tuple[int, ...],
        column_direction: tuple[int, ...],
        matrix_direction: tuple[tuple[int, ...], ...],
        reduction_direction: tuple[int, ...],
    ) -> None:
        record.retirement_activated_result = activated
        record.row_direction = row_direction
        record.column_direction = column_direction
        record.activation_gate = tuple(
            activation_gates(activated, self._current_pass_through)
        )
        if record.update_generated:
            record.matrix_update_directions = matrix_direction
            record.reduction_update_directions = reduction_direction

    def _apply_matrix_diagonal(self, package: _LiveMatrixPackage, diagonal: int) -> None:
        direction_matrix = _zero_matrix(self.config.n)
        for row in range(self.config.n):
            for column in range(self.config.n):
                if row + column == diagonal:
                    direction_matrix[row][column] = ternary_product(
                        package.row_direction[row],
                        package.column_direction[column],
                    )
        self._W = apply_matrix_update(
            self._W,
            direction_matrix,
            self.config.width,
        )

    def _advance_matrix_updates(self, injected: _LiveMatrixPackage | None) -> tuple[
        bool,
        int | None,
        list[tuple[int, int]],
    ]:
        """Advance/update matrix packages on one shared datapath edge."""

        completed_id: int | None = None
        applied: list[tuple[int, int]] = []
        next_packages: list[_LiveMatrixPackage] = []
        last_diagonal = 2 * self.config.n - 2

        for package in self._matrix_packages:
            diagonal = package.next_diagonal
            self._apply_matrix_diagonal(package, diagonal)
            applied.append((package.update_id, diagonal))
            if diagonal == last_diagonal:
                completed_id = package.update_id
            else:
                package.next_diagonal += 1
                next_packages.append(package)

        if injected is not None:
            self._apply_matrix_diagonal(injected, 0)
            applied.append((injected.update_id, 0))
            if last_diagonal == 0:
                completed_id = injected.update_id
            else:
                injected.next_diagonal = 1
                next_packages.append(injected)

        self._matrix_packages = next_packages
        completed = completed_id is not None
        if completed:
            self._w_generation += 1
        return completed, completed_id, applied

    def _advance_reduction_updates(
        self,
        injected: _LiveReductionPackage | None,
        load_reduction_weights: bool,
    ) -> tuple[bool, int | None]:
        """Advance the compact reduction delay and apply its old tail.

        ``loadReductionWeights`` has the same priority as the RTL register
        block: on that edge neither the delay pipe nor its resident state
        advances.  This is normally used only at a quiescent configuration
        boundary, but modeling the priority makes the contract explicit.
        """

        if load_reduction_weights:
            return False, None

        tail = self._reduction_pipe[-1]
        completed_id: int | None = None
        if tail is not None:
            self._R = apply_reduction_update(
                self._R,
                tail.direction,
                self.config.reduction_weight_width,
            )
            completed_id = tail.update_id
            self._r_generation += 1
            record = self._record_by_sample.get(tail.source_sample)
            if record is not None:
                record.reduction_update_cycle = self._cycle

        self._reduction_pipe = self._reduction_pipe[:-1]
        self._reduction_pipe.insert(0, injected)
        return completed_id is not None, completed_id

    def _shift_data_and_alignment(
        self,
        activation_context: SampleContext | None,
        activation_pop: bool,
        output_enqueue: bool,
    ) -> None:
        """Advance sample/skew/alignment positions for one array edge."""

        W_before = [row[:] for row in self._W]
        completed_tokens: list[_DataToken] = []
        active_tokens: list[_DataToken] = []
        final_age = 2 * self.config.n - 1

        for token in self._data_tokens:
            next_age = token.age + 1
            token.age = next_age
            if next_age <= final_age:
                diagonal = next_age - 1
                for row in range(self.config.n):
                    for column in range(self.config.n):
                        if row + column == diagonal:
                            token.observed_weights[row][column] = W_before[row][column]
            if next_age == final_age:
                observed = token.observed_weights
                if any(value is None for row in observed for value in row):
                    raise AssertionError("sample reached result boundary without all PE observations")
                observed_matrix = [
                    [int(value) for value in row]  # type: ignore[arg-type]
                    for row in observed
                ]
                raw = tuple(
                    matrix_multiply(
                        token.context.input_vector,
                        observed_matrix,
                        self.config.width,
                    )
                )
                completed_tokens.append(token)
                self._completed_tokens[token.context.sample_index] = token
                self._alignment.append(
                    AlignmentState(
                        sample_index=token.context.sample_index,
                        remaining=0,
                        raw_matrix_result=raw,
                    )
                )
                self._capture_pending.append(token.context.sample_index)
            else:
                active_tokens.append(token)

        self._data_tokens = active_tokens

        if activation_pop:
            assert activation_context is not None
            self._data_tokens.append(
                _DataToken(
                    context=activation_context,
                    age=0,
                    observed_weights=_zero_matrix(self.config.n),
                )
            )

        # The skew registers shift only with the shared advance.  A bubble is
        # represented by None; update packages do not depend on this queue.
        old_skew = self._skew_slots[:]
        self._skew_slots = [None] * self.config.n
        for stage in range(1, self.config.n):
            self._skew_slots[stage] = old_skew[stage - 1]
        if activation_pop:
            assert activation_context is not None
            self._skew_slots[0] = activation_context.sample_index

        # Existing fixed alignment stages move on this same edge.  A newly
        # completed vector is visible after the edge, so it cannot enqueue
        # until the following edge, matching the registered result FIFO input.
        shifted_alignment: deque[AlignmentState] = deque()
        skip_ready = output_enqueue
        for entry in self._alignment:
            if skip_ready and entry.remaining == 0:
                skip_ready = False
                continue
            shifted_alignment.append(
                entry
                if entry.remaining == 0
                else AlignmentState(
                    sample_index=entry.sample_index,
                    remaining=entry.remaining - 1,
                    raw_matrix_result=entry.raw_matrix_result,
                )
            )
        self._alignment = shifted_alignment

        # The result capture register clears on the next advancing edge.  A
        # newly completed sample is added above and therefore remains pending
        # for the next edge, just like verticalData[N] in the RTL.
        self._capture_pending.clear()
        self._capture_pending.extend(
            token.context.sample_index for token in completed_tokens
        )

    def _sample_positions_snapshot(self) -> tuple[SamplePosition, ...]:
        positions: list[SamplePosition] = []
        for token in sorted(self._data_tokens, key=lambda item: item.context.sample_index):
            positions.append(
                SamplePosition(
                    sample_index=token.context.sample_index,
                    age=token.age,
                    next_cell_anti_diagonal=token.age,
                    state="systolic",
                )
            )
        for sample in self._capture_pending:
            positions.append(
                SamplePosition(
                    sample_index=sample,
                    age=2 * self.config.n - 1,
                    next_cell_anti_diagonal=2 * self.config.n - 1,
                    state="result_capture",
                )
            )
        for entry in self._alignment:
            positions.append(
                SamplePosition(
                    sample_index=entry.sample_index,
                    age=2 * self.config.n - 1,
                    next_cell_anti_diagonal=2 * self.config.n - 1,
                    state="aligned" if entry.remaining == 0 else "alignment",
                )
            )
        positions.sort(key=lambda item: (item.sample_index, item.state))
        return tuple(positions)

    def _matrix_package_snapshot(self) -> tuple[MatrixUpdatePackage, ...]:
        return tuple(
            MatrixUpdatePackage(
                update_id=package.update_id,
                source_sample=package.source_sample,
                row_direction=package.row_direction,
                column_direction=package.column_direction,
                position=package.next_diagonal,
                generated_cycle=package.generated_cycle,
            )
            for package in self._matrix_packages
        )

    def _reduction_package_snapshot(self) -> tuple[ReductionUpdatePackage, ...]:
        return tuple(
            ReductionUpdatePackage(
                update_id=package.update_id,
                source_sample=package.source_sample,
                direction=package.direction,
                position=position,
                generated_cycle=package.generated_cycle,
            )
            for position, package in enumerate(self._reduction_pipe)
            if package is not None
        )

    def _make_snapshot(
        self,
        inputs: CycleInputs,
        *,
        activation_ready: bool = False,
        activation_accepted: bool = False,
        accepted_sample: int | None = None,
        weight_ready: bool = False,
        weight_accepted: bool = False,
        weight_popped: bool = False,
        result_valid: bool = False,
        result_retired: bool = False,
        result_enqueued: bool = False,
        retired_sample: int | None = None,
        result_last: bool = False,
        current_output_sample: int | None = None,
        current_output_result: tuple[int, ...] | None = None,
        current_output_raw: tuple[int, ...] | None = None,
        prediction: int = 0,
        learning_direction_value: int = 0,
        datapath_advance: bool = False,
        matrix_update_valid: bool = False,
        matrix_update_accepted: bool = False,
        matrix_update_complete: bool = False,
        completed_matrix_update_id: int | None = None,
        matrix_updates_applied: tuple[tuple[int, int], ...] = (),
        reduction_boundary_apply: bool = False,
        reduction_update_applied: bool = False,
        completed_reduction_update_id: int | None = None,
        reduction_update_generated: bool = False,
        reload_ready: bool = False,
        reload_accepted: bool = False,
        activation_popped: bool = False,
        datapath_bubble: bool = False,
    ) -> CycleSnapshot:
        return CycleSnapshot(
            cycle=self._cycle,
            activation_valid=inputs.activation_valid,
            activation_ready=activation_ready,
            activation_accepted=activation_accepted,
            accepted_sample=accepted_sample,
            weight_valid=inputs.weight_valid,
            weight_ready=weight_ready,
            weight_accepted=weight_accepted,
            weight_popped=weight_popped,
            result_valid=result_valid,
            result_ready=inputs.result_ready,
            result_retired=result_retired,
            result_enqueued=result_enqueued,
            retired_sample=retired_sample,
            result_last=result_last,
            current_output_sample=current_output_sample,
            current_output_result=current_output_result,
            current_output_raw=current_output_raw,
            prediction=prediction,
            learning_direction=learning_direction_value,
            datapath_advance=datapath_advance,
            matrix_update_valid=matrix_update_valid,
            matrix_update_accepted=matrix_update_accepted,
            matrix_update_complete=matrix_update_complete,
            completed_matrix_update_id=completed_matrix_update_id,
            matrix_updates_applied=matrix_updates_applied,
            reduction_boundary_apply=reduction_boundary_apply,
            reduction_update_applied=reduction_update_applied,
            completed_reduction_update_id=completed_reduction_update_id,
            reduction_update_generated=reduction_update_generated,
            weights_loaded=self._weights_loaded,
            reload_ready=reload_ready,
            reload_accepted=reload_accepted,
            w_generation=self._w_generation,
            r_generation=self._r_generation,
            W=_tuple_matrix(self._W),
            R=_tuple_vector(self._R),
            weight_fifo_contents=tuple(self._weight_fifo),
            weight_load_positions=tuple(self._weight_load_pipe),
            activation_fifo_contents=tuple(self._activation_fifo),
            sample_context_fifo_contents=tuple(self._sample_context_fifo),
            skew_slots=tuple(self._skew_slots),
            sample_positions=self._sample_positions_snapshot(),
            result_alignment_contents=tuple(self._alignment),
            result_alignment_delays=self.config.result_alignment_delays,
            output_fifo_contents=tuple(self._output_fifo),
            result_metadata_fifo_contents=tuple(self._result_metadata_fifo),
            matrix_update_packages=self._matrix_package_snapshot(),
            reduction_update_packages=self._reduction_package_snapshot(),
            input_bubble=inputs.activation_valid is False,
            activation_popped=activation_popped,
            datapath_bubble=datapath_bubble,
            pass_through=self._current_pass_through,
            reduce_output=inputs.reduce_output,
        )

    # ------------------------------------------------------------------
    # Main cycle transition
    # ------------------------------------------------------------------

    def step(
        self,
        inputs: CycleInputs | Mapping[str, Any] | Any | None = None,
        **overrides: Any,
    ) -> CycleSnapshot:
        """Simulate one clock edge and return its architectural snapshot.

        The method accepts a :class:`CycleInputs`, a mapping, an object with
        matching attributes, or direct keyword arguments.  Both snake_case
        and the RTL's camelCase names are accepted.
        """

        cycle_inputs = _coerce_inputs(inputs, overrides)
        self._current_pass_through = self._effective_pass_through(cycle_inputs)
        cycle_number = self._cycle

        if not cycle_inputs.reset_n:
            self._hardware_reset()
            snapshot = self._make_snapshot(cycle_inputs)
            self._cycle = cycle_number + 1
            self.snapshots.append(snapshot)
            return snapshot

        n = self.config.n
        pass_through = self._current_pass_through

        # Current output signals and handshakes are all combinational views of
        # pre-edge FIFO heads.  They are retained in the returned snapshot.
        output_head, metadata_head, context_head = self._head_state()
        (
            result_valid,
            prediction,
            learning_direction_value,
            current_output_result,
            current_activated,
            current_output_sample,
        ) = self._output_view(
            output_head,
            metadata_head,
            context_head,
            pass_through,
            cycle_inputs.reduce_output,
        )
        result_last = bool(result_valid and self._transmitted_result_row == n - 1)
        result_retired = bool(result_valid and cycle_inputs.result_ready)
        retired_sample = current_output_sample if result_retired else None

        context_full = len(self._sample_context_fifo) >= self.config.sample_context_depth
        sample_can_accept = (not context_full) or result_retired
        activation_full = len(self._activation_fifo) >= self.config.input_fifo_depth
        matrix_activation_ready = self._weights_loaded and not activation_full
        activation_ready = bool(matrix_activation_ready and sample_can_accept)
        activation_accepted = bool(cycle_inputs.activation_valid and activation_ready)

        weight_full = len(self._weight_fifo) >= n
        weight_ready = bool(not self._weights_loaded and not weight_full)
        weight_accepted = bool(cycle_inputs.weight_valid and weight_ready)
        weight_popped = bool(not self._weights_loaded and self._weight_fifo)

        reduction_boundary_live = bool(
            result_retired and context_head is not None and context_head.training_enable
        )
        reduction_tail_present = bool(
            self._weights_loaded and self._reduction_pipe[-1] is not None
        )
        reduction_boundary_busy = bool(
            reduction_boundary_live
            or reduction_tail_present
            or any(package is not None for package in self._reduction_pipe)
        )
        reload_ready = self._matrix_reload_ready(reduction_boundary_busy)
        reload_accepted = bool(cycle_inputs.reload_weights and reload_ready)

        output_full = len(self._output_fifo) >= self.config.output_fifo_depth
        output_blocked = bool(output_full and not result_retired and self._alignment and self._alignment[0].remaining == 0)
        if self._weights_loaded:
            datapath_advance = not output_blocked
        else:
            datapath_advance = weight_popped
        reduction_boundary_apply = bool(datapath_advance and reduction_tail_present)

        activation_popped = bool(
            self._weights_loaded
            and self._activation_fifo
            and datapath_advance
        )
        activation_context = self._activation_fifo[0] if activation_popped else None
        result_enqueue = bool(
            datapath_advance
            and self._alignment
            and self._alignment[0].remaining == 0
        )
        alignment_to_enqueue = self._alignment[0] if result_enqueue else None

        matrix_update_valid = bool(
            result_retired and context_head is not None and context_head.training_enable
        )
        matrix_update_accepted = bool(matrix_update_valid and datapath_advance)

        # The current result's update vectors use the snapshotted reduction
        # signs stored with that result, not the possibly newer resident R.
        injected_matrix: _LiveMatrixPackage | None = None
        injected_reduction: _LiveReductionPackage | None = None
        retirement_dirs: tuple[
            tuple[int, ...],
            tuple[int, ...],
            tuple[tuple[int, ...], ...],
            tuple[int, ...],
        ] | None = None
        if matrix_update_accepted:
            assert output_head is not None and metadata_head is not None and context_head is not None
            assert current_activated is not None
            sign_snapshot = metadata_head.reduction_weight_signs
            sign_weights = tuple(sign_snapshot)
            row_direction, column_direction, matrix_direction = matrix_update_directions(
                context_head.input_vector,
                current_activated,
                sign_weights,
                learning_direction_value,
                self.config.width,
                self.config.reduction_weight_width,
                pass_through,
            )
            reduction_direction = reduction_update_directions(
                current_activated,
                learning_direction_value,
            )
            update_id = self._next_update_id
            self._next_update_id += 1
            injected_matrix = _LiveMatrixPackage(
                update_id=update_id,
                source_sample=context_head.sample_index,
                row_direction=tuple(row_direction),
                column_direction=tuple(column_direction),
                next_diagonal=0,
                generated_cycle=cycle_number,
            )
            injected_reduction = _LiveReductionPackage(
                update_id=update_id,
                source_sample=context_head.sample_index,
                direction=tuple(reduction_direction),
                position=0,
                generated_cycle=cycle_number,
            )
            retirement_dirs = (
                tuple(row_direction),
                tuple(column_direction),
                _tuple_matrix(matrix_direction),
                tuple(reduction_direction),
            )

        # Capture pre-edge architectural state for computations whose RTL
        # register updates happen after the multiply/reduction on this edge.
        R_before = self._R[:]

        # External FIFO pushes are based on the same pre-edge ready signals.
        accepted_context: SampleContext | None = None
        if activation_accepted:
            input_vector = self._normalize_input_vector(cycle_inputs.activation_data)
            accepted_context = SampleContext(
                sample_index=self._next_sample_index,
                input_vector=input_vector,
                target=to_signed(cycle_inputs.target_data, self.config.target_width),
                training_enable=cycle_inputs.training_enable,
                accepted_cycle=cycle_number,
            )
            self._next_sample_index += 1

        pushed_weight: tuple[int, ...] | None = None
        if weight_accepted:
            pushed_weight = self._normalize_weight_vector(cycle_inputs.weight_data)

        # The matrix/data/update domain moves only on the shared advance.  The
        # data uses W_before; matrix updates are applied after those observations.
        if datapath_advance and self._weights_loaded:
            self._shift_data_and_alignment(
                activation_context,
                activation_popped,
                result_enqueue,
            )
        elif datapath_advance and not self._weights_loaded:
            # No activation/sample data can be accepted before the matrix is
            # loaded, but the weight loader itself is an advancing pipeline.
            pass

        # Matrix update waves advance on every array edge, including an edge
        # carrying a bubble.  Their live diagonal is applied after data use.
        matrix_complete = False
        completed_matrix_id: int | None = None
        matrix_applied: list[tuple[int, int]] = []
        if datapath_advance:
            matrix_complete, completed_matrix_id, matrix_applied = self._advance_matrix_updates(
                injected_matrix if matrix_update_accepted else None
            )

        # The reduction boundary follows the same advance, but its state block
        # gives an explicit reduction-weight load priority.
        reduction_complete = False
        completed_reduction_id: int | None = None
        if datapath_advance:
            reduction_complete, completed_reduction_id = self._advance_reduction_updates(
                injected_reduction if matrix_update_accepted else None,
                cycle_inputs.load_reduction_weights,
            )

        # Load streamed matrix weights.  The host order is bottom row first:
        # each popped vector enters row zero and shifts down one row per load
        # edge, exactly as the vertical load path does.
        if weight_popped:
            loaded_row = self._weight_fifo[0]
            self._weight_load_pipe = [loaded_row] + self._weight_load_pipe[:-1]
            for row in range(n):
                self._W[row] = (
                    list(self._weight_load_pipe[row])
                    if self._weight_load_pipe[row] is not None
                    else [0] * n
                )
            self._loaded_weight_count += 1
            if self._loaded_weight_count >= n:
                self._weights_loaded = True
                self._loaded_weight_count = 0

        # Resident reduction loading has priority over a boundary update, as
        # in the RTL always_ff block.  It is normally done while quiescent.
        if cycle_inputs.load_reduction_weights:
            self._R = list(self._normalize_reduction_vector(cycle_inputs.reduction_weight))

        # A legal reload starts a new weight-loading frame after the current
        # edge.  The PE contents are retained until the first new load edge.
        if reload_accepted:
            self._weights_loaded = False
            self._loaded_weight_count = 0
            self._weight_load_pipe = [None] * n

        # Apply FIFO state transitions.  All modeled FIFOs use signedFifo's
        # ordered pop/push behavior; output full + simultaneous pop is allowed.
        if weight_popped:
            self._weight_fifo.popleft()
        if pushed_weight is not None:
            self._weight_fifo.append(pushed_weight)

        if activation_popped:
            self._activation_fifo.popleft()
        if accepted_context is not None:
            self._activation_fifo.append(accepted_context)
            self._sample_context_fifo.append(accepted_context)
            self._accepted_activation_row = (
                self._accepted_activation_row + 1
            ) % n

        # An output enqueue captures raw data plus the pre-edge resident R.
        # The prediction is formed before any reduction boundary update on
        # this same edge, matching enqueuePrediction's combinational timing.
        enqueued_record: CycleSampleRecord | None = None
        if result_enqueue:
            assert alignment_to_enqueue is not None
            alignment_entry = alignment_to_enqueue
            token = self._completed_tokens.get(alignment_entry.sample_index)
            context_for_result = next(
                (
                    context
                    for context in self._sample_context_fifo
                    if context.sample_index == alignment_entry.sample_index
                ),
                None,
            )
            if context_for_result is None:
                # The sample context FIFO is ordered and should always contain
                # this entry before its result.  Treat an invariant violation
                # as a model error rather than silently mispairing metadata.
                raise AssertionError("aligned result has no sample context")
            if token is None:
                raise AssertionError("aligned result has no completed sample token")
            else:
                enqueued_record = self._record_for_enqueue(
                    token,
                    alignment_entry.raw_matrix_result,
                    pass_through,
                    R_before,
                )
            self._completed_tokens.pop(alignment_entry.sample_index, None)
            self._output_fifo.append(
                OutputEntry(
                    sample_index=alignment_entry.sample_index,
                    raw_matrix_result=alignment_entry.raw_matrix_result,
                    accepted_cycle=context_for_result.accepted_cycle,
                    result_enqueue_cycle=cycle_number,
                )
            )
            self._result_metadata_fifo.append(
                ResultMetadata(
                    sample_index=alignment_entry.sample_index,
                    prediction=enqueued_record.prediction,
                    reduction_weights_used=enqueued_record.R_used,
                    reduction_weight_signs=tuple(
                        ternary_sign(value) for value in enqueued_record.R_used
                    ),
                    result_enqueue_cycle=cycle_number,
                )
            )

        # Output retirement and its paired context/metadata pop are one event.
        if result_retired:
            if not self._output_fifo or not self._result_metadata_fifo or not self._sample_context_fifo:
                raise AssertionError("retired result FIFO transaction is not fully paired")
            retired_output = self._output_fifo.popleft()
            retired_metadata = self._result_metadata_fifo.popleft()
            retired_context = self._sample_context_fifo.popleft()
            if retired_output.sample_index != retired_context.sample_index or retired_output.sample_index != retired_metadata.sample_index:
                raise AssertionError("result, metadata, and sample context order diverged")
            record = self._record_by_sample.get(retired_context.sample_index)
            if record is None:
                raise AssertionError("retired result has no architectural record")
            record.result_retire_cycle = cycle_number
            if matrix_update_accepted:
                assert retirement_dirs is not None
                row_direction, column_direction, matrix_direction, reduction_direction = retirement_dirs
                self._update_record_at_retirement(
                    record,
                    current_activated if current_activated is not None else (),
                    row_direction,
                    column_direction,
                    matrix_direction,
                    reduction_direction,
                )
                record.matrix_update_cycle = cycle_number
                record.matrix_update_id = self._next_update_id - 1
                record.update_visible_at = cycle_number
            self._transmitted_result_row = (
                self._transmitted_result_row + 1
            ) % n

        # If a context was pushed and the output was simultaneously retired,
        # signedFifo semantics leave the new context at the tail.  The append
        # above already implements that order.  A result pop with no input is
        # the ordinary decrement case.

        # ``matrix_update_valid`` and reduction-boundary live events are now
        # represented in the queues; if a load was requested, the reduction
        # package intentionally did not enter its pipe due to load priority.
        if matrix_update_accepted:
            package = self._generated_matrix_updates
            live = injected_matrix
            assert live is not None
            package.append(
                MatrixUpdatePackage(
                    update_id=live.update_id,
                    source_sample=live.source_sample,
                    row_direction=live.row_direction,
                    column_direction=live.column_direction,
                    position=0,
                    generated_cycle=live.generated_cycle,
                )
            )
            if not cycle_inputs.load_reduction_weights:
                red = injected_reduction
                assert red is not None
                self._generated_reduction_updates.append(
                    ReductionUpdatePackage(
                        update_id=red.update_id,
                        source_sample=red.source_sample,
                        direction=red.direction,
                        position=0,
                        generated_cycle=red.generated_cycle,
                    )
                )

        snapshot = self._make_snapshot(
            cycle_inputs,
            activation_ready=activation_ready,
            activation_accepted=activation_accepted,
            accepted_sample=(accepted_context.sample_index if accepted_context else None),
            weight_ready=weight_ready,
            weight_accepted=weight_accepted,
            weight_popped=weight_popped,
            result_valid=result_valid,
            result_retired=result_retired,
            result_enqueued=result_enqueue,
            retired_sample=retired_sample,
            result_last=result_last,
            current_output_sample=current_output_sample,
            current_output_result=current_output_result,
            current_output_raw=(output_head.raw_matrix_result if output_head else None),
            prediction=prediction,
            learning_direction_value=learning_direction_value,
            datapath_advance=datapath_advance,
            matrix_update_valid=matrix_update_valid,
            matrix_update_accepted=matrix_update_accepted,
            matrix_update_complete=matrix_complete,
            completed_matrix_update_id=completed_matrix_id,
            matrix_updates_applied=tuple(matrix_applied),
            reduction_boundary_apply=reduction_boundary_apply,
            reduction_update_applied=reduction_complete,
            completed_reduction_update_id=completed_reduction_id,
            reduction_update_generated=bool(
                matrix_update_accepted and not cycle_inputs.load_reduction_weights
            ),
            reload_ready=reload_ready,
            reload_accepted=reload_accepted,
            activation_popped=activation_popped,
            datapath_bubble=bool(datapath_advance and not activation_popped),
        )
        self._cycle = cycle_number + 1
        self.snapshots.append(snapshot)
        return snapshot

    tick = step


def format_trace(snapshot: CycleSnapshot) -> str:
    """Return one compact human-readable line for a cycle snapshot."""

    accepted = (
        f"S{snapshot.accepted_sample}"
        if snapshot.accepted_sample is not None
        else "-"
    )
    retired = (
        f"S{snapshot.retired_sample}"
        if snapshot.retired_sample is not None
        else "-"
    )
    updates = ",".join(
        f"U{package.update_id}@diag{package.position}"
        for package in snapshot.matrix_update_packages
    ) or "-"
    input_q = ",".join(f"S{item.sample_index}" for item in snapshot.activation_fifo_contents) or "-"
    output_q = ",".join(f"S{item.sample_index}" for item in snapshot.output_fifo_contents) or "-"
    return (
        f"C{snapshot.cycle}:\n"
        f"  advance={int(snapshot.datapath_advance)} accept={accepted} retire={retired}\n"
        f"  Wgen={snapshot.w_generation} Rgen={snapshot.r_generation}\n"
        f"  updates=[{updates}] inputQ=[{input_q}] outputQ=[{output_q}]"
    )


def print_trace(snapshots: Sequence[CycleSnapshot]) -> None:
    """Print compact trace entries while retaining full snapshot state."""

    for snapshot in snapshots:
        print(format_trace(snapshot))


# Convenient aliases for external tests that use a shorter model name.
CycleModel = CycleReference
ArchitecturalCycleReference = CycleReference


__all__ = [
    "AlignmentState",
    "ArchitecturalCycleReference",
    "CycleConfig",
    "CycleInputs",
    "CycleModel",
    "CycleReference",
    "CycleSampleRecord",
    "CycleSnapshot",
    "MatrixUpdatePackage",
    "OutputEntry",
    "ReductionUpdatePackage",
    "ResultMetadata",
    "SampleContext",
    "SamplePosition",
    "format_trace",
    "print_trace",
]
