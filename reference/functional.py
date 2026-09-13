"""Independent architectural end-to-end reference model.

The model consumes one accepted sample per stream position.  It tracks only
the architectural matrix and reduction state, plus a sample-indexed queue of
complete learning packages.  It deliberately does not model PE registers,
skew, valid/ready, FIFOs, or any other cycle-level implementation state.
"""

from __future__ import annotations

from collections import defaultdict
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
    weighted_vector_reduction,
    to_signed,
)


@dataclass(frozen=True)
class ReferenceConfig:
    """Architectural parameters needed by the no-stall reference model."""

    n: int = 3
    width: int = 16
    fraction_bits: int = 4
    target_width: int | None = None
    reduction_weight_width: int = 8
    pass_through: bool = True
    update_visibility_delay: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.n, int) or isinstance(self.n, bool) or self.n < 1:
            raise ValueError("n must be a positive integer")
        if not isinstance(self.width, int) or isinstance(self.width, bool) or self.width < 1:
            raise ValueError("width must be a positive integer")
        if self.target_width is None:
            object.__setattr__(self, "target_width", self.width)
        if self.update_visibility_delay is None:
            object.__setattr__(self, "update_visibility_delay", 2 * self.n + 1)

        for name in ("width", "target_width", "reduction_weight_width"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(self.fraction_bits, int)
            or isinstance(self.fraction_bits, bool)
            or self.fraction_bits < 0
        ):
            raise ValueError("fraction_bits must be a non-negative integer")
        if (
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

    # RTL-style aliases are useful when copying parameter values from a testbench.
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
    def passThrough(self) -> bool:
        return self.pass_through

    @property
    def updateVisibilityDelay(self) -> int:
        return self.update_visibility_delay  # type: ignore[return-value]


AcceleratorConfig = ReferenceConfig
Config = ReferenceConfig


@dataclass(frozen=True)
class Sample:
    """One accepted architectural transaction."""

    x: tuple[int, ...]
    target: int
    training_enable: bool

    def __init__(
        self,
        x: Sequence[int],
        target: int,
        training_enable: bool = True,
    ) -> None:
        object.__setattr__(self, "x", tuple(int(value) for value in x))
        object.__setattr__(self, "target", int(target))
        object.__setattr__(self, "training_enable", bool(training_enable))

    @property
    def trainingEnable(self) -> bool:
        return self.training_enable


@dataclass(frozen=True)
class SampleRecord:
    """All architectural values associated with one simulated sample.

    ``row_direction`` and ``column_direction`` retain the RTL's combinational
    candidate signals.  The effective update-direction fields are zero when
    ``training_enable`` is false because no package is emitted for inference.
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
    update_visible_at: int | None

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
        """Directions that would be packaged if training were enabled.

        The effective direction fields are zero for an inference sample.  The
        candidate property keeps the RTL's combinational row/column direction
        calculation inspectable without treating it as an emitted update.
        """

        if self.update_generated:
            return self.matrix_update_directions
        row = self.row_direction
        column = self.column_direction
        return tuple(
            tuple(row_value * column_value for column_value in column)
            for row_value in row
        )

    @property
    def candidate_reduction_update_directions(self) -> tuple[int, ...]:
        """Directions that would be packaged if training were enabled."""

        if self.update_generated:
            return self.reduction_update_directions
        return tuple(
            self.learning_direction * (1 if value > 0 else -1 if value < 0 else 0)
            for value in self.activated_result
        )

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-friendly copies of the record fields."""

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
            "update_visible_at": self.update_visible_at,
        }


@dataclass(frozen=True)
class _UpdatePackage:
    source_sample: int
    matrix_direction: tuple[tuple[int, ...], ...]
    reduction_direction: tuple[int, ...]


def _read_config_value(config: Any, names: Sequence[str], default: Any = None) -> Any:
    if isinstance(config, Mapping):
        for name in names:
            if name in config:
                return config[name]
        return default
    for name in names:
        if hasattr(config, name):
            return getattr(config, name)
    return default


def _coerce_config(config: ReferenceConfig | Mapping[str, Any] | Any) -> ReferenceConfig:
    if isinstance(config, ReferenceConfig):
        return config
    if config is None:
        return ReferenceConfig()
    return ReferenceConfig(
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


def _copy_matrix(values: Sequence[Sequence[int]], n: int, width: int) -> list[list[int]]:
    if len(values) != n or any(len(row) != n for row in values):
        raise ValueError("W must be an N by N matrix")
    return [
        [to_signed(value, width) for value in row]
        for row in values
    ]


def _copy_vector(values: Sequence[int], n: int, width: int) -> list[int]:
    if len(values) != n:
        raise ValueError("R must be a vector of length N")
    return [to_signed(value, width) for value in values]


def _zero_matrix(n: int) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(0 for _ in range(n)) for _ in range(n))


def _coerce_sample(sample: Sample | Mapping[str, Any] | Sequence[Any] | Any) -> Sample:
    if isinstance(sample, Sample):
        return sample
    if isinstance(sample, Mapping):
        x = sample.get("x", sample.get("input_vector", sample.get("input")))
        target = sample.get("target")
        training = sample.get(
            "training_enable",
            sample.get("trainingEnable", sample.get("train", True)),
        )
        if x is None or target is None:
            raise ValueError("sample mapping must contain x/input_vector and target")
        return Sample(x, target, training)
    if hasattr(sample, "x") and hasattr(sample, "target"):
        return Sample(
            sample.x,
            sample.target,
            getattr(sample, "training_enable", getattr(sample, "trainingEnable", True)),
        )
    try:
        x, target, training = sample
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "sample must be Sample, mapping, or (x, target, training_enable)"
        ) from exc
    return Sample(x, target, training)


class FunctionalReference:
    """Architectural end-to-end model for continuous no-stall samples."""

    def __init__(
        self,
        config: ReferenceConfig | Mapping[str, Any] | Any = None,
        W: Sequence[Sequence[int]] | None = None,
        R: Sequence[int] | None = None,
    ) -> None:
        if W is None or R is None:
            raise ValueError("initial W and R are required")
        self.config = _coerce_config(config)
        self._initial_W = _copy_matrix(W, self.config.n, self.config.width)
        self._initial_R = _copy_vector(
            R,
            self.config.n,
            self.config.reduction_weight_width,
        )
        self.records: list[SampleRecord] = []
        self.reset()

    def reset(self) -> None:
        """Return the model to the constructor's initial state."""

        self._W = [row[:] for row in self._initial_W]
        self._R = self._initial_R[:]
        self._next_sample_index = 0
        self._pending_updates: dict[int, list[_UpdatePackage]] = defaultdict(list)
        self.records = []

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
    def next_sample_index(self) -> int:
        return self._next_sample_index

    @property
    def pending_update_count(self) -> int:
        return sum(len(packages) for packages in self._pending_updates.values())

    def _apply_updates_visible_at(self, sample_index: int) -> None:
        packages = self._pending_updates.pop(sample_index, ())
        for package in packages:
            self._W = apply_matrix_update(
                self._W,
                package.matrix_direction,
                self.config.width,
            )
            self._R = apply_reduction_update(
                self._R,
                package.reduction_direction,
                self.config.reduction_weight_width,
            )

    def _step(self, sample: Sample) -> SampleRecord:
        sample_index = self._next_sample_index
        self._apply_updates_visible_at(sample_index)

        input_vector = tuple(
            to_signed(value, self.config.width) for value in sample.x
        )
        if len(input_vector) != self.config.n:
            raise ValueError("sample input vector must have length N")
        target = to_signed(sample.target, self.config.target_width)
        W_used = tuple(tuple(row) for row in self._W)
        R_used = tuple(self._R)

        raw_result = tuple(
            matrix_multiply(input_vector, W_used, self.config.width)
        )
        activated_result = tuple(
            activate(raw_result, self.config.pass_through, self.config.matrix_result_width)
        )
        prediction = weighted_vector_reduction(
            activated_result,
            R_used,
            self.config.width,
            self.config.reduction_weight_width,
            self.config.fraction_bits,
        )
        direction = learning_direction(
            target,
            prediction,
            self.config.target_width,
            self.config.prediction_width,
        )

        row_direction, column_direction, candidate_matrix_direction = (
            matrix_update_directions(
                input_vector,
                activated_result,
                R_used,
                direction,
                self.config.width,
                self.config.reduction_weight_width,
                self.config.pass_through,
            )
        )
        candidate_reduction_direction = reduction_update_directions(
            activated_result,
            direction,
        )
        gates = tuple(activation_gates(activated_result, self.config.pass_through))

        update_generated = bool(sample.training_enable)
        if update_generated:
            matrix_direction = tuple(
                tuple(row) for row in candidate_matrix_direction
            )
            reduction_direction = tuple(candidate_reduction_direction)
            update_visible_at = sample_index + self.config.update_visibility_delay
            self._pending_updates[update_visible_at].append(
                _UpdatePackage(
                    source_sample=sample_index,
                    matrix_direction=matrix_direction,
                    reduction_direction=reduction_direction,
                )
            )
        else:
            matrix_direction = _zero_matrix(self.config.n)
            reduction_direction = tuple(0 for _ in range(self.config.n))
            update_visible_at = None

        record = SampleRecord(
            sample_index=sample_index,
            input_vector=input_vector,
            target=target,
            training_enable=bool(sample.training_enable),
            W_used=W_used,
            R_used=R_used,
            raw_matrix_result=raw_result,
            activated_result=activated_result,
            prediction=prediction,
            learning_direction=direction,
            row_direction=tuple(row_direction),
            column_direction=tuple(column_direction),
            matrix_update_directions=matrix_direction,
            reduction_update_directions=reduction_direction,
            activation_gate=gates,
            update_generated=update_generated,
            update_visible_at=update_visible_at,
        )
        self.records.append(record)
        self._next_sample_index += 1
        return record

    def flush_pending_updates(self) -> None:
        """Apply all queued packages in visibility order.

        This represents the end-of-stream boundary when no later sample needs
        to observe the state.  It creates no synthetic sample records.
        """

        while self._pending_updates:
            next_visibility = min(self._pending_updates)
            self._apply_updates_visible_at(next_visibility)

    def step(
        self,
        sample: Sample | Mapping[str, Any] | Sequence[Any] | Any,
    ) -> SampleRecord:
        """Simulate one sample from the current continuous stream position."""

        return self._step(_coerce_sample(sample))

    def run(
        self,
        samples: Sequence[Sample | Mapping[str, Any] | Sequence[Any] | Any],
        *,
        drain_updates: bool = True,
    ) -> list[SampleRecord]:
        """Simulate samples and optionally drain updates after the stream.

        With the default ``drain_updates=True``, ``final_W`` and ``final_R``
        include every update generated by the supplied stream.  Set it false
        to inspect state at the last simulated sample boundary, leaving
        packages whose visibility lies beyond that boundary pending.
        """

        records = [self.step(sample) for sample in samples]
        if drain_updates:
            self.flush_pending_updates()
        return records
