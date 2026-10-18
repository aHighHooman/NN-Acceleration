"""Exact-widening Fountain inference and serial full-sensitivity sign learning.

This model has no pipeline state. Each training step is an atomic reference
transaction, matching the hardware admission barrier rather than delayed LMS.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Sequence
from .arithmetic import clog2, to_signed, ternary_sign, saturating_lsb_update


@dataclass(frozen=True)
class ReferenceConfig:
    n: int = 5
    width: int = 8
    layers: int = 3
    fraction_bits: int = 0
    target_width: int | None = None
    reduction_weight_width: int = 8
    pass_through: bool = True
    reduce_output: bool = False

    def __post_init__(self):
        for name in ('n', 'width', 'layers', 'reduction_weight_width'):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.n < 3 or self.n % 2 == 0:
            raise ValueError('Fountain requires odd n>=3')
        if self.target_width is None:
            object.__setattr__(self, 'target_width', self.width)
        if not isinstance(self.target_width, int) or isinstance(self.target_width, bool) or self.target_width < 1:
            raise ValueError('target_width must be a positive integer')
        if not isinstance(self.fraction_bits, int) or isinstance(self.fraction_bits, bool) or self.fraction_bits < 0:
            raise ValueError('fraction_bits must be a nonnegative integer')

    @property
    def matrix_result_width(self):
        return self.width + self.layers * (self.width + clog2(self.n))

    @property
    def prediction_width(self):
        return self.matrix_result_width + clog2(self.n)

    @property
    def rescale_shift(self):
        return self.layers * self.fraction_bits + self.reduction_weight_width - 1


@dataclass(frozen=True)
class Sample:
    x: tuple[int, ...]
    target: int = 0
    training_enable: bool = False

    def __post_init__(self):
        object.__setattr__(self, 'x', tuple(self.x))


@dataclass(frozen=True)
class SampleRecord:
    sample_index: int
    input_vector: tuple[int, ...]
    target: int
    training_enable: bool
    W_used: tuple
    R_used: tuple[int, ...]
    activations: tuple[tuple[int, ...], ...]
    gates: tuple[tuple[bool, ...], ...]
    raw_matrix_result: tuple[int, ...]
    activated_result: tuple[int, ...]
    prediction: int
    learning_direction: int
    sensitivities: tuple[tuple[int, ...], ...]
    matrix_update_directions: tuple
    reduction_update_directions: tuple[int, ...]

    @property
    def update_generated(self):
        return self.training_enable


def normalize_weights(config, weights):
    # A single layer also accepts the original two-dimensional matrix shape.
    if config.layers == 1 and len(weights) == config.n and isinstance(weights[0][0], int):
        weights = [weights]
    if len(weights) != config.layers or any(len(matrix) != config.n or
            any(len(row) != config.n for row in matrix) for matrix in weights):
        raise ValueError('weights must have shape [layers][n][n]')
    return [[[to_signed(value, config.width) for value in row] for row in matrix] for matrix in weights]


def analyze(config: ReferenceConfig, weights, readout, sample: Sample, index=0):
    if not isinstance(sample, Sample):
        raise TypeError('sample must be a Sample')
    if len(sample.x) != config.n or len(readout) != config.n:
        raise ValueError('input/readout length must equal n')
    w = normalize_weights(config, weights)
    r = tuple(to_signed(value, config.reduction_weight_width) for value in readout)
    x = tuple(to_signed(value, config.width) for value in sample.x)
    activations, gates = [x], []
    raw = ()
    for matrix in w:
        raw = tuple(sum(activations[-1][i] * matrix[i][j] for i in range(config.n))
                    for j in range(config.n))
        gates.append(tuple(config.pass_through or value > 0 for value in raw))
        activations.append(tuple(value if config.pass_through or value >= 0 else 0 for value in raw))
    prediction = sum(a * coefficient for a, coefficient in zip(activations[-1], r)) >> config.rescale_shift
    assert to_signed(prediction, config.prediction_width) == prediction
    target = to_signed(sample.target, config.target_width)
    error = ternary_sign(target - prediction)
    sensitivity = [[0] * config.n for _ in range(config.layers)]
    sensitivity[-1] = [r[j] if gates[-1][j] else 0 for j in range(config.n)]
    for layer in range(config.layers - 2, -1, -1):
        sensitivity[layer] = [sum(w[layer+1][i][j] * sensitivity[layer+1][j]
                                 for j in range(config.n)) if gates[layer][i] else 0
                              for i in range(config.n)]
    direction = tuple(tuple(tuple(ternary_sign(activations[layer][i]) * error *
                        ternary_sign(sensitivity[layer][j]) if sample.training_enable else 0
                        for j in range(config.n)) for i in range(config.n))
                        for layer in range(config.layers))
    reduction = tuple(error * ternary_sign(a) if sample.training_enable else 0 for a in activations[-1])
    return SampleRecord(index, x, target, sample.training_enable,
        tuple(tuple(tuple(row) for row in matrix) for matrix in w), r,
        tuple(activations), tuple(gates), raw, activations[-1], prediction, error,
        tuple(tuple(row) for row in sensitivity), direction, reduction)


class FunctionalReference:
    def __init__(self, config: ReferenceConfig, W, R):
        if not isinstance(config, ReferenceConfig):
            raise TypeError('config must be a ReferenceConfig')
        if len(R) != config.n:
            raise ValueError('readout length must equal n')
        self.config = config
        self._initial_W = normalize_weights(config, W)
        self._initial_R = [to_signed(v, config.reduction_weight_width) for v in R]
        self.reset()

    def reset(self):
        self._W = [[row[:] for row in matrix] for matrix in self._initial_W]
        self._R = self._initial_R[:]
        self.records = []

    @property
    def W(self):
        return [[row[:] for row in matrix] for matrix in self._W]

    @property
    def R(self):
        return self._R[:]

    final_W = W
    final_R = R

    def step(self, sample: Sample):
        record = analyze(self.config, self._W, self._R, sample, len(self.records))
        if sample.training_enable:
            self._W = [[[saturating_lsb_update(value, record.matrix_update_directions[l][i][j], self.config.width)
                         for j, value in enumerate(row)] for i, row in enumerate(matrix)]
                       for l, matrix in enumerate(self._W)]
            self._R = [saturating_lsb_update(value, direction, self.config.reduction_weight_width)
                       for value, direction in zip(self._R, record.reduction_update_directions)]
        self.records.append(record)
        return record

    def run(self, samples: Sequence[Sample]):
        return [self.step(sample) for sample in samples]
