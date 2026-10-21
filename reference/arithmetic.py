"""Signed stored-integer helpers shared by the references, plus the original
single-layer arithmetic kept as the LAYERS=1 compatibility oracle.
Golden calculations never use floating point."""

from __future__ import annotations

from collections.abc import Sequence


def clog2(value: int) -> int:
    """Return the SystemVerilog ``$clog2`` value for a positive integer."""

    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("$clog2 requires a positive integer")
    return (value - 1).bit_length()


def to_signed(value: int, width: int) -> int:
    """Interpret the low ``width`` bits of ``value`` as a signed integer."""

    if not isinstance(width, int) or isinstance(width, bool) or width < 1:
        raise ValueError("bit width must be a positive integer")
    unsigned = int(value) & ((1 << width) - 1)
    return unsigned - (1 << width) if unsigned >> (width - 1) else unsigned


def ternary_sign(value: int) -> int:
    """Return the signed ternary sign ``-1``, ``0``, or ``+1``."""

    return (value > 0) - (value < 0)


def saturating_lsb_update(value: int, direction: int, width: int) -> int:
    """Apply one signed stored-LSB step with two's-complement saturation."""

    if direction not in (-1, 0, 1):
        raise ValueError("update direction must be -1, 0, or +1")
    limit = 1 << (width - 1)
    return max(-limit, min(limit - 1, to_signed(value, width) + direction))


def matrix_multiply(
    input_vector: Sequence[int],
    weights: Sequence[Sequence[int]],
    width: int,
) -> list[int]:
    """Single-layer ``input_vector * weights`` with ``weights[row][column]``
    orientation, signed 2*WIDTH products and 2*WIDTH+$clog2(N)-bit sums."""

    n = len(weights)
    if n < 1 or len(input_vector) != n or any(len(row) != n for row in weights):
        raise ValueError("input vector and weight matrix must both have size N")
    result_width = 2 * width + clog2(n)
    inputs = [to_signed(value, width) for value in input_vector]
    return [
        to_signed(sum(inputs[row] * to_signed(weights[row][column], width) for row in range(n)), result_width)
        for column in range(n)
    ]


def weighted_vector_reduction(
    activated_result: Sequence[int],
    reduction_weights: Sequence[int],
    width: int,
    reduction_weight_width: int,
    fraction_bits: int,
) -> int:
    """Single-layer scalar readout: sum every full-width product, then shift the
    complete sum once by FRACTION_BITS + REDUCTION_WEIGHT_WIDTH - 1."""

    n = len(activated_result)
    if n < 1 or len(reduction_weights) != n:
        raise ValueError("activated result and reduction weights must have size N")
    if fraction_bits < 0:
        raise ValueError("fraction_bits must be non-negative")
    matrix_width = 2 * width + clog2(n)
    accumulator = sum(
        to_signed(value, matrix_width) * to_signed(coefficient, reduction_weight_width)
        for value, coefficient in zip(activated_result, reduction_weights)
    )
    accumulator = to_signed(accumulator, matrix_width + reduction_weight_width + clog2(n))
    return to_signed(accumulator >> (fraction_bits + reduction_weight_width - 1), matrix_width + clog2(n))
