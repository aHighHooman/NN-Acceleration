"""Pure fixed-point arithmetic used by the accelerator reference model.

All values in this module are signed stored integers.  Width-limiting is
explicit at the same boundaries as the RTL; Python floating point is never
used for the golden calculation.
"""

from __future__ import annotations

from collections.abc import Sequence


def clog2(value: int) -> int:
    """Return the SystemVerilog ``$clog2`` value for a positive integer."""

    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("$clog2 requires a positive integer")
    return (value - 1).bit_length()


def _check_width(width: int) -> None:
    if not isinstance(width, int) or isinstance(width, bool) or width < 1:
        raise ValueError("bit width must be a positive integer")


def to_unsigned(value: int, width: int) -> int:
    """Return the low ``width`` bits of ``value`` as a non-negative integer."""

    _check_width(width)
    return int(value) & ((1 << width) - 1)


def to_signed(value: int, width: int) -> int:
    """Interpret the low ``width`` bits of ``value`` as a signed integer."""

    _check_width(width)
    unsigned = to_unsigned(value, width)
    sign_bit = 1 << (width - 1)
    return unsigned - (1 << width) if unsigned & sign_bit else unsigned


def signed_min(width: int) -> int:
    _check_width(width)
    return -(1 << (width - 1))


def signed_max(width: int) -> int:
    _check_width(width)
    return (1 << (width - 1)) - 1


def ternary_sign(value: int) -> int:
    """Return the signed ternary sign ``-1``, ``0``, or ``+1``."""

    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def ternary_product(left: int, right: int) -> int:
    """Multiply two already-ternary directions, preserving zero."""

    if left not in (-1, 0, 1) or right not in (-1, 0, 1):
        raise ValueError("ternary_product inputs must be -1, 0, or +1")
    return left * right


def matrix_result_width(width: int, n: int) -> int:
    """Width of one raw matrix-result element in the RTL."""

    _check_width(width)
    if n < 1:
        raise ValueError("N must be positive")
    return 2 * width + clog2(n)


def prediction_width(width: int, n: int) -> int:
    """Width of the reduced prediction in the RTL."""

    return matrix_result_width(width, n) + clog2(n)


def reduction_fraction_bits(reduction_weight_width: int) -> int:
    """Fractional bits contributed by a signed reduction coefficient."""

    _check_width(reduction_weight_width)
    return reduction_weight_width - 1


def rescale_shift(fraction_bits: int, reduction_weight_width: int) -> int:
    """Final right-shift amount used after the complete weighted sum."""

    if not isinstance(fraction_bits, int) or isinstance(fraction_bits, bool):
        raise ValueError("fraction_bits must be an integer")
    if fraction_bits < 0:
        raise ValueError("fraction_bits must be non-negative")
    return fraction_bits + reduction_fraction_bits(reduction_weight_width)


def matrix_multiply(
    input_vector: Sequence[int],
    weights: Sequence[Sequence[int]],
    width: int,
) -> list[int]:
    """Compute ``input_vector * weights`` with the RTL matrix widths.

    ``weights[row][column]`` is the architectural matrix orientation.  Each
    product is a signed ``2*WIDTH`` value and each column accumulation is
    performed at ``2*WIDTH+$clog2(N)`` bits.
    """

    _check_width(width)
    n = len(weights)
    if n < 1 or len(input_vector) != n:
        raise ValueError("input vector and weight matrix must both have size N")
    if any(len(row) != n for row in weights):
        raise ValueError("weight matrix must be square")

    result_width = matrix_result_width(width, n)
    inputs = [to_signed(value, width) for value in input_vector]
    matrix = [
        [to_signed(value, width) for value in row]
        for row in weights
    ]
    result: list[int] = []

    for column in range(n):
        accumulator = 0
        for row in range(n):
            product = to_signed(inputs[row] * matrix[row][column], 2 * width)
            # The explicit conversion models the signed extension into the
            # PE's result-width accumulator before each add.
            accumulator = to_signed(accumulator + product, result_width)
        result.append(accumulator)
    return result


def activate(raw_result: Sequence[int], pass_through: bool, width: int) -> list[int]:
    """Apply pass-through or the RTL's closed-at-zero ReLU."""

    _check_width(width)
    raw = [to_signed(value, width) for value in raw_result]
    if pass_through:
        return raw
    return [value if value > 0 else 0 for value in raw]


def activation_gates(
    activated_result: Sequence[int],
    pass_through: bool,
) -> list[bool]:
    """Return the per-lane matrix-update activation gates.

    This is intentionally the RTL condition ``passThrough || activated != 0``.
    Thus pass-through mode opens a gate even for a zero raw result, while ReLU
    closes it for non-positive pre-activations.
    """

    return [bool(pass_through) or activated != 0 for activated in activated_result]


def weighted_reduction_accumulator(
    activated_result: Sequence[int],
    reduction_weights: Sequence[int],
    width: int,
    reduction_weight_width: int,
) -> int:
    """Return the complete weighted sum at the RTL accumulator width."""

    _check_width(width)
    _check_width(reduction_weight_width)
    n = len(activated_result)
    if n < 1 or len(reduction_weights) != n:
        raise ValueError("activated result and reduction weights must have size N")

    matrix_width = matrix_result_width(width, n)
    product_width = matrix_width + reduction_weight_width
    accumulator_width = product_width + clog2(n)
    accumulator = 0

    for value, coefficient in zip(activated_result, reduction_weights):
        input_value = to_signed(value, matrix_width)
        weight_value = to_signed(coefficient, reduction_weight_width)
        product = to_signed(input_value * weight_value, product_width)
        # Every term is extended and added at ACCUMULATOR_WIDTH; no term is
        # individually rescaled.
        accumulator = to_signed(accumulator + product, accumulator_width)
    return accumulator


def fixed_point_rescale(
    value: int,
    shift: int,
    input_width: int,
    output_width: int | None = None,
) -> int:
    """Arithmetic-right-shift a signed fixed-width value and optionally narrow it."""

    _check_width(input_width)
    if not isinstance(shift, int) or isinstance(shift, bool) or shift < 0:
        raise ValueError("shift must be a non-negative integer")

    shifted = to_signed(to_signed(value, input_width) >> shift, input_width)
    return shifted if output_width is None else to_signed(shifted, output_width)


def weighted_vector_reduction(
    activated_result: Sequence[int],
    reduction_weights: Sequence[int],
    width: int,
    reduction_weight_width: int,
    fraction_bits: int,
) -> int:
    """Compute the RTL weighted readout and final fixed-point rescaling."""

    n = len(activated_result)
    accumulator = weighted_reduction_accumulator(
        activated_result,
        reduction_weights,
        width,
        reduction_weight_width,
    )
    accumulator_width = (
        matrix_result_width(width, n)
        + reduction_weight_width
        + clog2(n)
    )
    prediction = fixed_point_rescale(
        accumulator,
        rescale_shift(fraction_bits, reduction_weight_width),
        accumulator_width,
        prediction_width(width, n),
    )
    return prediction


def learning_direction(
    target: int,
    prediction: int,
    target_width: int,
    prediction_value_width: int,
) -> int:
    """Compare signed target/prediction values after RTL sign extension."""

    _check_width(target_width)
    _check_width(prediction_value_width)
    compare_width = max(target_width, prediction_value_width)
    compare_target = to_signed(to_signed(target, target_width), compare_width)
    compare_prediction = to_signed(
        to_signed(prediction, prediction_value_width),
        compare_width,
    )
    return ternary_sign(compare_target - compare_prediction)


def matrix_update_directions(
    input_vector: Sequence[int],
    activated_result: Sequence[int],
    reduction_weights: Sequence[int],
    learning_direction_value: int,
    width: int,
    reduction_weight_width: int,
    pass_through: bool,
) -> tuple[list[int], list[int], list[list[int]]]:
    """Generate row, column, and ternary outer-product update directions."""

    _check_width(width)
    _check_width(reduction_weight_width)
    n = len(input_vector)
    if n < 1 or len(activated_result) != n or len(reduction_weights) != n:
        raise ValueError("all update vectors must have size N")
    if learning_direction_value not in (-1, 0, 1):
        raise ValueError("learning direction must be -1, 0, or +1")

    gates = activation_gates(activated_result, pass_through)
    row_direction = [
        ternary_sign(to_signed(value, width)) for value in input_vector
    ]
    column_direction: list[int] = []
    for coefficient, gate in zip(reduction_weights, gates):
        coefficient_sign = ternary_sign(
            to_signed(coefficient, reduction_weight_width)
        )
        column_direction.append(
            ternary_product(learning_direction_value, coefficient_sign)
            if gate
            else 0
        )

    update_direction = [
        [ternary_product(row, column) for column in column_direction]
        for row in row_direction
    ]
    return row_direction, column_direction, update_direction


def reduction_update_directions(
    activated_result: Sequence[int],
    learning_direction_value: int,
) -> list[int]:
    """Generate ``learningDirection * sign(activated output[j])``."""

    if learning_direction_value not in (-1, 0, 1):
        raise ValueError("learning direction must be -1, 0, or +1")
    return [
        ternary_product(learning_direction_value, ternary_sign(value))
        for value in activated_result
    ]


def saturating_lsb_update(value: int, direction: int, width: int) -> int:
    """Apply one signed stored-LSB step with two's-complement saturation."""

    if direction not in (-1, 0, 1):
        raise ValueError("update direction must be -1, 0, or +1")
    current = to_signed(value, width)
    if direction > 0:
        return min(current + 1, signed_max(width))
    if direction < 0:
        return max(current - 1, signed_min(width))
    return current


def apply_matrix_update(
    weights: Sequence[Sequence[int]],
    update_direction: Sequence[Sequence[int]],
    width: int,
) -> list[list[int]]:
    """Return a new matrix after ternary one-LSB saturated updates."""

    _check_width(width)
    n = len(weights)
    if n < 1 or len(update_direction) != n:
        raise ValueError("weight and update matrices must have size N")
    if any(len(row) != n for row in weights) or any(
        len(row) != n for row in update_direction
    ):
        raise ValueError("weight and update matrices must be square")
    return [
        [
            saturating_lsb_update(weights[row][column], update_direction[row][column], width)
            for column in range(n)
        ]
        for row in range(n)
    ]


def apply_reduction_update(
    reduction_weights: Sequence[int],
    update_direction: Sequence[int],
    width: int,
) -> list[int]:
    """Return a new reduction vector after saturated one-LSB updates."""

    if len(reduction_weights) != len(update_direction):
        raise ValueError("reduction weights and directions must have equal length")
    return [
        saturating_lsb_update(value, direction, width)
        for value, direction in zip(reduction_weights, update_direction)
    ]


# Short aliases make the primitives convenient to use in small external tests
# without duplicating the implementation under multiple names.
matrix_product = matrix_multiply
weighted_reduction = weighted_vector_reduction
compare_target_prediction = learning_direction
