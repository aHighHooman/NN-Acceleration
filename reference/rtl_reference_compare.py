"""Generate RTL inputs and compare traced accelerator state with Python references.

The stimulus file contains one pre-edge input bundle per line.  The RTL trace
contains semantic post-edge state (never raw pointers or RAM layout).  This
module deliberately delegates all numerical and scheduling decisions to
FunctionalReference and CycleReference.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .cycle import CycleConfig, CycleInputs, CycleReference, CycleSnapshot
from .functional import FunctionalReference, ReferenceConfig, Sample


N = 3
WIDTH = 8
TARGET_WIDTH = 8
REDUCTION_WEIGHT_WIDTH = 8
FRACTION_BITS = 0
INITIAL_WEIGHT_MATRIX = ((1, 2, 3), (4, 5, 6), (7, 8, 9))
RELOADED_WEIGHT_MATRIX = ((-3, 2, 1), (5, -4, 2), (1, 3, -2))
INITIAL_REDUCTION_WEIGHTS = (16, 24, 32)


@dataclass(frozen=True)
class FunctionalComparison:
    name: str
    start_cycle: int
    end_cycle: int
    W: tuple[tuple[int, ...], ...]
    R: tuple[int, ...]
    samples: tuple[Sample, ...]
    pass_through: bool
    reduce_output: bool
    result_row_offset: int = 0


@dataclass(frozen=True)
class Scenario:
    name: str
    start_cycle: int
    end_cycle: int


@dataclass(frozen=True)
class DrivenCycle:
    """One RTL edge plus the quiescent-lifetime configuration on its pins."""

    inputs: CycleInputs
    pass_through: bool = True
    reduce_output: bool = False


@dataclass(frozen=True)
class ComparisonInputs:
    cycles: tuple[DrivenCycle, ...]
    scenarios: tuple[Scenario, ...]
    functional_comparisons: tuple[FunctionalComparison, ...]


def _driven(inputs: CycleInputs, *, pass_through: bool = True,
            reduce_output: bool = False) -> DrivenCycle:
    return DrivenCycle(inputs, pass_through, reduce_output)


def _idle_cycle(*, ready: bool = True, pass_through: bool = True,
                reduce_output: bool = False) -> DrivenCycle:
    return _driven(CycleInputs(result_ready=ready), pass_through=pass_through,
                   reduce_output=reduce_output)


def _reset_and_load_weights(weight_matrix: Sequence[Sequence[int]],
                            reduction_weights: Sequence[int],
                            *, pass_through: bool = True,
                            reduce_output: bool = False) -> list[DrivenCycle]:
    """Reset, load R, and stream W in the host order required by the RTL."""

    result = [
        _driven(CycleInputs(reset_n=False), pass_through=pass_through,
                reduce_output=reduce_output),
        _driven(CycleInputs(reset_n=False), pass_through=pass_through,
                reduce_output=reduce_output),
    ]
    # The first host vector ultimately occupies the bottom PE row, hence the
    # reverse row order.  This is interface stimulus, not a second model.
    for index, row in enumerate(reversed(weight_matrix)):
        result.append(
            _driven(CycleInputs(
                weight_valid=True,
                weight_data=tuple(row),
                reduction_weight=tuple(reduction_weights),
                load_reduction_weights=index == 0,
                result_ready=True,
            ), pass_through=pass_through, reduce_output=reduce_output)
        )
    result.append(_idle_cycle(pass_through=pass_through, reduce_output=reduce_output))
    return result


def _activation_cycle(x: Sequence[int], target: int, training: bool, *, ready: bool = True,
                      pass_through: bool = True, reduce_output: bool = False) -> DrivenCycle:
    return _driven(CycleInputs(
        activation_valid=True,
        activation_data=tuple(x),
        target_data=target,
        training_enable=training,
        result_ready=ready,
    ), pass_through=pass_through, reduce_output=reduce_output)


def define_cycle_inputs_and_comparisons() -> ComparisonInputs:
    cycles: list[DrivenCycle] = []
    scenarios: list[Scenario] = []
    functional_comparisons: list[FunctionalComparison] = []

    def add_scenario(name: str, scenario_cycles: Iterable[DrivenCycle],
                     samples_to_compare: Sequence[Sample] = (),
                     *, initial_weight_matrix: Sequence[Sequence[int]] = INITIAL_WEIGHT_MATRIX,
                     initial_reduction_weights: Sequence[int] = INITIAL_REDUCTION_WEIGHTS,
                     pass_through: bool = True, reduce_output: bool = False) -> None:
        start = len(cycles)
        cycles.extend(scenario_cycles)
        end = len(cycles) - 1
        scenarios.append(Scenario(name, start, end))
        if samples_to_compare:
            functional_comparisons.append(
                FunctionalComparison(
                    name, start, end,
                    tuple(tuple(v for v in row) for row in initial_weight_matrix),
                    tuple(initial_reduction_weights),
                    tuple(samples_to_compare), pass_through, reduce_output,
                )
            )

    # 1: continuous inference, including enough samples to demonstrate steady
    # one-result-per-clock retirement after the E0 -> E7 -> E8 fill latency.
    continuous_inference_samples = tuple(
        Sample((i + 1, (i % 3) - 1, 2 - (i % 4)), 20 - i, False) for i in range(12)
    )
    add_scenario("continuous_inference", [
        *_reset_and_load_weights(INITIAL_WEIGHT_MATRIX, INITIAL_REDUCTION_WEIGHTS),
        *[_activation_cycle(sample.x, sample.target, False)
          for sample in continuous_inference_samples],
        *[_idle_cycle() for _ in range(24)]], continuous_inference_samples)

    # 2: the first three training updates become visible to samples 7, 8, 9.
    continuous_training_samples = tuple(Sample((1, 2, 3), 127, True) for _ in range(10))
    add_scenario("continuous_training", [
        *_reset_and_load_weights(INITIAL_WEIGHT_MATRIX, INITIAL_REDUCTION_WEIGHTS),
        *[_activation_cycle(sample.x, sample.target, True)
          for sample in continuous_training_samples],
        *[_idle_cycle() for _ in range(28)]], continuous_training_samples)

    # 3: inference transactions remain interleaved but emit no update package.
    mixed_values = ((1, 2, 3), (-2, 3, 1), (3, -1, 2), (0, 2, -3),
                    (-1, -2, -3), (4, 1, 0), (2, 2, 1), (-3, 0, 2),
                    (1, -4, 3), (2, -2, -1))
    mixed_training_samples = tuple(Sample(x, 100 if i % 2 == 0 else -40, i % 2 == 0)
                                   for i, x in enumerate(mixed_values))
    add_scenario("mixed_training_inference", [
        *_reset_and_load_weights(INITIAL_WEIGHT_MATRIX, INITIAL_REDUCTION_WEIGHTS),
        *[_activation_cycle(sample.x, sample.target, sample.training_enable)
          for sample in mixed_training_samples],
        *[_idle_cycle() for _ in range(28)]], mixed_training_samples)

    # 4: bubbles are explicit invalid pre-edge bundles; update waves continue.
    bubble_cycles = _reset_and_load_weights(INITIAL_WEIGHT_MATRIX, INITIAL_REDUCTION_WEIGHTS)
    for index in range(14):
        if index in (1, 2, 5, 9, 10):
            bubble_cycles.append(_idle_cycle())
        else:
            bubble_cycles.append(_activation_cycle((1 + index % 2, 2, 3), 127, True))
    bubble_cycles.extend(_idle_cycle() for _ in range(28))
    add_scenario("input_bubbles", bubble_cycles)

    # 5: eight accepted contexts and six buffered outputs force a true array
    # stall while ready remains low; release then drains in original order.
    backpressure_cycles = _reset_and_load_weights(INITIAL_WEIGHT_MATRIX, INITIAL_REDUCTION_WEIGHTS)
    backpressure_cycles.extend(_activation_cycle((i + 1, 1, -1), 0, False, ready=False) for i in range(12))
    backpressure_cycles.extend(_idle_cycle(ready=False) for _ in range(16))
    backpressure_cycles.extend(_idle_cycle(ready=True) for _ in range(32))
    add_scenario("output_backpressure", backpressure_cycles)

    # 6: drain normal traffic, legally reload, stream a second matrix, and run
    # a separately checkable no-stall post-reload stream.
    pre_reload_samples = [Sample((1, 0, 1), 0, False), Sample((2, 1, -1), 0, False),
                          Sample((-1, 2, 0), 0, False)]
    post_reload_samples = tuple(Sample(x, target, False) for x, target in (
        ((2, -1, 3), 15), ((-1, 4, 2), -20), ((3, 0, -2), 7), ((1, 1, 1), 0)))
    reload_cycles = _reset_and_load_weights(INITIAL_WEIGHT_MATRIX, INITIAL_REDUCTION_WEIGHTS)
    reload_cycles.extend(
        _activation_cycle(sample.x, sample.target, False)
        for sample in pre_reload_samples
    )
    reload_cycles.extend(_idle_cycle() for _ in range(20))
    reload_cycles.append(_driven(CycleInputs(reload_weights=True, result_ready=True)))
    for row in reversed(RELOADED_WEIGHT_MATRIX):
        reload_cycles.append(_driven(CycleInputs(
            weight_valid=True, weight_data=tuple(row), result_ready=True)))
    reload_cycles.append(_idle_cycle())
    post_start_offset = len(reload_cycles)
    reload_cycles.extend(
        _activation_cycle(sample.x, sample.target, False)
        for sample in post_reload_samples
    )
    reload_cycles.extend(_idle_cycle() for _ in range(24))
    start = len(cycles)
    add_scenario("loading_reload", reload_cycles)
    functional_comparisons.append(FunctionalComparison(
        "loading_reload_post_reload", start + post_start_offset,
        len(cycles) - 1, RELOADED_WEIGHT_MATRIX, INITIAL_REDUCTION_WEIGHTS,
        post_reload_samples, True, False))

    # 7: run and drain pass-through/reduced mode, change both configuration
    # pins while quiescent, then run and drain ReLU/vector mode without reset.
    # Deliberately drain just one N=3 sample before changing modes. The stream
    # is quiescent even though the output frame position is 1; frame position
    # must not keep stream configuration live, but it must still block reload.
    pass_samples = (Sample((-2, 1, 3), 0, False),)
    relu_samples = tuple(Sample(x, t, False) for x, t in (
        ((-3, 1, 0), 2), ((2, -4, 1), -3), ((1, 1, -2), 4), ((-2, -1, 3), 1),
        ((4, 0, -1), 8), ((-1, 2, 2), -5)))
    transition_start = len(cycles)
    transition_cycles = _reset_and_load_weights(
        RELOADED_WEIGHT_MATRIX, INITIAL_REDUCTION_WEIGHTS,
        pass_through=True, reduce_output=True)
    pass_start = transition_start
    transition_cycles.extend(
        _activation_cycle(sample.x, sample.target, False,
                          pass_through=True, reduce_output=True)
        for sample in pass_samples
    )
    transition_cycles.extend(
        _idle_cycle(pass_through=True, reduce_output=True) for _ in range(24)
    )
    pass_end = transition_start + len(transition_cycles) - 1
    relu_start = pass_end + 1
    transition_cycles.extend(
        _activation_cycle(sample.x, sample.target, False,
                          pass_through=False, reduce_output=False)
        for sample in relu_samples
    )
    transition_cycles.extend(
        _idle_cycle(pass_through=False, reduce_output=False) for _ in range(24)
    )
    cycles.extend(transition_cycles)
    scenarios.append(Scenario("quiescent_configuration_transition", transition_start, len(cycles) - 1))
    functional_comparisons.extend((
        FunctionalComparison(
            "quiescent_pass_through_reduced", pass_start, pass_end,
            RELOADED_WEIGHT_MATRIX, INITIAL_REDUCTION_WEIGHTS,
            pass_samples, True, True,
        ),
        FunctionalComparison(
            "quiescent_relu_vector", relu_start, len(cycles) - 1,
            RELOADED_WEIGHT_MATRIX, INITIAL_REDUCTION_WEIGHTS,
            relu_samples, False, False, result_row_offset=len(pass_samples) % N,
        ),
    ))

    return ComparisonInputs(tuple(cycles), tuple(scenarios), tuple(functional_comparisons))


def write_stimulus(path: Path, comparison_inputs: ComparisonInputs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(f"{len(comparison_inputs.cycles)}\n")
        for cycle, item in enumerate(comparison_inputs.cycles):
            inputs = item.inputs
            weight = inputs.weight_data or (0,) * N
            activation = inputs.activation_data or (0,) * N
            reduction = inputs.reduction_weight or (0,) * N
            values = (
                cycle, int(inputs.reset_n), int(inputs.weight_valid), *weight,
                int(inputs.activation_valid), *activation, inputs.target_data,
                int(inputs.training_enable), int(inputs.result_ready), *reduction,
                int(inputs.load_reduction_weights), int(inputs.reload_weights),
                int(item.pass_through),
                int(item.reduce_output),
            )
            stream.write(" ".join(str(value) for value in values) + "\n")


def read_stimulus(path: Path) -> tuple[DrivenCycle, ...]:
    lines = path.read_text(encoding="ascii").splitlines()
    count = int(lines[0])
    if len(lines) != count + 1:
        raise ValueError(f"stimulus declares {count} cycles but contains {len(lines) - 1}")
    result = []
    for expected_cycle, line in enumerate(lines[1:]):
        v = [int(field) for field in line.split()]
        if len(v) != 20 or v[0] != expected_cycle:
            raise ValueError(f"malformed stimulus line for cycle {expected_cycle}")
        result.append(DrivenCycle(CycleInputs(reset_n=bool(v[1]), weight_valid=bool(v[2]),
            weight_data=tuple(v[3:6]), activation_valid=bool(v[6]),
            activation_data=tuple(v[7:10]), target_data=v[10], training_enable=bool(v[11]),
            result_ready=bool(v[12]), reduction_weight=tuple(v[13:16]),
            load_reduction_weights=bool(v[16]), reload_weights=bool(v[17])),
            pass_through=bool(v[18]), reduce_output=bool(v[19])))
    return tuple(result)


def _parse_counted(fields: list[str], width: int) -> tuple[tuple[int, ...], ...]:
    count = int(fields[1])
    values = tuple(int(value) for value in fields[2:])
    if len(values) != count * width:
        raise ValueError(f"bad {fields[0]} trace payload")
    return tuple(values[i * width:(i + 1) * width] for i in range(count))


def read_trace(path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    snapshots: list[dict[str, object]] = []
    retirements: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for line_number, line in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        f = line.split()
        if not f:
            continue
        if f[0] == "C":
            current = {"cycle": int(f[1])}
            snapshots.append(current)
        elif f[0] == "RT":
            if len(f) != 15:
                raise ValueError(f"bad RT record at trace line {line_number}")
            retirements.append({"cycle": int(f[1]), "raw": tuple(map(int, f[2:5])),
                "activated": tuple(map(int, f[5:8])), "prediction": int(f[8]),
                "direction": int(f[9]), "target": int(f[10]),
                "result": tuple(map(int, f[11:14])), "last": int(f[14])})
        elif current is None:
            raise ValueError(f"trace data before C at line {line_number}")
        elif f[0] == "W":
            values = tuple(map(int, f[1:])); current["W"] = tuple(values[i:i+N] for i in range(0, N*N, N))
        elif f[0] == "R":
            current["R"] = tuple(map(int, f[1:]))
        elif f[0] in ("WF", "AF", "OF"):
            current[{"WF": "weight_fifo", "AF": "activation_fifo", "OF": "output_fifo"}[f[0]]] = _parse_counted(f, N)
        elif f[0] == "SF":
            entries = _parse_counted(f, N + 2)
            current["sample_context_fifo"] = tuple((e[0], tuple(e[1:1+N]), bool(e[-1])) for e in entries)
        elif f[0] == "RF":
            entries = _parse_counted(f, N + 1)
            current["result_readout_fifo"] = tuple((e[0], tuple(e[1:])) for e in entries)
        else:
            raise ValueError(f"unknown trace record {f[0]} at line {line_number}")
    return snapshots, retirements


def _fail(cycle: int, field: str, expected: object, actual: object) -> None:
    raise AssertionError(f"cycle {cycle}:\n    {field}\n    expected {expected}\n    actual   {actual}")


def _compare_sequence(cycle: int, name: str, expected: Sequence[object], actual: Sequence[object]) -> None:
    if len(expected) != len(actual):
        _fail(cycle, f"{name} length", len(expected), len(actual))
    for index, (left, right) in enumerate(zip(expected, actual)):
        if left != right:
            _fail(cycle, f"{name} entry {index}", left, right)


def compare(stimulus_path: Path, trace_path: Path) -> tuple[int, int]:
    comparison_inputs = define_cycle_inputs_and_comparisons()
    inputs = read_stimulus(stimulus_path)
    if len(inputs) != len(comparison_inputs.cycles):
        raise AssertionError("stimulus file length does not match the deterministic scenario definitions")
    config = CycleConfig(n=N, width=WIDTH, fraction_bits=FRACTION_BITS,
                         target_width=TARGET_WIDTH, reduction_weight_width=REDUCTION_WEIGHT_WIDTH)
    cycle_model = CycleReference(config)
    expected = []
    for driven_cycle in inputs:
        cycle_model.reconfigure(
            pass_through=driven_cycle.pass_through,
            reduce_output=driven_cycle.reduce_output,
        )
        expected.append(cycle_model.step(driven_cycle.inputs))
    actual, retirements = read_trace(trace_path)
    if len(actual) != len(expected):
        raise AssertionError(f"trace has {len(actual)} snapshots; expected {len(expected)}")

    for exp, act in zip(expected, actual):
        cycle = exp.cycle
        if act.get("cycle") != cycle:
            _fail(cycle, "cycle number", cycle, act.get("cycle"))
        actual_W = act.get("W", ())
        if not isinstance(actual_W, tuple) or len(actual_W) != N:
            _fail(cycle, "W row count", N, len(actual_W) if isinstance(actual_W, tuple) else actual_W)
        for row in range(N):
            for column in range(N):
                if exp.W[row][column] != actual_W[row][column]:
                    _fail(cycle, f"W[{row}][{column}]", exp.W[row][column], actual_W[row][column])
        actual_R = act.get("R", ())
        if not isinstance(actual_R, tuple) or len(actual_R) != N:
            _fail(cycle, "R length", N, len(actual_R) if isinstance(actual_R, tuple) else actual_R)
        for lane in range(N):
            if exp.R[lane] != actual_R[lane]:
                _fail(cycle, f"R[{lane}]", exp.R[lane], actual_R[lane])
        for field in ("weight_fifo", "activation_fifo", "output_fifo"):
            left = getattr(exp, field); right = act.get(field)
            if isinstance(left, tuple) and isinstance(right, tuple):
                _compare_sequence(cycle, field, left, right)
            elif left != right:
                _fail(cycle, field, left, right)
        exp_context = tuple((e.target, e.input_signs, e.training_enable) for e in exp.sample_context_fifo)
        _compare_sequence(cycle, "sampleContextFifo", exp_context, act.get("sample_context_fifo", ()))
        exp_readout = tuple((e.prediction, e.reduction_weight_signs) for e in exp.result_readout_fifo)
        actual_readout = act.get("result_readout_fifo", ())
        _compare_sequence(cycle, "resultReadoutFifo", exp_readout, actual_readout)
        actual_output = act.get("output_fifo", ())
        if len(exp.output_fifo) != len(exp.result_readout_fifo):
            _fail(cycle, "reference output/readout FIFO occupancy pairing",
                  len(exp.output_fifo), len(exp.result_readout_fifo))
        if len(actual_output) != len(actual_readout):
            _fail(cycle, "output/readout FIFO occupancy pairing", len(actual_output), len(actual_readout))

    functional_comparisons = 0
    for comparison in comparison_inputs.functional_comparisons:
        model = FunctionalReference(ReferenceConfig(n=N, width=WIDTH, fraction_bits=FRACTION_BITS,
            target_width=TARGET_WIDTH, reduction_weight_width=REDUCTION_WEIGHT_WIDTH,
            pass_through=comparison.pass_through, reduce_output=comparison.reduce_output),
            comparison.W, comparison.R)
        records = model.run(comparison.samples)
        retired = [r for r in retirements
                   if comparison.start_cycle <= int(r["cycle"]) <= comparison.end_cycle]
        if len(retired) != len(records):
            raise AssertionError(
                f"{comparison.name}: retired {len(retired)} results; expected {len(records)}"
            )
        for index, (record, rtl) in enumerate(zip(records, retired)):
            cycle = int(rtl["cycle"])
            checks = (("raw matrix result", record.raw_matrix_result, rtl["raw"]),
                      ("activated result", record.activated_result, rtl["activated"]),
                      ("prediction", record.prediction, rtl["prediction"]),
                      ("learning direction", record.learning_direction, rtl["direction"]),
                      ("target", record.target, rtl["target"]))
            for label, left, right in checks:
                functional_comparisons += 1
                if left != right:
                    _fail(cycle, f"{comparison.name} sample {index} {label}", left, right)
            expected_output = ((record.prediction, 0, 0)
                               if model.config.reduce_output else record.activated_result)
            functional_comparisons += 1
            if expected_output != rtl["result"]:
                _fail(cycle, f"{comparison.name} sample {index} external result", expected_output, rtl["result"])
            expected_last = int(
                (comparison.result_row_offset + index) % N == N - 1
            )
            if expected_last != rtl["last"]:
                _fail(cycle, f"{comparison.name} sample {index} resultLast", expected_last, rtl["last"])
        final = actual[comparison.end_cycle]
        functional_comparisons += 2
        if tuple(tuple(row) for row in model.final_W) != final["W"]:
            _fail(comparison.end_cycle, f"{comparison.name} final W", model.final_W, final["W"])
        if tuple(model.final_R) != final["R"]:
            _fail(comparison.end_cycle, f"{comparison.name} final R", model.final_R, final["R"])

    # Scenario-specific evidence required beyond field-by-field comparison.
    training_comparison = next(
        item for item in comparison_inputs.functional_comparisons
        if item.name == "continuous_training"
    )
    training_model = FunctionalReference(ReferenceConfig(n=N, width=WIDTH, fraction_bits=FRACTION_BITS,
        target_width=TARGET_WIDTH, reduction_weight_width=REDUCTION_WEIGHT_WIDTH),
        INITIAL_WEIGHT_MATRIX, INITIAL_REDUCTION_WEIGHTS)
    training_records = training_model.run(training_comparison.samples)
    if any(record.W_used != INITIAL_WEIGHT_MATRIX or
           record.R_used != INITIAL_REDUCTION_WEIGHTS
           for record in training_records[:7]):
        raise AssertionError("a training update became visible before S7")
    for sample_index in (7, 8, 9):
        if training_records[sample_index].W_used == training_records[sample_index - 1].W_used or \
           training_records[sample_index].R_used == training_records[sample_index - 1].R_used:
            raise AssertionError(f"U{sample_index - 7} visibility evidence missing at S{sample_index}")

    inference = next(
        scenario for scenario in comparison_inputs.scenarios
        if scenario.name == "continuous_inference"
    )
    e0 = inference.start_cycle + len(
        _reset_and_load_weights(INITIAL_WEIGHT_MATRIX, INITIAL_REDUCTION_WEIGHTS)
    )
    if expected[e0 + 6].output_fifo or len(expected[e0 + 7].output_fifo) != 1:
        raise AssertionError("continuous inference did not demonstrate E0 -> E7 enqueue timing")
    inference_retirements = [int(r["cycle"]) for r in retirements
                             if inference.start_cycle <= int(r["cycle"]) <= inference.end_cycle]
    if inference_retirements != list(range(e0 + 8, e0 + 8 + 12)):
        raise AssertionError(f"continuous inference retirement cycles were {inference_retirements}")

    backpressure = next(
        scenario for scenario in comparison_inputs.scenarios
        if scenario.name == "output_backpressure"
    )
    states = [expected[i] for i in range(backpressure.start_cycle, backpressure.end_cycle + 1)]
    held = any(len(states[i].output_fifo) == config.output_fifo_depth and
               states[i].W == states[i-1].W and states[i].R == states[i-1].R and
               states[i].weight_fifo == states[i-1].weight_fifo and
               states[i].activation_fifo == states[i-1].activation_fifo and
               states[i].sample_context_fifo == states[i-1].sample_context_fifo and
               states[i].output_fifo == states[i-1].output_fifo and
               states[i].result_readout_fifo == states[i-1].result_readout_fifo
               for i in range(1, len(states)))
    if not held:
        raise AssertionError("output_backpressure did not produce a held architectural snapshot")
    simultaneous_full_pop_push = any(
        backpressure.start_cycle < cycle <= backpressure.end_cycle
        and len(expected[cycle - 1].output_fifo) == config.output_fifo_depth
        and inputs[cycle].inputs.result_ready
        and expected[cycle - 1].output_fifo
        and expected[cycle - 1].result_readout_fifo
        and expected[cycle - 1].sample_context_fifo
        and len(expected[cycle].output_fifo) == config.output_fifo_depth
        and expected[cycle].output_fifo != expected[cycle - 1].output_fifo
        for cycle in range(backpressure.start_cycle + 1, backpressure.end_cycle + 1)
    )
    if not simultaneous_full_pop_push:
        raise AssertionError(
            "output_backpressure did not exercise simultaneous full pop/push"
        )
    return len(expected), functional_comparisons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("stimulus", type=Path)
    check = sub.add_parser("compare")
    check.add_argument("stimulus", type=Path)
    check.add_argument("trace", type=Path)
    args = parser.parse_args()
    if args.command == "generate":
        comparison_inputs = define_cycle_inputs_and_comparisons()
        write_stimulus(args.stimulus, comparison_inputs)
        print(
            f"generated {len(comparison_inputs.cycles)} cycles across "
            f"{len(comparison_inputs.scenarios)} scenarios"
        )
    else:
        snapshots, functional_checks = compare(args.stimulus, args.trace)
        print(f"PASS: {snapshots} RTL post-edge snapshots; {functional_checks} FunctionalReference comparisons")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
