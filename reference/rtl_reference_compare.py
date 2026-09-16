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

from .arithmetic import apply_matrix_update, matrix_update_directions
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
PHASE6L_INITIAL_REDUCTION_WEIGHTS = (1, 1, 1)


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


def _input_cycle(x: Sequence[int], target: int, training: bool, *, ready: bool = True,
                 pass_through: bool = True, reduce_output: bool = False) -> DrivenCycle:
    return _driven(CycleInputs(
        input_valid=True,
        input_data=tuple(x),
        target_data=target,
        training_enable=training,
        result_ready=ready,
    ), pass_through=pass_through, reduce_output=reduce_output)


def _idle_cycles(count: int, **options: bool) -> list[DrivenCycle]:
    return [_idle_cycle(**options) for _ in range(count)]


def _sample_cycles(samples: Iterable[Sample], **options: bool) -> list[DrivenCycle]:
    """One input cycle per sample, carrying each sample's own training flag."""

    return [
        _input_cycle(s.x, s.target, s.training_enable, **options)
        for s in samples
    ]


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
        *_sample_cycles(continuous_inference_samples),
        *_idle_cycles(24)], continuous_inference_samples)

    # 2: the first three training updates become visible to samples 7, 8, 9.
    continuous_training_samples = tuple(Sample((1, 2, 3), 127, True) for _ in range(10))
    add_scenario("continuous_training", [
        *_reset_and_load_weights(INITIAL_WEIGHT_MATRIX, INITIAL_REDUCTION_WEIGHTS),
        *_sample_cycles(continuous_training_samples),
        *_idle_cycles(28)], continuous_training_samples)

    # 3: inference transactions remain interleaved but emit no update package.
    mixed_values = ((1, 2, 3), (-2, 3, 1), (3, -1, 2), (0, 2, -3),
                    (-1, -2, -3), (4, 1, 0), (2, 2, 1), (-3, 0, 2),
                    (1, -4, 3), (2, -2, -1))
    mixed_training_samples = tuple(Sample(x, 100 if i % 2 == 0 else -40, i % 2 == 0)
                                   for i, x in enumerate(mixed_values))
    add_scenario("mixed_training_inference", [
        *_reset_and_load_weights(INITIAL_WEIGHT_MATRIX, INITIAL_REDUCTION_WEIGHTS),
        *_sample_cycles(mixed_training_samples),
        *_idle_cycles(28)], mixed_training_samples)

    # 4: bubbles are explicit invalid pre-edge bundles; update waves continue.
    bubble_cycles = _reset_and_load_weights(INITIAL_WEIGHT_MATRIX, INITIAL_REDUCTION_WEIGHTS)
    for index in range(14):
        if index in (1, 2, 5, 9, 10):
            bubble_cycles.append(_idle_cycle())
        else:
            bubble_cycles.append(_input_cycle((1 + index % 2, 2, 3), 127, True))
    bubble_cycles.extend(_idle_cycles(28))
    add_scenario("input_bubbles", bubble_cycles)

    # 5: eight accepted contexts and six buffered outputs force a true array
    # stall while ready remains low; release then drains in original order.
    backpressure_cycles = _reset_and_load_weights(INITIAL_WEIGHT_MATRIX, INITIAL_REDUCTION_WEIGHTS)
    backpressure_cycles.extend(_input_cycle((i + 1, 1, -1), 0, False, ready=False) for i in range(12))
    backpressure_cycles.extend(_idle_cycles(16, ready=False))
    backpressure_cycles.extend(_idle_cycles(32, ready=True))
    add_scenario("output_backpressure", backpressure_cycles)

    # 6L: train one result before the result FIFO fills, then hold the real
    # datapath while its matrix wave and reduction-delay package are live.
    # The fifth result is training-enabled and was enqueued with R=(1,1,1).
    # The first retired package changes resident R to zero before that result
    # retires, so its stored positive signs must still make its matrix update.
    phase6l_samples = (
        Sample((1, 1, 1), -128, True),
        Sample((0, 0, 0), 0, False),
        Sample((0, 0, 0), 0, False),
        Sample((0, 0, 0), 0, False),
        Sample((1, 1, 1), -128, True),
        Sample((0, 0, 0), 0, False),
        Sample((0, 0, 0), 0, False),
        Sample((0, 0, 0), 0, False),
        Sample((0, 0, 0), 0, False),
        Sample((0, 0, 0), 0, False),
    )
    phase6l_cycles = _reset_and_load_weights(
        INITIAL_WEIGHT_MATRIX, PHASE6L_INITIAL_REDUCTION_WEIGHTS
    )
    # The first eight samples are accepted continuously.  The three idle
    # cycles leave four results buffered, then one ready cycle retires the
    # first result and injects the first learning package.
    phase6l_cycles.extend(_sample_cycles(phase6l_samples[:8], ready=False))
    phase6l_cycles.extend(_idle_cycles(3, ready=False))
    phase6l_cycles.extend(_sample_cycles(phase6l_samples[8:9], ready=True))
    # Two held cycles occur after the result FIFO becomes full.  The second
    # input is accepted on the first advancing edge after the stall.
    phase6l_cycles.extend(_idle_cycles(4, ready=False))
    phase6l_cycles.extend(_sample_cycles(phase6l_samples[9:10], ready=True))
    phase6l_cycles.extend(_idle_cycles(32, ready=True))
    add_scenario(
        "phase6L_training_backpressure",
        phase6l_cycles,
        phase6l_samples,
        initial_reduction_weights=PHASE6L_INITIAL_REDUCTION_WEIGHTS,
    )

    # 6: drain one sample, legally reload, stream a second matrix, and run a
    # separately checkable no-stall post-reload stream.
    pre_reload_samples = [Sample((1, 0, 1), 0, False)]
    post_reload_samples = tuple(Sample(x, target, False) for x, target in (
        ((2, -1, 3), 15), ((-1, 4, 2), -20), ((3, 0, -2), 7), ((1, 1, 1), 0)))
    reload_cycles = _reset_and_load_weights(INITIAL_WEIGHT_MATRIX, INITIAL_REDUCTION_WEIGHTS)
    reload_cycles.extend(_sample_cycles(pre_reload_samples))
    reload_cycles.extend(_idle_cycles(20))
    reload_cycles.append(_driven(CycleInputs(reload_weights=True, result_ready=True)))
    for row in reversed(RELOADED_WEIGHT_MATRIX):
        reload_cycles.append(_driven(CycleInputs(
            weight_valid=True, weight_data=tuple(row), result_ready=True)))
    reload_cycles.append(_idle_cycle())
    post_start_offset = len(reload_cycles)
    reload_cycles.extend(_sample_cycles(post_reload_samples))
    reload_cycles.extend(_idle_cycles(24))
    start = len(cycles)
    add_scenario("loading_reload", reload_cycles)
    functional_comparisons.append(FunctionalComparison(
        "loading_reload_post_reload", start + post_start_offset,
        len(cycles) - 1, RELOADED_WEIGHT_MATRIX, INITIAL_REDUCTION_WEIGHTS,
        post_reload_samples, True, False))

    # 7: run and drain pass-through/reduced mode, change both configuration
    # pins while quiescent, then run and drain ReLU/vector mode without reset.
    # Deliberately drain just one sample before changing modes to exercise the
    # stream-configuration boundary independently of matrix geometry.
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
        _sample_cycles(pass_samples, pass_through=True, reduce_output=True))
    transition_cycles.extend(
        _idle_cycles(24, pass_through=True, reduce_output=True))
    pass_end = transition_start + len(transition_cycles) - 1
    relu_start = pass_end + 1
    transition_cycles.extend(
        _sample_cycles(relu_samples, pass_through=False, reduce_output=False))
    transition_cycles.extend(
        _idle_cycles(24, pass_through=False, reduce_output=False))
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
            relu_samples, False, False,
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
            input_data = inputs.input_data or (0,) * N
            reduction = inputs.reduction_weight or (0,) * N
            values = (
                cycle, int(inputs.reset_n), int(inputs.weight_valid), *weight,
                int(inputs.input_valid), *input_data, inputs.target_data,
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
            weight_data=tuple(v[3:6]), input_valid=bool(v[6]),
            input_data=tuple(v[7:10]), target_data=v[10], training_enable=bool(v[11]),
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


def read_trace(
    path: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    snapshots: list[dict[str, object]] = []
    enqueues: list[dict[str, object]] = []
    retirements: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for line_number, line in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        f = line.split()
        if not f:
            continue
        if f[0] == "C":
            current = {"cycle": int(f[1])}
            snapshots.append(current)
        elif f[0] == "ENQ":
            if len(f) != N + 2:
                raise ValueError(f"bad ENQ record at trace line {line_number}")
            enqueues.append({
                "cycle": int(f[1]),
                "raw": tuple(map(int, f[2:])),
            })
        elif f[0] == "RT":
            if len(f) != 2*N + 5:
                raise ValueError(f"bad RT record at trace line {line_number}")
            v = [int(field) for field in f[1:]]
            retirements.append({
                "cycle": v[0],
                "activated": tuple(v[1:1 + N]),
                "prediction": v[1 + N],
                "direction": v[2 + N],
                "target": v[3 + N],
                "result": tuple(v[4 + N:4 + 2 * N]),
            })
        elif current is None:
            raise ValueError(f"trace data before C at line {line_number}")
        elif f[0] == "W":
            values = tuple(map(int, f[1:])); current["W"] = tuple(values[i:i+N] for i in range(0, N*N, N))
        elif f[0] == "R":
            current["R"] = tuple(map(int, f[1:]))
        elif f[0] == "PW":
            entries = _parse_counted(f, N)
            if len(entries) > 1:
                raise ValueError(f"bad PW trace payload at line {line_number}")
            current["pending_weight_row"] = entries[0] if entries else None
        elif f[0] == "IF":
            current["input_fifo"] = _parse_counted(f, N)
        elif f[0] == "SF":
            entries = _parse_counted(f, N + 2)
            current["sample_context_fifo"] = tuple((e[0], tuple(e[1:1+N]), bool(e[-1])) for e in entries)
        elif f[0] == "RF":
            entries = _parse_counted(f, 2*N + 1)
            current["result_fifo"] = tuple(
                (tuple(entry[:N]), entry[N], tuple(entry[N+1:]))
                for entry in entries
            )
        elif f[0] == "P":
            if len(f) != 11:
                raise ValueError(f"bad P record at trace line {line_number}")
            if int(f[1]) != int(current["cycle"]):
                raise ValueError(f"P record cycle does not match C at trace line {line_number}")
            current["datapath_progress"] = tuple(int(value) for value in f[2:])
        else:
            raise ValueError(f"unknown trace record {f[0]} at line {line_number}")
    return snapshots, enqueues, retirements


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
    expected_enqueues: list[dict[str, object]] = []
    for driven_cycle in inputs:
        cycle_model.reconfigure(
            pass_through=driven_cycle.pass_through,
            reduce_output=driven_cycle.reduce_output,
        )
        snapshot = cycle_model.step(driven_cycle.inputs)
        expected.append(snapshot)
        if cycle_model._last_enqueued_raw_result is not None:
            expected_enqueues.append({
                "cycle": snapshot.cycle,
                "raw": cycle_model._last_enqueued_raw_result,
            })
    actual, enqueues, retirements = read_trace(trace_path)
    if len(actual) != len(expected):
        raise AssertionError(f"trace has {len(actual)} snapshots; expected {len(expected)}")

    for exp, act in zip(expected, actual):
        cycle = exp.cycle
        if act.get("cycle") != cycle:
            _fail(cycle, "cycle number", cycle, act.get("cycle"))
        for name, left, right in (
            ("W", exp.W, act.get("W", ())),
            ("R", tuple(exp.R), act.get("R", ())),
            ("pending_weight_row", exp.pending_weight_row, act.get("pending_weight_row")),
            ("input_fifo", exp.input_fifo, act.get("input_fifo")),
            ("sampleContextFifo",
             tuple((e.target, e.input_signs, e.training_enable)
                   for e in exp.sample_context_fifo),
             act.get("sample_context_fifo", ())),
            ("resultFifo",
             tuple((e.activated_result, e.prediction, e.reduction_weight_signs)
                   for e in exp.result_fifo),
             act.get("result_fifo", ())),
        ):
            if isinstance(left, tuple) and isinstance(right, tuple):
                _compare_sequence(cycle, name, left, right)
            elif left != right:
                _fail(cycle, name, left, right)
        progress = act.get("datapath_progress")
        if not isinstance(progress, tuple) or len(progress) != 9:
            _fail(cycle, "datapath progress trace", "nine fields", progress)

    if len(enqueues) != len(expected_enqueues):
        raise AssertionError(
            f"trace has {len(enqueues)} matrix-result enqueues; expected {len(expected_enqueues)}"
        )
    for expected_enqueue, actual_enqueue in zip(expected_enqueues, enqueues):
        cycle = int(expected_enqueue["cycle"])
        if actual_enqueue.get("cycle") != cycle:
            _fail(cycle, "matrix-result enqueue cycle", cycle, actual_enqueue.get("cycle"))
        if actual_enqueue.get("raw") != expected_enqueue["raw"]:
            _fail(cycle, "raw matrix result at matrix-result handshake",
                  expected_enqueue["raw"], actual_enqueue.get("raw"))

    functional_comparisons = 0
    for comparison in comparison_inputs.functional_comparisons:
        model = FunctionalReference(
            ReferenceConfig(
                n=N, width=WIDTH, fraction_bits=FRACTION_BITS, target_width=TARGET_WIDTH,
                reduction_weight_width=REDUCTION_WEIGHT_WIDTH,
                pass_through=comparison.pass_through,
                reduce_output=comparison.reduce_output,
            ),
            comparison.W, comparison.R,
        )
        records = model.run(comparison.samples)
        window = [
            [r for r in rows
             if comparison.start_cycle <= int(r["cycle"]) <= comparison.end_cycle]
            for rows in (enqueues, retirements)
        ]
        enqueued, retired = window
        if len(enqueued) != len(records) or len(retired) != len(records):
            raise AssertionError(
                f"{comparison.name}: enqueued {len(enqueued)}, retired {len(retired)}; "
                f"expected {len(records)} each"
            )
        for index, (record, enq, rtl) in enumerate(zip(records, enqueued, retired)):
            cycle = int(rtl["cycle"])
            expected_output = ((record.prediction, 0, 0)
                               if comparison.reduce_output else record.activated_result)
            for at, label, left, right in (
                (int(enq["cycle"]), "raw matrix result", record.raw_matrix_result, enq["raw"]),
                (cycle, "activated result", record.activated_result, rtl["activated"]),
                (cycle, "prediction", record.prediction, rtl["prediction"]),
                (cycle, "learning direction", record.learning_direction, rtl["direction"]),
                (cycle, "target", record.target, rtl["target"]),
                (cycle, "external result", expected_output, rtl["result"]),
            ):
                functional_comparisons += 1
                if left != right:
                    _fail(at, f"{comparison.name} sample {index} {label}", left, right)
        final = actual[comparison.end_cycle]
        for label, left, right in (
            ("final W", tuple(tuple(row) for row in model.final_W), final["W"]),
            ("final R", tuple(model.final_R), final["R"]),
        ):
            functional_comparisons += 1
            if left != right:
                _fail(comparison.end_cycle, f"{comparison.name} {label}", left, right)

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
    if expected[e0 + 6].result_fifo or len(expected[e0 + 7].result_fifo) != 1:
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
    held = any(len(states[i].result_fifo) == config.output_fifo_depth and
               states[i].W == states[i-1].W and states[i].R == states[i-1].R and
               states[i].pending_weight_row == states[i-1].pending_weight_row and
               states[i].input_fifo == states[i-1].input_fifo and
               states[i].sample_context_fifo == states[i-1].sample_context_fifo and
               states[i].result_fifo == states[i-1].result_fifo
               for i in range(1, len(states)))
    if not held:
        raise AssertionError("output_backpressure did not produce a held architectural snapshot")
    simultaneous_full_pop_push = any(
        backpressure.start_cycle < cycle <= backpressure.end_cycle
        and len(expected[cycle - 1].result_fifo) == config.output_fifo_depth
        and inputs[cycle].inputs.result_ready
        and expected[cycle - 1].result_fifo
        and expected[cycle - 1].sample_context_fifo
        and len(expected[cycle].result_fifo) == config.output_fifo_depth
        and expected[cycle].result_fifo != expected[cycle - 1].result_fifo
        for cycle in range(backpressure.start_cycle + 1, backpressure.end_cycle + 1)
    )
    if not simultaneous_full_pop_push:
        raise AssertionError(
            "output_backpressure did not exercise simultaneous full pop/push"
        )

    phase6l = next(
        scenario for scenario in comparison_inputs.scenarios
        if scenario.name == "phase6L_training_backpressure"
    )
    phase6l_states = {
        snapshot.cycle: snapshot
        for snapshot in expected[phase6l.start_cycle:phase6l.end_cycle + 1]
    }
    phase6l_progress = {
        int(snapshot["cycle"]): snapshot["datapath_progress"]
        for snapshot in actual[phase6l.start_cycle:phase6l.end_cycle + 1]
    }
    stalled_cycles = [
        cycle for cycle, progress in phase6l_progress.items()
        if progress[0] == 0 and progress[1] == 1
    ]
    if len(stalled_cycles) < 2:
        raise AssertionError(
            "phase6L scenario did not produce multiple outputBlocked datapath-stall cycles"
        )
    if stalled_cycles != list(range(stalled_cycles[0], stalled_cycles[-1] + 1)):
        raise AssertionError("phase6L datapath stall was not held on consecutive cycles")
    first_stall = stalled_cycles[0]
    first_progress = phase6l_progress[first_stall]
    # P fields are datapathAdvance, outputBlocked, reductionUpdateBusy,
    # matrix-pipeline-busy, and result-alignment-busy.  Both update paths must
    # still be live at the first held edge.
    if first_progress[2] != 1 or first_progress[3] != 1:
        raise AssertionError(
            "phase6L stall did not begin with live reduction and matrix update work"
        )
    for cycle in stalled_cycles:
        before = phase6l_states[cycle - 1]
        during = phase6l_states[cycle]
        architectural_fields = (
            "W", "R", "pending_weight_row", "input_fifo", "sample_context_fifo",
            "result_fifo",
        )
        if any(getattr(before, field) != getattr(during, field) for field in architectural_fields):
            raise AssertionError(
                f"phase6L architectural state moved during the held stall at cycle {cycle}"
            )
        if len(during.result_fifo) != config.output_fifo_depth:
            raise AssertionError(
                f"phase6L stall at cycle {cycle} was not caused by a full output buffer"
            )
    for cycle in stalled_cycles[1:]:
        previous_progress = phase6l_progress[cycle - 1]
        current_progress = phase6l_progress[cycle]
        if previous_progress[5:] != current_progress[5:]:
            raise AssertionError(
                f"phase6L internal update/skew/alignment state advanced during cycle {cycle}"
            )

    phase6l_functional = next(
        item for item in comparison_inputs.functional_comparisons
        if item.name == "phase6L_training_backpressure"
    )
    phase6l_model = FunctionalReference(
        ReferenceConfig(
            n=N, width=WIDTH, fraction_bits=FRACTION_BITS,
            target_width=TARGET_WIDTH, reduction_weight_width=REDUCTION_WEIGHT_WIDTH,
        ),
        phase6l_functional.W,
        phase6l_functional.R,
    )
    phase6l_records = phase6l_model.run(phase6l_functional.samples)
    phase6l_retirements = [
        int(record["cycle"])
        for record in retirements
        if phase6l.start_cycle <= int(record["cycle"]) <= phase6l.end_cycle
    ]
    if len(phase6l_retirements) != len(phase6l_records):
        raise AssertionError("phase6L retirement count did not match FunctionalReference samples")
    buffered_result_index = 4
    buffered_retirement = phase6l_retirements[buffered_result_index]
    buffered_before_retire = phase6l_states[buffered_retirement - 1]
    stored_signs = buffered_before_retire.result_fifo[0].reduction_weight_signs
    resident_signs = tuple(1 if value > 0 else -1 if value < 0 else 0
                           for value in buffered_before_retire.R)
    if stored_signs != (1, 1, 1) or resident_signs != (0, 0, 0):
        raise AssertionError(
            "phase6L buffered result did not retain old R signs across the resident sign change"
        )
    buffered_record = phase6l_records[buffered_result_index]
    if buffered_record.R_used != PHASE6L_INITIAL_REDUCTION_WEIGHTS:
        raise AssertionError("FunctionalReference did not retain the buffered result's old R")
    _, _, current_sign_matrix_direction = matrix_update_directions(
        buffered_record.input_vector,
        buffered_record.activated_result,
        buffered_before_retire.R,
        buffered_record.learning_direction,
        WIDTH,
        REDUCTION_WEIGHT_WIDTH,
        True,
    )
    if tuple(tuple(row) for row in current_sign_matrix_direction) == buffered_record.matrix_update_directions:
        raise AssertionError("phase6L current-R counterfactual did not differ from stored-sign update")
    counterfactual_W = [row[:] for row in phase6l_functional.W]
    for index, record in enumerate(phase6l_records):
        if record.update_generated:
            direction = (
                current_sign_matrix_direction
                if index == buffered_result_index
                else record.matrix_update_directions
            )
            counterfactual_W = apply_matrix_update(counterfactual_W, direction, WIDTH)
    final_phase6l = phase6l_states[phase6l.end_cycle]
    if tuple(tuple(row) for row in counterfactual_W) == final_phase6l.W:
        raise AssertionError("phase6L final W did not expose the stored-sign choice")

    stall_before = phase6l_states[first_stall - 1]
    stall_during = phase6l_states[first_stall]
    resume = phase6l_states[stalled_cycles[-1] + 1]
    print(
        "PASS: phase6L evidence "
        f"stall_cycles={stalled_cycles[0]}..{stalled_cycles[-1]} "
        f"held_cycles={len(stalled_cycles)} "
        f"live_work=(reduction,matrix)=({first_progress[2]},{first_progress[3]}) "
        f"R_sign_change=cycle {buffered_retirement - 1} lanes=(0,1,2) +1->0 "
        f"stored_signs={stored_signs} resident_before_retire={resident_signs} "
        f"buffered_retire_cycle={buffered_retirement}"
    )
    print(
        "PASS: phase6L W/R states "
        f"before={stall_before.W}/{stall_before.R} "
        f"during={stall_during.W}/{stall_during.R} "
        f"after_resume={resume.W}/{resume.R} "
        f"final={final_phase6l.W}/{final_phase6l.R}"
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
