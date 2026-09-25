# Weight-Stationary Neural-Network Accelerator

A parameterized SystemVerilog accelerator for inference and on-chip supervised
learning. An `N × N` weight-stationary systolic array computes `x·W`, then ReLU
and a weighted sum produce a scalar prediction. Sign-sign least mean squares
(SSLMS) updates the matrix and reduction weights by at most one LSB per training
sample.

`nnAccelerator` exposes a parallel ready/valid interface in a single clock
domain. It sustains one sample per clock when the consumer keeps up.

| At the defaults | |
| --- | --- |
| Array | 3 × 3 PEs, weight-stationary |
| Numbers | signed Q12.4 inputs and weights, Q1.7 reduction weights |
| Throughput | 1 sample per clock, sustained |
| Latency | accept to result in 2N+1 = 7 edges, retire and first weight update on edge 2N+2 = 8 |
| Learning | SSLMS, −1 / 0 / +1 LSB per weight per training sample, saturating |
| Target | Cyclone V `5CSEMA5F31C6`; 100 MHz implementation target, not yet timing-closed |
| Verification | Python golden models, per-edge RTL comparison, UVM, directed benches |

Latency assumes no stalls and an always-ready consumer.

## Run the regression

From the repository root, with Python, PowerShell 7, and licensed Questa installed:

```powershell
pwsh -File scripts/run_modelsim.ps1
```

The scripts default to `C:\altera_lite\25.1std\questa_fse\win64`;
set `NN_ACCEL_QUESTA_BIN` to use another Questa installation. The suite runs
Python tests, parameter rejection, directed RTL benches, the golden comparison,
and UVM. See [Verification](#verification) for individual commands and coverage.

**Contents:**
[Architecture](#architecture) ·
[Processing element](#processing-element) ·
[Loading weights](#loading-weights) ·
[Timing](#timing-and-update-visibility) ·
[Learning rule](#learning-rule) ·
[Fixed point](#fixed-point-arithmetic) ·
[Two FIFOs](#transaction-storage) ·
[Backpressure](#backpressure) ·
[Lifecycle](#operating-lifecycle) ·
[Core interface](#core-interface) ·
[Parameters](#parameters) ·
[Verification](#verification) ·
[FPGA build](#fpga-build) ·
[Layout](#repository-layout)

## Architecture

![Block diagram of the nnAccelerator core and parallel interface](presentation/figures/01-architecture.svg)

In the diagrams, **blue** is forward data, **orange** is learning feedback,
and **grey-green** is storage and control. Figures follow the system theme.

The forward path:

1. An accepted input vector waits in a one-entry **input register**. It can
   leave and be replaced on the same edge, so the register never limits
   throughput.
2. The **skew** delays lane `i` by `i` clocks, so matching products meet on
   the same diagonal wavefront.
3. The **PE array** multiplies. Inputs move right, partial sums move down,
   and weights stay put.
4. The **alignment** stage delays column `j` by `N−1−j` so the whole result
   vector comes out on one edge.
5. **Activation** (ReLU or pass-through) is applied exactly once. The
   **weighted reduction** `Σ rⱼ·aⱼ` then forms the prediction `ŷ`.
6. The activated vector, `ŷ`, and the signs of `r` are pushed together into
   `resultFifo`.

Each sample's target, input signs, and training bit take a separate route into
`sampleContextFifo` when the sample is accepted. A result **retires** when the
consumer takes it, which pops the head of both FIFOs together. Nothing counts
cycles to pair a prediction with its target. Order alone does it, so bubbles
and stalls can't misalign them.

Retirement is also the only source of learning. A training sample's retirement
computes `e = sign(t − ŷ)` and sends one update package into PE(0,0), which
ripples across the array one anti-diagonal per clock. A matching reduction
update goes through a `2N−1` stage pipe and commits to the resident weights `r`.

## Processing element

![Processing element datapath](presentation/figures/02-processing-element.svg)

Each PE holds one weight. It multiplies the input passing through by that weight
and adds the result to the partial sum coming from above. It registers both
outputs: the input goes right, the partial sum goes down. Products are
`2·WIDTH` bits wide and nothing is truncated inside the array. An update steps
`w` by one LSB in the direction it's given and saturates at the signed limits.

## Loading weights

![Weight rows shifting into the array through the pending-row stage](presentation/figures/03-weight-loading.svg)

Weight rows arrive one at a time through a single pending-row register and
shift down the array. Send them in **reverse order**, row `N−1` first, so that
PE `(i, j)` ends up holding `W[i][j]`. `weightsLoaded` rises once the Nth row is
consumed. Inputs are accepted only after that point.

## Timing and update visibility

![Space-time chart of samples and the update wave for N = 3](presentation/figures/06-update-timing.svg)

Each line is one sample moving through the pipeline, one edge per step. The
chart shows N = 3 with one sample accepted on every edge. Only S0 trains, and
the consumer is always ready.

| Edge | What happens to sample S0 | N = 3 |
| --- | --- | ---: |
| E0 | accepted: vector into the input register, context into `sampleContextFifo` | 0 |
| E1 | leaves the input register and enters the skew | 1 |
| E2 … E2N | multiplies along anti-diagonals 0 … 2N−2 | 2 … 6 |
| E2N+1 | aligned vector activated and reduced to `ŷ`, pushed into `resultFifo` | 7 |
| E2N+2 | retires: `e` is formed and PE(0,0) takes its step on this edge | 8 |
| E2N+3 … E4N | the update wave crosses anti-diagonals 1 … 2N−2 | 9 … 12 |
| E4N+1 | the reduction update commits to `r` | 13 |

A PE multiply always uses the weight held *before* the edge. S6 (S2N) meets
PE(0,0) on the same edge as the update, so it still uses the old `w₀₀`. **S7
(S2N+1) is the first sample to see the updated matrix**, and it also sees the
updated `r`. The update wave moves in lockstep with S6 and reaches each
diagonal on the edge where S6 has just used it. S6 sees only old weights, S7
sees only new ones, and no sample ever sees half an update.

In general an update from sample S first affects sample S + 2N + 1, provided
there are no stalls. Updates start at retirement, so a consumer that holds off
`resultReady` also delays learning.

## Learning rule

![Worked example of the ternary outer-product update](presentation/figures/05-learning-rule.svg)

SSLMS uses only signs, avoiding multipliers in the weight-update logic.
Let `e = sign(t − ŷ)` be the ternary learning direction and
`gⱼ = passThrough ∨ (aⱼ ≠ 0)` the activation gate. ReLU's gate is closed at zero.

| Weight | Update direction | One step at defaults |
| --- | --- | --- |
| `rⱼ` (reduction) | `e · sign(aⱼ)` | 1 LSB = 1/128 |
| `Wᵢⱼ` (PE) | `sign(xᵢ) · e · sign(rⱼ) · gⱼ` | 1 LSB = 1/16 |

The matrix update is the outer product of two ternary vectors:

- `rowDirection = sign(x)`, taken from the original input.
- `columnDirection = e · sign(r) · g`.

Each PE receives two ternary directions. Every step saturates at the signed limits.

- The update uses the `sign(r)` **snapshotted when that result was formed**, so
  it matches the `r` that produced `ŷ`, even if `r` has changed since.
- `e` compares the target with the full-width prediction, with both
  sign-extended to a common width. It is `+1`, `0`, or `−1`.
- Inference samples (`trainingEnable = 0`) retire the same way but launch no
  update.

## Fixed-point arithmetic

![Bit-width ladder aligned on the binary point at default parameters](presentation/figures/04-fixed-point-widths.svg)

Full precision is retained until the final rescale:

- Products double the width.
- Column sums add `⌈log₂N⌉` guard bits, giving
  `MATRIX_RESULT_WIDTH = 2·WIDTH + ⌈log₂N⌉` with `2·FRACTION_BITS` fraction
  bits.
- The reduction keeps each full product and accumulates at
  `MATRIX_RESULT_WIDTH + REDUCTION_WEIGHT_WIDTH + ⌈log₂N⌉` bits.

The only precision loss is one arithmetic right shift by
`FRACTION_BITS + REDUCTION_WEIGHT_WIDTH − 1` once the sum is complete. It puts
`ŷ` back on the target's binary point, `FRACTION_BITS` fraction bits. The
narrowing to `PREDICTION_WIDTH = 2·WIDTH + 2·⌈log₂N⌉` then drops only
sign-extension bits.

A stored integer `v` represents `v / 2^FRACTION_BITS` for inputs, matrix
weights, targets, and predictions. The reduction weights default to Q1.7, a
sign bit plus seven fraction bits.

## Transaction storage

![Bit layouts of the two FIFO entries and their steady-state occupancy](presentation/figures/07-transaction-storage.svg)

Everything retirement needs is packed into two words:

- `sampleContextFifo` holds `{target, sign x[N], trainingEnable}` from
  acceptance until retirement.
- `resultFifo` holds `{activated vector, ŷ, sign r[N]}` from the moment the
  result forms until retirement.

The result FIFO does not retain the raw pre-activation vector.

At full speed with default depths, the context FIFO is full and the result
FIFO holds one entry. `IN_FLIGHT_DEPTH = 2N+2` is the smallest depth that
sustains one sample per clock. The `2N` result slots provide headroom for a
stalled consumer; their depth does not change unstalled latency.

## Backpressure

![Logic-analyzer trace of a consumer stall from the cycle reference model](presentation/figures/08-backpressure-trace.svg)

This is a real trace from `CycleReference` with N = 3 and the consumer stalled
for edges 9–18:

1. The context FIFO is already full, so admission closes at once.
2. The array keeps draining into `resultFifo` until it holds six results.
3. `arrayAdvance` falls, and every datapath and update register holds
   together, including update waves partway across the array.

When `resultReady` returns, retire, push, accept, and advance all restart on the
same edge.

```text
inputReady   = weightsLoaded ∧ (input register empty ∨ it is leaving) ∧ (context not full ∨ retire)
arrayAdvance = ¬(aligned result valid ∧ resultFifo full ∧ ¬retire)
```

## Operating lifecycle

![State diagram: reset, loading, quiescent, streaming](presentation/figures/09-lifecycle.svg)

`passThrough` and `reduceOutput` are stream settings, not per-sample fields.
They aren't stored in either FIFO, so the RTL always uses their live values.
The supported sequence is **configure, stream, drain, reconfigure**. Change a
mode only when the stream is quiescent, meaning no accepted sample, buffered
result, or learning update is left.

- `loadReductionWeights` copies all N coefficients at once and takes priority
  over learning. Pulse it only while quiescent.
- Assert `reloadWeights` only while `reloadReady` is high. That requires the
  matrix engine to be loaded and idle, both FIFOs to be empty, and the reduction
  update pipe to be empty. Pending update waves count as busy, so a reload can't
  overtake one.
- `rst_n` is synchronous. It clears every FIFO and pipeline, drops
  `weightsLoaded`, and sets `r` to zero. Reload the weights and `r` after a reset.

## Core interface

The `nnAccelerator` ports:

| Port | Dir | Meaning |
| --- | :---: | --- |
| `clk`, `rst_n` | in | Rising-edge clock and synchronous active-low reset |
| `weightData[N]`, `weightValid` / `weightReady` | in / out | One matrix row per handshake, row `N−1` first |
| `inputData[N]`, `targetData`, `trainingEnable`, `inputValid` / `inputReady` | in / out | One sample, all accepted together |
| `resultData[N]`, `resultValid` / `resultReady` | out / in | One retired result per handshake |
| `reductionWeight[N]`, `loadReductionWeights` | in | Load all of `r` at once |
| `passThrough` | in | 1 = no activation, 0 = ReLU |
| `reduceOutput` | in | 0 = activated vector, 1 = `ŷ` in lane 0 and zero in lanes `1…N−1` |
| `weightsLoaded`, `reloadWeights`, `reloadReady` | out / in / out | Loading state and reload handshake |

Interface details:

- Every result lane is `2·WIDTH + 2·⌈log₂N⌉` bits. In vector mode, lane `j`
  carries activated element `j`, sign-extended, with `2·FRACTION_BITS`
  fraction bits. In reduce mode, lane 0 carries `ŷ` with `FRACTION_BITS`
  fraction bits.
- Results are an ordered stream of independent samples, with no grouping or
  frame marker. Sending N input vectors in order still gives the N rows of
  `X·W` if you want a matrix product.

## Parameters

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `WIDTH` | `16` | Signed input and weight width, ≥ 1 |
| `N` | `3` | Array and vector dimension, ≥ 2 (tested for 2–4) |
| `FRACTION_BITS` | `4` | Fraction bits of inputs, weights, targets, and predictions, ≥ 0 |
| `TARGET_WIDTH` | `WIDTH` | Signed target width, ≥ 1, sign-extended for the comparison |
| `REDUCTION_WEIGHT_WIDTH` | `8` | Width of each `rⱼ`, ≥ 1, one sign bit plus `REDUCTION_WEIGHT_WIDTH − 1` fraction bits |
| `IN_FLIGHT_DEPTH` | `2*N+2` | Accepted but unretired samples, which is the `sampleContextFifo` depth, ≥ 1 |
| `OUTPUT_FIFO_DEPTH` | `2*N` | `resultFifo` depth, ≥ 1 |

The input register holds at most one vector. The generic `signedFifo`
backs both transaction FIFOs and supports any `DEPTH ≥ 1`.

## Verification

![Python references, golden RTL comparison, UVM, and directed benches](presentation/figures/11-verification.svg)

| Layer | Checks |
| --- | --- |
| `reference/arithmetic.py` | Numerical semantics: widths, rescale, saturation, ternary signs. Pinned by `test_arithmetic.py`. |
| `FunctionalReference` | End-to-end numbers and learning, one sample at a time. Update visibility in closed form, `2N+1`. |
| `CycleReference` | Latency, W/R evolution, FIFO contents, bubbles, backpressure, and reconfiguration. Derives update visibility by wave propagation for comparison with `FunctionalReference`. |
| Golden RTL comparison | Nine deterministic scenarios through `nnAcceleratorStateTrace_tb`; W, R, the pending row, and both FIFOs compared with `CycleReference` on every edge. |
| UVM (`uvm/`) | Seeded public-port traffic at N = 3, WIDTH = 8: ordering, loss/duplication, backpressure, reset/reload recovery, and exact pass-through inference while matrix weights are known. Training and inference with learned weights are checked for ordering and liveness. |
| Directed benches | Reduction arithmetic, the matrix core at N = 2/3/4, and update-wave propagation, saturation, and stalls. |

Run individual checks:

```powershell
pwsh -File scripts/run_rtl_reference_compare.ps1
pwsh -File scripts/run_uvm.ps1 -TestName nn_uvm_regression_test
pwsh -File scripts/run_uvm.ps1 -TestName nn_uvm_smoke_test
pwsh -File scripts/run_uvm.ps1 -TestName nn_uvm_regression_test -Seed 12345
```

The UVM log prints its seed, so `-Seed` replays a run. Stimulus uses `$urandom`
and explicit scenario checks, allowing it to run on the Questa FPGA Starter license.

## FPGA build

The Quartus project targets the DE1-SoC's Cyclone V `5CSEMA5F31C6`, with
`nnAccelerator` as its top level. All ports except `clk` use virtual pins to
characterize the core. A board integration needs its own interface and pin assignments.

The September 24, 2026 fit with Quartus Prime 25.1std Lite, at the defaults
(N = 3, WIDTH = 16), reports:

| Metric | Post-fit |
| --- | ---: |
| Logic | 1,565 / 32,070 ALMs (5%) |
| Registers | 1,168 |
| Block memory | 1,048 / 4,065,280 bits (<1%), 5 / 397 RAM blocks |
| DSP blocks | 18 / 87 (21%) |
| Pins | 1 physical clock pin, 258 virtual pins |
| Worst setup / hold slack | −3.665 ns / +0.143 ns |
| Lowest reported Fmax | 73.18 MHz |

`NN_Acceleration.sdc` sets a **100 MHz implementation target** (10 ns), distinct
from the board's 50 MHz oscillator. The fit does **not** meet the 100 MHz target.
The clock pin is unassigned and the fitter reports non-dedicated clock routing;
these figures characterize this fit, not a completed board implementation.
Reports are generated under `Quartus Stuff/output_files/` and are not tracked.

## Repository layout

```text
.
├── weightStationaryVariant/
│   ├── nnAccelerator.sv                          core: FIFOs, activation, reduction, retirement, learning
│   ├── weightStationaryMatrixMultiplier.sv       input register, skew, alignment, loading, arrayAdvance
│   ├── weightStationarySystolicArray.sv          PE grid and update-wave pipe
│   ├── weightStationaryProcessingElement.sv
│   ├── weightedVectorReduction.sv
│   └── *_tb.sv                                   directed benches and the state-trace bench
├── memory/signedFifo.sv
├── reference/          arithmetic, functional and cycle models, RTL comparison, unit tests
├── uvm/                UVM environment for nnAccelerator
├── scripts/            regression entry points and Questa discovery
├── presentation/figures/   the diagrams in this README
└── Quartus Stuff/      Quartus project, settings, and timing constraints
```
