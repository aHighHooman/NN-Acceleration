# Weight-Stationary Neural-Network Accelerator

A parameterized SystemVerilog accelerator for inference and on-chip supervised
learning. An `N × N` corner-fed, anti-diagonal-output weight-stationary systolic
array computes `x·W`, then ReLU
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
| Latency | accept to result in N+2 = 5 edges, retire and first weight update on edge N+3 = 6 |
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
2. The **skew** keeps a registered entry stage and supplies two delayed taps
   per row: lane `i` reaches the left boundary after `i` extra advances and the
   right boundary after `N−1−i` extra advances. Row zero has no right-side PEs.
3. The **PE array** computes from opposite corners. In the top-left region,
   activations move right and partial sums move down; in the bottom-right
   region, activations move left and partial sums move up.
4. Each **anti-diagonal PE**, where `i+j=N−1`, combines its local product and
   the two incoming partial sums. These PEs finish all output lanes together.
   Their existing MAC result registers are the output boundary; there is no
   separate output alignment or additional output register.
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
computes `e = sign(t − ŷ)` and sends one package into both opposite corners.
Updates converge inward with the same phase as computation:
`f(i,j) = min(i+j, 2N−2−i−j)`. A matching reduction update goes through an
`N`-stage pipe and commits to the resident weights `r`.

## Processing element

![Processing element datapath](presentation/figures/02-processing-element.svg)

Each PE holds one weight and an activation/partial-sum register pair. Its
compile-time role selects its neighbors: top-left PEs forward right/down,
bottom-right PEs forward left/up, and anti-diagonal PEs emit a complete column
sum. The anti-diagonal sum uses carry-save compression of the product and both
partial sums followed by a carry-propagate addition, within the same MAC stage.
Products are `2·WIDTH` bits wide; array sums retain all guard bits. An update
steps `w` by one LSB in the direction given and saturates at the signed limits.
A multiply reads the weight held before the edge, even if that edge updates it.

## Loading weights

![Host rows buffered, shifted from both ends, and captured together](presentation/figures/03-weight-loading.svg)

Send N rows in **reverse order**, row `N−1` first. Each accepted handshake
writes its logical row directly into an N×N staging bank. Host gaps are allowed;
the array does not begin loading until every row has arrived.

After the final handshake, `weightReady` falls and N consecutive load advances
reuse the MAC partial-sum paths as two-ended shift chains. At zero-based step
`q`, top lane `j` receives `W[N−1−q][j]` when `q≥j`, and bottom lane `j`
receives `W[q][j]` when `q≥N−j`; unused slots receive zero. Activations are
forced to zero and each anti-diagonal PE gates its lower partial sum during
loading, so its top chain loads independently of the bottom chain.

One further **capture edge** copies each PE's partial-sum register into its
resident weight. `weightsLoaded` rises only after that edge; inputs can then be
accepted. Resident weights remain unchanged throughout collection and shifting.
With uninterrupted row handshakes, loading takes N acceptance edges followed
by N shift edges and one capture edge. The staging bank costs N² additional
weight words; it preserves the existing host format and permits arbitrary gaps.

## Timing and update visibility

![Space-time chart of samples and the update wave for N = 3](presentation/figures/06-update-timing.svg)

The chart shows N = 3 with one sample accepted on every edge, only S0 training,
and an always-ready consumer. Every edge below assumes the datapath advances.

| Edge | What happens to sample S0 | N = 3 |
| --- | --- | ---: |
| E0 | accepted: vector into the input register, context into `sampleContextFifo` | 0 |
| E1 | leaves the input register and enters the registered skew entry | 1 |
| E2 … E(N+1) | multiplies along inward phases 0 … N−1 | 2 … 4 |
| E(N+2) | registered AD vector activated and reduced to `ŷ`, pushed into `resultFifo` | 5 |
| E(N+3) | retires: `e` is formed and both opposite corner PEs update | 6 |
| E(N+4) … E(2N+2) | the update crosses inward phases 1 … N−1 | 7 … 8 |
| E(2N+3) | the reduction update commits to `r` | 9 |

A PE multiply always uses the weight held *before* the edge. S4 (S(N+1))
meets the corner PEs on the same edge as S0's update, so it still uses the old
weights. **S5 (S(N+2)) is the first sample to see the updated matrix**, and it
also sees the updated `r`. Data and matrix updates visit each PE at the same
inward phase, so a sample observes a coherent generation throughout W.

An update from sample S first affects sample S + N + 2 during continuous,
unstalled traffic. Input bubbles and consumer backpressure change the sample
spacing; they preserve W/R coherence but do not retain this sample-index delay.
Updates start at retirement, so holding `resultReady` also delays learning.
The faster feedback intentionally changes training trajectories from the
conventional array's S + 2N + 1 schedule.

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

At full speed with default depths, the context FIFO holds `N+3` outstanding
samples and the result FIFO holds one entry. `N+3` context slots suffice for
unstalled one-sample-per-clock throughput. The default depths remain
`IN_FLIGHT_DEPTH = 2N+2` and `OUTPUT_FIFO_DEPTH = 2N`; the extra capacity absorbs
consumer stalls without changing unstalled latency.

## Backpressure

![Common advance control holds compute and learning together](presentation/figures/08-backpressure-trace.svg)

When the consumer stalls, results first accumulate in `resultFifo`. Admission
stops when the context FIFO fills. When a complete AD result is waiting and the
result FIFO is full, `arrayAdvance` falls: both skew taps, both partial-sum
directions, the AD result, valid state, and matrix/reduction updates hold together.
An input bubble advances these paths normally with an invalid sample.

When `resultReady` returns, the core can retire a result, enqueue the waiting
result, accept a new input, and advance on the same edge.

```text
inputReady   = weightsLoaded ∧ ¬acceptedReload ∧ (input register empty ∨ it is leaving) ∧ (context not full ∨ retire)
arrayAdvance = ¬(AD result valid ∧ resultFifo full ∧ ¬retire)  [while weightsLoaded]
```

While unloaded, `arrayAdvance` is asserted only for the N internal load shifts
and the capture edge; row collection itself does not advance the array.

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
- An accepted reload is an epoch boundary: it blocks input acceptance on that
  edge. Offer any simultaneous input again after the replacement weights load.
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
| `N` | `3` | Array and vector dimension, ≥ 2 (tested for 2/3/4/5/8) |
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
| `FunctionalReference` | End-to-end numbers and learning, one sample at a time. Update visibility in closed form, `N+2`, for continuous unstalled traffic. |
| `CycleReference` | Latency, W/R evolution, two-ended load shifts and capture, FIFO contents, bubbles, backpressure, reset, and reconfiguration. Derives update visibility by wave propagation for comparison with `FunctionalReference`. |
| Golden RTL comparison | Ten deterministic scenarios through `nnAcceleratorStateTrace_tb`; W, R, the staging bank and loader state, input stage, and both FIFOs compared with `CycleReference` on every edge. |
| UVM (`uvm/`) | Seeded public-port traffic at N = 3, WIDTH = 8: ordering, loss/duplication, backpressure, reset/reload recovery, and exact pass-through inference while matrix weights are known. Training and inference with learned weights are checked for ordering and liveness. |
| Directed benches | Reduction arithmetic, the matrix core at N = 2/3/4/5/8, and update-wave propagation, saturation, and stalls. |

Run individual checks:

```powershell
pwsh -File scripts/run_rtl_reference_compare.ps1
pwsh -File scripts/run_uvm.ps1 -TestName nn_uvm_regression_test
pwsh -File scripts/run_uvm.ps1 -TestName nn_uvm_smoke_test
pwsh -File scripts/run_uvm.ps1 -TestName nn_uvm_regression_test -Seed 12345
```

The UVM log prints its seed, so `-Seed` replays a run. Stimulus uses `$urandom`
and explicit scenario checks, allowing it to run on the Questa FPGA Starter license.

On Windows, a license-free alternative runs the Python tests, directed RTL suites,
parameter rejection, and the same integrated RTL/reference trace:

```powershell
pwsh -File scripts/run_verilator.ps1
pwsh -File scripts/run_verilator.ps1 -ConventionalBaseline
```

This needs [OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build) and a recent
[w64devkit](https://github.com/skeeto/w64devkit). The runner finds local bundles under
`build/tools/oss-cad-suite` and `build/tools/w64devkit`, or the directories named by
`NN_ACCEL_OSS_CAD_SUITE` and `NN_ACCEL_W64DEVKIT`. C++ model artifacts use a separate
space-free directory under the user profile; override it with
`NN_ACCEL_VERILATOR_BUILD_ROOT` when needed. Logs stay under `build/verilator`.
`-ConventionalBaseline` extracts the original core from commit `09a4097` and replays
the same signed, seeded-random vectors through the same independent dot-product
scoreboard. UVM remains a separate Questa check.

## FPGA build

The Quartus project targets the DE1-SoC's Cyclone V `5CSEMA5F31C6`, with
`nnAccelerator` as its top level. All ports except `clk` use virtual pins to
characterize the core. A board integration needs its own interface and pin assignments.

Fresh September 30, 2026 fits with Quartus Prime 25.1std Lite use the same
device, settings, and constraints at the defaults (N = 3, WIDTH = 16).
The conventional baseline is the core at `09a4097` with the pre-existing local
reload-ready simplification; the inward fit uses the current implementation.

| Metric | Conventional baseline | Inward / anti-diagonal |
| --- | ---: | ---: |
| Logic | 1,381 ALMs (4%) | 1,599 ALMs (5%) |
| Registers | 1,168 | 1,087 |
| Block memory | 1,048 bits, 5 RAM blocks | 1,048 bits, 5 RAM blocks |
| DSP blocks | 15 | 15 |
| Pins | 1 physical clock pin, 258 virtual pins | same |
| Worst setup / minimum hold slack | −2.438 ns / +0.123 ns | −2.322 ns / +0.163 ns |
| Lowest reported Fmax | 80.40 MHz | 81.16 MHz |
| Acceptance to result FIFO, unstalled | 7 clocks | 5 clocks |

This fit uses 15.8% more ALMs and 6.9% fewer registers. Both designs sustain
one sample per advancing clock with an always-ready consumer. At 50 MHz,
acceptance to result formation falls from 140 ns to 100 ns. At each design's
reported Fmax, it is approximately 87.1 ns versus 61.6 ns. One fit per design
does not establish a repeatable clock-frequency improvement.

The worst internal path remains the activation/reduction path into the result
FIFO. At the slow 85°C corner the inward design's worst path ending at a PE
psum register has −0.244 ns slack, versus −2.307 ns for the full core at that
corner. No extra AD/output register was added.

`NN_Acceleration.sdc` sets a **100 MHz implementation target** (10 ns), distinct
from the board's 50 MHz oscillator. Neither fit meets the 100 MHz target.
The clock pin is unassigned and the fitter reports non-dedicated clock routing;
these figures characterize this fit, not a completed board implementation.
The comparison reports are under `build/fpga-baseline/output_files/` and
`build/fpga-final/output_files/` (ignored). Normal project builds generate
`Quartus Stuff/output_files/`. See [the implementation report](LATENCY_SLASH_IMPLEMENTATION.md)
for validation results and the loading/learning tradeoffs.

## Repository layout

```text
.
├── weightStationaryVariant/
│   ├── nnAccelerator.sv                          core: FIFOs, activation, reduction, retirement, learning
│   ├── weightStationaryMatrixMultiplier.sv       input register, shared two-sided skew, staged loading, arrayAdvance
│   ├── weightStationarySystolicArray.sv          inward PE grid, unified valid pipe, inward update pipe
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
