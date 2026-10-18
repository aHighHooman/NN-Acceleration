# Weight-Stationary Neural-Network Accelerator — Fountain

The existing SystemVerilog accelerator now uses the Fountain center-out dataflow.
It supports a configurable chain of matrix stages, with a scalar weighted readout
only after the last stage. The implementation stays in the existing PE, array,
matrix-wrapper, accelerator, and readout files.

Inference can accept one vector per advancing clock when capacity and the
consumer permit. Training uses full backward sensitivities with saturating
sign-based weight steps, one training sample at a time. A training sample blocks
later admission until its prediction retires and its update completes.

See [the architecture specification](Fountain_Architecture_Specification.md)
and [the learning derivation](Fountain_Three_Layer_Learning_Analysis.md).

## Architecture

![Three-layer Fountain architecture](presentation/figures/01-architecture.svg)

`LAYERS` controls the number of matrix stages; three is the default. The diagram
shows that case at N=3, WIDTH=16 and F=4. Each stage stores weights
`W[layer][input][output]` and computes
`a[layer+1] = activation(a[layer] * W[layer])`. The scalar readout is
`prediction = sum(a[last][j] * r[j])`, followed by one final fixed-point rescale.

Inside each array, activations enter the center column and move horizontally
outward. Partial sums move vertically outward from two seeds. The center product
belongs to the lower sum only. A registered adder joins the two boundary sums
for each output column. These per-column joins are distinct from the final
scalar readout.

The complete output lanes retain the center-first front. They connect directly
to the next array without collecting and re-skewing the vector. Lane valids and
vector tags travel with the data. The final collector aligns lanes before the
whole-vector result/FIFO boundary.

## Timing

Let `h = (N-1)/2`, `G` be additional balanced route stages, and `J` the column
joining latency. One matrix stage shifts the center-first front by:

```text
D = h + 1 + G + J
raw lane j after K stages = K*D + abs(j-h)
final aligned vector     = K*D + h
result FIFO formation    = K*D + h + 1
first retirement         = K*D + h + 2
```

These are edge intervals from input acceptance, assuming no stalls and an
always-ready consumer. The input register supplies the center lane through a
zero-depth source-skew bypass; there is no additional common skew-entry register.
All pipeline dependencies count enabled steps when backpressured.

For `N=5`, `LAYERS=3`, `G=0`, `J=1`, the final raw lanes are ready at
`[14,13,12,13,14]`, the result FIFO forms at edge 15, and first retirement is edge
16. A single stage gives `[6,5,4,5,6]`, formation at 7, and retirement at 8.
These cycle counts do not establish an achieved frequency or wall-clock speedup.

## Loading and reconfiguration

The public signal names remain the same. Loading multiple layers uses the
existing reverse-row handshake repeatedly:

1. Send layer 0 rows `N-1` through `0` on `weightData[N]`.
2. Send layer 1 in the same order, then each remaining layer.
3. After `LAYERS*N` accepted row handshakes, the staging bank transfers one
   complete row per clock to resident PEs for `LAYERS*N` clocks.
4. One commit edge asserts `weightsLoaded`. Computation remains blocked until
   the entire chain is loaded.

Host gaps are supported. Loading uses explicit row writes, replacing the old
inward topology's two-ended partial-sum shifts. `LAYERS=1` retains the original
N-row host format and N-transfer-plus-one-commit timing.

`reloadWeights` is accepted only with `reloadReady`. Both the forward pipeline
and result FIFO must be empty, and no training sample/update may remain. An
accepted reload excludes input admission on that edge. Load all layers again.

`passThrough`, `reduceOutput`, and `loadReductionWeights` are stream-wide
configuration. Change/pulse them only when quiescent. The implementation rejects
busy readout loads and holds existing modes, recording an internal sticky
`configurationError` until reset. A readout-load edge excludes input admission.

`rst_n` is synchronous and takes priority over pipeline enable. It discards all
in-flight samples, loader state, and pending learning, clears resident readout
weights, and drops `weightsLoaded`. Reload matrix and readout weights afterward.

## Exact-width arithmetic

All arithmetic uses signed stored integers. There is no inter-layer narrowing,
rounding, or saturation in the forward matrix path. Each stage retains a full
sum, then applies ReLU or pass-through exactly once.

For input/weight width `WIDTH` and `C = ceil(log2(N))`:

```text
stage k output width, k starting at 1 = WIDTH + k*(WIDTH+C)
RAW_WIDTH                            = WIDTH + LAYERS*(WIDTH+C)
public RESULT_WIDTH                  = RAW_WIDTH + C
```

At `WIDTH=8`, `N=5`, three output widths are 19, 30, and 41 bits. Public lanes are
44 bits. The wider port is necessary for exact chaining; it is not an 8-bit
inter-layer quantization design.

Inputs and matrix weights have `FRACTION_BITS=F` fractional bits. Stage k's raw
output has `(k+1)*F`. The scalar readout coefficients have
`REDUCTION_WEIGHT_WIDTH-1` fractional bits. Accumulate all full products, then
shift once by `LAYERS*F + REDUCTION_WEIGHT_WIDTH - 1`; the scalar prediction
returns to F fractional bits.

- `reduceOutput=0`: sign-extended final activated vector, with `(LAYERS+1)*F`
  fractional bits.
- `reduceOutput=1`: rescaled scalar prediction in lane 0; other lanes are zero.

At `LAYERS=1`, port width and single-layer inference arithmetic match the former
accelerator. Training timing deliberately changes to the serialized policy.

## Learning

For squared error, define `e = target - prediction` and activation gates `g`.
The backward sensitivities are:

```text
s[last] = g[last] .* r
s[l]    = g[l] .* (W[l+1] * s[l+1])
W[l][i,j] += sign(a[l][i]) * sign(e) * sign(s[l][j])
r[j]       += sign(e) * sign(a[last+1][j])
```

Each stored weight update is one LSB, saturated to its signed range. Matrix
weights use the matrix-weight binary point; readout weights use their own.
ReLU gates close at zero; pass-through gates stay open.

A training sample is an admission barrier. Earlier inference samples may finish,
but no later sample enters until training finishes. Its target, layer-input
signs, derivative gates, and tag are retained. Resident weights stay unchanged
from its forward pass through backward sensitivity calculation.

The backward engine shares one multiplier/accumulator across the weighted
sensitivity sums. For nonzero error it takes `(LAYERS-1)*N*N` advancing clocks
after retirement, followed by one atomic update edge. Zero error or a single
layer skips the backward sums. Admission resumes after the commit. All matrices
and readout weights change together with no forward data in flight.

This implements full-sensitivity sign updates, not the cheaper approximation
that quantizes each intermediate error to its sign. It does not claim continuous
II=1 training or identical trajectories to the former delayed SSLMS wave.
Convergence on an application workload remains an empirical question. Avoid an
all-zero, bias-free ReLU initialization: its closed gates cannot bootstrap these
updates.

## Backpressure and capacity

The result FIFO absorbs downstream stalls. When it cannot accept an aligned
matrix result, one common `arrayAdvance` freezes the source delays, all arrays,
join routes, lane tags, and collector. FIFO pop/push on the same edge permits
continued advancement. Admission reserves outstanding capacity before accepting
a vector; the tag space must cover the maximum live sample count.

Training context is held for the single pending training sample. The former
sample-context FIFO and delayed matrix/readout update waves are no longer used.
The existing `signedFifo` still implements the public result buffer.

## Parameters and public interface

The public ports retain their names and roles from the previous accelerator.
`resultData[N]` now uses the layer-dependent width above. All weight/input lanes
remain `WIDTH` bits. `passThrough` applies to every matrix stage.

| Parameter | RTL default | Meaning |
| --- | ---: | --- |
| `N` | 3 | Odd square dimension, at least 3 |
| `WIDTH` | 16 | Input and stationary matrix-weight width |
| `LAYERS` | 3 | Number of dedicated matrix stages |
| `FRACTION_BITS` | 4 | Input/matrix-weight fractional bits |
| `TARGET_WIDTH` | `WIDTH` | Target width; prediction and target are sign-extended before comparison |
| `REDUCTION_WEIGHT_WIDTH` | 8 | Scalar-readout coefficient width |
| `TAG_WIDTH` | 16 | Internal vector tag width, 1 through 30 |
| `G`, `J` | 0, 1 | Balanced route stages and registered column-join latency |
| `OUTPUT_FIFO_DEPTH` | `2*N` | Result storage |
| `IN_FLIGHT_DEPTH` | `LAYERS*(N+G+J+2)+OUTPUT_FIFO_DEPTH+2` | Admission limit; must not exceed the tag space |

Even dimensions, rectangular layers, explicit requantization, independent lane
backpressure, and fixed-resource tiling remain extensions.

## Verification

Run the existing regression entry point with Python, PowerShell 7, and Questa:

```powershell
pwsh -File scripts/run_modelsim.ps1
```

Questa discovery defaults to `C:/altera_lite/25.1std/questa_fse/win64`; override it
with `NN_ACCEL_QUESTA_BIN`. On systems where sandbox DLL restrictions prevent
Questa from starting, run this command in a normal local terminal.

The passing suite includes:

- 68 Python tests: fixed-point arithmetic, exact-width inference, finite-difference
  checks of hidden update signs, clocked PE dependencies, loading, timing,
  bubbles, stalls, tag wrap, reset, and all-layer training.
- Seven directed matrix profiles at N=3/5/9, one/three layers, WIDTH=8/16,
  G=0/1, and J=1/2, including three-layer results wider than 64 bits.
- Seed/non-seed PE checks for configuration priority, signed extremes, saturated
  updates, same-edge old-weight reads, common enable, and reset priority.
- Expected-failure tests for unsupported dimensions, missing/mistagged MAC
  operands, bad join tags, busy configuration, and premature input.
- Four integrated RTL/reference profiles totaling 3,167 checked edges: N=3
  three-layer integer; N=5 three-layer integer; N=3 three-layer fixed-point with
  G=1/J=2; and N=3 single-layer fixed-point compatibility.
- The existing public-port UVM regression at `LAYERS=1`, with zero errors/fatals.
  Exact learned weights for all layers are checked by the integrated trace.

Individual commands:

```powershell
pwsh -File scripts/run_rtl_reference_compare.ps1 -N 5 -Layers 3
pwsh -File scripts/run_rtl_reference_compare.ps1 -FractionBits 2 -RouteStages 1 -JoinLatency 2
pwsh -File scripts/run_uvm.ps1 -Seed 12345
```

This change was validated with Questa.

## FPGA project

The existing Quartus project still uses `nnAccelerator` as top level on Cyclone V
`5CSEMA5F31C6`, with virtual data/control pins and a 100 MHz target. It explicitly
selects the bringup profile `N=5`, `WIDTH=8`, `LAYERS=3`, `FRACTION_BITS=0`, G=0/J=1.
These project overrides differ from the independently parameterized RTL defaults.

```powershell
cd 'Quartus Stuff'
quartus_sh --flow compile NN_Acceleration
```

The three-stage prototype completes synthesis, fit and assembly: 6,532 ALMs,
6,591 registers, 87/87 DSP blocks, and 2,650 memory bits. The shared backward
multiplier maps to logic to make the design fit. Worst reported core Fmax is
70.73 MHz; setup slack is -4.138 ns against the 100 MHz target, so timing at that
target does not close. The worst reported paths pass through the shared backward
engine. Data/control pins are virtual and lack external I/O constraints; the
unassigned clock uses non-dedicated routing. This is a core characterization,
not board sign-off. See specification section 14 for reports and limits. The
earlier 81.16 MHz figure belongs to the historical single-layer inward design
and is not a matched comparison.

## Repository layout

The implementation uses the existing file structure:

- `weightStationaryVariant/weightStationaryProcessingElement.sv`: tagged MAC,
  stationary weight, row loading and saturated update.
- `weightStationaryVariant/weightStationarySystolicArray.sv`: center-out grid,
  explicit column joins/routes, and the shared lane-delay helper.
- `weightStationaryVariant/weightStationaryMatrixMultiplier.sv`: source skew,
  direct stage chaining, multi-layer row loader, activation and final collector.
- `weightStationaryVariant/nnAccelerator.sv`: public control/FIFO, one readout,
  training admission barrier and serial backward engine.
- `weightStationaryVariant/weightedVectorReduction.sv`: full-width readout and
  an explicit configurable final shift.
- `reference/`, `uvm/`, and `scripts/`: updated existing models and regressions.
- `presentation/figures/01-architecture.svg`: current Fountain architecture,
  with the existing visual style and light/dark themes. Other figures describe
  the historical inward implementation.

The pre-Fountain implementation remains available in Git at `219a081`.
