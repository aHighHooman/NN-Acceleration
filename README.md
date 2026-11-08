# Fountain: A Weight-Stationary Neural-Network Accelerator

A parameterized SystemVerilog accelerator for small fully connected neural
networks. It runs inference and on-chip training, and targets an Intel Cyclone V
FPGA.

The core idea is the **Fountain dataflow**. In a regular systolic array, data
enters at one corner and moves across the grid. Fountain instead feeds each
input vector in at the **center column** and sends it **outward**, with partial
sums also moving outward from the middle rows. Each layer's outputs then leave
in the same center-first pattern its inputs arrived in. The next layer can
start on those outputs right away, without waiting for the whole vector or
re-skewing it. Each extra layer therefore adds only a small, fixed delay.

![Three-layer Fountain architecture](presentation/figures/01-architecture.svg)

## Features

- **Multi-layer network in hardware.** A configurable chain of `LAYERS` N×N
  matrix stages (three by default), each followed by ReLU or pass-through. A
  final Nx1 column of PEs reduces the last stage to one scalar prediction.
- **Pipelined inference.** Accepts up to one input vector per clock. A result
  FIFO and a single global stall signal handle downstream backpressure.
- **On-chip training.** Runs the gradient vectors back through the matrix stages
  themselves and applies saturating one-LSB sign updates to all weights.
- **Uniform fixed point.** Every stage takes the same activation format and
  requantizes its outputs back into it, so every multiplier is the same size
  whatever the depth.
- **Verified against a reference model.** Python functional and cycle-accurate
  models are compared edge-by-edge with the RTL in Questa.

## How it works

Each matrix stage holds its weights `W[layer][input][output]` stationary in a
grid of processing elements (PEs) and computes:

```text
a[layer+1] = requantize(activation(a[layer] * W[layer]))
prediction = sum(a[last][j] * r[j])          (scalar readout)
```

Inputs, activations and weights all have `FRACTION_BITS` fraction bits, so a
product has twice as many. Requantizing shifts the sum right by
`FRACTION_BITS`, which restores the binary point and truncates toward minus
infinity, then caps it at the `ACTIVATION_WIDTH` limits instead of wrapping.
Shifting is free in hardware; the cap is a compare and a mux.

Inside a stage:

1. Activations enter the center column and travel horizontally outward to
   both sides.
2. Partial sums start from two seed rows at the center and travel vertically
   outward, upward and downward.
3. A combinational adder at the edge of each output column joins the upper and
   lower sums into a complete output, which is then activated and requantized.
4. The output lanes keep the center-first timing and feed the next stage
   directly.

The readout works the same way, as an Nx1 column of PEs holding `r`. Each row
takes one lane of the last stage as it arrives, and the two sum chains join into
the prediction, so no lane is ever realigned into a vector.

### Latency

With `h = (N-1)/2`, each stage adds `D = h + 1` cycles. The column join has no
register of its own: the upper and lower boundary PEs capture their sums, and
the join, ReLU, requantization and the next stage's first multiply settle in the following
cycle. For N=5, if a stage's first PE captures a product at edge 1, the
boundary rows capture at edge 3 and the next stage's first PE captures at edge
4. For K stages, assuming no stalls:

| Event | Cycles after input is accepted |
| --- | --- |
| Last-stage lane `j` ready | `K*D + abs(j-h)` |
| Prediction ready | `(K+1)*D` |
| Prediction in result FIFO | `(K+1)*D + 1` |
| First result available | `(K+1)*D + 2` |

Example: N=5, three layers. The last stage's lanes are ready at cycles
`[11,10,9,10,11]`, the prediction at cycle 12, and the first result comes out at
cycle 14.

### Training

When a sample is submitted with `trainingEnable`, the accelerator:

1. Lets earlier inference samples finish and stops admitting new ones.
2. Runs the training sample forward and computes `e = target - prediction`.
3. Computes each layer's gradient vector by running the readout weights back
   through the matrix stages, in `LAYERS*(h+1)` cycles. Each entry is 0 where the
   condition fails:
   ```text
   grad[last] = r                              | output[last] > 0
   grad[l]    = requantize(W[l+1] * grad[l+1]) | output[l] > 0
   ```
4. Updates every weight by one LSB in a single cycle, saturating at its range
   limit:
   ```text
   W[l][i,j] += sign(a[l][i]) * sign(e) * sign(grad[l][j])
   r[j]      += sign(e) * sign(a[last+1][j])
   ```
5. Resumes admitting samples.

![Backpropagation path](presentation/figures/12-backprop-architecture.svg)

The backward pass multiplies on the other side of each matrix, `W * vector`
instead of `vector * W`. That equals `vector * W'`, so PE `(i,j)` also holds a
copy of `W[j][i]` and multiplies by it during the backward pass. The routes,
seeds and column joins stay exactly as they are; each array only takes its input
from the next array's output instead of the previous one's, so the same
center-first timing carries one array's outputs straight into the previous array.

Loading row `r` writes the forward weights of row `r` and the copies in column
`r`. Each copy takes the same one-LSB step as the weight it mirrors, so both
always hold the same matrix, and simulation asserts that they do.

The readout weights are held still, and the path from them to every lane is
feed-forward. The controller only has to wait for the longest path,
`(LAYERS-1)*(h+1) + h` cycles, and capture the signs on one more edge. The pass
needs no valid bits or lane skew, and stores only two sign bits per lane.

Array 0 never runs backward, so synthesis removes its copies. Gradient vectors
use the activation lanes, so the readout weights must fit an activation
(`REDUCTION_WEIGHT_WIDTH <= ACTIVATION_WIDTH`), and they are requantized like
activations after each array.

Inference stays fully pipelined. Training handles one sample at a time, and the
stages can't run inference while they run backward.
`reference/functional.py` is the executable statement of the update rule, and
`reference/test_functional.py` checks its signs against finite differences.

## Getting started

### Requirements

- Questa / ModelSim (Intel FPGA Starter Edition works)
- Python 3
- PowerShell 7
- Quartus Prime (only for FPGA builds)

### Run the test suite

```powershell
pwsh -File scripts/run_modelsim.ps1
```

This runs the Python reference tests, the unit testbenches for the PE, array
and readout, the parameter-rejection checks, and the integrated RTL-vs-reference
comparisons. The script looks for Questa in
`C:/altera_lite/25.1std/questa_fse/win64` by default. To use another location,
set `NN_ACCEL_QUESTA_BIN`.

To compare one configuration against the reference model:

```powershell
pwsh -File scripts/run_rtl_reference_compare.ps1 -N 5 -Layers 3
pwsh -File scripts/run_rtl_reference_compare.ps1 -FractionBits 2
```

### Build for FPGA

```powershell
cd 'Quartus Stuff'
quartus_sh --flow compile NN_Acceleration
```

The Quartus project targets a Cyclone V `5CSEMA5F31C6` (DE1-SoC) at 100 MHz. It
uses N=5, WIDTH=16, ACTIVATION_WIDTH=19, LAYERS=3, FRACTION_BITS=8. These overrides
differ from the RTL defaults.

## Using the accelerator

### Parameters

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `N` | 3 | Layer size (odd, ≥ 3) |
| `WIDTH` | 16 | Weight width in bits |
| `ACTIVATION_WIDTH` | 19 | Input, activation and gradient width in bits |
| `LAYERS` | 3 | Number of matrix stages |
| `FRACTION_BITS` | 8 | Fractional bits of inputs, activations and weights |
| `TARGET_WIDTH` | `WIDTH` | Training target width |
| `REDUCTION_WEIGHT_WIDTH` | 8 | Readout coefficient width |
| `OUTPUT_FIFO_DEPTH` | `2*N` | Result buffer depth |
| `IN_FLIGHT_DEPTH` | `LAYERS*(N+2)+OUTPUT_FIFO_DEPTH+2` | Maximum samples in flight |

### Interface

All streams use valid/ready handshakes, and `rst_n` is a synchronous
active-low reset.

| Port group | Signals | Purpose |
| --- | --- | --- |
| Weights | `weightData[N]`, `weightValid`, `weightReady`, `weightsLoaded` | Load matrix weights row by row |
| Reload | `reloadWeights`, `reloadReady` | Replace weights once the pipeline is idle |
| Input | `inputData[N]`, `targetData`, `trainingEnable`, `inputValid`, `inputReady` | Submit a sample, with a target if training |
| Readout | `reductionWeight[N]`, `loadReductionWeights` | Load the readout coefficients |
| Mode | `passThrough` | Disable ReLU on all stages |
| Output | `resultData`, `resultValid`, `resultReady` | Read the scalar prediction |

**Loading weights.** Send each layer's rows from `N-1` down to `0`, layer 0
first, for a total of `LAYERS*N` row handshakes. `weightsLoaded` goes high once
all layers are in place. Computation waits for it.

**Output format.** `resultData` is the scalar prediction, rescaled back to
`FRACTION_BITS` fractional bits. The accelerator has no vector output.

**Result width.** The prediction has `ACTIVATION_WIDTH + ceil(log2 N)` bits
and `FRACTION_BITS` fraction bits, whatever the depth. For example,
ACTIVATION_WIDTH=19 and N=5 give a 22-bit output port.

**Configuration.** Change `passThrough` and load readout weights only while the
accelerator is idle. Busy configuration attempts are rejected.

The cycle-accurate model in `reference/cycle.py` defines the exact handshake,
stall, reset and timing rules that the RTL is compared against.

## Results

### Simulation

All tests pass in Questa:

- 30 Python tests of the reference models: fixed-point arithmetic and
  requantization, timing, stalls, reset, the backward pass settling on the
  model's gradients, and finite-difference checks of the training update signs.
- Matrix-stage testbenches covering N=3/5/9, one, three and four layers,
  WIDTH=8/16 and FRACTION_BITS=0/2/8, including a narrow activation format that
  saturates often, and backward gradients against a dense calculation.
- Readout-column testbenches covering N=3/5/9 and the 19-bit lanes of the Quartus
  build: results against a dense sum, lane timing, stalls, loads and updates.
- 3,824 cycle-by-cycle RTL/reference comparisons across five configurations,
  hitting both caps in the forward and gradient lanes.
  Each one checks handshakes, results, every weight, every lane of every stage,
  and the readout output.

### FPGA (Cyclone V, N=5, WIDTH=16, ACTIVATION_WIDTH=19, FRACTION_BITS=8, three layers)

| Metric | Value |
| --- | --- |
| Logic | 7,576 ALMs |
| Registers | 7,436 |
| DSP blocks | 80 / 87 |
| Memory | 220 bits (1 RAM block) |
| Fmax | 72.0 MHz (100 MHz target not met) |

Every multiplier maps to one 18x19 DSP slot; 77 of the 80 pack two to a block,
so about 42 blocks would suffice even though the fitter spreads them over 80.
The critical path runs from a stage's boundary sums through the column join,
ReLU and requantization into the next stage's or the readout column's multiplier
and sum register, all in one cycle. Across seeds 1 to 3 Fmax is 72.0 to 77.6 MHz,
averaging 75.7 MHz; placement alone moves it by several MHz.
These numbers come from the default seed unless noted. I/O pins are virtual, so
they describe the core and are not a board-level sign-off.

## Repository layout

```text
weightStationaryVariant/
  weightStationaryProcessingElement.sv   PE: MAC, stationary weight, update logic
  weightStationarySystolicArray.sv       Center-out PE grid, combinational column joins
  weightStationaryMatrixMultiplier.sv    Layer chain, weight loader, activation
  weightedVectorReduction.sv             Scalar readout: Nx1 PE column
  nnAccelerator.sv                       Top level: control, FIFO, training control
  *_tb.sv                                Testbenches
memory/signedFifo.sv                     Result FIFO
reference/                               Python functional and cycle-accurate models, tests
scripts/                                 Simulation and comparison scripts
Quartus Stuff/                           Quartus project for Cyclone V
presentation/figures/                    Architecture diagrams
```

## Limitations and future work

- The 100 MHz timing target is not met on Cyclone V; the latest fit reaches
  72.0 MHz with the default seed and 75.7 MHz averaged over three seeds.
- The output is one scalar, so multi-output networks are not supported.
- Only square matrix stages with odd N are supported.
- Training handles one sample at a time; only inference is fully pipelined.
- It hasn't been shown yet how well training converges on a real workload. Do
  not start ReLU networks from all-zero weights, because closed gates cannot
  start learning.
- Requantization truncates and uses one format for every layer; rounding or a
  per-layer shift could recover precision.
- Possible extensions include rectangular layers, per-lane backpressure, and tiling large layers onto fixed hardware.

An earlier "inward" design, with data entering at two corners and meeting at
the anti-diagonal, is preserved at commit `219a081`.
