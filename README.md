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
- **On-chip training.** Backpropagates full error sensitivities through every
  layer and applies saturating one-LSB sign updates to all weights.
- **Exact arithmetic.** Signed fixed-point with no rounding or saturation
  between layers. Each stage widens just enough to stay exact.
- **Verified against a reference model.** Python functional and cycle-accurate
  models are compared edge-by-edge with the RTL in Questa.

## How it works

Each matrix stage holds its weights `W[layer][input][output]` stationary in a
grid of processing elements (PEs) and computes:

```text
a[layer+1] = activation(a[layer] * W[layer])
prediction = sum(a[last][j] * r[j])          (scalar readout)
```

Inside a stage:

1. Activations enter the center column and travel horizontally outward to
   both sides.
2. Partial sums start from two seed rows at the center and travel vertically
   outward, upward and downward.
3. A combinational adder at the edge of each output column joins the upper and
   lower sums into a complete output.
4. The output lanes keep the center-first timing and feed the next stage
   directly.

The readout works the same way, as an Nx1 column of PEs holding `r`. Each row
takes one lane of the last stage as it arrives, and the two sum chains join into
the prediction, so no lane is ever realigned into a vector.

### Latency

With `h = (N-1)/2`, each stage adds `D = h + 1` cycles. The column join has no
register of its own: the upper and lower boundary PEs capture their sums, and
the join, ReLU and the next stage's first multiply settle in the following
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
3. Computes the backward sensitivities on one shared multiply-accumulate unit
   in `(LAYERS-1)*N*N` cycles:
   ```text
   s[last] = g[last] .* r
   s[l]    = g[l] .* (W[l+1] * s[l+1])
   ```
4. Updates every weight by one LSB in a single cycle, saturating at its range
   limit:
   ```text
   W[l][i,j] += sign(a[l][i]) * sign(e) * sign(s[l][j])
   r[j]      += sign(e) * sign(a[last+1][j])
   ```
5. Resumes admitting samples.

![Backpropagation path](presentation/figures/12-backprop-architecture.svg)

Inference stays fully pipelined. Training handles one sample at a time.
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
uses N=5, WIDTH=8, LAYERS=3, FRACTION_BITS=0. These overrides
differ from the RTL defaults.

## Using the accelerator

### Parameters

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `N` | 3 | Layer size (odd, ≥ 3) |
| `WIDTH` | 16 | Input and weight width in bits |
| `LAYERS` | 3 | Number of matrix stages |
| `FRACTION_BITS` | 4 | Fractional bits of inputs and weights |
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

**Result width.** Results grow with depth so that no precision is lost. Stage k
has `WIDTH + k*(WIDTH + ceil(log2 N))` bits, and the output port adds another
`ceil(log2 N)` bits. For example, WIDTH=8 and N=5 give a 44-bit output port.

**Configuration.** Change `passThrough` and load readout weights only while the
accelerator is idle. Busy configuration attempts are rejected.

The cycle-accurate model in `reference/cycle.py` defines the exact handshake,
stall, reset and timing rules that the RTL is compared against.

## Results

### Simulation

All tests pass in Questa:

- 26 Python tests of the reference models: fixed-point arithmetic, timing,
  stalls, reset, and finite-difference checks of the training update signs.
- Matrix-stage testbenches covering N=3/5/9, one and three layers, WIDTH=8/16,
  including results wider than 64 bits.
- Readout-column testbenches covering N=3/5/9 and the 41-bit lanes of the Quartus
  build: results against a dense sum, lane timing, stalls, loads and updates.
- 3,270 cycle-by-cycle RTL/reference comparisons across four configurations.
  Each one checks handshakes, results, every weight, every lane of every stage,
  and the readout output.

### FPGA (Cyclone V, N=5, WIDTH=8, three layers)

| Metric | Value |
| --- | --- |
| Logic | 7,423 ALMs |
| Registers | 5,540 |
| DSP blocks | 87 / 87 |
| Memory | 440 bits (2 RAM blocks) |
| Fmax | 73.1 MHz (100 MHz target not met) |

The critical path runs from the last layer's weights through the shared backward
multiplier into the sensitivity registers. That multiplier is built from logic
because the forward stages use every DSP block. These numbers come from one fit
with the default seed. I/O pins are virtual, so they describe the core and are
not a board-level sign-off.

## Repository layout

```text
weightStationaryVariant/
  weightStationaryProcessingElement.sv   PE: MAC, stationary weight, update logic
  weightStationarySystolicArray.sv       Center-out PE grid, combinational column joins
  weightStationaryMatrixMultiplier.sv    Layer chain, weight loader, activation
  weightedVectorReduction.sv             Scalar readout: Nx1 PE column
  nnAccelerator.sv                       Top level: control, FIFO, training engine
  *_tb.sv                                Testbenches
memory/signedFifo.sv                     Result FIFO
reference/                               Python functional and cycle-accurate models, tests
scripts/                                 Simulation and comparison scripts
Quartus Stuff/                           Quartus project for Cyclone V
presentation/figures/                    Architecture diagrams
```

## Limitations and future work

- The 100 MHz timing target is not met on Cyclone V; the latest fit reaches
  73.1 MHz.
- The output is one scalar, so multi-output networks are not supported.
- Only square matrix stages with odd N are supported.
- Training handles one sample at a time; only inference is fully pipelined.
- It hasn't been shown yet how well training converges on a real workload. Do
  not start ReLU networks from all-zero weights, because closed gates cannot
  start learning.
- Possible extensions include rectangular layers, requantization between
  layers, per-lane backpressure, and tiling large layers onto fixed hardware.

An earlier "inward" design, with data entering at two corners and meeting at
the anti-diagonal, is preserved at commit `219a081`.
