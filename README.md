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
  matrix stages (three by default), each followed by ReLU or pass-through. An
  optional weighted scalar readout follows the last stage.
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
prediction = sum(a[last][j] * r[j])          (optional scalar readout)
```

Inside a stage:

1. Activations enter the center column and travel horizontally outward to
   both sides.
2. Partial sums start from two seed rows at the center and travel vertically
   outward, upward and downward.
3. A registered adder at the edge of each output column joins the upper and
   lower sums into a complete output.
4. The output lanes keep the center-first timing and feed the next stage
   directly. A collector at the end aligns the lanes into a full vector.

### Latency

With `h = (N-1)/2`, extra route stages `G`, and column-join latency `J`, each
stage adds `D = h + 1 + G + J` cycles. For K stages, assuming no stalls:

| Event | Cycles after input is accepted |
| --- | --- |
| Output lane `j` ready | `K*D + abs(j-h)` |
| Full vector in result FIFO | `K*D + h + 1` |
| First result available | `K*D + h + 2` |

Example: N=5, three layers, G=0, J=1. The output lanes are ready at cycles
`[14,13,12,13,14]` and the first result comes out at cycle 16.

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

Inference stays fully pipelined. Training handles one sample at a time.
[Fountain_Three_Layer_Learning_Analysis.md](Fountain_Three_Layer_Learning_Analysis.md)
derives the update rule.

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
pwsh -File scripts/run_rtl_reference_compare.ps1 -FractionBits 2 -RouteStages 1 -JoinLatency 2
```

### Build for FPGA

```powershell
cd 'Quartus Stuff'
quartus_sh --flow compile NN_Acceleration
```

The Quartus project targets a Cyclone V `5CSEMA5F31C6` (DE1-SoC) at 100 MHz. It
uses N=5, WIDTH=8, LAYERS=3, FRACTION_BITS=0, G=0 and J=1. These overrides
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
| `G`, `J` | 0, 1 | Extra route stages, column-join latency |
| `OUTPUT_FIFO_DEPTH` | `2*N` | Result buffer depth |
| `IN_FLIGHT_DEPTH` | `LAYERS*(N+G+J+2)+OUTPUT_FIFO_DEPTH+2` | Maximum samples in flight |

### Interface

All streams use valid/ready handshakes, and `rst_n` is a synchronous
active-low reset.

| Port group | Signals | Purpose |
| --- | --- | --- |
| Weights | `weightData[N]`, `weightValid`, `weightReady`, `weightsLoaded` | Load matrix weights row by row |
| Reload | `reloadWeights`, `reloadReady` | Replace weights once the pipeline is idle |
| Input | `inputData[N]`, `targetData`, `trainingEnable`, `inputValid`, `inputReady` | Submit a sample, with a target if training |
| Readout | `reductionWeight[N]`, `loadReductionWeights`, `reduceOutput` | Load and enable the scalar readout |
| Mode | `passThrough` | Disable ReLU on all stages |
| Output | `resultData[N]`, `resultValid`, `resultReady` | Read the result vector, or the scalar in lane 0 |

**Loading weights.** Send each layer's rows from `N-1` down to `0`, layer 0
first, for a total of `LAYERS*N` row handshakes. `weightsLoaded` goes high once
all layers are in place. Computation waits for it.

**Output format.** With `reduceOutput=0`, `resultData` carries the final
activated vector with `(LAYERS+1)*FRACTION_BITS` fractional bits. With
`reduceOutput=1`, lane 0 holds the scalar prediction, rescaled back to
`FRACTION_BITS` fractional bits.

**Result width.** Results grow with depth so that no precision is lost. Stage k
has `WIDTH + k*(WIDTH + ceil(log2 N))` bits, and the output port adds another
`ceil(log2 N)` bits. For example, WIDTH=8 and N=5 give a 44-bit output port.

**Configuration.** Change `passThrough` and `reduceOutput`, and load readout
weights, only while the accelerator is idle. Busy configuration attempts are
rejected.

[Fountain_Architecture_Specification.md](Fountain_Architecture_Specification.md)
covers the exact handshake, stall, reset and timing rules.

## Results

### Simulation

All tests pass in Questa:

- 25 Python tests of the reference models: fixed-point arithmetic, timing,
  stalls, reset, and finite-difference checks of the training update signs.
- Matrix-stage testbenches covering N=3/5/9, one and three layers, WIDTH=8/16,
  and several G/J settings, including results wider than 64 bits.
- 3,545 cycle-by-cycle RTL/reference comparisons across four configurations.
  Each one checks handshakes, results, every weight, and every output lane.

### FPGA (Cyclone V, N=5, WIDTH=8, three layers)

| Metric | Value |
| --- | --- |
| Logic | 6,296 ALMs |
| Registers | 4,882 |
| DSP blocks | 87 / 87 |
| Memory | 2,490 bits (7 RAM blocks) |
| Fmax | 71.0 MHz (100 MHz target not yet met) |

The critical path runs from the last layer's column join through the
activation and readout into the result FIFO. I/O pins are virtual, so these
numbers describe the core and are not a board-level sign-off.

## Repository layout

```text
weightStationaryVariant/
  weightStationaryProcessingElement.sv   PE: MAC, stationary weight, update logic
  weightStationarySystolicArray.sv       Center-out PE grid, routes, column joins
  weightStationaryMatrixMultiplier.sv    Layer chain, weight loader, activation, collector
  weightedVectorReduction.sv             Scalar readout
  nnAccelerator.sv                       Top level: control, FIFO, training engine
  *_tb.sv                                Testbenches
memory/signedFifo.sv                     Result FIFO
reference/                               Python functional and cycle-accurate models, tests
scripts/                                 Simulation and comparison scripts
Quartus Stuff/                           Quartus project for Cyclone V
presentation/figures/                    Architecture diagrams
```

## Documentation

- [Architecture specification](Fountain_Architecture_Specification.md):
  dataflow, timing, arithmetic, interfaces, verification and physical results
- [Learning analysis](Fountain_Three_Layer_Learning_Analysis.md): derivation
  and validation of the multi-layer training rule

## Limitations and future work

- The 100 MHz timing target is not yet met on Cyclone V.
- Only square layers with odd N are supported.
- Training handles one sample at a time; only inference is fully pipelined.
- It hasn't been shown yet how well training converges on a real workload. Do
  not start ReLU networks from all-zero weights, because closed gates cannot
  start learning.
- Possible extensions include rectangular layers, requantization between
  layers, per-lane backpressure, and tiling large layers onto fixed hardware.

An earlier "inward" design, with data entering at two corners and meeting at
the anti-diagonal, is preserved at commit `219a081`.
