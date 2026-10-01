# Latency Slash implementation

Implemented on `latency-slash`, September 30, 2026. The branch uses the corner-fed,
anti-diagonal-output weight-stationary array. The external `nnAccelerator`
interface, arithmetic widths, fixed-point scaling, and default FIFO depths are
preserved. Faster learning feedback is intentional. The AD MAC result register
is the sole array output register; no additional output stage was introduced.

## What changed

The upper-left triangle sends activations right and partial sums down. The
lower-right triangle sends activations left and partial sums up. Each
anti-diagonal PE combines both partial sums with its own product, using a
carry-save compressor and one carry-propagate addition into its existing result
register. Its registered result directly supplies the corresponding output
column. Every column becomes valid together, so the old output-alignment chains
are gone. Each row shares one activation-delay sequence between its two edge
taps, retaining the existing registered input/skew boundary.

Matrix learning follows the same inward phases as the samples. A package starts
at both corners on retirement, reaches the AD after N−1 more advancing edges,
and commits its reduction-weight update one advancing edge later. Same-edge MACs
use the old resident weight. This preserves coherent W/R generations while
reducing feedback delay. Stalls freeze data, validity, and learning together;
bubbles continue to advance the pipeline. Reload remains blocked until compute,
retirement, and the learning tail have drained.

The two-ended loader reuses the vertical compute paths. The host still supplies
N bottom-first rows, with arbitrary gaps between accepted rows. An N×N staging
bank reorders them into padded top and bottom streams over N shift edges. One
following capture edge latches all resident weights from their pre-edge psum
registers and asserts `weightsLoaded`. Padded streams flush old activations before
weight data reaches each PE, so load-mode activation gating is needed only at the
array edges. The AD suppresses its lower partial sum during loading.

An accepted reload takes priority over a simultaneous input offer: `inputReady`
falls on that boundary, and the producer must hold/reoffer the sample for the
next loaded epoch. This also fixes the inherited case where a sample could be
accepted into the input register on the same edge that unloaded the weights.

## Timing and practical effects

Edges below count from input acceptance at E0, with no stalls:

| Boundary | Conventional | Inward | N = 3, before → after |
| --- | ---: | ---: | ---: |
| Complete registered raw matrix vector | E(2N) | E(N+1) | E6 → E4 |
| Activated/reduced result enters FIFO | E(2N+1) | E(N+2) | E7 → E5 |
| Earliest result retirement | E(2N+2) | E(N+3) | E8 → E6 |
| First sample affected by a source sample's update, continuous traffic | source + 2N+1 | source + N+2 | S0 affects S7 → S5 |

Both designs can accept one sample per clock when running unstalled. The latency
saving does not multiply throughput at a fixed clock. Holding `resultReady`
delays retirement and therefore learning; bubbles and stalls invalidate the
simple sample-index feedback formula, while the coherent generation property
still holds. Faster feedback changes the training trajectory, so conventional
training outputs are not the expected numerical sequence for the new design.

Loading a new matrix takes N+1 clocks after the final host row is accepted,
versus one pending-row clock in the conventional loader. At N = 3 this is four
clocks versus one. The staging bank and stream-selection logic are a real cost
of retaining the host format while using both ends of the compute paths. This
cost is paid at initial load/reload, not between ordinary vectors. Reducing the
default FIFO capacities was left outside this change.

## Fitted comparison

Fresh full Quartus Prime 25.1std Lite builds used Cyclone V `5CSEMA5F31C6`, N = 3,
WIDTH = 16, the same 100 MHz SDC, virtual pins, fitter settings, and no added
pipeline stage. The baseline uses `09a409725e348b324f5cf87b61b907c2215d1e33`
with the pre-existing local removal of the redundant result-FIFO-empty reload
qualifier. The final build's six production source files match the current
working-tree files by SHA-256.

| Metric | Conventional baseline | Final inward implementation |
| --- | ---: | ---: |
| ALMs | 1,381 | 1,599 (+15.8%) |
| Registers | 1,168 | 1,087 (−6.9%) |
| DSP blocks | 15 | 15 |
| RAM blocks / memory bits | 5 / 1,048 | 5 / 1,048 |
| Lowest reported Fmax across slow corners | 80.40 MHz | 81.16 MHz |
| Worst setup slack at 100 MHz | −2.438 ns | −2.322 ns |
| Minimum hold slack across reported corners | +0.123 ns | +0.163 ns |

At a common 50 MHz, acceptance to result formation is 140 ns versus 100 ns.
Using each fit's reported Fmax gives approximately 87.1 ns versus 61.6 ns.
The small frequency increase is the result of this single matched fit, not
evidence that the topology always improves Fmax. The measured tradeoff here is
lower latency and fewer registers at higher ALM usage, with unchanged DSP/RAM
usage. The loader bank and new selection/merge logic contribute area; the fit
does not isolate each contribution.

The worst core path still crosses activation/reduction into the result FIFO.
At the slow 85°C corner, the final core's worst setup slack is −2.307 ns;
the worst path ending at a PE psum register is −0.244 ns, beginning at loader
`captureWeights`. Thus the AD merge is not the overall limiting path in this
fit. Neither architecture meets the 100 MHz target. These are core
characterization fits with an unassigned clock pin and virtual data pins;
board integration and interface timing remain separate work.

The complete comparison artifacts are locally available at:

- `build/fpga-baseline/output_files/NN_Acceleration.fit.summary`, `.sta.summary`, and `.sta.rpt`.
- `build/fpga-final/output_files/NN_Acceleration.fit.summary`, `.sta.summary`, and `.sta.rpt`.
- `build/fpga-final/output_files/critical-setup.rpt` and `array-setup.rpt`.

These generated artifacts are ignored by Git. The normal Quartus project in
`Quartus Stuff` points at the modified production sources; its historical
output files were not overwritten by the comparison builds.

## Verification completed

The cycle model follows the inward wiring and loading state machine. The
functional model independently uses the new closed-form feedback schedule.
The RTL benches retain independent signed dot-product scoreboards rather than
using the cycle model as the arithmetic oracle.

- **77 Python unit tests passed**, including loading/capture, reset, reload,
  latency, continuous learning, bubbles/backpressure, and coherent generations.
- **749 matrix-engine samples passed for each architecture**, across N = 2, 3,
  4, 5, and 8. Both cores ran the identical seeded random vectors and weights
  against the same independent long-integer scoreboard. Coverage includes
  signed extrema, streaming, input bubbles, blocked-output holds, and reload.
  New-loader atomic capture and reload-priority checks run on the inward core.
- **All PE roles and all five array sizes passed the inward learning bench**:
  signed/zero directions, saturation, old-weight same-edge MACs, overlapping
  updates, stalls, and coherent sample generations.
- **502 integrated RTL post-edge snapshots matched** across ten scenarios,
  with **332 functional-reference comparisons**. This includes saved reduction
  signs under backpressure, reset during a learning tail, reload, and context
  accounting. Weighted reduction tests and unsupported-N=1 diagnostics passed.
- **Questa compilation passed**, including the UVM environment. UVM simulation
  could not run because the installed license did not validate for this host.
  The executable directed/integrated results above were obtained with Verilator.
- **Full final Quartus compilation and fitting succeeded**; timing results and
  the remaining 100 MHz setup violation are reported above.

Run the completed simulator regression, including conventional replay, with:

```powershell
pwsh -File scripts/run_verilator.ps1 -ConventionalBaseline -BuildJobs 4
```

The script also runs the Python suite and integrated reference comparison.
See README for the portable Windows tool locations and environment overrides.
Logs are under `build/verilator/`; the full captured run is
`build/verilator-final-regression.log`. `scripts/run_modelsim.ps1` remains the
Questa/UVM path when a working simulator license is available.

README and the affected architecture, loader, timing, storage, stall, lifecycle,
and verification diagrams now describe the implemented topology. The earlier
[analysis](LATENCY_SLASH_ANALYSIS.md) is retained as historical design reasoning.
No commit or push was made; implementation and verification changes remain
reviewable in the working tree on `latency-slash`. Pre-existing local edits and
`PHASE6_COHESION_REVIEW.md` were preserved.
