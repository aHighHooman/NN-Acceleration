# Latency Slash: architecture and implementation assessment

Pre-implementation analysis, September 30, 2026. The user subsequently authorized
implementation on `latency-slash`; see [the implementation report](LATENCY_SLASH_IMPLEMENTATION.md)
for the completed changes, validation, and measured FPGA results. This document
records the reasoning before implementation.

**Assessment:** the corner-fed proposal is a credible way to shorten the matrix pipeline and remove output alignment. It is a contained redesign of the compute core, but supporting the current accelerator requires coordinated changes to weight loading, learning timing, stall/drain accounting, and the timing reference models. The surrounding transaction architecture can remain.

## Decisions from the follow-up

The selected direction is faster coherent learning feedback, two-ended weight loading over the compute psum paths, and the AD MAC register as the sole array output register. There is no deliberate legacy-feedback delay and no additional AD/output pipeline stage. The implementation report now contains the measured frequency, latency, and area on `latency-slash`.

Two-ended loading is selected because reuse of the compute directions avoids an additional opposite-direction load path through BR PEs. This can simplify the local grid, but improved placement is a hypothesis: the adapter's weight storage/distribution, bottom-edge injection routes, and capture-control fanout also contribute to the fitted result. Preserve the existing host row format with an adapter as the current planning assumption; its storage and scheduling remain to be designed.

## Repository and evidence

Fetched/pruned origin, ran `git pull --ff-only` (already up to date), and created branch `latency-slash` from local commit `09a409725e348b324f5cf87b61b907c2215d1e33`. Local main already contained origin/main and was ten commits ahead. The three existing modified source files and existing untracked `PHASE6_COHESION_REVIEW.md` were preserved. That review describes an older architecture; actual current RTL takes precedence.

Parallel reviews covered current RTL, verification, and the proposed geometry/learning schedule. Findings below distinguish existing behavior from derived behavior; none establish a fitted or simulated implementation of the new core.

## The current machine

The tracked design is a single-clock accelerator with a parallel ready/valid interface. SPI and CDC have already been removed. An accepted sample consists of a vector, target, and training flag. The vector enters a one-entry input register while its target, original input signs, and training flag enter `sampleContextFifo`.

The matrix wrapper registers the vector into input skew and delays row k by k additional advances. Activations move right; partial sums move down. PE `(k,j)` multiplies at phase `k+j`. Each PE retains a signed weight, registers its activation/psum outputs, and supports a saturating one-LSB learning step. Output column j is delayed by `N-1-j` to make a complete result vector.

ReLU or pass-through and weighted scalar reduction happen before a single N-wide `resultFifo` push. Each result stores the activated vector, prediction, and reduction-weight signs. A consumer handshake pops that result and its context together. FIFO order pairs them through bubbles and stalls; this pairing does not rely on a fixed cycle delay.

Retirement launches learning. The matrix update moves across phases `k+j`, matching the computation wave. Reduction coefficients update after a separate `2N-1`-advance delay. This synchronization makes one future vector see a coherent new matrix and matching reduction coefficients. At N=3 with continuous traffic, S0's learning first affects S7.

When the result FIFO cannot accept the pending complete vector, `arrayAdvance` freezes skew, PE state, matrix updates, and reduction updates together. Bubbles still advance; output stalls hold. Reload waits for sample computation, buffered transactions, and learning tails to drain.

Sources: [matrix wrapper](weightStationaryVariant/weightStationaryMatrixMultiplier.sv), [PE](weightStationaryVariant/weightStationaryProcessingElement.sv), [array and update taps](weightStationaryVariant/weightStationarySystolicArray.sv), [accelerator and result ownership](weightStationaryVariant/nnAccelerator.sv). Useful anchors are wrapper lines 74, 119, 142, 174; PE lines 37, 47, 51; array lines 20, 88; accelerator lines 36, 99, 149, 231, 243.

## What the proposed dataflow changes

For each column, split its reduction at row `r=N-1-j`. Rows above r still send products downward. Rows below r send products upward. The anti-diagonal PE combines those two partial sums with its own product. The bottom-right activation flow also reverses, so each row receives delayed copies from both ends.

The firing phase becomes

`f(k,j) = min(k+j, 2(N-1)-(k+j))`.

For N=3:

```text
Current MAC phases      Proposed MAC phases
0 1 2                   0 1 2*
1 2 3                   1 2*1
2 3 4                   2*1 0

* = complete column result at the anti-diagonal
```

Every contribution reaches its column sink at phase `N-1`. Specifically, a top contribution at row k arrives at `(k+j)+(r-k)=N-1`; a bottom contribution arrives at `2(N-1)-(k+j)+(k-r)=N-1`. The column includes every row exactly once. The arithmetic remains signed `x·W`, with the same full-width column sums and subsequent activation/reduction.

This requires role-specific PE input selection and routing, two activation injection paths, upward BR psum links, explicit column-to-output mapping from interior PEs, and AD accumulation. Output alignment storage disappears. The result FIFO already accepts complete vectors, so that boundary needs little conceptual change.

The external activation bandwidth need not double. This repo already holds the whole input vector in a register; each element can fan out into the two delay paths. A shared delay chain with two taps is also possible because both streams carry the same sequence at different delays. The extra cost is internal registers/routing, rather than mandatory additional host or RAM reads.

The spec's count of delay words is not a resource-equivalence claim. Removed output delays carry accumulator-width values; added input delays carry narrower activation values. Shared taps and unused right-side row-zero injection can further change the count. Synthesis must establish the actual area and routing cost.

## Latency: use matching boundaries

The selected design retains the existing input register and registered skew entry, uses the AD MAC register as the result register, and retains the result FIFO. The following timeline assumes the consumer is always ready and there are no stalls. E0 is acceptance.

| Event | Current | Proposed | N=3 current → proposed |
| --- | --- | --- | --- |
| Accepted into input register/context | E0 | E0 | 0 → 0 |
| Input register enters skew | E1 | E1 | 1 → 1 |
| First corner MAC | E2 | E2 | 2 → 2 |
| Complete raw vector becomes available | E2N | E(N+1) | 6 → 4 |
| Result FIFO captures prediction/vector | E(2N+1) | E(N+2) | 7 → 5 |
| First external retirement / learning launch | E(2N+2) | E(N+3) | 8 → 6 |

The saving is `N-1` clocks, with unchanged one-vector-per-advance throughput. At N=3 this is about a 29% reduction in accept-to-result edges (7 to 5), not a doubling of throughput.

The user resolved the spec's “one output register” wording: it means the AD MAC result register itself. Do not introduce another output register or split AD computation across an additional pipeline stage. Timing optimization must preserve the selected cycle latency. Specify latency at launch, complete raw result, FIFO enqueue, and retirement rather than mixing boundary conventions.

The new unstalled minimum context capacity would be `N+3` instead of `2N+2`. Existing depths are safe to retain initially and provide more headroom. FIFO depths and the achieved clock frequency remain separate throughput considerations.

## The most consequential hidden change: learning

The spec explains inference but omits the accelerator's online SSLMS update wave. Keeping its old `k+j` update taps while BR computation runs in reverse would let a sample observe different update generations at different PEs. Moving every weight update to one global edge would also break coherence for samples already in flight.

The straightforward adaptation is to apply each update package at the same phase f as the new computation. Both corners update at phase zero, then the two fronts converge on the AD. The existing pipelined row/column direction package can support this: shorten its phase span and select each PE's tap by f. A separate learning transport architecture is not inherently necessary. Retain saturating steps and the rule that a same-edge multiply uses the pre-update weight.

Reduction timing must change with it. With the latency table above, matrix updates reach the AD at `N-1` advances after retirement; reduction coefficients commit at N advances after retirement. The additional advance is necessary because reduction occurs at the result FIFO boundary, after the AD register. Simply shrinking both update pipes to the same number of registers gives the wrong W/R visibility relationship.

For a sample accepted at edge a, retirement launches its package at `U=a+N+3`. A subsequent vector accepted at b uses PE `(k,j)` at `b+2+f`. The update applies there at `U+f`. Since a same-edge MAC sees old W, the first vector to see the update everywhere is `b=U-1=a+N+2`. Its reduction also sees the matching updated coefficients when the R delay is N.

Thus the continuous-stream feedback distance changes from `2N+1` samples to `N+2`: S0 first affects S5 rather than S7 at N=3. This can change predictions, error signs, subsequent updates, and the entire training trajectory. Identical arithmetic does not imply identical training behavior. Earlier feedback is a design choice to evaluate, not an established improvement in convergence.

The user explicitly wants reduced weight staleness. Adopt the inward wave immediately at retirement, with no artificial delay to preserve the legacy feedback distance. The selected continuous-stream visibility contract is `N+2`, and the reduction-update delay is N advances. Changed training trajectories are an intentional consequence; convergence and numerical behavior still need evaluation under the new contract.

Saved prediction and reduction-weight signs must remain in each result entry. They describe the coefficients used to form that result, even if retirement occurs after later resident updates.

## Weight loading is a separate integration choice

Current loading consumes N bottom-first rows through one pending-row register. Every consumed row shifts down all columns and immediately updates each PE's resident weight. It has neither a complete matrix buffer nor the two-ended source streams assumed in the spec.

The proposed loader instead shifts through two column-dependent chains, then captures all weights once. For N=3 the three shift slots require:

| Slot | Top ports, columns 0/1/2 | Bottom ports, columns 0/1/2 |
| --- | --- | --- |
| 0 | W20 / pad / pad | pad / pad / pad |
| 1 | W10 / W11 / pad | pad / pad / W12 |
| 2 | W00 / W01 / W02 | pad / W21 / W22 |

The last bottom entries must come from an earlier host row. At N=3, saving W21 and W22 suffices for the stated continuous load stream. A complete matrix buffer is therefore not mandatory.

For larger N there is also a causal timing issue: at N=4, shift slot 1 needs W1,3 from below, but the existing reverse-row input delivers row 1 at slot 2. A direct unbuffered N-slot load cannot supply it. Preserving the current host interface requires buffering/lookahead and an adjusted load schedule, or collecting the matrix before shifting. Source bubbles make loading advance control part of that adapter. The permutation is free only when the supplying memory or host can already present it.

The analysis identified three viable choices:

- Preserve the host's bottom-first row format and build a reorder adapter for literal two-ended loading. Full-matrix staging is simple; structured lane buffers can be smaller.
- Change the host format to provide the permuted streams. This changes the public loading contract and its drivers/monitors.
- Preserve downward loading through every PE, while computation in BR flows upward. Select the above neighbor in load mode and the below neighbor in compute mode; AD forwards its load value down during loading. Keep the existing direct weight capture on every load advance. This avoids input reordering and preserves the host contract, at the cost of load-mode routing/muxing and a deliberate departure from the spec's mux-free loading objective.

The user selected the first direction: reuse the two-ended compute paths for loading, with the existing host row format preserved as the planning assumption. The third option remains a documented tradeoff, not the recommended implementation. The argument against two-ended loading is that its local routing simplicity is purchased with upstream reorder storage and distribution. Total placement efficiency cannot be inferred from PE mux counts alone. Load scheduling is outside the steady-state inference path because compute and loading do not overlap.

The selected two-ended loader's global capture edge needs explicit wrapper advance/readiness logic. Existing `weightsLoaded` rises after consuming the final pending row, whereas the new scheme is not ready until capture completes. Gate the AD lower psum during loading, use known padding in verification, suppress stale valid state, and test both short/empty up chains and reloads after learning. Weight capture is a loading event, not an additional compute/output pipeline stage.

## Stalls, timing, and verification

The proposed module interface omits `advance`. Integration needs it, or an equivalent hold mechanism. Its N-deep valid pipeline counts advancing compute slots, not unconditional clocks. Both skew paths, TL/BR/AD registers, result-valid state, and both learning paths must hold together. Reload/busy accounting must cover every remaining sample and update. Keeping the ordered result/context FIFOs makes transaction pairing reusable.

The spec's timing-headroom premise is stale. README documents a September 24 fit at 80.28 MHz with −2.457 ns setup slack. The local generated reports instead show 80.4 MHz and −2.438 ns; they are a historical fit, not a fresh build of this analysis checkout. Both miss the actual 100 MHz constraint in `Quartus Stuff/NN_Acceleration.sdc`. Neither supports assuming 110 MHz of established headroom.

The AD multiply plus two adds is a timing candidate, but it is not yet established as the new worst path. Existing combinational weighted reduction and control paths also remain. Interior result routing and counter-propagating links may affect placement. Compare fitted latency in nanoseconds and sustained samples/second, not just cycle count. If using carry-save arithmetic, compress all three operands (upper sum, lower sum, product) before the final carry-propagate add. Optimize arithmetic mapping and routing within the selected register boundaries; adding a pipeline stage is outside the user's chosen design. Evaluate the achieved clock after implementation and fitting rather than rejecting the experiment from the existing timing result.

The verification system already offers complementary checks:

- `reference/arithmetic.py` and its hand-calculated tests define widths, signed math, activation, reduction, and saturation. These mostly survive.
- `FunctionalReference` describes arithmetic and continuous-stream feedback visibility in closed form. Its `2N+1` assumption must become the chosen learning contract.
- `CycleReference` records the old W seen by each PE at its scheduled multiply, then forms the result from those observations. Its firing, loading, completion, update, and drain schedules must change independently of the closed-form model.
- The RTL trace bridge compares post-edge W/R, loading/input state, both transaction FIFOs, complete raw result handshakes, and retirements. Preserve these semantic comparisons while replacing probes into old alignment/horizontal/vertical structures.
- UVM exercises public-port ordering, backpressure, reset/reload, and known-weight inference. Its training checks cover order/liveness rather than exact learned predictions; a green UVM run alone cannot establish correct new learning timing.

The high-value additions are matching vector tags at each AD merge during streams/bubbles; unique asymmetric signed matrices for placement; both compute halves and learning waves held during a real output-full stall; coherent W/R update visibility; capture/reload/reset boundaries; and defined-edge latency checks. Extend parameter coverage to N=5/8: directed RTL currently covers N=2/3/4 and integrated traces are fixed at N=3. Retain an independent conventional-core baseline when eventually comparing both cores.

## Recommended next step and validation performed

The follow-up settles the output-register boundary, shorter learning feedback delay, and two-ended loading direction. The remaining loading work is to design the adapter storage/schedule under the existing host interface. When implementation is requested, establish a standalone inward core and compare it with the conventional core using the same arithmetic stimulus. Integrate stall handling and coherent updates before accepting full-accelerator learning results. Fit it and compare clock frequency, latency in nanoseconds, throughput, and area with a baseline under matching constraints. These are sequencing recommendations, not work performed now.

For this analysis, all **76 existing Python tests passed**. The standard regression compiled the core RTL and directed benches, then simulation failed before the first test because the installed Questa license reported an invalid host. The RTL trace and UVM simulations did not run; no fresh synthesis was performed. Log: `build/latency-slash-baseline.log`.

Independent arithmetic checks verified the column arrival-time identities and final two-ended weight placement for N=2,3,4,5,8. They support the schedule/permutation derivation, but are not RTL simulation, streaming verification, or physical timing evidence.

The strongest reason to pursue the change is lower latency with naturally aligned matrix outputs. The work remains concentrated in a few modules and their temporal models. Its main hidden cost is preserving the existing accelerator's learning and loading contracts while changing the geometry that currently makes them simple.
