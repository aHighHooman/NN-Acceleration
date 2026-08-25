# UVM starter and exercises

This directory is intentionally a **partial** UVM implementation.  It is small
enough to trace in a debugger, but its component boundaries are the same ones
you can extend for the asynchronous SPI stress work in GitHub issue #3.

## What is implemented

The starter is fixed at `N=2`, `WIDTH=8` and contains:

- `nn_matrix_item`: one complete matrix operation
- `nn_identity_sequence`: signed identity/pass-through smoke stimulus
- `nn_relu_sequence`: mixed-sign ReLU stimulus
- `nn_matrix_sequencer`: the typed sequencer
- `nn_spi_driver`: reset, weight load, activation send, result polling, reload
- `nn_result_monitor`: passive reconstruction of parallel MISO result rows
- `nn_scoreboard`: an independent `A x B` reference model and comparison
- `nn_matrix_stats`: an analysis subscriber with simple mode/sign counters
- three selectable tests, including a two-case starter regression

The data path is:

```text
sequence -> sequencer -> driver -> SPI pins -> DUT
                           |                  |
                           | expected        | MISO pins
                           v                 v
                        scoreboard <- result monitor
                           ^
                           |
                          stats
```

The existing directed testbenches remain valuable fast smoke tests.  The UVM
runner is separate and does not replace them.

The runner links Questa's precompiled `mtiUvm` library. In the current Quartus
25.1 installation this is UVM 1.1d, whose component APIs used by the starter
are the same core methodology you will encounter in UVM 1.2.

## Run it

From the repository root:

```powershell
pwsh -File scripts/run_uvm.ps1
pwsh -File scripts/run_uvm.ps1 -TestName nn_uvm_smoke_test
pwsh -File scripts/run_uvm.ps1 -TestName nn_uvm_relu_test
```

The script deliberately prefers Questa 25.1 over an older ModelSim found on
`PATH`.  On another machine, set `NN_ACCEL_QUESTA_BIN` to the folder containing
`vlog.exe` and `vsim.exe`.

## Suggested exercises

Do these in order; each step adds one UVM idea without requiring a rewrite.

### 1. Consume the clock-ratio knob

`nn_matrix_item` already has an `sclk_half_period_ns` extension field, and the
interface clock generator reads `bus.sclk_half_period_ns`. In
`nn_spi_driver::run_phase`, assign the item field to the interface before
driving a transaction:

```systemverilog
vif.sclk_half_period_ns = req.sclk_half_period_ns;
```

After confirming that your Questa license includes the `svverification`
feature, mark the relevant item fields `rand`, add constraints, and create
`nn_random_clock_sequence`. Send 10 randomized items and print the seed from
the simulator log when a failure occurs. Initially keep matrix values in a
small range such as `[-8:7]` so failures are easy to inspect.

Coverage goal: bins for fast/equal/slow `sclk` relative to `clk`, crossed with
ReLU/pass-through.

The FPGA Starter license installed with this repository's Quartus setup can
run the directed UVM components, but its current license does not include
constrained randomization or covergroups. Keep `nn_matrix_stats` until you have
that feature, then replace it with a `uvm_subscriber` containing a covergroup.

### 2. Add input bubbles

The transaction also contains `input_bubbles`, which the starter driver leaves
unused.  Insert that many complete SCLK cycles with CS high between selected
weight or activation vectors.  Confirm that the same scoreboard passes.

Coverage goal: zero, one, and multiple bubbles crossed with weight/activation
traffic.

### 3. Move prediction to a passive input monitor

The starter sends the expected matrix directly from the driver.  That is easy
to understand but it cannot detect a bug in the driver itself.

Create an `nn_input_monitor` that:

1. Observes `weightCs_n/weightMosi` and
   `activationCs_n/activationMosi`.
2. Reconstructs each lane MSB-first.
3. Discards a word when CS rises before `WIDTH` bits.
4. Groups `N` weight rows and `N` activation rows into an `nn_matrix_item`.
5. Publishes the item to the scoreboard's expected FIFO.

After it works, remove `expected_ap` from the driver.  This is a useful lesson
in keeping UVM checking independent from UVM stimulus.

### 4. Partial and extra-clock frames

Add a lower-level `nn_spi_frame_item` with fields like:

```systemverilog
rand bit is_weight;
rand int unsigned bits_to_send;
rand int unsigned extra_clocks;
rand data_t lane_data[N];
```

Constrain `bits_to_send` to `1..WIDTH`, then write directed sequences for:

- CS released after `WIDTH/2` bits
- exactly `WIDTH` bits
- one and several extra clocks

Keep the matrix-level sequence as a virtual sequence layered over these frame
sequences.  Do not teach the scoreboard to ignore malformed output; instead,
make the input monitor model the documented abort semantics.

### 5. Reset injection

Add a reset agent or a virtual sequencer so reset stimulus and SPI stimulus do
not both write `rst_n`.  Inject reset in these phases:

- idle
- halfway through a weight word
- between weight rows
- halfway through an activation word
- while an output row is pending or paused

Flush incomplete predictions and observed rows on reset.  Add a post-reset
identity transaction to prove recovery, rather than considering “no output” a
pass.

### 6. Per-lane skew

The starter driver asserts every CS lane together.  Add per-lane start delays
and send the same vector with skewed CS/MOSI timing.  Decide whether the driver
should use a `fork` per lane or separate lane agents.  The latter is more UVM
infrastructure; the former is sufficient for this fixed parallel interface.

Check that the DUT forwards a vector only after all lanes hold complete words.

Coverage goal: identity of first/last lane, maximum skew, traffic type, and
whether back-to-back frames were used.

### 7. Protocol assertions

Put interface-level SVA in a separate checker module and bind it to the top.
Good first properties are:

- MISO and `misoValid` are zero/low when result CS is high.
- A result lane does not lose validity during an unpaused word.
- A complete input is not accepted twice.
- Reset clears any partial input or output frame.

Assertions complement UVM: UVM checks transaction outcomes, while SVA checks
cycle-level protocol rules.

### 8. Parameter regression

Finally, parameterize the interface/top and compile separate simulations for
`N=2`, `3`, and `4`.  Treat parameters as build-time configurations, not as
runtime sequence randomization.  Update the PowerShell runner to invoke each
configuration and retain a separate log for each.

## A practical completion target

A strong completion of issue #3 would include randomized clock ratios, reset
phase, malformed frames, lane skew, and back-to-back traffic; passive prediction;
protocol assertions; a coverage report; and a deterministic CI seed plus a way
to rerun any failing seed locally.
