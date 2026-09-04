`timescale 1ns/1ps

// Core-level ready/valid interface used by the UVM environment.  The
// interface deliberately stops at the matrix core; no SPI signals belong in
// this verification layer.
interface nn_core_if #(
    parameter int WIDTH = 8,
    parameter int N = 3
);
    localparam int RESULT_WIDTH = 2*WIDTH + $clog2(N);

    logic clk;
    logic rst_n;

    logic signed [WIDTH-1:0] weightData [N];
    logic weightValid;
    logic weightReady;

    logic signed [WIDTH-1:0] activationData [N];
    logic activationValid;
    logic activationReady;

    logic signed [RESULT_WIDTH-1:0] resultData [N];
    logic resultValid;
    logic resultReady;
    logic passThrough;
    logic resultLast;

    logic weightsLoaded;
    logic reloadWeights;
    logic reloadReady;

    // Input clocking is sampled in the clocking block's input region, before
    // the DUT's nonblocking assignments update FIFO/state registers.  This
    // makes passive monitors observe the same transfer represented by
    // valid && ready at the active clock edge.
    clocking monitor_cb @(posedge clk);
        default input #1step;
        input rst_n;
        input weightData, weightValid, weightReady;
        input activationData, activationValid, activationReady;
        input resultData, resultValid, resultReady, resultLast;
        input passThrough, weightsLoaded, reloadWeights, reloadReady;
    endclocking

    // A small set of cycle-level properties is kept here because these are
    // easier to express against sampled signals than in the end-to-end
    // scoreboard.  Mathematical correctness is intentionally not checked by
    // these assertions.
    generate
        for (genvar lane = 0; lane < N; lane++) begin : protocol_assertions
            property p_result_stable_while_waiting;
                @(posedge clk) disable iff (!rst_n)
                    resultValid && !resultReady |=>
                    resultValid && $stable(resultData[lane]) && $stable(resultLast);
            endproperty
            assert property (p_result_stable_while_waiting)
                else $error("NN_UVM_RESULT_STABLE: result lane %0d changed while stalled", lane);
        end
    endgenerate

    // Reset is synchronous in the core's clk domain.  Checking the cycle
    // after reset is asserted avoids depending on pre-reset FIFO contents.
    property p_reset_clears_interface;
        @(posedge clk) !rst_n |=>
            !weightsLoaded && !resultValid && !reloadReady;
    endproperty
    assert property (p_reset_clears_interface)
        else $error("NN_UVM_RESET: core interface did not return to reset state");

endinterface
