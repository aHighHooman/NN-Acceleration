`timescale 1ns/1ps

// Public nnAccelerator interface used by the UVM environment.  This is a
// black-box boundary: no matrix-engine or FIFO implementation signals are
// present here.
interface nn_core_if #(
    parameter int WIDTH = 8,
    parameter int N = 3,
    parameter int TARGET_WIDTH = WIDTH,
    parameter int REDUCTION_WEIGHT_WIDTH = 8
);
    localparam int RESULT_WIDTH = 2*WIDTH + 2*$clog2(N);

    logic clk;
    logic rst_n;

    logic signed [WIDTH-1:0] weightData [N];
    logic weightValid;
    logic weightReady;

    logic signed [WIDTH-1:0] inputData [N];
    logic signed [TARGET_WIDTH-1:0] targetData;
    logic trainingEnable;
    logic inputValid;
    logic inputReady;

    logic signed [REDUCTION_WEIGHT_WIDTH-1:0] reductionWeight [N];
    logic loadReductionWeights;
    logic passThrough;
    logic reduceOutput;

    logic signed [RESULT_WIDTH-1:0] resultData [N];
    logic resultValid;
    logic resultReady;

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
        input inputData, targetData, trainingEnable;
        input inputValid, inputReady;
        input reductionWeight, loadReductionWeights, passThrough, reduceOutput;
        input resultData, resultValid, resultReady;
        input weightsLoaded, reloadWeights, reloadReady;
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
                    resultValid && $stable(resultData[lane]);
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
