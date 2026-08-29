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
            property p_weight_stable_while_waiting;
                @(posedge clk) disable iff (!rst_n)
                    weightValid && !weightReady |=>
                    weightValid && $stable(weightData[lane]);
            endproperty
            assert property (p_weight_stable_while_waiting)
                else $error("NN_UVM_WEIGHT_STABLE: weight lane %0d changed while stalled", lane);

            property p_activation_stable_while_waiting;
                @(posedge clk) disable iff (!rst_n)
                    activationValid && !activationReady |=>
                    activationValid && $stable(activationData[lane]);
            endproperty
            assert property (p_activation_stable_while_waiting)
                else $error("NN_UVM_ACT_STABLE: activation lane %0d changed while stalled", lane);

            property p_result_stable_while_waiting;
                @(posedge clk) disable iff (!rst_n)
                    resultValid && !resultReady |=>
                    resultValid && $stable(resultData[lane]) && $stable(resultLast);
            endproperty
            assert property (p_result_stable_while_waiting)
                else $error("NN_UVM_RESULT_STABLE: result lane %0d changed while stalled", lane);
        end
    endgenerate

    property p_last_requires_valid;
        @(posedge clk) disable iff (!rst_n)
            resultLast |-> resultValid;
    endproperty
    assert property (p_last_requires_valid)
        else $error("NN_UVM_RESULT_LAST: resultLast asserted without resultValid");

    // Reset is synchronous in the core's clk domain.  Checking the cycle
    // after reset is asserted avoids depending on pre-reset FIFO contents.
    property p_reset_clears_interface;
        @(posedge clk) !rst_n |=>
            !weightsLoaded && !resultValid && !reloadReady;
    endproperty
    assert property (p_reset_clears_interface)
        else $error("NN_UVM_RESET: core interface did not return to reset state");

    // The testbench intentionally uses only legal reload requests.  This
    // assertion catches accidental pulses made before the core is quiescent.
    property p_reload_is_ready;
        @(posedge clk) disable iff (!rst_n)
            reloadWeights |-> reloadReady;
    endproperty
    assert property (p_reload_is_ready)
        else $error("NN_UVM_RELOAD: reloadWeights asserted while reloadReady was low");

    // A procedural framing checker complements resultLast's local assertion:
    // each accepted row must be numbered 0..N-1, with last only on row N-1.
    int unsigned accepted_result_row;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            accepted_result_row <= 0;
        end else if (resultValid && resultReady) begin
            if (resultLast !== (accepted_result_row == N-1))
                $error("NN_UVM_FRAMING: resultLast mismatch at accepted row %0d", accepted_result_row);

            if (resultLast)
                accepted_result_row <= 0;
            else
                accepted_result_row <= accepted_result_row + 1;
        end
    end

endinterface
