`timescale 1ns/1ps

// Small FIFOs exercise context replacement on completion, delayed consumption,
// full-output holds, and target sign extension independently of the Python model.
module nnAcceleratorFeedback_case #(
    parameter int N = 3,
    parameter int TARGET_WIDTH = 12
) (input logic clk, output logic done);
    localparam int WIDTH = 8;
    localparam int RESULT_WIDTH = 2*WIDTH + 2*$clog2(N);
    logic rst_n, weightValid, weightReady, inputValid, inputReady;
    logic trainingEnable, loadReductionWeights, resultValid, resultReady;
    logic weightsLoaded, reloadWeights, reloadReady;
    logic signed [WIDTH-1:0] weightData[N], inputData[N], reductionWeight[N];
    logic signed [TARGET_WIDTH-1:0] targetData;
    logic signed [RESULT_WIDTH-1:0] resultData[N];
    integer completed, consumed;
    integer expectedPrediction[3];
    logic signed [WIDTH-1:0] observedW[N][N];
    for (genvar row = 0; row < N; row++) begin : observe_row
        for (genvar col = 0; col < N; col++) begin : observe_col
            assign observedW[row][col] =
                dut.matrixEngine.systolicArray.row_loop[row].col_loop[col].pe.weightReg;
        end
    end
    nnAccelerator #(.WIDTH(WIDTH), .N(N), .FRACTION_BITS(0),
        .TARGET_WIDTH(TARGET_WIDTH), .REDUCTION_WEIGHT_WIDTH(WIDTH),
        .IN_FLIGHT_DEPTH(1), .OUTPUT_FIFO_DEPTH(1)) dut (
        .clk(clk), .rst_n(rst_n), .weightData(weightData), .weightValid(weightValid),
        .weightReady(weightReady), .inputData(inputData), .targetData(targetData),
        .trainingEnable(trainingEnable), .inputValid(inputValid), .inputReady(inputReady),
        .reductionWeight(reductionWeight), .loadReductionWeights(loadReductionWeights),
        .reduceOutput(1'b1), .resultData(resultData), .resultValid(resultValid),
        .resultReady(resultReady), .weightsLoaded(weightsLoaded),
        .reloadWeights(reloadWeights), .reloadReady(reloadReady), .passThrough(1'b0)
    );

    always @(posedge clk) begin
        if (!rst_n) begin completed = 0; consumed = 0; end
        else begin
            if (dut.matrixResultPush) completed++;
            if (resultValid && resultReady) begin
                if (consumed >= 3 || resultData[0] !== expectedPrediction[consumed])
                    $fatal(1, "N=%0d output %0d prediction mismatch", N, consumed);
                for (int lane = 1; lane < N; lane++)
                    if (resultData[lane] !== 0) $fatal(1, "nonzero reduced tail");
                consumed++;
            end
        end
    end

    task automatic check_weights(input int w, input int r);
        for (int row = 0; row < N; row++) begin
            for (int col = 0; col < N; col++)
                if (observedW[row][col] !== w)
                    $fatal(1, "N=%0d W[%0d][%0d]=%0d expected %0d", N, row, col,
                           observedW[row][col], w);
            if (dut.residentReductionWeight[row] !== r)
                $fatal(1, "N=%0d R[%0d]=%0d expected %0d", N, row,
                       dut.residentReductionWeight[row], r);
        end
    endtask

    initial begin
        done = 0; rst_n = 0; weightValid = 0; inputValid = 0;
        trainingEnable = 1; loadReductionWeights = 0; resultReady = 0; reloadWeights = 0;
        targetData = (1 << (TARGET_WIDTH-1)) - 1;
        for (int lane = 0; lane < N; lane++) begin
            weightData[lane] = 1; inputData[lane] = 1; reductionWeight[lane] = 64;
        end
        expectedPrediction[0] = (N*N*64) >>> 7;
        expectedPrediction[1] = (2*N*N*65) >>> 7;
        expectedPrediction[2] = (3*N*N*66) >>> 7;
        repeat (2) @(negedge clk);
        rst_n = 1; weightValid = 1; loadReductionWeights = 1;
        repeat (N) begin
            @(posedge clk); #1ps;
            @(negedge clk); loadReductionWeights = 0;
        end
        weightValid = 0;
        wait (weightsLoaded); @(negedge clk);
        inputValid = 1;
        // One-entry context fills; the second input replaces it on the first
        // result's completion edge while the output consumer remains held.
        @(posedge clk); #1ps;
        if (inputReady) $fatal(1, "context did not fill after acceptance");
        repeat (N+1) begin
            @(posedge clk); #1ps;
            if (completed != 0 || observedW[0][0] != 1)
                $fatal(1, "N=%0d early completion/update", N);
        end
        @(posedge clk); #1ps;
        if (completed != 1 || consumed != 0 || observedW[0][0] != 2 ||
            !dut.matrixEngine.inputVectorValid || dut.sampleContextFifo.values != 1)
            $fatal(1, "N=%0d formation/replacement did not update without consumption", N);
        @(negedge clk); inputValid = 0;
        repeat (N+4) @(negedge clk);
        check_weights(2, 65);
        if (completed != 1 || consumed != 0 || dut.arrayAdvance)
            $fatal(1, "N=%0d pending second result did not stall at full output", N);
        reloadWeights = 1;
        repeat (3) begin
            @(posedge clk); #1ps;
            check_weights(2, 65);
            if (reloadReady || !weightsLoaded || completed != 1)
                $fatal(1, "N=%0d blocked output allowed reload/duplicate learning", N);
        end
        @(negedge clk); reloadWeights = 0; resultReady = 1;
        repeat (N+5) @(negedge clk);
        check_weights(3, 66);
        if (completed != 2 || consumed != 2 || !reloadReady || !dut.sampleContextEmpty)
            $fatal(1, "N=%0d outputs/updates failed to drain", N);
        // A negative target wider than the prediction's original input width
        // must reverse the update. Consumption must not update a second time.
        targetData = -(1 << (TARGET_WIDTH-1)); inputValid = 1;
        @(posedge clk); @(negedge clk); inputValid = 0;
        repeat (2*N+6) @(negedge clk);
        check_weights(2, 65);
        if (completed != 3 || consumed != 3 || !reloadReady)
            $fatal(1, "N=%0d signed-target training failed", N);
        $display("PASS: N=%0d one-entry FIFOs, formation feedback, replacement, full hold, reload, signed targets", N);
        done = 1;
    end
endmodule

module nnAcceleratorFeedback_tb;
    logic clk = 0;
    always #5 clk = ~clk;
    logic done[5];
    nnAcceleratorFeedback_case #(.N(2)) c2(clk, done[0]);
    nnAcceleratorFeedback_case #(.N(3)) c3(clk, done[1]);
    nnAcceleratorFeedback_case #(.N(4)) c4(clk, done[2]);
    nnAcceleratorFeedback_case #(.N(5)) c5(clk, done[3]);
    nnAcceleratorFeedback_case #(.N(8)) c8(clk, done[4]);
    initial begin
        wait (done[0] && done[1] && done[2] && done[3] && done[4]);
        $display("PASS: integrated formation-feedback suites completed"); $finish;
    end
    initial begin #100000; $fatal(1, "feedback regression timed out"); end
endmodule
