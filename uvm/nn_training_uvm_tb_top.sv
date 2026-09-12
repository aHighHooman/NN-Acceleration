`timescale 1ns/1ps

module nn_training_uvm_tb_top;
    import uvm_pkg::*;
    import nn_training_uvm_pkg::*;
    `include "uvm_macros.svh"

    localparam int TARGET_WIDTH = WIDTH;
    localparam int REDUCTION_WEIGHT_WIDTH = 8;
    localparam int UPDATE_STAGES = 2*N - 1;

    nn_training_if #(
        WIDTH, N, TARGET_WIDTH, REDUCTION_WEIGHT_WIDTH
    ) bus();

    nnAccelerator #(
        .WIDTH(WIDTH),
        .N(N),
        .FRACTION_BITS(4),
        .TARGET_WIDTH(TARGET_WIDTH),
        .REDUCTION_WEIGHT_WIDTH(REDUCTION_WEIGHT_WIDTH),
        .INPUT_FIFO_DEPTH(2*N),
        .OUTPUT_FIFO_DEPTH(2*N)
    ) dut (
        .clk(bus.clk),
        .rst_n(bus.rst_n),
        .weightData(bus.weightData),
        .weightValid(bus.weightValid),
        .weightReady(bus.weightReady),
        .activationData(bus.activationData),
        .targetData(bus.targetData),
        .trainingEnable(bus.trainingEnable),
        .activationValid(bus.activationValid),
        .activationReady(bus.activationReady),
        .reductionWeight(bus.reductionWeight),
        .loadReductionWeights(bus.loadReductionWeights),
        .reduceOutput(bus.reduceOutput),
        .resultData(bus.resultData),
        .resultTargetData(bus.resultTargetData),
        .learningDirection(bus.learningDirection),
        .rowDirection(bus.rowDirection),
        .columnDirection(bus.columnDirection),
        .matrixUpdateValid(bus.matrixUpdateValid),
        .resultValid(bus.resultValid),
        .resultReady(bus.resultReady),
        .resultLast(bus.resultLast),
        .weightsLoaded(bus.weightsLoaded),
        .reloadWeights(bus.reloadWeights),
        .reloadReady(bus.reloadReady),
        .passThrough(bus.passThrough)
    );

    assign bus.samplePush = dut.samplePush;
    assign bus.samplePop = dut.samplePop;
    assign bus.bufferedTrainingEnable = dut.trainingEnableHead;
    assign bus.matrixUpdateAccepted = dut.matrixUpdateAccepted;
    assign bus.matrixUpdateComplete = dut.matrixUpdateComplete;
    assign bus.arrayAdvance = dut.matrixEngine.arrayAdvance;
    assign bus.reductionUpdateEmpty = dut.reductionUpdateEmpty;

    generate
        for (genvar stage = 0; stage < UPDATE_STAGES; stage++) begin : update_debug
            assign bus.updateStageValid[stage] =
                dut.matrixEngine.systolicArr.updateValidPipe[stage];
        end
        for (genvar row = 0; row < N; row++) begin : matrix_row_debug
            for (genvar col = 0; col < N; col++) begin : matrix_col_debug
                assign bus.residentMatrixWeight[row][col] =
                    dut.matrixEngine.systolicArr.row_loop[row].col_loop[col].mb.weightReg;
            end
        end
        for (genvar lane = 0; lane < N; lane++) begin : reduction_debug
            assign bus.residentReductionWeight[lane] =
                dut.residentReductionWeight[lane];
        end
    endgenerate

    initial begin
        bus.clk = 1'b0;
        forever #5 bus.clk = ~bus.clk;
    end

    initial begin
        bus.rst_n = 1'b0;
        bus.weightValid = 1'b0;
        bus.activationValid = 1'b0;
        bus.trainingEnable = 1'b0;
        bus.resultReady = 1'b0;
        bus.loadReductionWeights = 1'b0;
        bus.reduceOutput = 1'b1;
        bus.reloadWeights = 1'b0;
        bus.passThrough = 1'b1;
        bus.targetData = '0;
        for (int lane = 0; lane < N; lane++) begin
            bus.weightData[lane] = '0;
            bus.activationData[lane] = '0;
            bus.reductionWeight[lane] = '0;
        end

        uvm_config_db #(virtual nn_training_if #(
            WIDTH, N, TARGET_WIDTH, REDUCTION_WEIGHT_WIDTH
        ))::set(null, "uvm_test_top", "training_vif", bus);
        run_test();
    end

    initial begin
        #2_000_000;
        `uvm_fatal("TB_TIMEOUT", "Phase 5F training UVM test exceeded 2 ms")
    end

endmodule
