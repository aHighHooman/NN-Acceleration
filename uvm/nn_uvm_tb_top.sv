`timescale 1ns/1ps

module nn_uvm_tb_top;
    import uvm_pkg::*;
    import nn_uvm_pkg::*;
    `include "uvm_macros.svh"

    nn_core_if #(WIDTH, N, TARGET_WIDTH, REDUCTION_WEIGHT_WIDTH) bus();

    nnAccelerator #(
        .WIDTH(WIDTH),
        .N(N),
        .FRACTION_BITS(FRACTION_BITS),
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
        .inputData(bus.inputData),
        .targetData(bus.targetData),
        .trainingEnable(bus.trainingEnable),
        .inputValid(bus.inputValid),
        .inputReady(bus.inputReady),
        .reductionWeight(bus.reductionWeight),
        .loadReductionWeights(bus.loadReductionWeights),
        .reduceOutput(bus.reduceOutput),
        .resultData(bus.resultData),
        .resultValid(bus.resultValid),
        .resultReady(bus.resultReady),
        .weightsLoaded(bus.weightsLoaded),
        .reloadWeights(bus.reloadWeights),
        .reloadReady(bus.reloadReady),
        .passThrough(bus.passThrough)
    );

    initial begin
        bus.clk = 1'b0;
        forever #5 bus.clk = ~bus.clk;
    end

    initial begin
        bus.rst_n = 1'b0;
        bus.weightValid = 1'b0;
        bus.inputValid = 1'b0;
        bus.targetData = '0;
        bus.trainingEnable = 1'b0;
        bus.loadReductionWeights = 1'b0;
        bus.passThrough = 1'b1;
        bus.reduceOutput = 1'b0;
        bus.resultReady = 1'b0;
        bus.reloadWeights = 1'b0;
        for (int lane = 0; lane < N; lane++) begin
            bus.weightData[lane] = '0;
            bus.inputData[lane] = '0;
            bus.reductionWeight[lane] = '0;
        end

        uvm_config_db #(virtual nn_core_if #(WIDTH, N, TARGET_WIDTH,
                                             REDUCTION_WEIGHT_WIDTH))::set(
            null, "uvm_test_top.env*", "vif", bus);
        run_test();
    end

    initial begin
        #2_000_000;
        `uvm_fatal("TB_TIMEOUT", "core-level UVM test exceeded 2 ms")
    end

endmodule
