`timescale 1ns/1ps

module nn_uvm_tb_top;
    import uvm_pkg::*;
    import nn_uvm_pkg::*;
    `include "uvm_macros.svh"

    nn_core_if #(WIDTH, N) bus();
    logic signed [1:0] noRowDirection[N], noColumnDirection[N];

    matrixMultiplierWeightStationary #(
        .WIDTH(WIDTH),
        .N(N),
        .INPUT_FIFO_DEPTH(2*N),
        .OUTPUT_FIFO_DEPTH(2*N)
    ) dut (
        .clk(bus.clk),
        .rst_n(bus.rst_n),
        .weightData(bus.weightData),
        .weightValid(bus.weightValid),
        .weightReady(bus.weightReady),
        .activationData(bus.activationData),
        .activationValid(bus.activationValid),
        .activationReady(bus.activationReady),
        .rowDirection(noRowDirection),
        .columnDirection(noColumnDirection),
        .matrixUpdateValid(1'b0),
        .reductionUpdateData('0),
        .matrixUpdateAccepted(),
        .matrixUpdateComplete(),
        .datapathAdvance(),
        .resultEnqueue(),
        .resultEnqueueData(),
        .reductionUpdateBoundaryValid(),
        .reductionUpdateBoundaryData(),
        .resultData(bus.resultData),
        .resultValid(bus.resultValid),
        .resultReady(bus.resultReady),
        .resultLast(bus.resultLast),
        .weightsLoaded(bus.weightsLoaded),
        .reloadWeights(bus.reloadWeights),
        .reloadReady(bus.reloadReady)
    );

    initial begin
        bus.clk = 1'b0;
        forever #5 bus.clk = ~bus.clk;
    end

    initial begin
        bus.rst_n = 1'b0;
        bus.weightValid = 1'b0;
        bus.activationValid = 1'b0;
        bus.resultReady = 1'b0;
        bus.reloadWeights = 1'b0;
        for (int lane = 0; lane < N; lane++) begin
            bus.weightData[lane] = '0;
            bus.activationData[lane] = '0;
            noRowDirection[lane] = 2'sd0;
            noColumnDirection[lane] = 2'sd0;
        end

        uvm_config_db #(virtual nn_core_if #(WIDTH, N))::set(
            null, "uvm_test_top.env*", "vif", bus);
        run_test();
    end

    initial begin
        #2_000_000;
        `uvm_fatal("TB_TIMEOUT", "core-level UVM test exceeded 2 ms")
    end

endmodule
