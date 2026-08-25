`timescale 1ns/1ps

module nn_uvm_tb_top;
    import uvm_pkg::*;
    import nn_uvm_pkg::*;
    `include "uvm_macros.svh"

    nn_uvm_if bus();

    matrixMultiplierWeightStationarySPI #(
        .WIDTH(nn_uvm_pkg::WIDTH),
        .N(nn_uvm_pkg::N),
        .INPUT_FIFO_DEPTH(2*nn_uvm_pkg::N),
        .OUTPUT_FIFO_DEPTH(2*nn_uvm_pkg::N)
    ) dut (
        .clk(bus.clk),
        .rst_n(bus.rst_n),
        .weightReady(bus.weightReady),
        .activationReady(bus.activationReady),
        .passThrough(bus.passThrough),
        .weightsLoaded(bus.weightsLoaded),
        .reloadWeights(bus.reloadWeights),
        .reloadReady(bus.reloadReady),
        .sclk(bus.sclk),
        .cs_n(bus.cs_n),
        .miso(bus.miso),
        .misoValid(bus.misoValid),
        .weightCs_n(bus.weightCs_n),
        .weightMosi(bus.weightMosi),
        .activationCs_n(bus.activationCs_n),
        .activationMosi(bus.activationMosi)
    );

    initial begin
        bus.clk = 0;
        forever #5 bus.clk = ~bus.clk;
    end

    initial begin
        bus.sclk = 0;
        #2;
        forever begin
            #(bus.sclk_half_period_ns * 1ns);
            bus.sclk = ~bus.sclk;
        end
    end

    initial begin
        // Prevent X-driven pins before the UVM driver reaches run_phase.
        bus.rst_n = 0;
        bus.passThrough = 1;
        bus.reloadWeights = 0;
        for (int lane = 0; lane < nn_uvm_pkg::N; lane++) begin
            bus.cs_n[lane] = 1;
            bus.weightCs_n[lane] = 1;
            bus.activationCs_n[lane] = 1;
            bus.weightMosi[lane] = 0;
            bus.activationMosi[lane] = 0;
        end

        uvm_config_db #(virtual nn_uvm_if)::set(null, "uvm_test_top.env.*", "vif", bus);
        run_test();
    end

    initial begin
        #200_000;
        `uvm_fatal("TB_TIMEOUT", "UVM starter test exceeded 200 us")
    end

endmodule
