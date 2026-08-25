`timescale 1ns/1ps

// Fixed-size interface for the first UVM learning environment.
//
// Keeping N and WIDTH fixed makes the class code easier to study.  A suggested
// exercise in EXERCISES.md is to move these values into a shared configuration
// object and run the same environment at N=3 and N=4.
interface nn_uvm_if;
    localparam int WIDTH = 8;
    localparam int N = 2;
    localparam int RESULT_WIDTH = 2*WIDTH + $clog2(N);

    logic clk;
    logic sclk;
    logic rst_n;

    logic weightReady;
    logic activationReady;
    logic passThrough;
    logic weightsLoaded;
    logic reloadWeights;
    logic reloadReady;

    logic cs_n[N];
    logic miso[N];
    logic misoValid[N];
    logic weightCs_n[N];
    logic weightMosi[N];
    logic activationCs_n[N];
    logic activationMosi[N];

    // The top-level clock generator reads this before every half-cycle.  The
    // starter tests leave it at seven ns; a constrained-random clock-ratio test
    // can change it between transactions without rewriting the testbench top.
    int unsigned sclk_half_period_ns = 7;

endinterface
