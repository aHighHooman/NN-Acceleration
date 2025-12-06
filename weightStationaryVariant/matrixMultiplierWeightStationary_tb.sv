`timescale 1ns / 1ps

module matrixMultiplierWeightStationary_tb;

    // Parameters
    parameter int WIDTH = 16;
    parameter int N = 3;

    // Signals
    logic        clk;
    logic        rst_n;
    logic signed [2*WIDTH-1:0] resultMatrix [N][N];

    // DUT Instantiation
    matrixMultiplierWeightStationary #(
        .WIDTH(WIDTH),
        .N(N)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .resultMatrix(resultMatrix)
    );

    // Clock Generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 10ns period
    end

    // Test Sequence
    initial begin
        // 1. Initialize Inputs
        initialize_memory();

        // 2. Reset
        rst_n = 0;
        #20;
        rst_n = 1;
        $display("Reset released. Starting simulation...");

        // 3. Wait for operation to complete
        // The weight loader takes about N*2 cycles to load?
        // Data orchestrator takes some time too.
        // Let's wait enough time for the systolic array to fill and compute.
        // N=3. 
        // Weight loading: ~6-9 cycles.
        // Computation: ~2*N + N cycles latency?
        // Let's wait 100 cycles to be safe and observe the waves/output.
        repeat (100) @(posedge clk);

        // 4. Check Results
        display_results();

        $finish;
    end

    // Task to initialize memory contents
    // We use hierarchical access to set the memory values directly.
    task initialize_memory();
        int k;
        $display("Initializing Memory...");

        // Initialize Matrix A (Input) in ALL memInput blocks
        // We load the entire flattened matrix into each memory bank.
        // Unrolled for N=3 to avoid illegal generate block index error
        
        // Bank 0
        dut.memInput.memArray[0].memInst.mem[0] = 1;
        dut.memInput.memArray[0].memInst.mem[1] = 2;
        dut.memInput.memArray[0].memInst.mem[2] = 3;
        dut.memInput.memArray[0].memInst.mem[3] = 4;
        dut.memInput.memArray[0].memInst.mem[4] = 5;
        dut.memInput.memArray[0].memInst.mem[5] = 6;
        dut.memInput.memArray[0].memInst.mem[6] = 7;
        dut.memInput.memArray[0].memInst.mem[7] = 8;
        dut.memInput.memArray[0].memInst.mem[8] = 9;

        // Bank 1
        dut.memInput.memArray[1].memInst.mem[0] = 1;
        dut.memInput.memArray[1].memInst.mem[1] = 2;
        dut.memInput.memArray[1].memInst.mem[2] = 3;
        dut.memInput.memArray[1].memInst.mem[3] = 4;
        dut.memInput.memArray[1].memInst.mem[4] = 5;
        dut.memInput.memArray[1].memInst.mem[5] = 6;
        dut.memInput.memArray[1].memInst.mem[6] = 7;
        dut.memInput.memArray[1].memInst.mem[7] = 8;
        dut.memInput.memArray[1].memInst.mem[8] = 9;

        // Bank 2
        dut.memInput.memArray[2].memInst.mem[0] = 1;
        dut.memInput.memArray[2].memInst.mem[1] = 2;
        dut.memInput.memArray[2].memInst.mem[2] = 3;
        dut.memInput.memArray[2].memInst.mem[3] = 4;
        dut.memInput.memArray[2].memInst.mem[4] = 5;
        dut.memInput.memArray[2].memInst.mem[5] = 6;
        dut.memInput.memArray[2].memInst.mem[6] = 7;
        dut.memInput.memArray[2].memInst.mem[7] = 8;
        dut.memInput.memArray[2].memInst.mem[8] = 9;


        // Initialize Matrix B (Weights) in ALL memWeight blocks
        // We load the entire flattened matrix into each memory bank.
        
        // Bank 0
        dut.memWeight.memArray[0].memInst.mem[0] = 1;
        dut.memWeight.memArray[0].memInst.mem[1] = 0;
        dut.memWeight.memArray[0].memInst.mem[2] = 0;
        dut.memWeight.memArray[0].memInst.mem[3] = 0;
        dut.memWeight.memArray[0].memInst.mem[4] = 1;
        dut.memWeight.memArray[0].memInst.mem[5] = 0;
        dut.memWeight.memArray[0].memInst.mem[6] = 0;
        dut.memWeight.memArray[0].memInst.mem[7] = 0;
        dut.memWeight.memArray[0].memInst.mem[8] = 1;

        // Bank 1
        dut.memWeight.memArray[1].memInst.mem[0] = 1;
        dut.memWeight.memArray[1].memInst.mem[1] = 0;
        dut.memWeight.memArray[1].memInst.mem[2] = 0;
        dut.memWeight.memArray[1].memInst.mem[3] = 0;
        dut.memWeight.memArray[1].memInst.mem[4] = 1;
        dut.memWeight.memArray[1].memInst.mem[5] = 0;
        dut.memWeight.memArray[1].memInst.mem[6] = 0;
        dut.memWeight.memArray[1].memInst.mem[7] = 0;
        dut.memWeight.memArray[1].memInst.mem[8] = 1;

        // Bank 2
        dut.memWeight.memArray[2].memInst.mem[0] = 1;
        dut.memWeight.memArray[2].memInst.mem[1] = 0;
        dut.memWeight.memArray[2].memInst.mem[2] = 0;
        dut.memWeight.memArray[2].memInst.mem[3] = 0;
        dut.memWeight.memArray[2].memInst.mem[4] = 1;
        dut.memWeight.memArray[2].memInst.mem[5] = 0;
        dut.memWeight.memArray[2].memInst.mem[6] = 0;
        dut.memWeight.memArray[2].memInst.mem[7] = 0;
        dut.memWeight.memArray[2].memInst.mem[8] = 1;
    endtask

    // Captured results
    logic signed [2*WIDTH-1:0] captured_results [N][N];

    // Monitor Process
    initial begin
        // Initialize captured results
        for (int r = 0; r < N; r++) begin
            for (int c = 0; c < N; c++) begin
                captured_results[r][c] = 0;
            end
        end

        // Wait for Reset
        wait(rst_n == 0);
        wait(rst_n == 1);

        // Wait for Weight Loading to finish
        // We can monitor the writeEnable signal inside the DUT
        wait(dut.writeEnableWeightToSyst == 1);
        wait(dut.writeEnableWeightToSyst == 0);
        $display("Weight loading complete. Waiting for results...");

        // Capture results
        // The results stream out of the bottom of the systolic array.
        // dut.outputs[j] corresponds to the result of Column j.
        // As rows of A flow through, we expect rows of C to appear at the output.
        // Due to the systolic nature (skewed inputs), the results might also be skewed.
        // However, let's just capture valid non-zero outputs or capture for a fixed window.
        // Based on dataOrchestrator, inputs are fed over a window.
        
        // Let's capture for a sufficient number of cycles.
        // We assume the first valid result appears after some latency.
        // We'll just log what we see for now to help the user debug/verify.
        
        repeat (20) @(posedge clk) begin
            $display("Time %0t: Output Row = {%d, %d, %d}", $time, dut.outputs[0], dut.outputs[1], dut.outputs[2]);
        end
    end

    task display_results();
        $display("\n--- Simulation Finished ---");
        $display("Please check the 'Output Row' logs above to verify the streaming output.");
        $display("Note: Top-level resultMatrix is undriven in the current architecture.");
        $display("---------------------------");
    endtask

endmodule
