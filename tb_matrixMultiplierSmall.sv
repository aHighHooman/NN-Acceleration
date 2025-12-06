`timescale 1ns / 1ps

module tb_matrixMultiplierSmall;

    // Parameters
    localparam int WIDTH = 16;
    localparam int N = 3;
    localparam int CLK_PERIOD = 10; // 10ns clock period

    // Testbench signals
    logic        clk;
    logic        rst_n;
    logic signed [2*WIDTH-1:0] resultMatrix [N][N];

    // Instantiate the Design Under Test (DUT)
    matrixMultiplierSmall #(
        .WIDTH(WIDTH),
        .N(N)
    ) dut (
        .clk          (clk),
        .rst_n        (rst_n),
        .resultMatrix (resultMatrix)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #((CLK_PERIOD / 2)) clk = ~clk;
    end

    // Test data types
    typedef logic signed [WIDTH-1:0] matrix_t [N][N];
    typedef logic signed [2*WIDTH-1:0] result_matrix_t [N][N];

    // Test Case 1: A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    //              B = Identity
    matrix_t matrixA = '{
        '{-2, 2, -3},
        '{4, -5, 6},
        '{-7, 8, -9}
    };

    matrix_t matrixB = '{
        '{1, 0, 0},
        '{0, 1, 0},
        '{0, 0, 1}
    };
    
    // Expected Result C = A_transpose * B
    // A_transpose = [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
    // C = A_transpose * I = A_transpose
    result_matrix_t expectedC = '{
        '{-1, 2, -3}, // C[0][j] = A_t[0] . B[col j] = (1,4,7) . B[col j]
        '{4, -5, 6}, // C[1][j] = A_t[1] . B[col j] = (2,5,8) . B[col j]
        '{-7, 8, -9}  // C[2][j] = A_t[2] . B[col j] = (3,6,9) . B[col j]
    };
    // Whoops, my manual calculation was wrong. Let's trace C[i][j].
    // C[i][j] = A_t[row i] . B[col j]
    // C[0][0] = (1,4,7) . (1,0,0) = 1
    // C[0][1] = (1,4,7) . (0,1,0) = 4
    // C[0][2] = (1,4,7) . (0,0,1) = 7
    // C[1][0] = (2,5,8) . (1,0,0) = 2
    // C[1][1] = (2,5,8) . (0,1,0) = 5
    // C[1][2] = (2,5,8) . (0,0,1) = 8
    // C[2][0] = (3,6,9) . (1,0,0) = 3
    // C[2][1] = (3,6,9) . (0,1,0) = 6
    // C[2][2] = (3,6,9) . (0,0,1) = 9
    
    result_matrix_t expectedC_test1 = '{
        '{1, 4, 7},
        '{2, 5, 8},
        '{3, 6, 9}
    };


    // Main test sequence
    initial begin
        $display("Testbench started.");
        
        // Run Test 1
        run_test(matrixA, matrixB, expectedC_test1, "Test 1: A * I");

        // --- Add more tests here ---
        // Example:
        // matrix_t matrixA_test2 = ...
        // matrix_t matrixB_test2 = ...
        // result_matrix_t expectedC_test2 = ...
        // run_test(matrixA_test2, matrixB_test2, expectedC_test2, "Test 2: ...");
        
        $display("All tests complete.");
        $finish;
    end

    // Task to run a single test case
    task run_test(
        input matrix_t testA,
        input matrix_t testB,
        input result_matrix_t expected_C,
        input string test_name
    );
        int errors;
        $display("---------------------------------");
        $display("Starting: %s", test_name);

        // 1. Initialize memory while in reset
        rst_n = 0;
        clk = 0; // ensure clock is 0
        
        // Load memA (Column-Major)
        // memA[k*N + j] = A[j][k]
        /* This loop is removed as it's both incorrect and causes the compile error
        for (int j = 0; j < N; j++) begin // Column index
            for (int i = 0; i < N; i++) begin // Row index
                // Use hierarchical path to load memory
                // memA[j] (memBlock) -> memArray[j] (gen block) -> memInst (memory) -> mem (logic array)
                // The orchestrator accesses memA[i] at base addr i*N
                // We need to load A[col i] starting at memA[i*N]
                dut.memA.memArray[i].memInst.mem[j] = testA[j][i];
            end
        end
        */

        // Correction: Orchestrator reads A[col i] from memA[i] at addrs i*N, i*N+1, i*N+2
        // So, we must load memArray[i].memInst.mem[i*N + k] = testA[k][i]
        
        // Unrolled loops to fix vsim-3745 and correct loading logic:
        
        // Load memA: Column 'i' of testA -> memA.memArray[i]
        // Load Column 0 (testA[0][0], testA[1][0], testA[2][0]) into memArray[0] at addrs 0,1,2
        dut.memA.memArray[0].memInst.mem[0] = testA[0][0];
        dut.memA.memArray[0].memInst.mem[1] = testA[0][1];
        dut.memA.memArray[0].memInst.mem[2] = testA[0][2];
        dut.memA.memArray[0].memInst.mem[25] = 0;

        // Load Column 1 (testA[0][1], testA[1][1], testA[2][1]) into memArray[1] at addrs 3,4,5
        dut.memA.memArray[1].memInst.mem[3] = testA[1][0];
        dut.memA.memArray[1].memInst.mem[4] = testA[1][1];
        dut.memA.memArray[1].memInst.mem[5] = testA[1][2];
        dut.memA.memArray[1].memInst.mem[25] = 0;

        // Load Column 2 (testA[0][2], testA[1][2], testA[2][2]) into memArray[2] at addrs 6,7,8
        dut.memA.memArray[2].memInst.mem[6] = testA[2][0];
        dut.memA.memArray[2].memInst.mem[7] = testA[2][1];
        dut.memA.memArray[2].memInst.mem[8] = testA[2][2];
        dut.memA.memArray[2].memInst.mem[25] = 0;


        // Load memB (Row-Major)
        // Orchestrator reads B[row i] from memB[i] at addrs i, i+N, i+2N
        // So, we must load memArray[i].memInst.mem[i + j*N] = testB[i][j]

        // Load Row 0 (testB[0][0], testB[0][1], testB[0][2]) into memArray[0] at addrs 0,3,6
        dut.memB.memArray[0].memInst.mem[0] = testB[0][0];
        dut.memB.memArray[0].memInst.mem[3] = testB[1][0];
        dut.memB.memArray[0].memInst.mem[6] = testB[2][0];
        dut.memB.memArray[0].memInst.mem[25] = 0;
        

        // Load Row 1 (testB[1][0], testB[1][1], testB[1][2]) into memArray[1] at addrs 1,4,7
        dut.memB.memArray[1].memInst.mem[1] = testB[0][1];
        dut.memB.memArray[1].memInst.mem[4] = testB[1][1];
        dut.memB.memArray[1].memInst.mem[7] = testB[2][1];
        dut.memB.memArray[1].memInst.mem[25] = 0;

        // Load Row 2 (testB[2][0], testB[2][1], testB[2][2]) into memArray[2] at addrs 2,5,8
        dut.memB.memArray[2].memInst.mem[2] = testB[0][2];
        dut.memB.memArray[2].memInst.mem[5] = testB[1][2];
        dut.memB.memArray[2].memInst.mem[8] = testB[2][2];
        dut.memB.memArray[2].memInst.mem[25] = 0;
        
        // Wait 1 cycle for memories to be loaded (not strictly needed, but good practice)
        #CLK_PERIOD;

        // 2. Apply reset
        rst_n = 0;
        repeat (1) @(posedge clk);
        
        // 3. Release reset
        rst_n = 1;
        $display("Reset released. Running computation...");
        
        // 4. Wait for calculation to finish
        // Calculation for C[N-1][N-1] is stable around 3*N cycles
        // Let's wait for 3*N + 5 = 14 cycles
        repeat (3*N + 5) @(posedge clk);
        
        // 5. Check results
        errors = 0;
        for (int i = 0; i < N; i++) begin
            for (int j = 0; j < N; j++) begin
                if (resultMatrix[i][j] != expected_C[i][j]) begin
                    $error("FAIL: [%s] Mismatch at C[%0d][%0d]: Got %d, Expected %d",
                           test_name, i, j, resultMatrix[i][j], expected_C[i][j]);
                    errors = errors + 1;
                end
            end
        end

        if (errors == 0) begin
            $display("PASS: %s", test_name);
        end else begin
            $display("FAIL: %s had %0d errors.", test_name, errors);
        end
        
        // Optional: Display final matrix
        $display("Final Result Matrix:");
        for (int i = 0; i < N; i++) begin
            $display("  Row %0d: %d, %d, %d", i, resultMatrix[i][0], resultMatrix[i][1], resultMatrix[i][2]);
        end
        $display("---------------------------------");
        
    endtask

endmodule