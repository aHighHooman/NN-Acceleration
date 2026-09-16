`timescale 1ns / 1ps

// Parameterized engine-level checks.  The accelerator-level traffic, learning
// timing, and deep state ownership live in the trace/reference path and UVM.
module matrixMultiplierWeightStationary_testcase #(
    parameter int WIDTH = 16,
    parameter int N = 3
) (
    output logic done
);
    localparam int CLK_PERIOD = 10;
    localparam int RESULT_WIDTH = 2*WIDTH + $clog2(N);

    typedef logic signed [WIDTH-1:0] data_t;
    typedef logic signed [RESULT_WIDTH-1:0] result_t;
    typedef data_t matrix_t[N][N];
    typedef result_t result_matrix_t[N][N];

    logic clk, rst_n;
    data_t weightData[N], inputData[N];
    logic weightValid, weightReady, inputValid, inputReady;
    logic signed [1:0] noRowDirection[N], noColumnDirection[N];
    result_t resultData[N];
    logic resultValid, resultReady;
    logic weightsLoaded, reloadWeights, reloadReady;

    matrixMultiplierWeightStationary #(.WIDTH(WIDTH), .N(N)) dut (
        .clk(clk), .rst_n(rst_n),
        .weightData(weightData), .weightValid(weightValid), .weightReady(weightReady),
        .inputData(inputData), .inputValid(inputValid),
        .inputReady(inputReady),
        .rowDirection(noRowDirection), .columnDirection(noColumnDirection),
        .matrixUpdateValid(1'b0), .arrayAdvance(),
        .resultData(resultData), .resultValid(resultValid), .resultReady(resultReady),
        .weightsLoaded(weightsLoaded), .reloadWeights(reloadWeights),
        .reloadReady(reloadReady)
    );

    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    initial begin
        matrix_t identity_weights, basic_inputs;
        matrix_t signed_weights, signed_inputs;
        matrix_t positive_weights, positive_inputs;
        matrix_t negative_weights, negative_inputs;
        data_t min_data, max_data;
        data_t held_sample[N];

        min_data = {1'b1, {(WIDTH-1){1'b0}}};
        max_data = {1'b0, {(WIDTH-1){1'b1}}};
        for (int row = 0; row < N; row++) begin
            for (int col = 0; col < N; col++) begin
                identity_weights[row][col] = (row == col) ? 1 : 0;
                basic_inputs[row][col] = row * N + col + 1;
                signed_inputs[row][col] = (row == col) ? -3 :
                                               data_t'(row + 2*col + 1);
                signed_weights[row][col] = (row == col) ? 2 :
                                           (((row + col) % 2) ? -1 : 1);
                positive_inputs[row][col] = min_data;
                positive_weights[row][col] = min_data;
                negative_inputs[row][col] = min_data;
                negative_weights[row][col] = max_data;
            end
            held_sample[row] = row + 1;
        end

        done = 1'b0;
        rst_n = 1'b0;
        weightValid = 1'b0;
        inputValid = 1'b0;
        resultReady = 1'b0;
        reloadWeights = 1'b0;
        for (int lane = 0; lane < N; lane++) begin
            weightData[lane] = '0;
            inputData[lane] = '0;
            noRowDirection[lane] = 2'sd0;
            noColumnDirection[lane] = 2'sd0;
        end

        repeat (3) @(posedge clk);
        @(negedge clk) rst_n = 1'b1;

        // One direct result-boundary stall proves valid/data stability without
        // making any internal pending-row register part of the contract.
        send_weights("identity weights", identity_weights);
        check_result_backpressure(held_sample, identity_weights);

        // Deterministic matrix multiplication and signed arithmetic.
        send_matrix_and_check("basic identity multiplication",
                              basic_inputs, identity_weights);
        request_weight_reload();
        send_weights("signed weights", signed_weights);
        send_matrix_and_check("signed multiplication", signed_inputs,
                              signed_weights);

        // The edge cases require the full accumulation width, not a
        // per-product truncation.  Keep both signs of the boundary case.
        request_weight_reload();
        send_weights("positive accumulation-edge weights", positive_weights);
        send_matrix_and_check("positive accumulation width",
                              positive_inputs, positive_weights);
        request_weight_reload();
        send_weights("negative accumulation-edge weights", negative_weights);
        send_matrix_and_check("negative accumulation width",
                              negative_inputs, negative_weights);

        $display("PASS: %0dx%0d deterministic arithmetic, signed edges, and result backpressure",
                 N, N);
        done = 1'b1;
    end

    task send_weights(input string label, input matrix_t weight_matrix);
        $display("%0dx%0d: loading %s", N, N, label);
        for (int row = N-1; row >= 0; row--) begin
            @(negedge clk);
            while (!weightReady) @(negedge clk);
            for (int lane = 0; lane < N; lane++)
                weightData[lane] = weight_matrix[row][lane];
            weightValid = 1'b1;
            @(posedge clk);
        end
        @(negedge clk) weightValid = 1'b0;
        wait(weightsLoaded);
    endtask

    task send_inputs(input matrix_t input_matrix);
        for (int row = 0; row < N; row++) begin
            @(negedge clk);
            while (!inputReady) @(negedge clk);
            for (int lane = 0; lane < N; lane++)
                inputData[lane] = input_matrix[row][lane];
            inputValid = 1'b1;
            @(posedge clk);
        end
        @(negedge clk) inputValid = 1'b0;
    endtask

    task send_matrix_and_check(
        input string label,
        input matrix_t input_matrix,
        input matrix_t weight_matrix
    );
        result_matrix_t expected, actual;
        int errors;
        errors = 0;
        for (int row = 0; row < N; row++)
            for (int col = 0; col < N; col++) begin
                longint signed sum;
                sum = 0;
                for (int k = 0; k < N; k++)
                    sum += $signed(input_matrix[row][k]) *
                           $signed(weight_matrix[k][col]);
                expected[row][col] = result_t'(sum);
            end

        resultReady = 1'b1;
        fork
            send_inputs(input_matrix);
            begin
                for (int row = 0; row < N; row++) begin
                    bit accepted;
                    accepted = 1'b0;
                    while (!accepted) begin
                        @(negedge clk);
                        @(posedge clk);
                        if (resultValid && resultReady) begin
                            for (int col = 0; col < N; col++)
                                actual[row][col] = resultData[col];
                            accepted = 1'b1;
                        end
                    end
                end
            end
        join
        @(negedge clk) resultReady = 1'b0;

        for (int row = 0; row < N; row++)
            for (int col = 0; col < N; col++)
                if (actual[row][col] !== expected[row][col]) begin
                    $error("%0dx%0d %s mismatch [%0d][%0d]: got %0d expected %0d",
                           N, N, label, row, col,
                           actual[row][col], expected[row][col]);
                    errors++;
                end
        if (errors != 0)
            $fatal(1, "FAIL: %0dx%0d %s had %0d errors", N, N, label, errors);
        $display("PASS: %0dx%0d %s", N, N, label);
    endtask

    task check_result_backpressure(
        input data_t sample[N],
        input matrix_t weight_matrix
    );
        result_t expected[N], held[N];
        int guard;
        for (int col = 0; col < N; col++) begin
            longint signed sum;
            sum = 0;
            for (int row = 0; row < N; row++)
                sum += $signed(sample[row]) * $signed(weight_matrix[row][col]);
            expected[col] = result_t'(sum);
        end

        resultReady = 1'b0;
        send_single_input(sample);
        guard = 0;
        while (!resultValid) begin
            @(negedge clk);
            if (++guard > 1000)
                $fatal(1, "%0dx%0d result-boundary stall timeout", N, N);
        end

        for (int lane = 0; lane < N; lane++)
            held[lane] = resultData[lane];
        repeat (3) begin
            @(posedge clk); #1;
            if (!resultValid)
                $fatal(1, "%0dx%0d result valid dropped while ready was low", N, N);
            for (int lane = 0; lane < N; lane++)
                if (resultData[lane] !== held[lane])
                    $fatal(1, "%0dx%0d result lane %0d changed under backpressure",
                           N, N, lane);
        end

        @(negedge clk) resultReady = 1'b1;
        @(posedge clk);
        for (int lane = 0; lane < N; lane++)
            if (held[lane] !== expected[lane])
                $fatal(1, "%0dx%0d stalled result lane %0d got %0d expected %0d",
                       N, N, lane, held[lane], expected[lane]);
        @(negedge clk) resultReady = 1'b0;
        $display("PASS: %0dx%0d direct result backpressure stability", N, N);
    endtask

    task send_single_input(input data_t sample[N]);
        @(negedge clk);
        while (!inputReady) @(negedge clk);
        for (int lane = 0; lane < N; lane++)
            inputData[lane] = sample[lane];
        inputValid = 1'b1;
        @(posedge clk);
        @(negedge clk) inputValid = 1'b0;
    endtask

    task request_weight_reload();
        wait(reloadReady);
        @(negedge clk) reloadWeights = 1'b1;
        @(posedge clk);
        @(negedge clk) reloadWeights = 1'b0;
        wait(!weightsLoaded);
    endtask
endmodule

module matrixMultiplierWeightStationary_tb;
    logic done2, done3, done4;
    matrixMultiplierWeightStationary_testcase #(.N(2)) test_2x2 (.done(done2));
    matrixMultiplierWeightStationary_testcase #(.N(3)) test_3x3 (.done(done3));
    matrixMultiplierWeightStationary_testcase #(.N(4)) test_4x4 (.done(done4));

    initial begin
        wait(done2 && done3 && done4);
        $display("PASS: N=2, N=3, and N=4 matrix-engine suites completed.");
        $finish;
    end
endmodule
