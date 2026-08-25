`timescale 1ns / 1ps

// Runs the same functional, bubble, backpressure, and signed-data checks for
// each supported square-array size.
module matrixMultiplierWeightStationary_testcase #(
    parameter int WIDTH = 16,
    parameter int N = 3
) (
    output logic done
);
    localparam int CLK_PERIOD = 10;
    localparam int RESULT_WIDTH = 2*WIDTH + $clog2(N);
    localparam bit NO_BUBBLES = 1'b0;
    localparam bit WITH_BUBBLES = 1'b1;
    localparam bit NO_BACKPRESSURE = 1'b0;
    localparam bit WITH_BACKPRESSURE = 1'b1;
    localparam bit RELU_RESULTS = 1'b0;
    localparam bit RAW_RESULTS = 1'b1;

    typedef logic signed [WIDTH-1:0] data_t;
    typedef logic signed [RESULT_WIDTH-1:0] result_t;
    typedef data_t matrix_t[N][N];
    typedef result_t result_matrix_t[N][N];

    logic clk, rst_n;
    data_t weightData[N], activationData[N];
    logic weightValid, weightReady, activationValid, activationReady;
    result_t resultData[N];
    logic resultValid, resultReady, resultLast;
    logic passThrough;
    logic weightsLoaded, reloadWeights, reloadReady;

    matrixMultiplierWeightStationary #(.WIDTH(WIDTH), .N(N)) dut (
        .clk(clk), .rst_n(rst_n),
        .weightData(weightData), .weightValid(weightValid), .weightReady(weightReady),
        .activationData(activationData), .activationValid(activationValid),
        .activationReady(activationReady), .resultData(resultData),
        .resultValid(resultValid), .resultReady(resultReady), .passThrough(passThrough),
        .resultLast(resultLast),
        .weightsLoaded(weightsLoaded), .reloadWeights(reloadWeights), .reloadReady(reloadReady)
    );

    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    initial begin
        matrix_t weight_identity, input_basic, input_arbitrary;
        matrix_t weight_arbitrary;
        matrix_t input_signed, weight_signed;
        matrix_t input_signed_edge;
        matrix_t input_signed_mixed, weight_signed_mixed;
        matrix_t input_positive_overflow, weight_positive_overflow;
        matrix_t input_negative_overflow, weight_negative_overflow;

        data_t min_data, max_data;

        min_data = {1'b1, {(WIDTH-1){1'b0}}};
        max_data = {1'b0, {(WIDTH-1){1'b1}}};

        // Keep the inputs beside the scenarios that use them. The formulas give
        // every supported N a deterministic mix of positive and negative data.
        for (int row = 0; row < N; row++)
            for (int col = 0; col < N; col++) begin
                weight_identity[row][col] = (row == col) ? 1 : 0;
                input_basic[row][col] = row * N + col + 1;
                input_arbitrary[row][col] = (row * 3 + col * 2 + 1) % 7 - 3;
                weight_arbitrary[row][col] = (row * 2 + col * 3 + 2) % 9 - 4;
                input_signed[row][col] = (row * 5 + col * 3 + 2) % 11 - 5;
                weight_signed[row][col] = (row * 4 + col * 5 + 1) % 13 - 6;
                input_signed_edge[row][col] = input_arbitrary[row][col];
                input_signed_mixed[row][col] = (row * 7 + col * 5 + 2) % 17 - 8;
                weight_signed_mixed[row][col] = (row * 11 + col * 3 + 1) % 15 - 7;
                input_positive_overflow[row][col] = min_data;
                weight_positive_overflow[row][col] = min_data;
                input_negative_overflow[row][col] = min_data;
                weight_negative_overflow[row][col] = max_data;
            end

        done = 1'b0;
        rst_n = 1'b0;
        weightValid = 1'b0;
        activationValid = 1'b0;
        resultReady = 1'b0;
        passThrough = 1'b1;
        reloadWeights = 1'b0;
        for (int lane = 0; lane < N; lane++) begin
            weightData[lane] = '0;
            activationData[lane] = '0;
        end

        repeat (3) @(posedge clk);
        @(negedge clk) rst_n = 1'b1;

        // Two input matrices are transmitted back-to-back under one stationary weight matrix.
        send_weights("identity weights", weight_identity, NO_BUBBLES);
        fork
            begin
                send_input("basic input", input_basic, NO_BUBBLES);
                send_input("back-to-back arbitrary input", input_arbitrary, NO_BUBBLES);
            end
            begin
                expect_result("basic input x identity weights",
                              input_basic, weight_identity, WITH_BACKPRESSURE, RAW_RESULTS);
                expect_result("arbitrary input x identity weights",
                              input_arbitrary, weight_identity, WITH_BACKPRESSURE, RAW_RESULTS);
            end
        join

        request_weight_reload();
        send_weights("arbitrary weights with input bubbles", weight_arbitrary, WITH_BUBBLES);
        fork
            send_input("basic input with input bubbles", input_basic, WITH_BUBBLES);
            expect_result("basic input x arbitrary weights",
                          input_basic, weight_arbitrary, WITH_BACKPRESSURE, RAW_RESULTS);
        join

        request_weight_reload();
        send_weights("signed weights", weight_signed, NO_BUBBLES);
        fork
            send_input("signed input", input_signed, NO_BUBBLES);
            expect_result("signed input x signed weights pass-through",
                          input_signed, weight_signed, WITH_BACKPRESSURE, RAW_RESULTS);
        join

        @(negedge clk) passThrough = 1'b0;
        fork
            send_input("signed input for ReLU", input_signed, NO_BUBBLES);
            expect_result("signed input x signed weights ReLU",
                          input_signed, weight_signed, WITH_BACKPRESSURE, RELU_RESULTS);
        join

        request_weight_reload();
        send_weights("signed identity weights", weight_identity, NO_BUBBLES);
        fork
            send_input("signed edge-value input", input_signed_edge, NO_BUBBLES);
            expect_result("signed edge-value input x identity weights ReLU",
                          input_signed_edge, weight_identity, NO_BACKPRESSURE, RELU_RESULTS);
        join

        @(negedge clk) passThrough = 1'b1;
        request_weight_reload();
        send_weights("additional signed weights", weight_signed_mixed, NO_BUBBLES);
        fork
            send_input("additional signed input pass-through", input_signed_mixed, NO_BUBBLES);
            expect_result("additional signed input x signed weights pass-through",
                          input_signed_mixed, weight_signed_mixed,
                          NO_BACKPRESSURE, RAW_RESULTS);
        join

        @(negedge clk) passThrough = 1'b0;
        fork
            send_input("additional signed input ReLU", input_signed_mixed, NO_BUBBLES);
            expect_result("additional signed input x signed weights ReLU",
                          input_signed_mixed, weight_signed_mixed,
                          NO_BACKPRESSURE, RELU_RESULTS);
        join

        @(negedge clk) passThrough = 1'b1;
        request_weight_reload();
        send_weights("positive-overflow weights", weight_positive_overflow, NO_BUBBLES);
        fork
            send_input("positive-overflow input", input_positive_overflow, NO_BUBBLES);
            expect_result("full-width positive accumulation",
                          input_positive_overflow, weight_positive_overflow,
                          NO_BACKPRESSURE, RAW_RESULTS);
        join

        request_weight_reload();
        send_weights("negative-overflow weights", weight_negative_overflow, NO_BUBBLES);
        fork
            send_input("negative-overflow input", input_negative_overflow, NO_BUBBLES);
            expect_result("full-width negative accumulation",
                          input_negative_overflow, weight_negative_overflow,
                          NO_BACKPRESSURE, RAW_RESULTS);
        join

        $display("\nPASS: all %0dx%0d weight-stationary tests completed.", N, N);
        done = 1'b1;
    end

    task send_weights(input string label, input matrix_t weight_matrix, input bit add_bubbles);
        $display("\n=== %0dx%0d: Loading %s ===", N, N, label);
        for (int row = N-1; row >= 0; row--) begin
            if (add_bubbles && row == N-2) begin
                @(negedge clk) weightValid = 1'b0;
                @(negedge clk);
            end
            @(negedge clk);
            while (!weightReady) @(negedge clk);
            for (int col = 0; col < N; col++) weightData[col] = weight_matrix[row][col];
            weightValid = 1'b1;
            @(posedge clk);
        end
        @(negedge clk) weightValid = 1'b0;
        wait(weightsLoaded);
    endtask

    task send_input(input string label, input matrix_t input_matrix, input bit add_bubbles);
        $display("\n%0dx%0d: Sending %s", N, N, label);
        for (int row = 0; row < N; row++) begin
            if (add_bubbles && row == 1) begin
                @(negedge clk) activationValid = 1'b0;
                @(negedge clk);
            end
            @(negedge clk);
            while (!activationReady) @(negedge clk);
            for (int k = 0; k < N; k++) activationData[k] = input_matrix[row][k];
            activationValid = 1'b1;
            @(posedge clk);
        end
        @(negedge clk) activationValid = 1'b0;
    endtask

    task expect_result(input string label, input matrix_t input_matrix,
                       input matrix_t weight_matrix,
                       input bit add_backpressure, input bit expected_pass_through);
        result_matrix_t actual, expected;
        result_t product;
        int errors;
        errors = 0;

        for (int row = 0; row < N; row++)
            for (int col = 0; col < N; col++) begin
                expected[row][col] = '0;
                for (int k = 0; k < N; k++) begin
                    product = input_matrix[row][k] * weight_matrix[k][col];
                    expected[row][col] += product;
                end
                if (!expected_pass_through && expected[row][col][RESULT_WIDTH-1])
                    expected[row][col] = '0;
            end

        for (int row = 0; row < N; row++) begin
            bit accepted;
            accepted = 1'b0;
            while (!accepted) begin
                @(negedge clk);
                resultReady = !add_backpressure || (($time / CLK_PERIOD) % 4 != 1);
                @(posedge clk);
                if (resultValid && resultReady) begin
                    for (int col = 0; col < N; col++) actual[row][col] = resultData[col];
                    if (resultLast !== (row == N-1)) begin
                        $error("%0dx%0d %s resultLast mismatch on row %0d", N, N, label, row);
                        errors++;
                    end
                    accepted = 1'b1;
                end
            end
        end
        @(negedge clk) resultReady = 1'b0;
        for (int row = 0; row < N; row++)
            for (int col = 0; col < N; col++)
                if (actual[row][col] !== expected[row][col]) begin
                    $error("%0dx%0d %s mismatch [%0d][%0d]: got %0d expected %0d",
                           N, N, label, row, col, actual[row][col], expected[row][col]);
                    errors++;
                end
        if (errors) $fatal(1, "FAIL: %0dx%0d %s had %0d errors", N, N, label, errors);
        $display("PASS: %0dx%0d %s", N, N, label);
    endtask

    task request_weight_reload();
        wait(reloadReady);
        @(negedge clk) reloadWeights = 1'b1;
        @(posedge clk);
        @(negedge clk) reloadWeights = 1'b0;
        wait(!weightsLoaded);
    endtask

    result_t heldResult[N];
    logic heldLast, holdingResult;
    always_ff @(posedge clk) begin
        if (!rst_n) holdingResult <= 1'b0;
        else if (resultValid && !resultReady) begin
            if (holdingResult) begin
                for (int i = 0; i < N; i++)
                    assert(resultData[i] == heldResult[i]) else $error("Output changed under backpressure");
                assert(resultLast == heldLast) else $error("resultLast changed under backpressure");
            end
            for (int i = 0; i < N; i++) heldResult[i] <= resultData[i];
            heldLast <= resultLast;
            holdingResult <= 1'b1;
        end else holdingResult <= 1'b0;
    end
endmodule

// Keep the original top-level name, but run all three array dimensions.
module matrixMultiplierWeightStationary_tb;
    logic done2, done3, done4;
    matrixMultiplierWeightStationary_testcase #(.N(2)) test_2x2 (.done(done2));
    matrixMultiplierWeightStationary_testcase #(.N(3)) test_3x3 (.done(done3));
    matrixMultiplierWeightStationary_testcase #(.N(4)) test_4x4 (.done(done4));

    initial begin
        wait(done2 && done3 && done4);
        $display("\nPASS: 2x2, 3x3, and 4x4 test suites completed.");
        $finish;
    end
endmodule
