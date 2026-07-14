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

    typedef logic signed [WIDTH-1:0] data_t;
    typedef logic signed [2*WIDTH-1:0] result_t;
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
        matrix_t identity, a_basic, a_arbitrary, b_arbitrary;
        matrix_t a_signed, b_signed, a_signed_edge, a_signed_mixed, b_signed_mixed;

        build_test_matrices(identity, a_basic, a_arbitrary, b_arbitrary,
                            a_signed, b_signed, a_signed_edge,
                            a_signed_mixed, b_signed_mixed);
        initialize_signals();
        apply_reset();

        // Two A matrices are transmitted back-to-back under one stationary B.
        send_weights("identity weights", identity, 1'b0);
        fork
            begin
                send_activations("basic A", a_basic, 1'b0);
                send_activations("back-to-back arbitrary A", a_arbitrary, 1'b0);
            end
            begin
                receive_and_check("basic A x identity", a_basic, identity, 1'b1, 1'b1);
                receive_and_check("arbitrary A x identity", a_arbitrary, identity, 1'b1, 1'b1);
            end
        join

        request_weight_reload();
        send_weights("arbitrary weights with input bubbles", b_arbitrary, 1'b1);
        fork
            send_activations("basic A with input bubbles", a_basic, 1'b1);
            receive_and_check("basic A x arbitrary B", a_basic, b_arbitrary, 1'b1, 1'b1);
        join

        request_weight_reload();
        send_weights("signed weights", b_signed, 1'b0);
        fork
            send_activations("signed A", a_signed, 1'b0);
            receive_and_check("signed A x signed B pass-through", a_signed, b_signed, 1'b1, 1'b1);
        join

        set_pass_through(1'b0);
        fork
            send_activations("signed A for ReLU", a_signed, 1'b0);
            receive_and_check("signed A x signed B ReLU", a_signed, b_signed, 1'b1, 1'b0);
        join

        request_weight_reload();
        send_weights("signed identity weights", identity, 1'b0);
        fork
            send_activations("signed edge-value A", a_signed_edge, 1'b0);
            receive_and_check("signed edge-value A x identity ReLU",
                              a_signed_edge, identity, 1'b0, 1'b0);
        join

        set_pass_through(1'b1);
        request_weight_reload();
        send_weights("additional signed weights", b_signed_mixed, 1'b0);
        fork
            send_activations("additional signed A pass-through", a_signed_mixed, 1'b0);
            receive_and_check("additional signed A x B pass-through",
                              a_signed_mixed, b_signed_mixed, 1'b0, 1'b1);
        join

        set_pass_through(1'b0);
        fork
            send_activations("additional signed A ReLU", a_signed_mixed, 1'b0);
            receive_and_check("additional signed A x B ReLU",
                              a_signed_mixed, b_signed_mixed, 1'b0, 1'b0);
        join

        $display("\nPASS: all %0dx%0d weight-stationary tests completed.", N, N);
        done = 1'b1;
    end

    task build_test_matrices(
        output matrix_t identity, output matrix_t a_basic, output matrix_t a_arbitrary,
        output matrix_t b_arbitrary, output matrix_t a_signed, output matrix_t b_signed,
        output matrix_t a_signed_edge, output matrix_t a_signed_mixed,
        output matrix_t b_signed_mixed
    );
        for (int row = 0; row < N; row++)
            for (int col = 0; col < N; col++) begin
                identity[row][col]    = (row == col) ? 1 : 0;
                a_basic[row][col]     = row * N + col + 1;
                a_arbitrary[row][col] = (row * 3 + col * 2 + 1) % 7 - 3;
                b_arbitrary[row][col] = (row * 2 + col * 3 + 2) % 9 - 4;
                a_signed[row][col]    = (row * 5 + col * 3 + 2) % 11 - 5;
                b_signed[row][col]    = (row * 4 + col * 5 + 1) % 13 - 6;
                a_signed_edge[row][col] = (row * 3 + col * 2 + 1) % 7 - 3;
                a_signed_mixed[row][col] = (row * 7 + col * 5 + 2) % 17 - 8;
                b_signed_mixed[row][col] = (row * 11 + col * 3 + 1) % 15 - 7;
            end
    endtask

    task initialize_signals();
        done = 1'b0;
        rst_n = 1'b0;
        weightValid = 1'b0;
        activationValid = 1'b0;
        resultReady = 1'b0;
        passThrough = 1'b1;
        reloadWeights = 1'b0;
        for (int i = 0; i < N; i++) begin
            weightData[i] = '0;
            activationData[i] = '0;
        end
    endtask

    task apply_reset();
        repeat (3) @(posedge clk);
        @(negedge clk) rst_n = 1'b1;
    endtask

    task send_weights(input string label, input matrix_t matrixB, input bit add_bubbles);
        $display("\n=== %0dx%0d: Loading %s ===", N, N, label);
        display_input_matrix("Matrix B", matrixB);
        for (int row = N-1; row >= 0; row--) begin
            if (add_bubbles && row == N-2) begin
                @(negedge clk) weightValid = 1'b0;
                @(negedge clk);
            end
            @(negedge clk);
            while (!weightReady) @(negedge clk);
            for (int col = 0; col < N; col++) weightData[col] = matrixB[row][col];
            weightValid = 1'b1;
            @(posedge clk);
        end
        @(negedge clk) weightValid = 1'b0;
        wait(weightsLoaded);
    endtask

    task send_activations(input string label, input matrix_t matrixA, input bit add_bubbles);
        $display("\n%0dx%0d: Sending %s", N, N, label);
        display_input_matrix("Matrix A", matrixA);
        for (int row = 0; row < N; row++) begin
            if (add_bubbles && row == 1) begin
                @(negedge clk) activationValid = 1'b0;
                @(negedge clk);
            end
            @(negedge clk);
            while (!activationReady) @(negedge clk);
            for (int k = 0; k < N; k++) activationData[k] = matrixA[row][k];
            activationValid = 1'b1;
            @(posedge clk);
        end
        @(negedge clk) activationValid = 1'b0;
    endtask

    task receive_and_check(input string label, input matrix_t matrixA, input matrix_t matrixB,
                           input bit add_backpressure, input bit expected_pass_through);
        result_matrix_t actual, expected;
        int errors;
        errors = 0;
        multiply_reference(matrixA, matrixB, expected);
        if (!expected_pass_through)
            for (int row = 0; row < N; row++)
                for (int col = 0; col < N; col++)
                    if (expected[row][col][2*WIDTH-1]) expected[row][col] = '0;
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
        display_result_matrix("Expected A x B", expected);
        display_result_matrix("Received A x B", actual);
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

    task set_pass_through(input bit enabled);
        @(negedge clk) passThrough = enabled;
        $display("%0dx%0d: passThrough=%0b", N, N, enabled);
    endtask

    task multiply_reference(input matrix_t a, input matrix_t b, output result_matrix_t c);
        for (int row = 0; row < N; row++)
            for (int col = 0; col < N; col++) begin
                c[row][col] = '0;
                for (int k = 0; k < N; k++) c[row][col] += a[row][k] * b[k][col];
            end
    endtask

    task display_input_matrix(input string label, input matrix_t matrix);
        $display("%s:", label);
        for (int row = 0; row < N; row++) begin
            $write("  [");
            for (int col = 0; col < N; col++) begin
                $write("%0d", matrix[row][col]);
                if (col < N-1) $write(", ");
            end
            $display("]");
        end
    endtask

    task display_result_matrix(input string label, input result_matrix_t matrix);
        $display("%s:", label);
        for (int row = 0; row < N; row++) begin
            $write("  [");
            for (int col = 0; col < N; col++) begin
                $write("%0d", matrix[row][col]);
                if (col < N-1) $write(", ");
            end
            $display("]");
        end
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
