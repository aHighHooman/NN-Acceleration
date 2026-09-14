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
    localparam int unsigned RANDOM_SEED = 32'h5eed_2026;

    typedef logic signed [WIDTH-1:0] data_t;
    typedef logic signed [RESULT_WIDTH-1:0] result_t;
    typedef data_t matrix_t[N][N];
    typedef result_t result_matrix_t[N][N];

    function automatic data_t random_data();
        logic [WIDTH-1:0] value;

        value = '0;
        for (int bit_index = 0; bit_index < WIDTH; bit_index += 32)
            value = (value << 32) | $urandom();
        return data_t'(value);
    endfunction

    logic clk, rst_n;
    data_t weightData[N], activationData[N];
    logic weightValid, weightReady, activationValid, activationReady;
    logic signed [1:0] noRowDirection[N], noColumnDirection[N];
    result_t resultData[N];
    logic resultValid, resultReady, resultLast;
    logic weightsLoaded, reloadWeights, reloadReady;
    logic streamQuiescent;

    matrixMultiplierWeightStationary #(.WIDTH(WIDTH), .N(N)) dut (
        .clk(clk), .rst_n(rst_n),
        .weightData(weightData), .weightValid(weightValid), .weightReady(weightReady),
        .activationData(activationData), .activationValid(activationValid),
        .activationReady(activationReady),
        .rowDirection(noRowDirection),
        .columnDirection(noColumnDirection), .matrixUpdateValid(1'b0),
        .datapathAdvance(), .resultEnqueue(), .resultEnqueueData(),
        .resultData(resultData),
        .resultValid(resultValid), .resultReady(resultReady),
        .resultLast(resultLast),
        .weightsLoaded(weightsLoaded), .reloadWeights(reloadWeights), .reloadReady(reloadReady)
    );

    // Verification-only stream state deliberately excludes outputRowIndex.
    // The matrix reload boundary is stricter because it also requires that
    // frame position to be zero.
    assign streamQuiescent = dut.activationEmpty && !dut.skewBusy &&
                             !dut.pipelineBusy && !dut.resultAlignBusy &&
                             dut.outputEmpty;

    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    initial begin
        matrix_t weight_identity, activation_basic, activation_arbitrary;
        matrix_t weight_arbitrary;
        matrix_t activation_signed, weight_signed;
        matrix_t activation_signed_edge;
        matrix_t activation_signed_mixed, weight_signed_mixed;
        matrix_t activation_positive_overflow, weight_positive_overflow;
        matrix_t activation_negative_overflow, weight_negative_overflow;

        data_t min_data, max_data;
        int unsigned random_seed;

        min_data = {1'b1, {(WIDTH-1){1'b0}}};
        max_data = {1'b0, {(WIDTH-1){1'b1}}};
        random_seed = RANDOM_SEED;
        void'($urandom(random_seed));

        for (int row = 0; row < N; row++)
            for (int col = 0; col < N; col++) begin
                weight_identity[row][col] = (row == col) ? 1 : 0;
                activation_basic[row][col] = row * N + col + 1;
                activation_arbitrary[row][col] = random_data();
                weight_arbitrary[row][col] = random_data();
                activation_signed[row][col] = random_data();
                weight_signed[row][col] = random_data();
                activation_signed_edge[row][col] =
                    (row == col) ? 1 : ((row == 0 && col == 1) ? -1 : 0);
                activation_signed_mixed[row][col] = random_data();
                weight_signed_mixed[row][col] = random_data();
                activation_positive_overflow[row][col] = min_data;
                weight_positive_overflow[row][col] = min_data;
                activation_negative_overflow[row][col] = min_data;
                weight_negative_overflow[row][col] = max_data;
            end

        done = 1'b0;
        rst_n = 1'b0;
        weightValid = 1'b0;
        activationValid = 1'b0;
        resultReady = 1'b0;
        reloadWeights = 1'b0;
        for (int lane = 0; lane < N; lane++) begin
            weightData[lane] = '0;
            activationData[lane] = '0;
            noRowDirection[lane] = 2'sd0;
            noColumnDirection[lane] = 2'sd0;
        end

        repeat (3) @(posedge clk);
        @(negedge clk) rst_n = 1'b1;

        // Two activation matrices are transmitted back-to-back under one stationary weight matrix.
        send_weights("identity weights", weight_identity, NO_BUBBLES);
        if (N == 3)
            check_reload_frame_bookkeeping(weight_identity);
        fork
            begin
                send_activations("basic activations", activation_basic, NO_BUBBLES);
                send_activations("back-to-back arbitrary activations",
                                 activation_arbitrary, NO_BUBBLES);
            end
            begin
                expect_result("basic activations x identity weights",
                              activation_basic, weight_identity,
                              WITH_BACKPRESSURE);
                expect_result("arbitrary activations x identity weights",
                              activation_arbitrary, weight_identity,
                              WITH_BACKPRESSURE);
            end
        join

        request_weight_reload();
        send_weights("arbitrary weights with weight bubbles", weight_arbitrary, WITH_BUBBLES);
        fork
            send_activations("basic activations with activation bubbles",
                             activation_basic, WITH_BUBBLES);
            expect_result("basic activations x arbitrary weights",
                          activation_basic, weight_arbitrary,
                          WITH_BACKPRESSURE);
        join

        request_weight_reload();
        send_weights("signed weights", weight_signed, NO_BUBBLES);
        fork
            send_activations("signed activations", activation_signed, NO_BUBBLES);
            expect_result("signed activations x signed weights",
                          activation_signed, weight_signed,
                          WITH_BACKPRESSURE);
        join

        fork
            send_activations("repeated signed activations", activation_signed, NO_BUBBLES);
            expect_result("repeated signed activations x signed weights",
                          activation_signed, weight_signed,
                          WITH_BACKPRESSURE);
        join

        request_weight_reload();
        send_weights("signed identity weights", weight_identity, NO_BUBBLES);
        fork
            send_activations("signed edge-value activations",
                             activation_signed_edge, NO_BUBBLES);
            expect_result("signed edge-value activations x identity weights",
                          activation_signed_edge, weight_identity,
                          NO_BACKPRESSURE);
        join

        request_weight_reload();
        send_weights("additional signed weights", weight_signed_mixed, NO_BUBBLES);
        fork
            send_activations("additional signed activations",
                             activation_signed_mixed, NO_BUBBLES);
            expect_result("additional signed activations x signed weights",
                          activation_signed_mixed, weight_signed_mixed,
                          NO_BACKPRESSURE);
        join

        fork
            send_activations("repeated additional signed activations",
                             activation_signed_mixed, NO_BUBBLES);
            expect_result("repeated additional signed activations x signed weights",
                          activation_signed_mixed, weight_signed_mixed,
                          NO_BACKPRESSURE);
        join

        request_weight_reload();
        send_weights("positive-overflow weights", weight_positive_overflow, NO_BUBBLES);
        fork
            send_activations("positive-overflow activations",
                             activation_positive_overflow, NO_BUBBLES);
            expect_result("full-width positive accumulation",
                          activation_positive_overflow, weight_positive_overflow,
                          NO_BACKPRESSURE);
        join

        request_weight_reload();
        send_weights("negative-overflow weights", weight_negative_overflow, NO_BUBBLES);
        fork
            send_activations("negative-overflow activations",
                             activation_negative_overflow, NO_BUBBLES);
            expect_result("full-width negative accumulation",
                          activation_negative_overflow, weight_negative_overflow,
                          NO_BACKPRESSURE);
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

    task send_activations(
        input string label,
        input matrix_t activation_matrix,
        input bit add_bubbles
    );
        $display("\n%0dx%0d: Sending %s", N, N, label);
        for (int row = 0; row < N; row++) begin
            if (add_bubbles && row == 1) begin
                @(negedge clk) activationValid = 1'b0;
                @(negedge clk);
            end
            @(negedge clk);
            while (!activationReady) @(negedge clk);
            for (int k = 0; k < N; k++) activationData[k] = activation_matrix[row][k];
            activationValid = 1'b1;
            @(posedge clk);
        end
        @(negedge clk) activationValid = 1'b0;
    endtask

    task expect_result(input string label, input matrix_t activation_matrix,
                       input matrix_t weight_matrix,
                       input bit add_backpressure);
        result_matrix_t actual, expected;
        result_t product;
        int errors;
        errors = 0;

        for (int row = 0; row < N; row++)
            for (int col = 0; col < N; col++) begin
                expected[row][col] = '0;
                for (int k = 0; k < N; k++) begin
                    product = activation_matrix[row][k] * weight_matrix[k][col];
                    expected[row][col] += product;
                end
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

    task send_single_activation(input data_t sample[N]);
        @(negedge clk);
        while (!activationReady) @(negedge clk);
        for (int lane = 0; lane < N; lane++) activationData[lane] = sample[lane];
        activationValid = 1'b1;
        @(posedge clk);
        @(negedge clk) activationValid = 1'b0;
    endtask

    task wait_for_result_handshake(input bit expected_last, input string label);
        int guard;

        guard = 0;
        while (1) begin
            if (resultValid && resultReady) begin
                if (resultLast !== expected_last)
                    $fatal(1, "%0dx%0d %s resultLast mismatch", N, N, label);
                @(posedge clk);
                break;
            end
            @(negedge clk);
            guard++;
            if (guard > 1000)
                $fatal(1, "%0dx%0d %s result handshake timeout", N, N, label);
        end
    endtask

    task wait_for_stream_quiescent(input string label);
        int guard;

        guard = 0;
        while (!streamQuiescent) begin
            @(negedge clk);
            guard++;
            if (guard > 1000)
                $fatal(1, "%0dx%0d %s stream quiescence timeout", N, N, label);
        end
    endtask

    task check_reload_frame_bookkeeping(input matrix_t weight_matrix);
        data_t sample0[N], sample1[N], sample2[N];

        for (int lane = 0; lane < N; lane++) begin
            sample0[lane] = lane + 1;
            sample1[lane] = lane + 2;
            sample2[lane] = lane + 3;
        end

        // Case A: one N=3 row drains completely. Stream configuration may
        // change at this boundary, but the partial output frame may not reload.
        resultReady = 1'b1;
        send_single_activation(sample0);
        wait_for_result_handshake(1'b0, "one-sample drain");
        wait_for_stream_quiescent("one-sample drain");
        if (dut.outputRowIndex !== 1 || reloadReady !== 1'b0)
            $fatal(1, "%0dx%0d one-sample drain frame/reload state mismatch", N, N);
        @(negedge clk) reloadWeights = 1'b1;
        @(posedge clk);
        @(negedge clk) reloadWeights = 1'b0;
        if (weightsLoaded !== 1'b1 || dut.outputRowIndex !== 1)
            $fatal(1, "%0dx%0d partial-frame reload was accepted", N, N);

        // Case D: output backpressure holds the frame position and keeps
        // reload blocked until the actual result handshake.
        resultReady = 1'b0;
        send_single_activation(sample1);
        while (!resultValid) @(negedge clk);
        repeat (3) begin
            @(negedge clk);
            if (dut.outputRowIndex !== 1 || reloadReady !== 1'b0)
                $fatal(1, "%0dx%0d output position changed under backpressure", N, N);
        end
        @(negedge clk) reloadWeights = 1'b1;
        @(posedge clk);
        @(negedge clk) reloadWeights = 1'b0;
        if (weightsLoaded !== 1'b1 || dut.outputRowIndex !== 1)
            $fatal(1, "%0dx%0d backpressured partial-frame reload was accepted", N, N);
        resultReady = 1'b1;
        wait_for_result_handshake(1'b0, "backpressure release");
        wait_for_stream_quiescent("backpressure release");
        if (dut.outputRowIndex !== 2 || reloadReady !== 1'b0)
            $fatal(1, "%0dx%0d post-handshake frame/reload state mismatch", N, N);

        // Cases B/C: the final row wraps the output position, presents the
        // only last marker, and makes a complete-frame reload legal.
        send_single_activation(sample2);
        wait_for_result_handshake(1'b1, "complete N-row frame");
        wait_for_stream_quiescent("complete N-row frame");
        if (dut.outputRowIndex !== 0 || reloadReady !== 1'b1)
            $fatal(1, "%0dx%0d complete-frame reload state mismatch", N, N);
        request_weight_reload();
        send_weights("focused complete-frame reload", weight_matrix, NO_BUBBLES);
        $display("PASS: 3x3 partial-frame quiescence, handshake-only output position, resultLast, and reload boundary");
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
