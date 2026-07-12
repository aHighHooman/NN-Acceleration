`timescale 1ns / 1ps

module matrixMultiplierWeightStationary_tb;
    localparam int WIDTH = 16;
    localparam int N = 3;
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
    logic weightsLoaded, reloadWeights, reloadReady;

    matrixMultiplierWeightStationary #(.WIDTH(WIDTH), .N(N)) dut (
        .clk(clk), .rst_n(rst_n),
        .weightData(weightData), .weightValid(weightValid), .weightReady(weightReady),
        .activationData(activationData), .activationValid(activationValid),
        .activationReady(activationReady), .resultData(resultData),
        .resultValid(resultValid), .resultReady(resultReady), .resultLast(resultLast),
        .weightsLoaded(weightsLoaded), .reloadWeights(reloadWeights), .reloadReady(reloadReady)
    );

    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    initial begin
        matrix_t identity, a_basic, a_arbitrary, b_arbitrary, a_signed, b_signed;

        identity = '{'{1,0,0}, '{0,1,0}, '{0,0,1}};
        a_basic = '{'{1,2,3}, '{4,5,6}, '{7,8,9}};
        a_arbitrary = '{'{1,2,3}, '{0,1,4}, '{5,6,0}};
        b_arbitrary = '{'{7,8,9}, '{2,3,4}, '{1,0,6}};
        a_signed = '{'{-2,3,1}, '{4,-1,2}, '{0,5,-3}};
        b_signed = '{'{1,-2,4}, '{-3,0,2}, '{5,1,-1}};

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
                receive_and_check("basic A x identity", a_basic, identity, 1'b1);
                receive_and_check("arbitrary A x identity", a_arbitrary, identity, 1'b1);
            end
        join

        request_weight_reload();
        send_weights("arbitrary weights with input bubbles", b_arbitrary, 1'b1);
        fork
            send_activations("basic A with input bubbles", a_basic, 1'b1);
            receive_and_check("basic A x arbitrary B", a_basic, b_arbitrary, 1'b1);
        join

        request_weight_reload();
        send_weights("signed weights", b_signed, 1'b0);
        fork
            send_activations("signed A", a_signed, 1'b0);
            receive_and_check("signed A x signed B", a_signed, b_signed, 1'b1);
        join

        $display("\nPASS: all continuous FIFO weight-stationary tests completed.");
        $finish;
    end

    task initialize_signals();
        rst_n = 1'b0;
        weightValid = 1'b0;
        activationValid = 1'b0;
        resultReady = 1'b0;
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
        $display("\n=== Loading %s ===", label);
        display_input_matrix("Matrix B", matrixB);

        // Bottom-to-top is the physical loading order of the stationary array.
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
        $display("Weights are stationary in the PE array.");
    endtask

    task send_activations(input string label, input matrix_t matrixA, input bit add_bubbles);
        $display("\nSending %s:", label);
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

    task receive_and_check(
        input string label,
        input matrix_t matrixA,
        input matrix_t matrixB,
        input bit add_backpressure
    );
        result_matrix_t actual, expected;
        int errors;

        errors = 0;
        multiply_reference(matrixA, matrixB, expected);
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
                        $error("%s resultLast mismatch on row %0d", label, row);
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
                    $error("%s mismatch [%0d][%0d]: got %0d expected %0d",
                           label, row, col, actual[row][col], expected[row][col]);
                    errors++;
                end

        if (errors) $fatal(1, "FAIL: %s had %0d errors", label, errors);
        $display("PASS: %s", label);
    endtask

    task request_weight_reload();
        wait(reloadReady);
        @(negedge clk) reloadWeights = 1'b1;
        @(posedge clk);
        @(negedge clk) reloadWeights = 1'b0;
        wait(!weightsLoaded);
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
        if (!rst_n) begin
            holdingResult <= 1'b0;
        end else if (resultValid && !resultReady) begin
            if (holdingResult) begin
                for (int i = 0; i < N; i++)
                    assert(resultData[i] == heldResult[i]) else $error("Output changed under backpressure");
                assert(resultLast == heldLast) else $error("resultLast changed under backpressure");
            end
            for (int i = 0; i < N; i++) heldResult[i] <= resultData[i];
            heldLast <= resultLast;
            holdingResult <= 1'b1;
        end else begin
            holdingResult <= 1'b0;
        end
    end

endmodule
