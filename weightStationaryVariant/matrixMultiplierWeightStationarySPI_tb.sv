`timescale 1ns / 1ps

module matrixMultiplierWeightStationarySPI_tb;
    localparam int WIDTH = 8;
    localparam int N = 2;
    localparam int FRACTION_BITS = 4;
    localparam int REDUCTION_WEIGHT_WIDTH = 16;
    localparam int REDUCTION_FRACTION_BITS = REDUCTION_WEIGHT_WIDTH - 1;
    localparam int RESCALE_SHIFT = FRACTION_BITS
                                   + REDUCTION_FRACTION_BITS;
    localparam int SCALE = 1 << FRACTION_BITS;
    localparam int PREDICTION_WIDTH = 2*WIDTH + 2*$clog2(N);

    typedef logic signed [WIDTH-1:0] data_t;
    typedef logic signed [PREDICTION_WIDTH-1:0] result_t;

    logic clk, sclk, rst_n;
    logic weightReady, activationReady, passThrough, reduceOutput;
    logic signed [REDUCTION_WEIGHT_WIDTH-1:0] reductionWeight[N];
    logic weightsLoaded, reloadWeights, reloadReady;
    logic cs_n[N], miso[N], misoValid[N];
    logic weightCs_n[N], weightMosi[N];
    logic activationCs_n[N], activationMosi[N];

    matrixMultiplierWeightStationarySPI #(
        .WIDTH(WIDTH),
        .N(N),
        .FRACTION_BITS(FRACTION_BITS),
        .REDUCTION_WEIGHT_WIDTH(REDUCTION_WEIGHT_WIDTH)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .weightReady(weightReady), .activationReady(activationReady),
        .passThrough(passThrough), .reduceOutput(reduceOutput),
        .reductionWeight(reductionWeight), .weightsLoaded(weightsLoaded),
        .reloadWeights(reloadWeights), .reloadReady(reloadReady),
        .sclk(sclk), .cs_n(cs_n), .miso(miso), .misoValid(misoValid),
        .weightCs_n(weightCs_n), .weightMosi(weightMosi),
        .activationCs_n(activationCs_n), .activationMosi(activationMosi)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        sclk = 1'b0;
        #2;
        forever #7 sclk = ~sclk;
    end

    initial begin
        data_t weight_vector[N], activation_vector[N];

        rst_n = 1'b0;
        passThrough = 1'b1;
        reduceOutput = 1'b0;
        reloadWeights = 1'b0;
        for (int lane = 0; lane < N; lane++) begin
            reductionWeight[lane] = '0;
            cs_n[lane] = 1'b1;
            weightCs_n[lane] = 1'b1;
            activationCs_n[lane] = 1'b1;
            weightMosi[lane] = 1'b0;
            activationMosi[lane] = 1'b0;
        end

        repeat (3) @(posedge sclk);
        repeat (3) @(posedge clk);
        rst_n = 1'b1;

        // Fixed-point identity weights preserve the transmitted activation
        // values in real units; raw matrix codes retain the 2*FRACTION_BITS
        // binary point.
        weight_vector[0] = 0;
        weight_vector[1] = SCALE;
        send_weight_vector(weight_vector);

        weight_vector[0] = SCALE;
        weight_vector[1] = 0;
        send_weight_vector(weight_vector);
        wait(weightsLoaded);

        // Reduction disabled: nonzero coefficients must have no effect and
        // the ordinary activated vector must remain intact.
        reductionWeight[0] = 111;
        reductionWeight[1] = -222;
        activation_vector[0] = 2;
        activation_vector[1] = -3;
        send_activation_vector(activation_vector);

        activation_vector[0] = 4;
        activation_vector[1] = 5;
        send_activation_vector(activation_vector);

        expect_configured_row("reduction-disabled pass-through row 0", 2*SCALE, -3*SCALE);
        expect_configured_row("reduction-disabled pass-through row 1", 4*SCALE, 5*SCALE);

        passThrough = 1'b0;

        activation_vector[0] = -6;
        activation_vector[1] = 7;
        send_activation_vector(activation_vector);

        activation_vector[0] = 8;
        activation_vector[1] = -9;
        send_activation_vector(activation_vector);

        expect_configured_row("reduction-disabled ReLU row 0", -6*SCALE, 7*SCALE);
        expect_configured_row("reduction-disabled ReLU row 1", 8*SCALE, -9*SCALE);

        // Pass-through + weighted reduction. Mixed-sign raw values and
        // mixed-sign coefficients create both positive and negative terms.
        passThrough = 1'b1;
        reductionWeight[0] = -16384; // -0.5
        reductionWeight[1] = 16384;  //  0.5
        reduceOutput = 1'b1;

        activation_vector[0] = 10;
        activation_vector[1] = 20;
        send_activation_vector(activation_vector);

        activation_vector[0] = -10;
        activation_vector[1] = -20;
        send_activation_vector(activation_vector);

        expect_configured_row("signed pass-through reduction row 0", 10*SCALE, 20*SCALE);
        expect_configured_row("signed pass-through reduction row 1", -10*SCALE, -20*SCALE);

        // Exact cancellation is checked as a separate, deterministic workload.
        // Configuration changes occur only after both prior results are read.
        reductionWeight[0] = -16384;
        reductionWeight[1] = 16384;

        activation_vector[0] = 2;
        activation_vector[1] = 3;
        send_activation_vector(activation_vector);

        activation_vector[0] = 5;
        activation_vector[1] = 4;
        send_activation_vector(activation_vector);

        expect_configured_row("fractional result truncates after sum", 2*SCALE, 3*SCALE);
        expect_configured_row("negative arithmetic rescale", 5*SCALE, 4*SCALE);

        // ReLU must precede reduction. The two orderings intentionally produce
        // different fixed-point predictions after the final rescale.
        passThrough = 1'b0;
        reductionWeight[0] = -32768; // -1.0
        reductionWeight[1] = 24576;  //  0.75

        activation_vector[0] = -6;
        activation_vector[1] = 7;
        send_activation_vector(activation_vector);

        activation_vector[0] = 8;
        activation_vector[1] = -9;
        send_activation_vector(activation_vector);

        expect_relu_before_reduction("ReLU-before-reduction row 0", -6*SCALE, 7*SCALE);
        expect_relu_before_reduction("ReLU-before-reduction row 1", 8*SCALE, -9*SCALE);

        // Reload a matrix that produces the largest reachable positive raw
        // result (32768) in both lanes: (-128*-128) + (-128*-128).
        request_weight_reload();
        weight_vector[0] = -128;
        weight_vector[1] = -128;
        send_weight_vector(weight_vector);
        send_weight_vector(weight_vector);
        wait(weightsLoaded);

        passThrough = 1'b1;
        reductionWeight[0] = 32767;
        reductionWeight[1] = 32767;

        activation_vector[0] = -128;
        activation_vector[1] = -128;
        send_activation_vector(activation_vector);
        send_activation_vector(activation_vector);
        send_activation_vector(activation_vector);
        send_activation_vector(activation_vector);

        // Each product (32768*32767) fits in 31 signed bits, but their sum
        // needs 32 signed bits before the one final rescale. The bridge has a
        // result staging register plus the SPI shifter, so a second back-to-back
        // matrix is queued to make
        // the third prediction encounter the existing output backpressure.
        check_stalled_prediction("reduced output backpressure", 4095);
        expect_configured_row("accumulation-width row 0", 32768, 32768);
        expect_configured_row("accumulation-width row 1", 32768, 32768);
        expect_configured_row("back-to-back accumulation-width row 0", 32768, 32768);
        expect_configured_row("back-to-back accumulation-width row 1", 32768, 32768);

        // The most-negative 16-bit coefficient exercises the full signed
        // product path; two -2^30 terms also reach exactly -2^31.
        reductionWeight[0] = -32768;
        reductionWeight[1] = -32768;

        send_activation_vector(activation_vector);
        send_activation_vector(activation_vector);

        expect_configured_row("full signed-product-width row 0", 32768, 32768);
        expect_configured_row("full signed-product-width row 1", 32768, 32768);

        $display("PASS: asynchronous-clock SPI integration test completed.");
        $finish;
    end

    initial begin
        #100_000;
        $fatal(1, "FAIL: asynchronous-clock SPI integration test timed out.");
    end

    task send_weight_vector(input data_t vector[N]);
        wait(weightReady);
        @(negedge sclk);
        for (int lane = 0; lane < N; lane++)
            weightCs_n[lane] = 1'b0;

        for (int bitIndex = WIDTH-1; bitIndex >= 0; bitIndex--) begin
            for (int lane = 0; lane < N; lane++)
                weightMosi[lane] = vector[lane][bitIndex];
            @(posedge sclk);
            @(negedge sclk);
        end

        for (int lane = 0; lane < N; lane++)
            weightCs_n[lane] = 1'b1;
    endtask

    task send_activation_vector(input data_t vector[N]);
        wait(activationReady);
        @(negedge sclk);
        for (int lane = 0; lane < N; lane++)
            activationCs_n[lane] = 1'b0;

        for (int bitIndex = WIDTH-1; bitIndex >= 0; bitIndex--) begin
            for (int lane = 0; lane < N; lane++)
                activationMosi[lane] = vector[lane][bitIndex];
            @(posedge sclk);
            @(negedge sclk);
        end

        for (int lane = 0; lane < N; lane++)
            activationCs_n[lane] = 1'b1;
    endtask

    task request_weight_reload();
        wait(reloadReady);
        @(negedge clk) reloadWeights = 1'b1;
        @(posedge clk);
        @(negedge clk) reloadWeights = 1'b0;
        wait(!weightsLoaded);
    endtask

    task expect_configured_row(
        input string label,
        input integer raw0,
        input integer raw1
    );
        result_t activated0, activated1;
        result_t expected0, expected1;
        longint signed activatedFull0, activatedFull1;
        longint signed fullReduction;

        activatedFull0 = (!passThrough && raw0 < 0) ? 0 : raw0;
        activatedFull1 = (!passThrough && raw1 < 0) ? 0 : raw1;
        activated0 = activatedFull0;
        activated1 = activatedFull1;
        if (reduceOutput) begin
            fullReduction = activatedFull0 * $signed(reductionWeight[0])
                            + activatedFull1 * $signed(reductionWeight[1]);
            expected0 = fullReduction >>> RESCALE_SHIFT;
            expected1 = '0;
        end else begin
            expected0 = activated0;
            expected1 = activated1;
        end
        expect_result_row(label, expected0, expected1);
    endtask

    task expect_relu_before_reduction(
        input string label,
        input integer raw0,
        input integer raw1
    );
        result_t reduce_after_relu;
        result_t relu_after_reduce;
        result_t raw_sum;
        longint signed reduceAfterReluFull;
        longint signed rawFull;

        reduceAfterReluFull = ((raw0 < 0) ? 0 : raw0)
                              * $signed(reductionWeight[0])
                              + ((raw1 < 0) ? 0 : raw1)
                              * $signed(reductionWeight[1]);
        rawFull = raw0 * $signed(reductionWeight[0])
                  + raw1 * $signed(reductionWeight[1]);
        reduce_after_relu = reduceAfterReluFull >>> RESCALE_SHIFT;
        raw_sum = rawFull >>> RESCALE_SHIFT;
        relu_after_reduce = (raw_sum < 0) ? 0 : raw_sum;
        if (reduce_after_relu == relu_after_reduce)
            $fatal(1, "FAIL: %s does not distinguish activation/reduction order.", label);
        expect_result_row(label, reduce_after_relu, 0);
    endtask

    task check_stalled_prediction(input string label, input result_t expected);
        result_t held_prediction;

        wait(dut.resultValid && !dut.resultReady);
        held_prediction = dut.resultData[0];
        if (held_prediction !== expected || dut.resultData[1] !== '0)
            $fatal(1, "FAIL: %s initially got [%0d, %0d], expected [%0d, 0].",
                   label, dut.resultData[0], dut.resultData[1], expected);
        repeat (5) begin
            @(posedge clk);
            if (!dut.resultValid || dut.resultReady
                || dut.resultData[0] !== held_prediction
                || dut.resultData[1] !== '0)
                $fatal(1, "FAIL: %s prediction changed while stalled.", label);
        end
        $display("PASS: %s held prediction %0d stable", label, held_prediction);
    endtask

    task expect_result_row(
        input string label,
        input result_t expected0,
        input result_t expected1
    );
        result_t actual[N];
        bit result_available;

        result_available = 1'b0;

        // Poll only through top-level signals. Failed polls raise CS again before
        // the next SCLK edge, so no result bit is consumed.
        while (!result_available) begin
            @(negedge sclk);
            for (int lane = 0; lane < N; lane++) cs_n[lane] = 1'b0;
            #1;
            result_available = 1'b1;
            for (int lane = 0; lane < N; lane++)
                result_available &= misoValid[lane];
            if (!result_available)
                for (int lane = 0; lane < N; lane++) cs_n[lane] = 1'b1;
        end

        for (int bitIndex = PREDICTION_WIDTH-1; bitIndex >= 0; bitIndex--) begin
            @(posedge sclk);
            for (int lane = 0; lane < N; lane++) begin
                assert(misoValid[lane])
                    else $fatal(1, "FAIL: MISO lane %0d was not valid.", lane);
                actual[lane][bitIndex] = miso[lane];
            end
        end

        @(negedge sclk);
        for (int lane = 0; lane < N; lane++) cs_n[lane] = 1'b1;

        if (actual[0] !== expected0 || actual[1] !== expected1)
            $fatal(1, "FAIL: %s got [%0d, %0d], expected [%0d, %0d].",
                   label, actual[0], actual[1], expected0, expected1);
        $display("PASS: %s = [%0d, %0d]", label, actual[0], actual[1]);
    endtask

endmodule
