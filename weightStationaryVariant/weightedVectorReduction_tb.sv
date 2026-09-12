`timescale 1ns / 1ps

module weightedVectorReduction_tb;
    localparam int MATRIX_RESULT_WIDTH = 8;
    localparam int REDUCTION_WEIGHT_WIDTH = 8;
    localparam int N = 4;
    localparam int FRACTION_BITS = 2;
    localparam int PREDICTION_WIDTH = MATRIX_RESULT_WIDTH + $clog2(N);
    localparam int RESCALE_SHIFT = FRACTION_BITS
                                   + REDUCTION_WEIGHT_WIDTH - 1;

    logic signed [MATRIX_RESULT_WIDTH-1:0] inputData [N];
    logic signed [REDUCTION_WEIGHT_WIDTH-1:0] reductionWeight [N];
    logic signed [PREDICTION_WIDTH-1:0] prediction;

    weightedVectorReduction #(
        .MATRIX_RESULT_WIDTH(MATRIX_RESULT_WIDTH),
        .REDUCTION_WEIGHT_WIDTH(REDUCTION_WEIGHT_WIDTH),
        .N(N),
        .FRACTION_BITS(FRACTION_BITS)
    ) dut (
        .inputData(inputData),
        .reductionWeight(reductionWeight),
        .prediction(prediction)
    );

    initial begin
        if (REDUCTION_WEIGHT_WIDTH != 8 ||
            (1.0 / (1 << (REDUCTION_WEIGHT_WIDTH-1))) != 0.0078125)
            $fatal(1, "8-bit reduction coefficients are not Q1.7 with a 1/128 LSB");
        if ($bits(dut.product[0]) !=
                MATRIX_RESULT_WIDTH + REDUCTION_WEIGHT_WIDTH ||
            $bits(dut.accumulator) !=
                MATRIX_RESULT_WIDTH + REDUCTION_WEIGHT_WIDTH + $clog2(N) ||
            $bits(prediction) != MATRIX_RESULT_WIDTH + $clog2(N))
            $fatal(1, "weighted reduction intermediate/output widths changed");

        set_values(64, 32, 16, 8, 64, 64, 64, 64);
        check_prediction("positive fractional weights", 7680, 15);

        set_values(80, -48, 32, -16, 64, 32, -64, 96);
        check_prediction("mixed positive and negative inputs", 0, 0);

        set_values(7, 6, 5, 4, -12, 8, -4, 8);
        check_prediction("mixed positive and negative weights", -24, -1);

        set_values(12, 12, -12, -12, 12, -12, 12, -12);
        check_prediction("cancellation between terms", 0, 0);

        // The first term is (-128)*(-128) = +16384. That value does not fit
        // in 15 signed bits, so it requires the full
        // MATRIX_RESULT_WIDTH+REDUCTION_WEIGHT_WIDTH product width before the
        // remaining terms cancel it down to one.
        set_values(-128, 127, -128, 127, -128, 127, 127, -128);
        check_prediction("full product width with signed cancellation", 1, 0);

        set_values(127, 127, 127, 127, 127, 127, 127, 127);
        check_prediction("sum requiring accumulation bits", 64516, 126);

        set_values(-128, -128, -128, -128, 127, 127, 127, 127);
        check_prediction("negative arithmetic rescale", -65024, -127);

        $display("PASS: full-width products/accumulation, one final arithmetic rescale, and architectural prediction width.");
        $finish;
    end

    task set_values(
        input integer input0, input integer input1,
        input integer input2, input integer input3,
        input integer weight0, input integer weight1,
        input integer weight2, input integer weight3
    );
        inputData[0] = input0;
        inputData[1] = input1;
        inputData[2] = input2;
        inputData[3] = input3;
        reductionWeight[0] = weight0;
        reductionWeight[1] = weight1;
        reductionWeight[2] = weight2;
        reductionWeight[3] = weight3;
    endtask

    task check_prediction(
        input string label,
        input integer expectedAccumulator,
        input integer expectedPrediction
    );
        #1;
        if (dut.accumulator !== expectedAccumulator)
            $fatal(1, "FAIL: %s accumulator: got %0d, expected %0d",
                   label, dut.accumulator, expectedAccumulator);
        if (prediction !== PREDICTION_WIDTH'(expectedPrediction))
            $fatal(1, "FAIL: %s prediction: got %0d, expected %0d",
                   label, prediction, expectedPrediction);
        if (prediction !== PREDICTION_WIDTH'(expectedAccumulator >>> RESCALE_SHIFT))
            $fatal(1, "FAIL: %s did not shift the completed sum once", label);
        $display("PASS: %s", label);
    endtask

endmodule
