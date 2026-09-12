`timescale 1ns / 1ps

module weightedVectorReduction_tb;
    localparam int MATRIX_RESULT_WIDTH = 8;
    localparam int REDUCTION_WEIGHT_WIDTH = 6;
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
        set_values(64, 32, 16, 8, 16, 16, 16, 16);
        check_prediction("positive fractional weights", 1920, 15);

        set_values(80, -48, 32, -16, 16, 8, -16, 24);
        check_prediction("mixed positive and negative inputs", 0, 0);

        set_values(7, 6, 5, 4, -3, 2, -1, 2);
        check_prediction("mixed positive and negative weights", -6, -1);

        set_values(12, 12, -12, -12, 3, -3, 3, -3);
        check_prediction("cancellation between terms", 0, 0);

        // The first term is (-128)*(-32) = +4096. That value does not fit in
        // 13 signed bits, so it requires the full
        // MATRIX_RESULT_WIDTH+REDUCTION_WEIGHT_WIDTH product width before the
        // remaining terms cancel it down to one.
        set_values(-128, 127, -128, 127, -32, 31, 31, -32);
        check_prediction("full product width with signed cancellation", 1, 0);

        set_values(127, 127, 127, 127, 31, 31, 31, 31);
        check_prediction("sum requiring accumulation bits", 15748, 123);

        set_values(-128, -128, -128, -128, 31, 31, 31, 31);
        check_prediction("negative arithmetic rescale", -15872, -124);

        $display("PASS: weighted vector reduction tests completed.");
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
