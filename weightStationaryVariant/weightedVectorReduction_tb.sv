`timescale 1ns / 1ps

module weightedVectorReduction_tb;
    localparam int INPUT_WIDTH = 8;
    localparam int WEIGHT_WIDTH = 6;
    localparam int N = 4;
    localparam int OUTPUT_WIDTH = INPUT_WIDTH + WEIGHT_WIDTH + $clog2(N);

    logic signed [INPUT_WIDTH-1:0] inputData [N];
    logic signed [WEIGHT_WIDTH-1:0] reductionWeight [N];
    logic signed [OUTPUT_WIDTH-1:0] prediction;

    weightedVectorReduction #(
        .INPUT_WIDTH(INPUT_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .N(N)
    ) dut (
        .inputData(inputData),
        .reductionWeight(reductionWeight),
        .prediction(prediction)
    );

    initial begin
        set_values(1, 2, 3, 4, 5, 6, 7, 8);
        check_prediction("positive inputs and weights", 70);

        set_values(5, -4, 3, -2, 2, 3, 4, 5);
        check_prediction("mixed positive and negative inputs", 0);

        set_values(7, 6, 5, 4, -3, 2, -1, 4);
        check_prediction("mixed positive and negative weights", 2);

        set_values(12, 12, -12, -12, 3, -3, 3, -3);
        check_prediction("cancellation between terms", 0);

        // The first term is (-128)*(-32) = +4096. That value does not fit in
        // 13 signed bits, so it requires the full INPUT_WIDTH+WEIGHT_WIDTH
        // product width before the remaining terms cancel it down to one.
        set_values(-128, 127, -128, 127, -32, 31, 31, -32);
        check_prediction("full product width with signed cancellation", 1);

        set_values(127, 127, 127, 127, 31, 31, 31, 31);
        check_prediction("sum requiring accumulation bits", 15748);

        set_values(-128, -128, -128, -128, 31, 31, 31, 31);
        check_prediction("negative sum near accumulation limit", -15872);

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

    task check_prediction(input string label, input integer expected);
        #1;
        if (prediction !== OUTPUT_WIDTH'(expected))
            $fatal(1, "FAIL: %s: got %0d, expected %0d", label, prediction, expected);
        $display("PASS: %s", label);
    endtask

endmodule
