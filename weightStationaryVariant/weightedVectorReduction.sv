module weightedVectorReduction #(
    parameter int MATRIX_RESULT_WIDTH = 16,
    parameter int REDUCTION_WEIGHT_WIDTH = 8,
    parameter int N = 3,
    parameter int FRACTION_BITS = 4
)(
    input  logic signed [MATRIX_RESULT_WIDTH-1:0]    inputData [N],
    input  logic signed [REDUCTION_WEIGHT_WIDTH-1:0] reductionWeight [N],
    output logic signed [MATRIX_RESULT_WIDTH+$clog2(N)-1:0] prediction
);

    localparam int REDUCTION_FRACTION_BITS = REDUCTION_WEIGHT_WIDTH - 1;
    localparam int PRODUCT_WIDTH = MATRIX_RESULT_WIDTH
                                   + REDUCTION_WEIGHT_WIDTH;
    localparam int ACCUMULATOR_WIDTH = PRODUCT_WIDTH + $clog2(N);
    localparam int PREDICTION_WIDTH = MATRIX_RESULT_WIDTH + $clog2(N);
    localparam int RESCALE_SHIFT = FRACTION_BITS + REDUCTION_FRACTION_BITS;

    initial begin
        if (MATRIX_RESULT_WIDTH < 1 || REDUCTION_WEIGHT_WIDTH < 1 ||
            N < 1 || FRACTION_BITS < 0)
            $fatal(1, "reduction widths/N>=1 and FRACTION_BITS>=0");
    end

    logic signed [PRODUCT_WIDTH-1:0] product [N];
    logic signed [ACCUMULATOR_WIDTH-1:0] extendedProduct [N];
    logic signed [ACCUMULATOR_WIDTH-1:0] accumulator;
    logic signed [ACCUMULATOR_WIDTH-1:0] rescaledAccumulator;

    genvar term;
    generate
        for (term = 0; term < N; term = term + 1) begin : reduction_terms
            // A signed MATRIX_RESULT_WIDTH x REDUCTION_WEIGHT_WIDTH product
            // is exactly PRODUCT_WIDTH bits, so every bit is preserved.
            // Keep the operands at native width: explicit sign extension
            // to PRODUCT_WIDTH makes synthesis build a truncated
            // PRODUCT_WIDTH x PRODUCT_WIDTH multiplier instead.
            assign product[term] = inputData[term] * reductionWeight[term];
            assign extendedProduct[term] =
                {{(ACCUMULATOR_WIDTH-PRODUCT_WIDTH)
                   {product[term][PRODUCT_WIDTH-1]}},
                 product[term]};
        end
    endgenerate

    always_comb begin
        // Every add occurs at ACCUMULATOR_WIDTH; no term is rescaled first.
        accumulator = '0;
        for (int termIndex = 0; termIndex < N; termIndex++)
            accumulator = accumulator + extendedProduct[termIndex];
    end

    // The sum has 2*FRACTION_BITS + REDUCTION_FRACTION_BITS fractional bits.
    // Shift only the full sum to restore the input/target binary point.
    assign rescaledAccumulator = accumulator >>> RESCALE_SHIFT;
    assign prediction = $signed(
        rescaledAccumulator[PREDICTION_WIDTH-1:0]
    );

endmodule
