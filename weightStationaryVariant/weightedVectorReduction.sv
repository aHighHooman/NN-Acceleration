module weightedVectorReduction #(
    parameter int MATRIX_RESULT_WIDTH = 16,
    parameter int REDUCTION_WEIGHT_WIDTH = 16,
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

    logic signed [PRODUCT_WIDTH-1:0] product [N];
    logic signed [ACCUMULATOR_WIDTH-1:0] extendedProduct [N];
    logic signed [ACCUMULATOR_WIDTH-1:0] accumulator;
    logic signed [ACCUMULATOR_WIDTH-1:0] rescaledAccumulator;

    genvar term;
    generate
        for (term = 0; term < N; term = term + 1) begin : reduction_terms
            logic signed [PRODUCT_WIDTH-1:0] extendedInput;
            logic signed [PRODUCT_WIDTH-1:0] extendedWeight;

            // Explicit signed extension makes the multiply expression
            // PRODUCT_WIDTH wide, preserving every matrix/coefficient bit.
            assign extendedInput =
                {{REDUCTION_WEIGHT_WIDTH
                   {inputData[term][MATRIX_RESULT_WIDTH-1]}},
                 inputData[term]};
            assign extendedWeight =
                {{MATRIX_RESULT_WIDTH
                   {reductionWeight[term][REDUCTION_WEIGHT_WIDTH-1]}},
                 reductionWeight[term]};
            assign product[term] = extendedInput * extendedWeight;
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

    // inputData retains the matrix result's 2*FRACTION_BITS binary point.
    // Reduction coefficients contribute REDUCTION_FRACTION_BITS more.  Shift
    // only the completed full-width sum so prediction returns to the input and
    // target binary-point position.
    assign rescaledAccumulator = accumulator >>> RESCALE_SHIFT;
    assign prediction = $signed(
        rescaledAccumulator[PREDICTION_WIDTH-1:0]
    );

endmodule
