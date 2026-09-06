module weightedVectorReduction #(
    parameter int INPUT_WIDTH = 16,
    parameter int WEIGHT_WIDTH = 16,
    parameter int N = 3
)(
    input  logic signed [INPUT_WIDTH-1:0]  inputData [N],
    input  logic signed [WEIGHT_WIDTH-1:0] reductionWeight [N],
    output logic signed [INPUT_WIDTH+WEIGHT_WIDTH+$clog2(N)-1:0] prediction
);

    localparam int PRODUCT_WIDTH = INPUT_WIDTH + WEIGHT_WIDTH;
    localparam int OUTPUT_WIDTH = PRODUCT_WIDTH + $clog2(N);

    logic signed [PRODUCT_WIDTH-1:0] product [N];
    logic signed [OUTPUT_WIDTH-1:0] extendedProduct [N];

    genvar term;
    generate
        for (term = 0; term < N; term = term + 1) begin : reduction_terms
            logic signed [PRODUCT_WIDTH-1:0] extendedInput;
            logic signed [PRODUCT_WIDTH-1:0] extendedWeight;

            assign extendedInput = {{WEIGHT_WIDTH{inputData[term][INPUT_WIDTH-1]}},
                                    inputData[term]};
            assign extendedWeight = {{INPUT_WIDTH{reductionWeight[term][WEIGHT_WIDTH-1]}},
                                     reductionWeight[term]};
            assign product[term] = extendedInput * extendedWeight;
            assign extendedProduct[term] =
                {{(OUTPUT_WIDTH-PRODUCT_WIDTH){product[term][PRODUCT_WIDTH-1]}},
                 product[term]};
        end
    endgenerate

    always_comb begin
        prediction = '0;
        for (int termIndex = 0; termIndex < N; termIndex++)
            prediction = prediction + extendedProduct[termIndex];
    end

endmodule
