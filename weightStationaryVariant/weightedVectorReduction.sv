// Scalar readout: an Nx1 column of PEs (row 0 at the top). Each row takes one
// lane of the last matrix stage on its center-first front, so no lane is
// realigned. Sums run in two chains, rows C-1..0 upward and rows C..N-1
// downward; a combinational join adds the chain ends.
module weightedVectorReduction #(
    parameter int MATRIX_RESULT_WIDTH = 16,
    parameter int REDUCTION_WEIGHT_WIDTH = 8,
    parameter int N = 3,
    parameter int RESCALE_SHIFT = REDUCTION_WEIGHT_WIDTH-1
)(
    input logic clk, rst_n, advance,
    input logic cfg_we,
    input logic signed [REDUCTION_WEIGHT_WIDTH-1:0] cfg_data[N],
    input logic update_valid,
    input logic signed [1:0] row_direction[N], column_direction,
    input logic signed [MATRIX_RESULT_WIDTH-1:0] in_data[N],
    input logic in_valid[N],
    output logic signed [MATRIX_RESULT_WIDTH+$clog2(N)-1:0] prediction,
    output logic predictionValid,
    output logic signed [REDUCTION_WEIGHT_WIDTH-1:0] weights[N]
);
    localparam int C = (N-1)/2;
    localparam int SUM_WIDTH = MATRIX_RESULT_WIDTH+REDUCTION_WEIGHT_WIDTH+$clog2(N);

    logic signed [SUM_WIDTH-1:0] sumIn[N], sumOut[N];
    logic signed [1:0] updateDirection[N];
    logic [N-1:0] activationValid;
    genvar i;

    initial begin
        if (MATRIX_RESULT_WIDTH<1 || REDUCTION_WEIGHT_WIDTH<1 || N<3 || N%2==0 || RESCALE_SHIFT<0)
            $fatal(1, "Fountain requires positive reduction widths and odd N>=3");
    end

    always_comb begin
        // Sums: rows C-1 and C seed the upward and downward chains.
        sumIn[C-1] = '0;
        sumIn[C]   = '0;
        for (int i=0; i<C-1; i++) sumIn[i] = sumOut[i+1];
        for (int i=C+1; i<N; i++) sumIn[i] = sumOut[i-1];
        // Ternary signs need comparisons, not a DSP multiplier per PE.
        for (int i=0; i<N; i++)
            updateDirection[i] = (row_direction[i]==0 || column_direction==0) ? 2'sd0 :
                (row_direction[i]==column_direction ? 2'sd1 : -2'sd1);
    end

    generate
    for (i=0; i<N; i++) begin : rows
        weightStationaryProcessingElement #(.INPUT_WIDTH(MATRIX_RESULT_WIDTH),
            .WIDTH(REDUCTION_WEIGHT_WIDTH), .RESULT_WIDTH(SUM_WIDTH)) pe (
            .clk, .rst_n, .advance, .backward(1'b0),
            .cfg_we, .cfg_data(cfg_data[i]), .cfg_transposed_we(1'b0), .cfg_transposed_data('0),
            .update_valid, .update_direction(updateDirection[i]), .transposed_update_direction(2'sd0),
            .activation_in(in_data[i]), .activation_valid(in_valid[i]),
            .sum_in(sumIn[i]), .sum_out(sumOut[i]),
            .activation_out(), .activation_out_valid(activationValid[i]),
            .weight(weights[i])
        );
    end
    endgenerate

    // Shift only the complete sum to restore the input/target binary point.
    assign prediction = (sumOut[0] + sumOut[N-1]) >>> RESCALE_SHIFT;
    assign predictionValid = activationValid[0];

    // synthesis translate_off
    always_ff @(posedge clk) begin
        if (rst_n) begin
            if (advance) assert (activationValid[0]==activationValid[N-1])
                else $fatal(1, "Fountain reduction join valid mismatch");
            if (update_valid) assert (activationValid=='0)
                else $fatal(1, "Reduction update requires an idle column");
        end
    end
    // synthesis translate_on
endmodule
