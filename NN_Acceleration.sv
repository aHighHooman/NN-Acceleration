module NN_Acceleration #(
    parameter int WIDTH = 16,
    parameter int N = 3
)(
    input  logic        clk,
    input  logic        rst_n,
    input  logic signed [WIDTH-1:0] matrixA [N][N],
    input  logic signed [WIDTH-1:0] matrixB [N][N],
    output logic signed [2*WIDTH-1:0] resultMatrix [N][N]
);

matrixMultiplierSmall #(
    .WIDTH(WIDTH),
    .N(N)
) matMult (
    .clk          (clk),
    .rst_n        (rst_n),
    .matrixA      (matrixA),
    .matrixB      (matrixB),
    .resultMatrix (resultMatrix)
);

endmodule