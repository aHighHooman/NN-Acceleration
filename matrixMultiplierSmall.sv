module matrixMultiplierSmall #(
    parameter int WIDTH = 6,
    parameter int N = 3
)(
    input  logic        clk,
    input  logic        rst_n,
    input  logic signed [WIDTH-1:0] matrixA [N][N],
    input  logic signed [WIDTH-1:0] matrixB [N][N],
    output logic signed [2*WIDTH-1:0] resultMatrix [N][N]
);
    
    // Needed for internal connections, sadly looks like the synthesizer isn'st smart enough to optimize these away
    logic signed [WIDTH-1:0] rowData [N];
    logic signed [WIDTH-1:0] colData [N];
    
    dataOrchestrator #(
        .WIDTH(WIDTH),
        .N(N)
    ) dataOrch (
        .clk      (clk),
        .rst_n    (rst_n),
        .matrixA  (matrixA),
        .matrixB  (matrixB),
        .row      (rowData),
        .col      (colData)
    );

    systolicArray #(
        .WIDTH(WIDTH),
        .N(N)
    ) systolicArr (
        .clk          (clk),
        .rst_n        (rst_n),
        .row          (rowData),
        .col          (colData),
        .resultMatrix (resultMatrix)
    );
    
endmodule