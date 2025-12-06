module matrixMultiplierSmall #(
    parameter int WIDTH = 16,
    parameter int N = 3
)(
    input  logic        clk,
    input  logic        rst_n,
    output logic signed [2*WIDTH-1:0] resultMatrix [N][N]
);
    
    logic signed [WIDTH-1:0] rowData_OrchToSyst [N];
    logic signed [WIDTH-1:0] colData_OrchToSyst [N];

    logic        [7:0]        addrA_OrchToMem   [N];
    logic        [7:0]        addrB_OrchToMem   [N];

    logic signed [WIDTH-1:0]  colA_MemToOrch    [N];
    logic signed [WIDTH-1:0]  rowB_MemToOrch    [N];
    
    memBlock #(
        .WIDTH(WIDTH),
        .DEPTH(256),
        .N(N)
    ) memA (
        .clk      (clk),
        .rst_n    (rst_n),
        .addr     (addrA_OrchToMem),
        .dataOut  (colA_MemToOrch)
    );

    memBlock #(
        .WIDTH(WIDTH),
        .DEPTH(256),
        .N(N)
    ) memB (
        .clk      (clk),
        .rst_n    (rst_n),
        .addr     (addrB_OrchToMem),
        .dataOut  (rowB_MemToOrch)
    );

    dataOrchestrator #(
        .WIDTH(WIDTH),
        .N(N)
    ) dataOrch (
        .clk      (clk),
        .rst_n    (rst_n),
        .colA     (colA_MemToOrch),
        .rowB     (rowB_MemToOrch),
        .row      (rowData_OrchToSyst),
        .col      (colData_OrchToSyst),
        .addrA    (addrA_OrchToMem),
        .addrB    (addrB_OrchToMem)
    );

    systolicArray #(
        .WIDTH(WIDTH),
        .N(N)
    ) systolicArr (
        .clk          (clk),
        .rst_n        (rst_n),
        .row          (rowData_OrchToSyst),
        .col          (colData_OrchToSyst),
        .resultMatrix (resultMatrix)
    );
    
endmodule