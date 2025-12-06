module matrixMultiplierWeightStationary #(
    parameter int WIDTH = 16,
    parameter int N = 3
)(
    input  logic        clk,
    input  logic        rst_n,
    output logic signed [2*WIDTH-1:0] resultMatrix [N][N]
);
    
    logic signed [WIDTH-1:0] rowData_OrchToSyst    [N];
    logic signed [WIDTH-1:0] colData_WeightToSyst  [N];

    logic        [7:0]         addrA_OrchToMem      [N];
    logic        [7:0]         addrB_WeightToMem    [N];

    logic signed [WIDTH-1:0]   rowA_MemToOrch       [N];
    logic signed [WIDTH-1:0]   colB_MemToWeight     [N];
    
    logic                      writeEnableWeightToSyst;
    logic signed [2*WIDTH-1:0] outputs             [N];
    
    memBlock #(
        .WIDTH(WIDTH),
        .DEPTH(256),
        .N(N)
    ) memInput (
        .clk      (clk),
        .rst_n    (rst_n),
        .addr     (addrA_OrchToMem),
        .dataOut  (rowA_MemToOrch)
    );

    memBlock #(
        .WIDTH(WIDTH),
        .DEPTH(256),
        .N(N)
    ) memWeight (
        .clk      (clk),
        .rst_n    (rst_n),
        .addr     (addrB_WeightToMem),
        .dataOut  (colB_MemToWeight)
    );

    dataOrchestratorWeightStationary #(
        .WIDTH(WIDTH),
        .N(N)
    ) dataOrch (
        .clk      (clk),
        .rst_n    (rst_n),
        .writeEnable (writeEnableWeightToSyst),
        .rowA     (rowA_MemToOrch),
        .row      (rowData_OrchToSyst),
        .addrA    (addrA_OrchToMem)
    );

    weightLoader #(
        .WIDTH(WIDTH),
        .N(N)
    ) weightLoader (
        .clk      (clk),
        .rst_n    (rst_n),
        .colB     (colB_MemToWeight),
        .col      (colData_WeightToSyst),
        .addrB    (addrB_WeightToMem),
        .writeEnable (writeEnableWeightToSyst)
    );

    systolicArrayWeightStationary #(
        .WIDTH(WIDTH),
        .N(N)
    ) systolicArr (
        .clk          (clk),
        .rst_n        (rst_n),
        .row          (rowData_OrchToSyst),
        .col          (colData_WeightToSyst),
        .writeEnable  (writeEnableWeightToSyst),
        .result       (outputs)
    );
    

endmodule