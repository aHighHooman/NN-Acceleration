module nnAccelerator #(
    parameter int WIDTH = 16,
    parameter int N = 3,
    parameter int INPUT_FIFO_DEPTH = 2*N,
    parameter int OUTPUT_FIFO_DEPTH = 2*N
)(
    input  logic                              clk,
    input  logic                              rst_n,
    input  logic signed [WIDTH-1:0]           weightData [N],
    input  logic                              weightValid,
    output logic                              weightReady,
    input  logic signed [WIDTH-1:0]           activationData [N],
    input  logic                              activationValid,
    output logic                              activationReady,
    output logic signed [2*WIDTH+$clog2(N)-1:0] resultData [N],
    output logic                              resultValid,
    input  logic                              resultReady,
    output logic                              resultLast,
    output logic                              weightsLoaded,
    input  logic                              reloadWeights,
    output logic                              reloadReady,
    input  logic                              passThrough
);

    localparam int RESULT_WIDTH = 2*WIDTH + $clog2(N);

    logic signed [RESULT_WIDTH-1:0] rawResultData[N];

    matrixMultiplierWeightStationary #(
        .WIDTH(WIDTH), .N(N),
        .INPUT_FIFO_DEPTH(INPUT_FIFO_DEPTH),
        .OUTPUT_FIFO_DEPTH(OUTPUT_FIFO_DEPTH)
    ) matrixEngine (
        .clk(clk), .rst_n(rst_n),
        .weightData(weightData), .weightValid(weightValid), .weightReady(weightReady),
        .activationData(activationData), .activationValid(activationValid),
        .activationReady(activationReady), .resultData(rawResultData),
        .resultValid(resultValid), .resultReady(resultReady),
        .resultLast(resultLast), .weightsLoaded(weightsLoaded),
        .reloadWeights(reloadWeights), .reloadReady(reloadReady)
    );

    activationLayer #(.WIDTH(RESULT_WIDTH), .N(N)) resultActivation (
        .inputData(rawResultData), .passThrough(passThrough), .outputData(resultData)
    );

endmodule
