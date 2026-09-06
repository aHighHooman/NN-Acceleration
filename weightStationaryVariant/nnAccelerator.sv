module nnAccelerator #(
    parameter int WIDTH = 16,
    parameter int N = 3,
    parameter int REDUCTION_WEIGHT_WIDTH = WIDTH,
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
    input  logic signed [REDUCTION_WEIGHT_WIDTH-1:0] reductionWeight [N],
    input  logic                              reduceOutput,
    output logic signed [2*WIDTH+REDUCTION_WEIGHT_WIDTH+2*$clog2(N)-1:0] resultData [N],
    output logic                              resultValid,
    input  logic                              resultReady,
    output logic                              resultLast,
    output logic                              weightsLoaded,
    input  logic                              reloadWeights,
    output logic                              reloadReady,
    input  logic                              passThrough
);

    localparam int ACTIVATED_WIDTH = 2*WIDTH + $clog2(N);
    localparam int PREDICTION_WIDTH = ACTIVATED_WIDTH
                                      + REDUCTION_WEIGHT_WIDTH + $clog2(N);

    logic signed [ACTIVATED_WIDTH-1:0] rawResultData[N];
    logic signed [ACTIVATED_WIDTH-1:0] activatedData[N];
    logic signed [PREDICTION_WIDTH-1:0] prediction;

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

    activationLayer #(.WIDTH(ACTIVATED_WIDTH), .N(N)) resultActivation (
        .inputData(rawResultData), .passThrough(passThrough), .outputData(activatedData)
    );

    weightedVectorReduction #(
        .INPUT_WIDTH(ACTIVATED_WIDTH),
        .WEIGHT_WIDTH(REDUCTION_WEIGHT_WIDTH),
        .N(N)
    ) weightedReadout (
        .inputData(activatedData),
        .reductionWeight(reductionWeight),
        .prediction(prediction)
    );

    always_comb begin
        for (int lane = 0; lane < N; lane++) begin
            if (reduceOutput)
                resultData[lane] = (lane == 0) ? prediction : '0;
            else
                resultData[lane] =
                    {{(PREDICTION_WIDTH-ACTIVATED_WIDTH)
                       {activatedData[lane][ACTIVATED_WIDTH-1]}},
                     activatedData[lane]};
        end
    end

endmodule
