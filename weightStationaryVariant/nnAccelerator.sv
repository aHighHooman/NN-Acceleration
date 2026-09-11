module nnAccelerator #(
    parameter int WIDTH = 16,
    parameter int N = 3,
    parameter int TARGET_WIDTH = WIDTH,
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
    input  logic signed [TARGET_WIDTH-1:0]    targetData,
    input  logic                              activationValid,
    output logic                              activationReady,
    input  logic signed [REDUCTION_WEIGHT_WIDTH-1:0] reductionWeight [N],
    input  logic                              reduceOutput,
    output logic signed [2*WIDTH+REDUCTION_WEIGHT_WIDTH+2*$clog2(N)-1:0] resultData [N],
    output logic signed [TARGET_WIDTH-1:0]    resultTargetData,
    output logic signed [1:0]                 learningDirection,
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
    localparam int COMPARE_WIDTH = (PREDICTION_WIDTH > TARGET_WIDTH)
                                   ? PREDICTION_WIDTH : TARGET_WIDTH;

    logic signed [ACTIVATED_WIDTH-1:0] rawResultData[N];
    logic signed [ACTIVATED_WIDTH-1:0] activatedData[N];
    logic signed [PREDICTION_WIDTH-1:0] prediction;
    logic signed [COMPARE_WIDTH-1:0] comparePrediction, compareTarget;
    logic matrixActivationValid, matrixActivationReady;
    logic matrixResultValid, matrixResultReady, matrixResultLast;
    logic targetPush, targetPop, targetFull, targetEmpty;
    logic targetCanAccept;

    // The activation vector and target are one input transaction.  Gate the
    // matrix valid as well as the external ready so neither side can advance
    // alone when the target queue applies backpressure.
    assign targetPop             = resultValid && resultReady;
    assign targetCanAccept       = !targetFull || targetPop;
    assign activationReady       = matrixActivationReady && targetCanAccept;
    assign matrixActivationValid = activationValid && targetCanAccept;
    assign targetPush            = activationValid && activationReady;

    assign resultValid       = matrixResultValid && !targetEmpty;
    assign matrixResultReady = resultReady && !targetEmpty;
    assign resultLast        = matrixResultLast && !targetEmpty;

    // Compare the full weighted-reduction result with the aligned FIFO head.
    // Assignment to the wider signed signals sign-extends either narrower side.
    assign comparePrediction = prediction;
    assign compareTarget     = resultTargetData;

    always_comb begin
        learningDirection = 2'sd0;
        if (resultValid) begin
            if (compareTarget > comparePrediction)
                learningDirection = 2'sd1;
            else if (compareTarget < comparePrediction)
                learningDirection = -2'sd1;
        end
    end

    // FIFO order, rather than a cycle count, carries each scalar target to the
    // result transaction produced by the corresponding activation vector.
    signedFifo #(
        .WIDTH(TARGET_WIDTH),
        .DEPTH(INPUT_FIFO_DEPTH)
    ) targetFifo (
        .clk(clk), .rst_n(rst_n),
        .push(targetPush), .pushData(targetData),
        .pop(targetPop), .popData(resultTargetData),
        .full(targetFull), .empty(targetEmpty), .values()
    );

    matrixMultiplierWeightStationary #(
        .WIDTH(WIDTH), .N(N),
        .INPUT_FIFO_DEPTH(INPUT_FIFO_DEPTH),
        .OUTPUT_FIFO_DEPTH(OUTPUT_FIFO_DEPTH)
    ) matrixEngine (
        .clk(clk), .rst_n(rst_n),
        .weightData(weightData), .weightValid(weightValid), .weightReady(weightReady),
        .activationData(activationData), .activationValid(matrixActivationValid),
        .activationReady(matrixActivationReady), .resultData(rawResultData),
        .resultValid(matrixResultValid), .resultReady(matrixResultReady),
        .resultLast(matrixResultLast), .weightsLoaded(weightsLoaded),
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
