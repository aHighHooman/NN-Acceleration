module nnAccelerator #(
    parameter int WIDTH = 16,
    parameter int N = 3,
    parameter int FRACTION_BITS = 4,
    parameter int TARGET_WIDTH = WIDTH,
    parameter int REDUCTION_WEIGHT_WIDTH = 8,
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
    input  logic                              trainingEnable,
    input  logic                              activationValid,
    output logic                              activationReady,
    input  logic signed [REDUCTION_WEIGHT_WIDTH-1:0] reductionWeight [N],
    input  logic                              loadReductionWeights,
    input  logic                              reduceOutput,
    output logic signed [2*WIDTH+2*$clog2(N)-1:0] resultData [N],
    output logic signed [TARGET_WIDTH-1:0]    resultTargetData,
    output logic signed [1:0]                 learningDirection,
    output logic signed [1:0]                 rowDirection [N],
    output logic signed [1:0]                 columnDirection [N],
    output logic                              matrixUpdateValid,
    output logic                              resultValid,
    input  logic                              resultReady,
    output logic                              resultLast,
    output logic                              weightsLoaded,
    input  logic                              reloadWeights,
    output logic                              reloadReady,
    input  logic                              passThrough
);

    localparam int MATRIX_RESULT_WIDTH = 2*WIDTH + $clog2(N);
    localparam int PREDICTION_WIDTH = MATRIX_RESULT_WIDTH + $clog2(N);
    localparam int COMPARE_WIDTH = (PREDICTION_WIDTH > TARGET_WIDTH)
                                   ? PREDICTION_WIDTH : TARGET_WIDTH;
    localparam logic signed [REDUCTION_WEIGHT_WIDTH-1:0]
        REDUCTION_WEIGHT_MIN = {1'b1, {(REDUCTION_WEIGHT_WIDTH-1){1'b0}}};
    localparam logic signed [REDUCTION_WEIGHT_WIDTH-1:0]
        REDUCTION_WEIGHT_MAX = {1'b0, {(REDUCTION_WEIGHT_WIDTH-1){1'b1}}};
    localparam logic signed [REDUCTION_WEIGHT_WIDTH-1:0]
        REDUCTION_WEIGHT_ONE = {{(REDUCTION_WEIGHT_WIDTH-1){1'b0}}, 1'b1};

    logic signed [MATRIX_RESULT_WIDTH-1:0] rawResultData[N];
    logic signed [MATRIX_RESULT_WIDTH-1:0] activatedData[N];
    logic signed [REDUCTION_WEIGHT_WIDTH-1:0] residentReductionWeight[N];
    logic signed [PREDICTION_WIDTH-1:0] prediction;
    logic signed [COMPARE_WIDTH-1:0] comparePrediction, compareTarget;
    logic signed [2*N-1:0] inputSignPushData, inputSignHead;
    logic trainingEnableHead;
    logic signed [1:0] reductionDirection[N];
    logic signed [2*N-1:0] reductionUpdateData;
    logic reductionBoundaryValidPipe[N+1];
    logic signed [2*N-1:0] reductionBoundaryDataPipe[N+1];
    logic signed [N*REDUCTION_WEIGHT_WIDTH-1:0] reductionSnapshotPushData;
    logic signed [N*REDUCTION_WEIGHT_WIDTH-1:0] reductionSnapshotHead;
    logic signed [REDUCTION_WEIGHT_WIDTH-1:0] sampleReductionWeight[N];
    logic signed [REDUCTION_WEIGHT_WIDTH-1:0] updatedReductionWeight[N];
    logic matrixUpdateAccepted, matrixUpdateComplete;
    logic matrixDatapathAdvance, matrixResultEnqueue, matrixResultPop;
    logic matrixReloadReady, reductionBoundaryBusy;
    logic reductionUpdateBoundaryValid;
    logic signed [2*N-1:0] reductionUpdateBoundaryData;
    logic reductionSnapshotFull, reductionSnapshotEmpty;
    logic matrixActivationValid, matrixActivationReady;
    logic matrixResultValid, matrixResultReady, matrixResultLast;
    logic targetPush, targetPop, targetFull, targetEmpty;
    logic inputSignFull, inputSignEmpty;
    logic trainingEnableFull, trainingEnableEmpty;
    logic samplePush, samplePop, sampleCanAccept;
    logic readoutHeadValid;

    // The activation vector, target, input signs, and training-enable bit are
    // one input transaction. Gate the matrix valid as well as the external
    // ready so no part can advance alone when any metadata queue applies
    // backpressure.
    assign samplePop             = resultValid && resultReady;
    assign sampleCanAccept       = (!targetFull && !inputSignFull &&
                                    !trainingEnableFull) || samplePop;
    assign activationReady       = matrixActivationReady && sampleCanAccept;
    assign matrixActivationValid = activationValid && sampleCanAccept;
    assign samplePush            = activationValid && activationReady;
    assign targetPush            = samplePush;
    assign targetPop             = samplePop;

    assign readoutHeadValid  = matrixResultValid && !targetEmpty &&
                               !inputSignEmpty && !trainingEnableEmpty &&
                               !reductionSnapshotEmpty;
    assign resultValid       = readoutHeadValid;
    assign matrixResultReady = resultReady && !targetEmpty && !inputSignEmpty &&
                               !trainingEnableEmpty && !reductionSnapshotEmpty;
    assign matrixResultPop   = matrixResultValid && matrixResultReady;
    assign resultLast        = matrixResultLast && resultValid;
    assign matrixUpdateValid = samplePop && trainingEnableHead;

    always_comb begin
        reductionBoundaryBusy = reductionUpdateBoundaryValid;
        for (int stage = 0; stage <= N; stage++)
            reductionBoundaryBusy |= reductionBoundaryValidPipe[stage];
    end
    assign reloadReady = matrixReloadReady && !reductionBoundaryBusy;

    // Compare the rescaled architectural prediction with the aligned FIFO
    // head. Assignment to the wider signed signals sign-extends either side.
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

    // Capture the original input signs as a compact vector. Two-bit signed
    // values encode the complete ternary set: 2'b01, 2'b00, and 2'b11.
    always_comb begin
        inputSignPushData = '0;
        for (int lane = 0; lane < N; lane++) begin
            if (activationData[lane] == '0)
                inputSignPushData[2*lane +: 2] = 2'sd0;
            else if (activationData[lane][WIDTH-1])
                inputSignPushData[2*lane +: 2] = -2'sd1;
            else
                inputSignPushData[2*lane +: 2] = 2'sd1;
        end
    end

    // Both update vectors describe the FIFO-head sample. Matrix-update valid
    // is a training-enabled result handshake, so no package is emitted for an
    // inference sample or while an output is stalled. The resident weights
    // here are the values used by this sample; their sequential update takes
    // effect only after the handshake edge.
    always_comb begin
        for (int lane = 0; lane < N; lane++) begin
            rowDirection[lane] = $signed(inputSignHead[2*lane +: 2]);
            columnDirection[lane] = 2'sd0;
            reductionDirection[lane] = 2'sd0;

            if (activatedData[lane] != '0) begin
                if (activatedData[lane][MATRIX_RESULT_WIDTH-1])
                    reductionDirection[lane] = -learningDirection;
                else
                    reductionDirection[lane] = learningDirection;
            end

            if ((passThrough ||
                 (!rawResultData[lane][MATRIX_RESULT_WIDTH-1] &&
                  (rawResultData[lane] != '0))) &&
                (sampleReductionWeight[lane] != '0)) begin
                if (sampleReductionWeight[lane][REDUCTION_WEIGHT_WIDTH-1])
                    columnDirection[lane] = -learningDirection;
                else
                    columnDirection[lane] = learningDirection;
            end
        end
    end

    // A matrix package and its reduction package are generated together.
    // Packing the N ternary directions keeps the sideband compact while
    // successive packages overlap in the update wave.
    always_comb begin
        reductionUpdateData = '0;
        for (int lane = 0; lane < N; lane++)
            reductionUpdateData[2*lane +: 2] = reductionDirection[lane];
    end

    // FIFO order, rather than a cycle count, carries each sample's metadata to
    // the result transaction produced by the corresponding activation vector.
    signedFifo #(
        .WIDTH(TARGET_WIDTH),
        .DEPTH(INPUT_FIFO_DEPTH)
    ) targetFifo (
        .clk(clk), .rst_n(rst_n),
        .push(targetPush), .pushData(targetData),
        .pop(targetPop), .popData(resultTargetData),
        .full(targetFull), .empty(targetEmpty), .values()
    );

    signedFifo #(
        .WIDTH(2*N),
        .DEPTH(INPUT_FIFO_DEPTH)
    ) inputSignFifo (
        .clk(clk), .rst_n(rst_n),
        .push(samplePush), .pushData(inputSignPushData),
        .pop(samplePop), .popData(inputSignHead),
        .full(inputSignFull), .empty(inputSignEmpty), .values()
    );

    signedFifo #(
        .WIDTH(1),
        .DEPTH(INPUT_FIFO_DEPTH)
    ) trainingEnableFifo (
        .clk(clk), .rst_n(rst_n),
        .push(samplePush), .pushData(trainingEnable),
        .pop(samplePop), .popData(trainingEnableHead),
        .full(trainingEnableFull), .empty(trainingEnableEmpty), .values()
    );

    // Carry the stage-zero learning boundary to readout under the same advance
    // enable as the matrix datapath. At the output edge, the boundary update
    // and the first result behind it use the same saturated next state.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int lane = 0; lane < N; lane++)
                residentReductionWeight[lane] <= '0;
            for (int stage = 0; stage <= N; stage++) begin
                reductionBoundaryValidPipe[stage] <= 1'b0;
                reductionBoundaryDataPipe[stage] <= '0;
            end
        end else if (loadReductionWeights) begin
            for (int lane = 0; lane < N; lane++)
                residentReductionWeight[lane] <= reductionWeight[lane];
        end else begin
            if (matrixDatapathAdvance) begin
                for (int stage = N; stage > 0; stage--) begin
                    reductionBoundaryValidPipe[stage] <=
                        reductionBoundaryValidPipe[stage-1];
                    reductionBoundaryDataPipe[stage] <=
                        reductionBoundaryDataPipe[stage-1];
                end
                reductionBoundaryValidPipe[0] <=
                    reductionUpdateBoundaryValid;
                reductionBoundaryDataPipe[0] <=
                    reductionUpdateBoundaryData;
                if (reductionBoundaryValidPipe[N])
                    for (int lane = 0; lane < N; lane++)
                        residentReductionWeight[lane] <=
                            updatedReductionWeight[lane];
            end
        end
    end

    always_comb begin
        for (int lane = 0; lane < N; lane++) begin
            updatedReductionWeight[lane] = residentReductionWeight[lane];
            if (reductionBoundaryValidPipe[N]) begin
                case ($signed(reductionBoundaryDataPipe[N][2*lane +: 2]))
                    2'sd1:
                        if (residentReductionWeight[lane] != REDUCTION_WEIGHT_MAX)
                            updatedReductionWeight[lane] =
                                residentReductionWeight[lane] + REDUCTION_WEIGHT_ONE;
                    -2'sd1:
                        if (residentReductionWeight[lane] != REDUCTION_WEIGHT_MIN)
                            updatedReductionWeight[lane] =
                                residentReductionWeight[lane] - REDUCTION_WEIGHT_ONE;
                    default: updatedReductionWeight[lane] =
                                 residentReductionWeight[lane];
                endcase
            end
            reductionSnapshotPushData[lane*REDUCTION_WEIGHT_WIDTH +:
                                      REDUCTION_WEIGHT_WIDTH] =
                updatedReductionWeight[lane];
        end
    end
    for (genvar reductionLane = 0; reductionLane < N; reductionLane++) begin : reduction_snapshot_lanes
        assign sampleReductionWeight[reductionLane] =
            reductionSnapshotHead[reductionLane*REDUCTION_WEIGHT_WIDTH +:
                                  REDUCTION_WEIGHT_WIDTH];
    end

    signedFifo #(
        .WIDTH(N*REDUCTION_WEIGHT_WIDTH), .DEPTH(OUTPUT_FIFO_DEPTH)
    ) reductionSnapshotFifo (
        .clk(clk), .rst_n(rst_n),
        .push(matrixResultEnqueue), .pushData(reductionSnapshotPushData),
        .pop(matrixResultPop), .popData(reductionSnapshotHead),
        .full(reductionSnapshotFull), .empty(reductionSnapshotEmpty), .values()
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
        .rowDirection(rowDirection), .columnDirection(columnDirection),
        .matrixUpdateValid(matrixUpdateValid),
        .reductionUpdateData(reductionUpdateData),
        .matrixUpdateAccepted(matrixUpdateAccepted),
        .matrixUpdateComplete(matrixUpdateComplete),
        .datapathAdvance(matrixDatapathAdvance),
        .resultEnqueue(matrixResultEnqueue),
        .resultSidebandFull(reductionSnapshotFull),
        .reductionUpdateBoundaryValid(reductionUpdateBoundaryValid),
        .reductionUpdateBoundaryData(reductionUpdateBoundaryData),
        .resultValid(matrixResultValid), .resultReady(matrixResultReady),
        .resultLast(matrixResultLast), .weightsLoaded(weightsLoaded),
        .reloadWeights(reloadWeights && reloadReady),
        .reloadReady(matrixReloadReady)
    );

    activationLayer #(.WIDTH(MATRIX_RESULT_WIDTH), .N(N)) resultActivation (
        .inputData(rawResultData), .passThrough(passThrough), .outputData(activatedData)
    );

    weightedVectorReduction #(
        .MATRIX_RESULT_WIDTH(MATRIX_RESULT_WIDTH),
        .REDUCTION_WEIGHT_WIDTH(REDUCTION_WEIGHT_WIDTH),
        .N(N),
        .FRACTION_BITS(FRACTION_BITS)
    ) weightedReadout (
        .inputData(activatedData),
        .reductionWeight(sampleReductionWeight),
        .prediction(prediction)
    );

    always_comb begin
        for (int lane = 0; lane < N; lane++) begin
            if (reduceOutput)
                resultData[lane] = (lane == 0) ? prediction : '0;
            else
                resultData[lane] =
                    {{(PREDICTION_WIDTH-MATRIX_RESULT_WIDTH)
                       {activatedData[lane][MATRIX_RESULT_WIDTH-1]}},
                     activatedData[lane]};
        end
    end

endmodule
