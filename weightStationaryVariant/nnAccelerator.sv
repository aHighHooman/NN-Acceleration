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
    // The resident reduction vector commits at the same advancing edge as
    // the last PE on the matrix update wave.  There are 2N-1 anti-diagonals
    // from PE(0,0) through PE(N-1,N-1).
    localparam int REDUCTION_BOUNDARY_STAGES = 2*N-1;
    // Sample metadata remains resident until its corresponding result is
    // consumed, so its lifetime is longer than the matrix activation FIFO's.
    localparam int SAMPLE_CONTEXT_DEPTH =
        (INPUT_FIFO_DEPTH > (2*N + 2)) ? INPUT_FIFO_DEPTH : (2*N + 2);
    localparam int SAMPLE_CONTEXT_WIDTH = TARGET_WIDTH + 2*N + 1;
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
    logic signed [MATRIX_RESULT_WIDTH-1:0] rawResultEnqueueData[N];
    logic signed [MATRIX_RESULT_WIDTH-1:0] activatedResultEnqueueData[N];
    logic signed [REDUCTION_WEIGHT_WIDTH-1:0] residentReductionWeight[N];
    logic signed [PREDICTION_WIDTH-1:0] prediction;
    logic signed [PREDICTION_WIDTH-1:0] enqueuePrediction;
    logic signed [2*N-1:0] reductionWeightSignPushData;
    logic signed [2*N-1:0] reductionWeightSignHead;
    logic signed [PREDICTION_WIDTH+2*N-1:0] resultMetadataPushData;
    logic signed [PREDICTION_WIDTH+2*N-1:0] resultMetadataHead;
    logic signed [COMPARE_WIDTH-1:0] comparePrediction, compareTarget;
    logic signed [2*N-1:0] inputSignPushData, inputSignHead;
    logic trainingEnableHead;
    logic signed [1:0] reductionDirection[N];
    logic signed [2*N-1:0] reductionUpdateData;
    logic reductionBoundaryValidPipe[REDUCTION_BOUNDARY_STAGES];
    logic signed [2*N-1:0]
        reductionBoundaryDataPipe[REDUCTION_BOUNDARY_STAGES];
    logic matrixUpdateAccepted, matrixUpdateComplete;
    logic matrixDatapathAdvance, matrixResultEnqueue, matrixResultPop;
    logic matrixReloadReady, reductionBoundaryBusy, reductionBoundaryApply;
    logic reductionUpdateBoundaryValid;
    logic signed [2*N-1:0] reductionUpdateBoundaryData;
    logic resultMetadataPush, resultMetadataPop;
    logic matrixActivationValid, matrixActivationReady;
    logic matrixResultValid, matrixResultReady, matrixResultLast;
    logic signed [SAMPLE_CONTEXT_WIDTH-1:0] sampleContextPushData;
    logic signed [SAMPLE_CONTEXT_WIDTH-1:0] sampleContextHead;
    logic sampleContextFull, sampleContextEmpty;
    logic samplePush, samplePop, sampleCanAccept;
    logic readoutHeadValid;

    // The activation vector, target, input signs, and training-enable bit are
    // one input transaction. Gate the matrix valid as well as the external
    // ready so no part can advance alone when the sample-context FIFO applies
    // backpressure.
    assign samplePop             = resultValid && resultReady;
    assign sampleCanAccept       = !sampleContextFull || samplePop;
    assign activationReady       = matrixActivationReady && sampleCanAccept;
    assign matrixActivationValid = activationValid && sampleCanAccept;
    assign samplePush            = activationValid && activationReady;

    // Reduction metadata is a transaction sideband pushed and popped with
    // each ordinary matrix result.  It never participates in forward-path
    // flow control; the FIFO counts are structurally identical to the result
    // stream and are therefore not another reason for a result to wait.
    assign readoutHeadValid  = matrixResultValid && !sampleContextEmpty;
    assign resultValid       = readoutHeadValid;
    assign matrixResultReady = resultReady && !sampleContextEmpty;
    assign matrixResultPop   = matrixResultValid && matrixResultReady;
    assign resultLast        = matrixResultLast && resultValid;
    assign matrixUpdateValid = samplePop && trainingEnableHead;
    assign reductionBoundaryApply = matrixDatapathAdvance &&
                                    reductionBoundaryValidPipe[REDUCTION_BOUNDARY_STAGES-1];

    always_comb begin
        reductionBoundaryBusy = reductionUpdateBoundaryValid ||
                                reductionBoundaryApply;
        for (int stage = 0; stage < REDUCTION_BOUNDARY_STAGES; stage++)
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
    // inference sample or while an output is stalled. The reduction-weight
    // sign is captured with the ordinary result when it enters the result
    // buffer; this keeps a buffered prediction and its learning package tied
    // together even when a later boundary has already advanced the resident
    // vector.
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

            if ((passThrough || activatedData[lane] != '0) &&
                ($signed(reductionWeightSignHead[2*lane +: 2]) != 0)) begin
                if ($signed(reductionWeightSignHead[2*lane +: 2]) < 0)
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

    // Capture only the reduction result and the ternary sign of the resident
    // coefficient with each normal result transaction.  No reduction event is
    // created for an input bubble, and no update-only item can get in front of
    // a ready result.
    always_comb begin
        reductionWeightSignPushData = '0;
        for (int lane = 0; lane < N; lane++) begin
            if (residentReductionWeight[lane][REDUCTION_WEIGHT_WIDTH-1])
                reductionWeightSignPushData[2*lane +: 2] = -2'sd1;
            else if (residentReductionWeight[lane] != '0)
                reductionWeightSignPushData[2*lane +: 2] = 2'sd1;
            else
                reductionWeightSignPushData[2*lane +: 2] = 2'sd0;
        end
        resultMetadataPushData = {
            enqueuePrediction, reductionWeightSignPushData
        };
    end

    assign sampleContextPushData = {
        targetData, inputSignPushData, trainingEnable
    };
    assign resultTargetData = sampleContextHead[SAMPLE_CONTEXT_WIDTH-1 -: TARGET_WIDTH];
    assign inputSignHead = sampleContextHead[2*N:1];
    assign trainingEnableHead = sampleContextHead[0];

    assign prediction = resultMetadataHead[PREDICTION_WIDTH+2*N-1:2*N];
    assign reductionWeightSignHead = resultMetadataHead[2*N-1:0];
    assign resultMetadataPush = matrixResultEnqueue;
    assign resultMetadataPop  = matrixResultPop;

    // FIFO order, rather than a cycle count, carries the complete sample
    // context to the result transaction produced by the corresponding
    // activation vector.
    signedFifo #(
        .WIDTH(SAMPLE_CONTEXT_WIDTH),
        .DEPTH(SAMPLE_CONTEXT_DEPTH)
    ) sampleContextFifo (
        .clk(clk), .rst_n(rst_n),
        .push(samplePush), .pushData(sampleContextPushData),
        .pop(samplePop), .popData(sampleContextHead),
        .full(sampleContextFull), .empty(sampleContextEmpty), .values()
    );

    // Ordinary result metadata is pushed and popped with each matrix result.
    // It is not an event stream and its fullness is deliberately not part of
    // the array's advance decision.
    signedFifo #(
        .WIDTH(PREDICTION_WIDTH+2*N),
        .DEPTH(OUTPUT_FIFO_DEPTH+1)
    ) resultMetadataFifo (
        .clk(clk), .rst_n(rst_n),
        .push(resultMetadataPush), .pushData(resultMetadataPushData),
        .pop(resultMetadataPop), .popData(resultMetadataHead),
        .full(), .empty(), .values()
    );

    // Carry the compact reduction update through the same advancing slots as
    // the matrix update wave.  The boundary commits the resident vector on an
    // advancing edge, including a useful bubble edge; it has no readout event
    // to drain and cannot hold a result valid.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int lane = 0; lane < N; lane++)
                residentReductionWeight[lane] <= '0;
            for (int stage = 0; stage < REDUCTION_BOUNDARY_STAGES; stage++) begin
                reductionBoundaryValidPipe[stage] <= 1'b0;
                reductionBoundaryDataPipe[stage] <= '0;
            end
        end else if (loadReductionWeights) begin
            for (int lane = 0; lane < N; lane++)
                residentReductionWeight[lane] <= reductionWeight[lane];
        end else begin
            if (matrixDatapathAdvance) begin
                for (int stage = REDUCTION_BOUNDARY_STAGES-1; stage > 0; stage--) begin
                    reductionBoundaryValidPipe[stage] <=
                        reductionBoundaryValidPipe[stage-1];
                    reductionBoundaryDataPipe[stage] <=
                        reductionBoundaryDataPipe[stage-1];
                end
                reductionBoundaryValidPipe[0] <=
                    reductionUpdateBoundaryValid;
                reductionBoundaryDataPipe[0] <=
                    reductionUpdateBoundaryData;
            end

            if (reductionBoundaryApply) begin
                for (int lane = 0; lane < N; lane++) begin
                    case ($signed(reductionBoundaryDataPipe[REDUCTION_BOUNDARY_STAGES-1][2*lane +: 2]))
                        2'sd1: begin
                            if (residentReductionWeight[lane] != REDUCTION_WEIGHT_MAX)
                                residentReductionWeight[lane] <=
                                    residentReductionWeight[lane] + REDUCTION_WEIGHT_ONE;
                        end
                        -2'sd1: begin
                            if (residentReductionWeight[lane] != REDUCTION_WEIGHT_MIN)
                                residentReductionWeight[lane] <=
                                    residentReductionWeight[lane] - REDUCTION_WEIGHT_ONE;
                        end
                        default: residentReductionWeight[lane] <=
                                     residentReductionWeight[lane];
                    endcase
                end
            end
        end
    end

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
        .resultEnqueueData(rawResultEnqueueData),
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

    activationLayer #(.WIDTH(MATRIX_RESULT_WIDTH), .N(N)) enqueueActivation (
        .inputData(rawResultEnqueueData), .passThrough(passThrough),
        .outputData(activatedResultEnqueueData)
    );

    weightedVectorReduction #(
        .MATRIX_RESULT_WIDTH(MATRIX_RESULT_WIDTH),
        .REDUCTION_WEIGHT_WIDTH(REDUCTION_WEIGHT_WIDTH),
        .N(N),
        .FRACTION_BITS(FRACTION_BITS)
    ) weightedReadoutAtEnqueue (
        .inputData(activatedResultEnqueueData),
        .reductionWeight(residentReductionWeight),
        .prediction(enqueuePrediction)
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
