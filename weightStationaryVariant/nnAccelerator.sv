module nnAccelerator #(
    parameter int WIDTH = 16,
    parameter int N = 3,
    parameter int FRACTION_BITS = 4,
    parameter int TARGET_WIDTH = WIDTH,
    parameter int REDUCTION_WEIGHT_WIDTH = 8,
    parameter int IN_FLIGHT_DEPTH = 2*N + 2,
    parameter int OUTPUT_FIFO_DEPTH = 2*N
)(
    input  logic                              clk,
    input  logic                              rst_n,
    input  logic signed [WIDTH-1:0]           weightData [N],
    input  logic                              weightValid,
    output logic                              weightReady,
    input  logic signed [WIDTH-1:0]           inputData [N],
    input  logic signed [TARGET_WIDTH-1:0]    targetData,
    input  logic                              trainingEnable,
    input  logic                              inputValid,
    output logic                              inputReady,
    input  logic signed [REDUCTION_WEIGHT_WIDTH-1:0] reductionWeight [N],
    input  logic                              loadReductionWeights,
    input  logic                              reduceOutput,
    output logic signed [2*WIDTH+2*$clog2(N)-1:0] resultData [N],
    output logic                              resultValid,
    input  logic                              resultReady,
    output logic                              weightsLoaded,
    input  logic                              reloadWeights,
    output logic                              reloadReady,
    input  logic                              passThrough
);

    localparam int MATRIX_RESULT_WIDTH = 2*WIDTH + $clog2(N);
    localparam int PREDICTION_WIDTH = MATRIX_RESULT_WIDTH + $clog2(N);
    // Matrix updates reach the AD after N-1 advances; reduction observes the
    // resulting registered vector one edge later at result formation.
    localparam int REDUCTION_UPDATE_DELAY = N;
    localparam int SAMPLE_CONTEXT_WIDTH = TARGET_WIDTH + 2*N + 1;
    localparam int COMPARE_WIDTH = (PREDICTION_WIDTH > TARGET_WIDTH)
                                   ? PREDICTION_WIDTH : TARGET_WIDTH;
    localparam logic signed [REDUCTION_WEIGHT_WIDTH-1:0]
        REDUCTION_WEIGHT_MIN = {1'b1, {(REDUCTION_WEIGHT_WIDTH-1){1'b0}}};
    localparam logic signed [REDUCTION_WEIGHT_WIDTH-1:0]
        REDUCTION_WEIGHT_MAX = {1'b0, {(REDUCTION_WEIGHT_WIDTH-1){1'b1}}};
    localparam logic signed [REDUCTION_WEIGHT_WIDTH-1:0]
        REDUCTION_WEIGHT_ONE = {{(REDUCTION_WEIGHT_WIDTH-1){1'b0}}, 1'b1};

    initial begin
        if (WIDTH < 1 || N < 2 || FRACTION_BITS < 0 ||
            TARGET_WIDTH < 1 || REDUCTION_WEIGHT_WIDTH < 1 ||
            IN_FLIGHT_DEPTH < 1 || OUTPUT_FIFO_DEPTH < 1)
            $fatal(1, "WIDTH>=1, N>=2, FRACTION_BITS>=0, widths/depths>=1");
    end

    logic signed [MATRIX_RESULT_WIDTH-1:0] rawResultData[N];
    logic signed [MATRIX_RESULT_WIDTH-1:0] activatedMatrixResultData[N];
    logic signed [MATRIX_RESULT_WIDTH-1:0] activatedData[N];
    logic signed [REDUCTION_WEIGHT_WIDTH-1:0] residentReductionWeight[N];
    logic signed [PREDICTION_WIDTH-1:0] prediction;
    logic signed [PREDICTION_WIDTH-1:0] enqueuePrediction;
    localparam int RESULT_ENTRY_WIDTH = N*MATRIX_RESULT_WIDTH +
                                        PREDICTION_WIDTH;
    logic signed [RESULT_ENTRY_WIDTH-1:0] resultFifoPushData;
    logic signed [RESULT_ENTRY_WIDTH-1:0] resultFifoHead;
    // The completing sample's target and live prediction feed the update package.
    logic signed [TARGET_WIDTH-1:0] resultTargetData;
    logic signed [1:0] learningDirection;
    logic signed [COMPARE_WIDTH-1:0] comparePrediction, compareTarget;
    logic signed [2*N-1:0] inputSignPushData, inputSignHead;
    logic trainingEnableHead;
    // Internal update wiring connects the matrix package to matrixEngine
    // and the reduction package to the resident-vector update pipe.
    logic signed [1:0] rowDirection[N], columnDirection[N];
    logic matrixUpdateValid;
    logic signed [1:0] reductionDirection[N];
    logic signed [2*N-1:0] reductionUpdateData;
    logic reductionUpdateValidPipe[REDUCTION_UPDATE_DELAY];
    logic signed [2*N-1:0]
        reductionUpdateDirectionPipe[REDUCTION_UPDATE_DELAY];
    logic arrayAdvance, matrixResultPush;
    logic matrixReloadReady, reductionUpdateBusy, applyReductionUpdate;
    logic resultFifoPush, resultFifoPop, resultFifoCanAccept;
    logic resultFifoFull, resultFifoEmpty;
    logic matrixInputValid, matrixInputReady;
    logic matrixResultValid, matrixResultReady;
    logic signed [SAMPLE_CONTEXT_WIDTH-1:0] sampleContextPushData;
    logic signed [SAMPLE_CONTEXT_WIDTH-1:0] sampleContextHead;
    logic sampleContextFull, sampleContextEmpty;
    logic samplePush, samplePop, sampleCanAccept;

    // The vector, target, signs, and training bit form one transaction. Gate matrix
    // valid and external ready together when the sample-context FIFO backpressures.
    assign sampleCanAccept       = !sampleContextFull || samplePop;
    assign inputReady       = matrixInputReady && sampleCanAccept;
    assign matrixInputValid = inputValid && sampleCanAccept;
    assign samplePush       = inputValid && inputReady;

    // The result FIFO holds the complete architectural result and controls capacity;
    // a full FIFO can accept a matrix result on the same edge its head retires.
    assign resultFifoPop      = resultValid && resultReady;
    assign resultFifoCanAccept = !resultFifoFull || resultFifoPop;
    assign matrixResultReady  = resultFifoCanAccept;
    assign matrixResultPush   = matrixResultValid && matrixResultReady;
    assign resultFifoPush     = matrixResultPush;
    // Learning and context completion happen before output storage. Consumption
    // only frees output capacity; it never issues a learning package.
    assign resultValid        = !resultFifoEmpty;
    assign samplePop          = matrixResultPush;
    assign matrixUpdateValid = samplePop && trainingEnableHead;
    assign applyReductionUpdate = arrayAdvance &&
                                  reductionUpdateValidPipe[REDUCTION_UPDATE_DELAY-1];

    always_comb begin
        reductionUpdateBusy = matrixUpdateValid;
        for (int stage = 0; stage < REDUCTION_UPDATE_DELAY; stage++)
            reductionUpdateBusy |= reductionUpdateValidPipe[stage];
    end
    assign reloadReady = matrixReloadReady && sampleContextEmpty &&
                         resultFifoEmpty && !reductionUpdateBusy;

    // Compare the live rescaled prediction with the oldest incomplete context.
    // Assignment to the wider signed signals sign-extends either side.
    assign comparePrediction = enqueuePrediction;
    assign compareTarget     = resultTargetData;

    always_comb begin
        learningDirection = 2'sd0;
        if (matrixResultValid) begin
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
            if (inputData[lane] == '0)
                inputSignPushData[2*lane +: 2] = 2'sd0;
            else if (inputData[lane][WIDTH-1])
                inputSignPushData[2*lane +: 2] = -2'sd1;
            else
                inputSignPushData[2*lane +: 2] = 2'sd1;
        end
    end

    // Both updates use the completing vector and the resident coefficients that
    // produced its live prediction, before any same-edge coefficient update.
    always_comb begin
        for (int lane = 0; lane < N; lane++) begin
            rowDirection[lane] = $signed(inputSignHead[2*lane +: 2]);
            columnDirection[lane] = 2'sd0;
            reductionDirection[lane] = 2'sd0;

            if (activatedMatrixResultData[lane] != '0) begin
                if (activatedMatrixResultData[lane][MATRIX_RESULT_WIDTH-1])
                    reductionDirection[lane] = -learningDirection;
                else
                    reductionDirection[lane] = learningDirection;
            end

            if ((passThrough || activatedMatrixResultData[lane] != '0) &&
                (residentReductionWeight[lane] != '0)) begin
                if (residentReductionWeight[lane][REDUCTION_WEIGHT_WIDTH-1])
                    columnDirection[lane] = -learningDirection;
                else
                    columnDirection[lane] = learningDirection;
            end
        end
    end

    // Matrix and reduction packages are generated together; packed ternary
    // directions keep the sideband compact as update waves overlap.
    always_comb begin
        reductionUpdateData = '0;
        for (int lane = 0; lane < N; lane++)
            reductionUpdateData[2*lane +: 2] = reductionDirection[lane];
    end

    // Store only the output payload. Learning has already consumed the live
    // activation, prediction, and reduction signs on this same handshake.
    always_comb begin
        resultFifoPushData = '0;
        for (int lane = 0; lane < N; lane++)
            resultFifoPushData[lane*MATRIX_RESULT_WIDTH +: MATRIX_RESULT_WIDTH] =
                activatedMatrixResultData[lane];
        resultFifoPushData[N*MATRIX_RESULT_WIDTH +: PREDICTION_WIDTH] =
            enqueuePrediction;
    end

    assign sampleContextPushData = {
        targetData, inputSignPushData, trainingEnable
    };
    assign resultTargetData = sampleContextHead[SAMPLE_CONTEXT_WIDTH-1 -: TARGET_WIDTH];
    assign inputSignHead = sampleContextHead[2*N:1];
    assign trainingEnableHead = sampleContextHead[0];

    always_comb begin
        for (int lane = 0; lane < N; lane++)
            activatedData[lane] =
                resultFifoHead[lane*MATRIX_RESULT_WIDTH +: MATRIX_RESULT_WIDTH];
    end
    assign prediction = resultFifoHead[N*MATRIX_RESULT_WIDTH +: PREDICTION_WIDTH];

    // FIFO order pairs each completing vector with its original sample context.
    signedFifo #(
        .WIDTH(SAMPLE_CONTEXT_WIDTH),
        .DEPTH(IN_FLIGHT_DEPTH)
    ) sampleContextFifo (
        .clk(clk), .rst_n(rst_n),
        .push(samplePush), .pushData(sampleContextPushData),
        .pop(samplePop), .popData(sampleContextHead),
        .full(sampleContextFull), .empty(sampleContextEmpty)
    );

    // Output buffering is a side branch of learning. Preserve the prediction
    // actually computed even if resident coefficients change before consumption.
    signedFifo #(
        .WIDTH(RESULT_ENTRY_WIDTH),
        .DEPTH(OUTPUT_FIFO_DEPTH)
    ) resultFifo (
        .clk(clk), .rst_n(rst_n),
        .push(resultFifoPush), .pushData(resultFifoPushData),
        .pop(resultFifoPop), .popData(resultFifoHead),
        .full(resultFifoFull), .empty(resultFifoEmpty)
    );

    // Stage zero uses the live reduction package; the pipe commits the resident
    // vector on the matrix update wave's advancing slots, including bubbles.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int lane = 0; lane < N; lane++)
                residentReductionWeight[lane] <= '0;
            for (int stage = 0; stage < REDUCTION_UPDATE_DELAY; stage++) begin
                reductionUpdateValidPipe[stage] <= 1'b0;
                reductionUpdateDirectionPipe[stage] <= '0;
            end
        end else if (loadReductionWeights) begin
            for (int lane = 0; lane < N; lane++)
                residentReductionWeight[lane] <= reductionWeight[lane];
        end else begin
            if (arrayAdvance) begin
                for (int stage = REDUCTION_UPDATE_DELAY-1; stage > 0; stage--) begin
                    reductionUpdateValidPipe[stage] <=
                        reductionUpdateValidPipe[stage-1];
                    reductionUpdateDirectionPipe[stage] <=
                        reductionUpdateDirectionPipe[stage-1];
                end
                reductionUpdateValidPipe[0] <=
                    matrixUpdateValid;
                reductionUpdateDirectionPipe[0] <=
                    reductionUpdateData;
            end

            if (applyReductionUpdate) begin
                for (int lane = 0; lane < N; lane++) begin
                    case ($signed(reductionUpdateDirectionPipe[REDUCTION_UPDATE_DELAY-1][2*lane +: 2]))
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

    weightStationaryMatrixMultiplier #(
        .WIDTH(WIDTH), .N(N)
    ) matrixEngine (
        .clk(clk), .rst_n(rst_n),
        .weightData(weightData), .weightValid(weightValid), .weightReady(weightReady),
        .inputData(inputData), .inputValid(matrixInputValid),
        .inputReady(matrixInputReady), .resultData(rawResultData),
        .rowDirection(rowDirection), .columnDirection(columnDirection),
        .matrixUpdateValid(matrixUpdateValid),
        .arrayAdvance(arrayAdvance),
        .resultValid(matrixResultValid), .resultReady(matrixResultReady),
        .weightsLoaded(weightsLoaded),
        .reloadWeights(reloadWeights && reloadReady),
        .reloadReady(matrixReloadReady)
    );

    // This is the sole activation operation.  It runs on the aligned raw
    // matrix result immediately before the result FIFO captures the entry;
    // ReLU is closed at zero, matching the column gate above.  Zero passes
    // through as zero, so only the sign bit selects, not a full compare.
    always_comb begin
        for (int lane = 0; lane < N; lane++)
            activatedMatrixResultData[lane] =
                (passThrough || !rawResultData[lane][MATRIX_RESULT_WIDTH-1])
                    ? rawResultData[lane] : '0;
    end

    weightedVectorReduction #(
        .MATRIX_RESULT_WIDTH(MATRIX_RESULT_WIDTH),
        .REDUCTION_WEIGHT_WIDTH(REDUCTION_WEIGHT_WIDTH),
        .N(N),
        .FRACTION_BITS(FRACTION_BITS)
    ) weightedReadoutAtMatrixResult (
        .inputData(activatedMatrixResultData),
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
