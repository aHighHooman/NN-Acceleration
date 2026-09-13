`timescale 1ns / 1ps

// Integration-level directed verification for the Phase 5H streaming
// learning boundary. The scoreboard follows every accepted sample and the
// structurally aligned matrix/reduction state across bubbles and stalls.
module nnAccelerator_tb;
    localparam int WIDTH = 8;
    localparam int TARGET_WIDTH = 7;
    localparam int N = 2;
    localparam int FRACTION_BITS = 4;
    localparam int REDUCTION_WEIGHT_WIDTH = 8;
    localparam int REDUCTION_FRACTION_BITS = REDUCTION_WEIGHT_WIDTH - 1;
    localparam int RESCALE_SHIFT = FRACTION_BITS
                                   + REDUCTION_FRACTION_BITS;
    localparam int SCALE = 1 << FRACTION_BITS;
    localparam int HALF = 1 << (REDUCTION_FRACTION_BITS - 1);
    localparam int PREDICTION_WIDTH = 2*WIDTH + 2*$clog2(N);
    localparam int SAMPLE_COUNT = 21;
    localparam int UPDATE_STAGES = 2*N - 1;

    typedef logic signed [PREDICTION_WIDTH-1:0] result_t;
    typedef logic signed [TARGET_WIDTH-1:0] target_t;

    logic clk, rst_n;
    logic signed [WIDTH-1:0] weightData[N], activationData[N];
    target_t targetData, resultTargetData;
    logic weightValid, weightReady, activationValid, activationReady;
    logic trainingEnable;
    logic signed [REDUCTION_WEIGHT_WIDTH-1:0] reductionWeight[N];
    result_t resultData[N];
    logic signed [1:0] learningDirection;
    logic signed [1:0] rowDirection[N], columnDirection[N];
    logic matrixUpdateValid, loadReductionWeights;
    logic reduceOutput, resultValid, resultReady, resultLast;
    logic weightsLoaded, reloadWeights, reloadReady, passThrough;

    integer cycleCount, acceptedCount, consumedCount;
    integer lastAcceptedCycle;
    target_t expectedTarget[0:SAMPLE_COUNT-1];
    logic expectedTrainingEnable[0:SAMPLE_COUNT-1];
    logic signed [WIDTH-1:0] acceptedInput[0:SAMPLE_COUNT-1][N];
    logic signed [WIDTH-1:0] sampleMatrixWeight[0:SAMPLE_COUNT-1][N][N];
    integer sampleMatrixVersion[0:SAMPLE_COUNT-1];
    logic signed [REDUCTION_WEIGHT_WIDTH-1:0]
        sampleReductionWeight[0:SAMPLE_COUNT-1][N];
    integer sampleReductionVersion[0:SAMPLE_COUNT-1];
    logic signed [WIDTH-1:0] modelMatrixWeight[N][N];
    logic signed [REDUCTION_WEIGHT_WIDTH-1:0] modelReductionWeight[N];
    logic signed [REDUCTION_WEIGHT_WIDTH-1:0]
        architecturalReductionWeight[0:SAMPLE_COUNT][N];
    logic signed [1:0] pendingReductionDirection[0:SAMPLE_COUNT-1][N];
    integer acceptedUpdateSampleIndex[0:SAMPLE_COUNT-1];
    integer reductionQueueHead, reductionQueueTail, reductionQueueCount;
    integer matrixEntryCount, acceptedUpdateCount, completedUpdateCount;
    integer matrixVersion, matrixGeneration;
    integer lastConsumedCycle;
    bit sawBackToBack, sawBubble;
    bit sawPositive, sawZero, sawNegative;
    bit sawOverlappingUpdates, sawUpdateWaveStall;
    bit sawOldWeightVersion, sawUpdatedWeightVersion;
    bit sawReductionIncrement, sawReductionDecrement;
    bit sawValidStallWithUpdateWave;
    bit sawStreamingBoundary;
    bit sawReadoutBoundaryEvaluation;

    nnAccelerator #(
        .WIDTH(WIDTH), .N(N), .TARGET_WIDTH(TARGET_WIDTH),
        .FRACTION_BITS(FRACTION_BITS),
        .REDUCTION_WEIGHT_WIDTH(REDUCTION_WEIGHT_WIDTH),
        .INPUT_FIFO_DEPTH(4), .OUTPUT_FIFO_DEPTH(2)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .weightData(weightData), .weightValid(weightValid), .weightReady(weightReady),
        .activationData(activationData), .targetData(targetData),
        .trainingEnable(trainingEnable),
        .activationValid(activationValid), .activationReady(activationReady),
        .reductionWeight(reductionWeight),
        .loadReductionWeights(loadReductionWeights), .reduceOutput(reduceOutput),
        .resultData(resultData), .resultTargetData(resultTargetData),
        .learningDirection(learningDirection),
        .rowDirection(rowDirection), .columnDirection(columnDirection),
        .matrixUpdateValid(matrixUpdateValid),
        .resultValid(resultValid), .resultReady(resultReady), .resultLast(resultLast),
        .weightsLoaded(weightsLoaded), .reloadWeights(reloadWeights),
        .reloadReady(reloadReady), .passThrough(passThrough)
    );

    always #5 clk = ~clk;

    // Keep the matrix result and complete weighted sum at full precision,
    // then perform the one required final rescale.
    function automatic result_t phase5_prediction(
        input logic signed [WIDTH-1:0] x0,
        input logic signed [WIDTH-1:0] x1,
        input logic signed [WIDTH-1:0] weight00,
        input logic signed [WIDTH-1:0] weight01,
        input logic signed [WIDTH-1:0] weight10,
        input logic signed [WIDTH-1:0] weight11,
        input logic signed [REDUCTION_WEIGHT_WIDTH-1:0] reduction0,
        input logic signed [REDUCTION_WEIGHT_WIDTH-1:0] reduction1,
        input logic applyPassThrough
    );
        integer xw0, xw1;
        integer fullReduction;
        begin
            xw0 = x0 * weight00 + x1 * weight10;
            xw1 = x0 * weight01 + x1 * weight11;
            if (!applyPassThrough) begin
                if (xw0 < 0) xw0 = 0;
                if (xw1 < 0) xw1 = 0;
            end
            fullReduction = xw0 * reduction0 + xw1 * reduction1;
            phase5_prediction = fullReduction >>> RESCALE_SHIFT;
        end
    endfunction

    function automatic logic signed [1:0] ternary_sign(input integer value);
        if (value > 0)
            ternary_sign = 2'sd1;
        else if (value < 0)
            ternary_sign = -2'sd1;
        else
            ternary_sign = 2'sd0;
    endfunction

    function automatic logic signed [1:0] ternary_product(
        input logic signed [1:0] left,
        input logic signed [1:0] right
    );
        if ((left == 2'sd0) || (right == 2'sd0))
            ternary_product = 2'sd0;
        else if (left == right)
            ternary_product = 2'sd1;
        else
            ternary_product = -2'sd1;
    endfunction

    always @(posedge clk) begin
        result_t expectedPredictionNow;
        logic signed [1:0] expectedDirectionNow;
        integer rawResult[N];
        integer liveUpdateStages;

        if (!rst_n) begin
            cycleCount        = 0;
            acceptedCount     = 0;
            consumedCount     = 0;
            matrixEntryCount  = 0;
            acceptedUpdateCount = 0;
            completedUpdateCount = 0;
            matrixVersion = 0;
            matrixGeneration = 0;
            lastConsumedCycle = -2;
            lastAcceptedCycle = -2;
            sawBackToBack     = 0;
            sawBubble         = 0;
            sawPositive       = 0;
            sawZero           = 0;
            sawNegative       = 0;
            sawOverlappingUpdates = 0;
            sawUpdateWaveStall = 0;
            sawOldWeightVersion = 0;
            sawUpdatedWeightVersion = 0;
            sawReductionIncrement = 0;
            sawReductionDecrement = 0;
            sawValidStallWithUpdateWave = 0;
            sawStreamingBoundary = 0;
            sawReadoutBoundaryEvaluation = 0;
            reductionQueueHead = 0;
            reductionQueueTail = 0;
            reductionQueueCount = 0;
            for (int lane = 0; lane < N; lane++) begin
                modelReductionWeight[lane] = 0;
                for (int generation = 0; generation <= SAMPLE_COUNT; generation++)
                    architecturalReductionWeight[generation][lane] = 0;
            end
            modelMatrixWeight[0][0] = 2*SCALE;
            modelMatrixWeight[0][1] = -1*SCALE;
            modelMatrixWeight[1][0] = 3*SCALE;
            modelMatrixWeight[1][1] = 4*SCALE;
        end else begin
            cycleCount = cycleCount + 1;

            liveUpdateStages = 0;
            for (int stage = 0; stage < UPDATE_STAGES; stage++)
                if (dut.matrixEngine.systolicArr.updateValidPipe[stage])
                    liveUpdateStages = liveUpdateStages + 1;
            if (liveUpdateStages >= 2)
                sawOverlappingUpdates = 1;
            if (resultValid && !resultReady && (liveUpdateStages != 0))
                sawValidStallWithUpdateWave = 1;

            if (dut.targetPush !== (activationValid && activationReady))
                $fatal(1, "target push did not equal the external sample handshake");
            if (dut.targetPush !==
                (dut.matrixActivationValid && dut.matrixActivationReady))
                $fatal(1, "activation and target were not accepted atomically");
            if (dut.targetPop !== (resultValid && resultReady))
                $fatal(1, "target pop did not equal the result handshake");
            if (dut.samplePush !== dut.targetPush ||
                dut.inputSignFifo.values !== dut.targetFifo.values ||
                dut.trainingEnableFifo.values !== dut.targetFifo.values)
                $fatal(1, "sample metadata FIFOs lost alignment");
            if (matrixUpdateValid !==
                ((resultValid && resultReady) && dut.trainingEnableHead))
                $fatal(1, "matrix update valid did not match buffered training enable");
            if (matrixUpdateValid && !dut.matrixEngine.arrayAdvance)
                $fatal(1, "matrix update package was presented while the array was stalled");
            if (dut.matrixUpdateAccepted !== matrixUpdateValid)
                $fatal(1, "matrix update acceptance did not match the generated package");
            if (dut.reductionSnapshotFifo.values !==
                dut.matrixEngine.fifo_banks[0].outputFifo.values)
                $fatal(1, "matrix-result and reduction-snapshot FIFOs lost alignment");
            if (dut.matrixDatapathAdvance &&
                dut.reductionBoundaryValidPipe[N] &&
                dut.matrixResultEnqueue)
                sawReadoutBoundaryEvaluation = 1;
            for (int lane = 0; lane < N; lane++) begin
                if (dut.residentReductionWeight[lane] !==
                    modelReductionWeight[lane])
                    $fatal(1, "resident reduction weight %0d got %0d, expected %0d",
                           lane, dut.residentReductionWeight[lane],
                           modelReductionWeight[lane]);
            end

            if (loadReductionWeights) begin
                for (int lane = 0; lane < N; lane++) begin
                    modelReductionWeight[lane] = reductionWeight[lane];
                    // Loads in this directed test occur only after all pending
                    // update waves drain. They replace the reduction half of
                    // the current complete architectural network version.
                    architecturalReductionWeight[acceptedUpdateCount][lane] =
                        reductionWeight[lane];
                end
            end

            // row lane 0 is the leading edge of a sample wave. All other
            // lanes encounter their matching PE diagonal on later advances.
            if (dut.matrixEngine.arrayAdvance &&
                dut.matrixEngine.validData_OrchToSyst[0]) begin
                if (matrixEntryCount >= acceptedCount)
                    $fatal(1, "matrix sample entered without an accepted input");
                for (int rowIndex = 0; rowIndex < N; rowIndex++)
                    for (int columnIndex = 0; columnIndex < N; columnIndex++)
                        sampleMatrixWeight[matrixEntryCount][rowIndex][columnIndex] =
                            modelMatrixWeight[rowIndex][columnIndex];
                sampleMatrixVersion[matrixEntryCount] = matrixVersion;
                for (int lane = 0; lane < N; lane++)
                    sampleReductionWeight[matrixEntryCount][lane] =
                        architecturalReductionWeight[matrixGeneration][lane];
                sampleReductionVersion[matrixEntryCount] = matrixVersion;
                matrixEntryCount = matrixEntryCount + 1;
            end

            // Stage 0 crosses the leading array boundary after this edge's
            // multiply. The next sample entering PE(0,0) therefore uses this
            // package consistently at every later anti-diagonal.
            if (dut.matrixEngine.arrayAdvance &&
                dut.matrixEngine.systolicArr.updateValidPipe[0]) begin
                for (int rowIndex = 0; rowIndex < N; rowIndex++) begin
                    for (int columnIndex = 0; columnIndex < N; columnIndex++) begin
                        case (ternary_product(
                                  dut.matrixEngine.systolicArr.updateRowPipe[0][rowIndex],
                                  dut.matrixEngine.systolicArr.updateColumnPipe[0][columnIndex]))
                            2'sd1: begin
                                if (modelMatrixWeight[rowIndex][columnIndex] !=
                                    {1'b0, {(WIDTH-1){1'b1}}})
                                    modelMatrixWeight[rowIndex][columnIndex] =
                                        modelMatrixWeight[rowIndex][columnIndex] + 1;
                            end
                            -2'sd1: begin
                                if (modelMatrixWeight[rowIndex][columnIndex] !=
                                    {1'b1, {(WIDTH-1){1'b0}}})
                                    modelMatrixWeight[rowIndex][columnIndex] =
                                        modelMatrixWeight[rowIndex][columnIndex] - 1;
                            end
                            default: modelMatrixWeight[rowIndex][columnIndex] =
                                         modelMatrixWeight[rowIndex][columnIndex];
                        endcase
                    end
                end
                matrixVersion = matrixVersion + 1;
                matrixGeneration = matrixGeneration + 1;
            end

            if (activationValid && activationReady) begin
                if (acceptedCount >= SAMPLE_COUNT)
                    $fatal(1, "accepted more samples than the test supplied");

                if (cycleCount == lastAcceptedCycle + 1)
                    sawBackToBack = 1;
                if ((lastAcceptedCycle >= 0) &&
                    (cycleCount > lastAcceptedCycle + 1))
                    sawBubble = 1;
                lastAcceptedCycle = cycleCount;

                expectedTarget[acceptedCount] = targetData;
                expectedTrainingEnable[acceptedCount] = trainingEnable;
                for (int lane = 0; lane < N; lane++)
                    acceptedInput[acceptedCount][lane] = activationData[lane];
                acceptedCount = acceptedCount + 1;
            end

            if (resultValid && resultReady) begin
                if (consumedCount >= acceptedCount)
                    $fatal(1, "result was consumed without an accepted sample");
                if (consumedCount >= matrixEntryCount)
                    $fatal(1, "result was consumed before its matrix-weight snapshot");
                if (resultTargetData !== expectedTarget[consumedCount])
                    $fatal(1, "target %0d got %0d, expected %0d",
                           consumedCount, resultTargetData,
                           expectedTarget[consumedCount]);
                if (dut.trainingEnableHead !==
                    expectedTrainingEnable[consumedCount])
                    $fatal(1, "training enable %0d got %0b, expected %0b",
                           consumedCount, dut.trainingEnableHead,
                           expectedTrainingEnable[consumedCount]);
                if (matrixUpdateValid !==
                    expectedTrainingEnable[consumedCount])
                    $fatal(1, "sample %0d matrix update valid got %0b, expected %0b",
                           consumedCount, matrixUpdateValid,
                           expectedTrainingEnable[consumedCount]);
                if (sampleMatrixVersion[consumedCount] !=
                    sampleReductionVersion[consumedCount])
                    $fatal(1, "sample %0d has mismatched snapshotted matrix/reduction versions %0d/%0d",
                           consumedCount, sampleMatrixVersion[consumedCount],
                           sampleReductionVersion[consumedCount]);
                for (int lane = 0; lane < N; lane++)
                    if (dut.sampleReductionWeight[lane] !==
                        sampleReductionWeight[consumedCount][lane])
                        $fatal(1, "sample %0d reduction snapshot lane %0d got %0d, expected %0d",
                               consumedCount, lane,
                               dut.sampleReductionWeight[lane],
                               sampleReductionWeight[consumedCount][lane]);
                if (sampleMatrixVersion[consumedCount] == 0)
                    sawOldWeightVersion = 1;
                else
                    sawUpdatedWeightVersion = 1;
                expectedPredictionNow = phase5_prediction(
                    acceptedInput[consumedCount][0],
                    acceptedInput[consumedCount][1],
                    sampleMatrixWeight[consumedCount][0][0],
                    sampleMatrixWeight[consumedCount][0][1],
                    sampleMatrixWeight[consumedCount][1][0],
                    sampleMatrixWeight[consumedCount][1][1],
                    sampleReductionWeight[consumedCount][0],
                    sampleReductionWeight[consumedCount][1],
                    passThrough);
                if ($signed(expectedTarget[consumedCount]) >
                    $signed(expectedPredictionNow))
                    expectedDirectionNow = 2'sd1;
                else if ($signed(expectedTarget[consumedCount]) <
                         $signed(expectedPredictionNow))
                    expectedDirectionNow = -2'sd1;
                else
                    expectedDirectionNow = 2'sd0;

                rawResult[0] = acceptedInput[consumedCount][0]
                               * sampleMatrixWeight[consumedCount][0][0]
                               + acceptedInput[consumedCount][1]
                               * sampleMatrixWeight[consumedCount][1][0];
                rawResult[1] = acceptedInput[consumedCount][0]
                               * sampleMatrixWeight[consumedCount][0][1]
                               + acceptedInput[consumedCount][1]
                               * sampleMatrixWeight[consumedCount][1][1];

                if (resultData[0] !== expectedPredictionNow ||
                    resultData[1] !== '0)
                    $fatal(1, "prediction %0d got [%0d, %0d], expected [%0d, 0]",
                           consumedCount, resultData[0], resultData[1],
                           expectedPredictionNow);
                if (learningDirection !== expectedDirectionNow)
                    $fatal(1, "learning direction %0d got %0d, expected %0d",
                           consumedCount, learningDirection,
                           expectedDirectionNow);

                for (int lane = 0; lane < N; lane++) begin
                    if (rowDirection[lane] !==
                        ternary_sign(acceptedInput[consumedCount][lane]))
                        $fatal(1, "row direction %0d:%0d got %0d",
                               consumedCount, lane, rowDirection[lane]);
                    if (columnDirection[lane] !==
                        ((passThrough || (rawResult[lane] > 0))
                         ? ternary_product(
                               expectedDirectionNow,
                               ternary_sign(
                                   sampleReductionWeight[consumedCount][lane]))
                         : 2'sd0))
                        $fatal(1, "column direction %0d:%0d got %0d",
                               consumedCount, lane, columnDirection[lane]);

                    if (expectedTrainingEnable[consumedCount])
                        pendingReductionDirection[reductionQueueTail][lane] =
                            ternary_product(
                                expectedDirectionNow,
                                ternary_sign(
                                    (passThrough || (rawResult[lane] > 0))
                                    ? rawResult[lane] : 0));
                end

                if (expectedTrainingEnable[consumedCount]) begin
                    acceptedUpdateSampleIndex[acceptedUpdateCount] = consumedCount;
                    // A training transaction defines the reduction half of
                    // the next architectural network version immediately from
                    // the sample snapshot and its independently predicted
                    // learning direction. DUT reduction commit timing is not
                    // part of this expected-value model.
                    for (int lane = 0; lane < N; lane++) begin
                        architecturalReductionWeight[acceptedUpdateCount+1][lane] =
                            architecturalReductionWeight[acceptedUpdateCount][lane];
                        case (pendingReductionDirection[reductionQueueTail][lane])
                            2'sd1:
                                if (architecturalReductionWeight[acceptedUpdateCount][lane] !=
                                    {1'b0, {(REDUCTION_WEIGHT_WIDTH-1){1'b1}}})
                                    architecturalReductionWeight[acceptedUpdateCount+1][lane] =
                                        architecturalReductionWeight[acceptedUpdateCount][lane] + 1;
                            -2'sd1:
                                if (architecturalReductionWeight[acceptedUpdateCount][lane] !=
                                    {1'b1, {(REDUCTION_WEIGHT_WIDTH-1){1'b0}}})
                                    architecturalReductionWeight[acceptedUpdateCount+1][lane] =
                                        architecturalReductionWeight[acceptedUpdateCount][lane] - 1;
                            default:
                                architecturalReductionWeight[acceptedUpdateCount+1][lane] =
                                    architecturalReductionWeight[acceptedUpdateCount][lane];
                        endcase
                    end
                    acceptedUpdateCount = acceptedUpdateCount + 1;
                    reductionQueueTail = reductionQueueTail + 1;
                    reductionQueueCount = reductionQueueCount + 1;
                end

                case (expectedDirectionNow)
                    2'sd1:  sawPositive = 1;
                    2'sd0:  sawZero = 1;
                    -2'sd1: sawNegative = 1;
                    default: $fatal(1, "non-ternary learning direction %0d",
                                    learningDirection);
                endcase
                if ((consumedCount > 0) &&
                    (sampleMatrixVersion[consumedCount] !=
                     sampleMatrixVersion[consumedCount-1]) &&
                    (cycleCount == lastConsumedCycle + 1))
                    sawStreamingBoundary = 1;
                lastConsumedCycle = cycleCount;
                consumedCount = consumedCount + 1;
            end


            // The sideband package advances from PE stage zero to readout and
            // commits on the same edge as the first result behind it.
            if (dut.matrixDatapathAdvance &&
                dut.reductionBoundaryValidPipe[N]) begin
                if (reductionQueueCount == 0)
                    $fatal(1, "reference reduction queue underflow");
                for (int lane = 0; lane < N; lane++) begin
                    if ($signed(dut.reductionBoundaryDataPipe[N][2*lane +: 2]) !==
                        pendingReductionDirection[reductionQueueHead][lane])
                        $fatal(1, "reduction boundary package mismatch at lane %0d",
                               lane);
                    case (pendingReductionDirection[reductionQueueHead][lane])
                        2'sd1: if (modelReductionWeight[lane] !=
                                      {1'b0, {(REDUCTION_WEIGHT_WIDTH-1){1'b1}}}) begin
                            sawReductionIncrement = 1;
                            modelReductionWeight[lane] =
                                modelReductionWeight[lane] + 1;
                        end
                        -2'sd1: if (modelReductionWeight[lane] !=
                                       {1'b1, {(REDUCTION_WEIGHT_WIDTH-1){1'b0}}}) begin
                            sawReductionDecrement = 1;
                            modelReductionWeight[lane] =
                                modelReductionWeight[lane] - 1;
                        end
                        default: modelReductionWeight[lane] =
                                     modelReductionWeight[lane];
                    endcase
                end
                reductionQueueHead = reductionQueueHead + 1;
                reductionQueueCount = reductionQueueCount - 1;
                completedUpdateCount = completedUpdateCount + 1;
            end
        end
    end

    initial begin
        target_t heldTarget;
        result_t heldPrediction;
        logic signed [1:0] heldDirection;
        logic heldTrainingEnable;
        logic signed [WIDTH-1:0] inferenceMatrixWeight[N][N];
        logic signed [REDUCTION_WEIGHT_WIDTH-1:0]
            inferenceReductionWeight[N];

        clk = 0;
        rst_n = 0;
        weightValid = 0;
        activationValid = 0;
        trainingEnable = 0;
        resultReady = 0;
        reduceOutput = 1;
        passThrough = 1;
        reloadWeights = 0;
        loadReductionWeights = 0;
        targetData = 0;
        reductionWeight[0] = HALF;
        reductionWeight[1] = HALF;
        weightData[0] = 0;
        weightData[1] = 0;
        activationData[0] = 0;
        activationData[1] = 0;

        repeat (3) @(posedge clk);
        @(negedge clk) rst_n = 1;

        @(negedge clk) loadReductionWeights = 1;
        @(posedge clk);
        @(negedge clk) loadReductionWeights = 0;

        // Load fixed-point [[2, -1], [3, 4]] in reverse-row order.
        send_weight_row(3*SCALE, 4*SCALE);
        send_weight_row(2*SCALE, -1*SCALE);
        wait(weightsLoaded);

        // Four consecutive accepted samples carry the mixed training pattern
        // training pattern 0,1,0,1.  All four are in flight before the first
        // result is released, and the live input is returned to zero after
        // sample 3, so it disagrees with that buffered training transaction.
        @(negedge clk);
        activationData[0] = 7*SCALE;
        activationData[1] = -1*SCALE;
        targetData = 2*SCALE;
        trainingEnable = 0;
        activationValid = 1;
        if (!activationReady)
            $fatal(1, "sample 0 was unexpectedly backpressured");
        @(posedge clk);
        @(negedge clk);
        activationData[0] = 2*SCALE;
        activationData[1] = -1*SCALE;
        targetData = -3*SCALE;
        trainingEnable = 1;
        if (!activationReady)
            $fatal(1, "sample 1 was unexpectedly backpressured");
        @(posedge clk);
        @(negedge clk);
        activationData[0] = 5*SCALE;
        activationData[1] = -1*SCALE;
        targetData = -1*SCALE;
        trainingEnable = 0;
        if (!activationReady)
            $fatal(1, "sample 2 was unexpectedly backpressured");
        @(posedge clk);
        @(negedge clk);
        activationData[0] = 2*SCALE;
        activationData[1] = 0;
        targetData = 2*SCALE;
        trainingEnable = 1;
        if (!activationReady)
            $fatal(1, "sample 3 was unexpectedly backpressured");
        @(posedge clk);
        @(negedge clk);
        activationValid = 0;
        trainingEnable = 0;
        if (acceptedCount != 4)
            $fatal(1, "the alternating training pattern was not accepted consecutively");

        // Keep replacement inputs arriving on the same three clocks that
        // samples 0, 1, and 2 leave the full metadata FIFO. Sample 1 launches
        // the first update; sample 4 is the last sample at its old complete-
        // network boundary, and sample 5 is the first one behind it.
        // This leaves old- and new-version samples resident concurrently.
        wait(resultValid);
        @(negedge clk);
        activationData[0] = 3*SCALE;
        activationData[1] = SCALE;
        targetData = 0;
        trainingEnable = 0;
        activationValid = 1;
        resultReady = 1;
        @(posedge clk);
        @(negedge clk);
        activationData[0] = -2*SCALE;
        activationData[1] = SCALE;
        targetData = SCALE;
        @(posedge clk);
        @(negedge clk);
        activationData[0] = SCALE;
        activationData[1] = SCALE;
        targetData = -SCALE;
        @(posedge clk);
        @(negedge clk);
        activationValid = 0;
        resultReady = 0;
        if (acceptedCount != 7)
            $fatal(1, "version-boundary replacements accepted %0d samples, expected 7",
                   acceptedCount);
        wait(resultValid);
        heldTarget = resultTargetData;
        heldPrediction = resultData[0];
        heldDirection = learningDirection;
        heldTrainingEnable = dut.trainingEnableHead;
        if (heldTarget !== 2*SCALE)
            $fatal(1, "unexpected stalled tuple: prediction=%0d target=%0d direction=%0d",
                   heldPrediction, heldTarget, heldDirection);
        repeat (4) begin
            @(negedge clk);
            if (!resultValid || resultTargetData !== heldTarget ||
                resultData[0] !== heldPrediction ||
                learningDirection !== heldDirection ||
                dut.trainingEnableHead !== heldTrainingEnable || dut.targetPop)
                $fatal(1, "prediction or buffered metadata changed while output was stalled");
        end

        // Drain the boundary sequence. Only sample indices 1 and 3 may
        // create matrix/reduction update packages.
        resultReady = 1;
        wait_for_consumed(7);
        if (sampleMatrixVersion[4] != 0 ||
            sampleMatrixVersion[5] != 1 ||
            sampleMatrixVersion[6] != 1)
            $fatal(1, "focused boundary versions got [%0d,%0d,%0d], expected [0,1,1]",
                   sampleMatrixVersion[4], sampleMatrixVersion[5],
                   sampleMatrixVersion[6]);
        if (sampleReductionWeight[4][0] != HALF ||
            sampleReductionWeight[4][1] != HALF ||
            sampleReductionWeight[5][0] != HALF-1 ||
            sampleReductionWeight[5][1] != HALF+1 ||
            sampleReductionWeight[6][0] != HALF-1 ||
            sampleReductionWeight[6][1] != HALF+1)
            $fatal(1, "focused boundary did not snapshot old/new reduction states");
        if (acceptedUpdateCount != 2 ||
            acceptedUpdateSampleIndex[0] != 1 ||
            acceptedUpdateSampleIndex[1] != 3)
            $fatal(1, "alternating training pattern updated samples [%0d,%0d], expected [1,3]",
                   acceptedUpdateSampleIndex[0], acceptedUpdateSampleIndex[1]);
        wait(!dut.matrixEngine.pipelineBusy);

        // The first post-wave inference sample carries the twice-updated
        // matrix and reduction state without a readout catch-up bubble.
        send_sample_with_bubbles(SCALE, 0, SCALE, 0, 1);
        wait_for_consumed(8);
        if (completedUpdateCount != 2 || reductionQueueCount != 0)
            $fatal(1, "two sideband updates did not cross the learning boundary");

        // Inference-only traffic continues to predict through bubbles and
        // output backpressure without changing either resident weight set.
        inferenceMatrixWeight[0][0] =
            dut.matrixEngine.systolicArr.row_loop[0].col_loop[0].mb.weightReg;
        inferenceMatrixWeight[0][1] =
            dut.matrixEngine.systolicArr.row_loop[0].col_loop[1].mb.weightReg;
        inferenceMatrixWeight[1][0] =
            dut.matrixEngine.systolicArr.row_loop[1].col_loop[0].mb.weightReg;
        inferenceMatrixWeight[1][1] =
            dut.matrixEngine.systolicArr.row_loop[1].col_loop[1].mb.weightReg;
        for (int lane = 0; lane < N; lane++)
            inferenceReductionWeight[lane] = dut.residentReductionWeight[lane];
        resultReady = 0;
        send_sample_with_bubbles(-SCALE, SCALE, 0, 0, 2);
        send_sample_with_bubbles(0, -SCALE, -SCALE, 0, 1);
        wait(resultValid);
        repeat (3) @(negedge clk);
        resultReady = 1;
        wait_for_consumed(10);
        wait(!dut.matrixEngine.pipelineBusy);
        @(negedge clk);
        if (acceptedUpdateCount != 2)
            $fatal(1, "an inference-only sample generated a learning update");
        if (dut.matrixEngine.systolicArr.row_loop[0].col_loop[0].mb.weightReg !== inferenceMatrixWeight[0][0] ||
            dut.matrixEngine.systolicArr.row_loop[0].col_loop[1].mb.weightReg !== inferenceMatrixWeight[0][1] ||
            dut.matrixEngine.systolicArr.row_loop[1].col_loop[0].mb.weightReg !== inferenceMatrixWeight[1][0] ||
            dut.matrixEngine.systolicArr.row_loop[1].col_loop[1].mb.weightReg !== inferenceMatrixWeight[1][1])
            $fatal(1, "inference changed a matrix PE weight");
        for (int lane = 0; lane < N; lane++)
            if (dut.residentReductionWeight[lane] !==
                inferenceReductionWeight[lane])
                $fatal(1, "inference changed reduction weight %0d", lane);

        // Back-to-back training results place packages in successive update
        // stages.  Freeze arrayAdvance while both are live and prove that the
        // matrix wave and reduction sideband resume at the same boundary.
        send_sample_with_bubbles(2*SCALE, 0, 2*SCALE, 1, 2); // 1 -> +1
        send_sample_with_bubbles(2*SCALE, -1*SCALE, -3*SCALE, 1, 0); // -2.5 -> -1
        stall_active_update_wave();
        wait_for_consumed(12);
        send_sample_with_bubbles(0, -1*SCALE, -7*SCALE/2, 1, 1);   // -3.5 -> 0
        wait_for_consumed(13);
        wait(!dut.matrixEngine.pipelineBusy);
        send_sample_with_bubbles(SCALE, 0, 0, 0, 0);
        wait_for_consumed(14);
        wait_for_reduction_updates();

        // Exercise both saturation endpoints without relying on arithmetic
        // overflow. The first sample requests a decrement of a resident MIN;
        // the second requests an increment of a freshly reloaded MAX.
        reductionWeight[0] = {1'b0, {(REDUCTION_WEIGHT_WIDTH-1){1'b1}}};
        reductionWeight[1] = {1'b1, {(REDUCTION_WEIGHT_WIDTH-1){1'b0}}};
        load_reduction_vector();
        send_sample_with_bubbles(2*SCALE, SCALE, -(1 << (TARGET_WIDTH-1)), 1, 1);
        wait_for_consumed(15);
        wait(!dut.matrixEngine.pipelineBusy);
        send_sample_with_bubbles(SCALE, 0, 0, 0, 0);
        wait_for_consumed(16);
        wait_for_reduction_updates();
        @(negedge clk);
        if (dut.residentReductionWeight[1] !==
            {1'b1, {(REDUCTION_WEIGHT_WIDTH-1){1'b0}}})
            $fatal(1, "negative reduction-weight saturation failed");

        load_reduction_vector();
        send_sample_with_bubbles(0, SCALE, (1 << (TARGET_WIDTH-1))-1, 1, 1);
        wait_for_consumed(17);
        wait(!dut.matrixEngine.pipelineBusy);
        send_sample_with_bubbles(SCALE, 0, 0, 0, 0);
        wait_for_consumed(18);
        wait_for_reduction_updates();
        @(negedge clk);
        if (dut.residentReductionWeight[0] !==
            {1'b0, {(REDUCTION_WEIGHT_WIDTH-1){1'b1}}})
            $fatal(1, "positive reduction-weight saturation failed");

        // ReLU blocks the negative second pre-activation from the column
        // package and from the reduction update, while row signs still report
        // the original +/− input vector.
        reductionWeight[0] = HALF;
        reductionWeight[1] = HALF;
        load_reduction_vector();
        passThrough = 0;
        send_sample_with_bubbles(7*SCALE, -1*SCALE,
                                 (1 << (TARGET_WIDTH-1))-1, 1, 1);
        wait_for_consumed(19);

        // A nonzero error with an all-zero input must update neither the
        // reduction weights nor any matrix PE: row directions are all zero
        // and the activated vector is all zero.
        passThrough = 1;
        send_sample_with_bubbles(0, 0, SCALE, 1, 1);
        wait_for_consumed(20);
        wait(!dut.matrixEngine.pipelineBusy);
        send_sample_with_bubbles(SCALE, 0, 0, 0, 0);
        wait_for_consumed(SAMPLE_COUNT);
        @(negedge clk) resultReady = 0;
        wait(!dut.matrixEngine.pipelineBusy);
        @(negedge clk);
        for (int lane = 0; lane < N; lane++) begin
            if (dut.residentReductionWeight[lane] !==
                modelReductionWeight[lane])
                $fatal(1, "zero activation changed reduction weight %0d", lane);
        end
        if (dut.matrixEngine.systolicArr.row_loop[0].col_loop[0].mb.weightReg !==
                modelMatrixWeight[0][0] ||
            dut.matrixEngine.systolicArr.row_loop[0].col_loop[1].mb.weightReg !==
                modelMatrixWeight[0][1] ||
            dut.matrixEngine.systolicArr.row_loop[1].col_loop[0].mb.weightReg !==
                modelMatrixWeight[1][0] ||
            dut.matrixEngine.systolicArr.row_loop[1].col_loop[1].mb.weightReg !==
                modelMatrixWeight[1][1])
            $fatal(1, "zero input package changed a matrix PE");

        if (acceptedCount != SAMPLE_COUNT || consumedCount != SAMPLE_COUNT)
            $fatal(1, "got %0d accepted samples and %0d consumed pairs, expected %0d/%0d",
                   acceptedCount, consumedCount, SAMPLE_COUNT, SAMPLE_COUNT);
        if (!sawBackToBack || !sawBubble)
            $fatal(1, "did not observe both consecutive and bubbled accepted samples");
        if (!sawPositive || !sawZero || !sawNegative)
            $fatal(1, "did not observe all three learning-direction outcomes");
        if (!sawOverlappingUpdates)
            $fatal(1, "did not observe multiple learning packages in successive update stages");
        if (!sawUpdateWaveStall)
            $fatal(1, "did not complete the directed update-wave stall");
        if (!sawValidStallWithUpdateWave)
            $fatal(1, "did not stall a valid result while a matrix update wave was live");
        if (!sawStreamingBoundary)
            $fatal(1, "learning boundary inserted a bubble between ready results");
        if (!sawReadoutBoundaryEvaluation)
            $fatal(1, "no reduction update coincided with its first behind-boundary result");
        if (!sawOldWeightVersion || !sawUpdatedWeightVersion)
            $fatal(1, "did not observe both sides of the shared matrix/reduction version boundary");
        if (!sawReductionIncrement || !sawReductionDecrement)
            $fatal(1, "did not observe both signed reduction-weight LSB steps");
        if (acceptedUpdateCount != completedUpdateCount)
            $fatal(1, "accepted/completed learning packages differ: %0d/%0d",
                   acceptedUpdateCount, completedUpdateCount);
        if (REDUCTION_WEIGHT_WIDTH != 8 ||
            (1.0 / (1 << REDUCTION_FRACTION_BITS)) != 0.0078125)
            $fatal(1, "reduction weight LSB is not the required Q1.7 1/128");

        $display("PASS: Phase 5H structurally aligned streaming matrix/reduction boundaries, stalls, saturation, and inference stability.");
        $finish;
    end

    task send_weight_row(input integer lane0, input integer lane1);
        @(negedge clk);
        while (!weightReady) @(negedge clk);
        weightData[0] = lane0;
        weightData[1] = lane1;
        weightValid = 1;
        @(posedge clk);
        @(negedge clk) weightValid = 0;
    endtask

    task send_sample_with_bubbles(
        input integer lane0,
        input integer lane1,
        input integer target,
        input logic train,
        input integer bubbleCycles
    );
        activationValid = 0;
        repeat (bubbleCycles) @(negedge clk);
        activationData[0] = lane0;
        activationData[1] = lane1;
        targetData = target;
        trainingEnable = train;
        activationValid = 1;
        while (!activationReady) @(negedge clk);
        @(posedge clk);
        @(negedge clk) activationValid = 0;
    endtask

    task load_reduction_vector();
        wait_for_reduction_updates();
        @(negedge clk) loadReductionWeights = 1;
        @(posedge clk);
        @(negedge clk) loadReductionWeights = 0;
    endtask

    task stall_active_update_wave();
        logic heldStageValid[UPDATE_STAGES];
        logic signed [1:0] heldStageRow[UPDATE_STAGES][N];
        logic signed [1:0] heldStageColumn[UPDATE_STAGES][N];
        logic signed [WIDTH-1:0] heldMatrixWeight[N][N];
        begin
            wait ((dut.matrixEngine.systolicArr.updateValidPipe[0] &&
                   dut.matrixEngine.systolicArr.updateValidPipe[1]) ||
                  (dut.matrixEngine.systolicArr.updateValidPipe[1] &&
                   dut.matrixEngine.systolicArr.updateValidPipe[2]));
            @(negedge clk);
            force dut.matrixEngine.arrayAdvance = 1'b0;
            #1;

            heldMatrixWeight[0][0] = dut.matrixEngine.systolicArr.row_loop[0].col_loop[0].mb.weightReg;
            heldMatrixWeight[0][1] = dut.matrixEngine.systolicArr.row_loop[0].col_loop[1].mb.weightReg;
            heldMatrixWeight[1][0] = dut.matrixEngine.systolicArr.row_loop[1].col_loop[0].mb.weightReg;
            heldMatrixWeight[1][1] = dut.matrixEngine.systolicArr.row_loop[1].col_loop[1].mb.weightReg;
            for (int stage = 0; stage < UPDATE_STAGES; stage++) begin
                heldStageValid[stage] =
                    dut.matrixEngine.systolicArr.updateValidPipe[stage];
                for (int lane = 0; lane < N; lane++) begin
                    heldStageRow[stage][lane] =
                        dut.matrixEngine.systolicArr.updateRowPipe[stage][lane];
                    heldStageColumn[stage][lane] =
                        dut.matrixEngine.systolicArr.updateColumnPipe[stage][lane];
                end
            end
            repeat (3) begin
                @(posedge clk); #1;
                if (dut.matrixUpdateComplete)
                    $fatal(1, "matrix update completed during arrayAdvance stall");
                for (int stage = 0; stage < UPDATE_STAGES; stage++) begin
                    if (dut.matrixEngine.systolicArr.updateValidPipe[stage] !==
                        heldStageValid[stage])
                        $fatal(1, "matrix update stage %0d moved during stall", stage);
                    for (int lane = 0; lane < N; lane++) begin
                        if (dut.matrixEngine.systolicArr.updateRowPipe[stage][lane] !==
                                heldStageRow[stage][lane] ||
                            dut.matrixEngine.systolicArr.updateColumnPipe[stage][lane] !==
                                heldStageColumn[stage][lane])
                            $fatal(1, "matrix update package changed during stall at stage %0d lane %0d",
                                   stage, lane);
                    end
                end
                if (dut.matrixEngine.systolicArr.row_loop[0].col_loop[0].mb.weightReg !== heldMatrixWeight[0][0] ||
                    dut.matrixEngine.systolicArr.row_loop[0].col_loop[1].mb.weightReg !== heldMatrixWeight[0][1] ||
                    dut.matrixEngine.systolicArr.row_loop[1].col_loop[0].mb.weightReg !== heldMatrixWeight[1][0] ||
                    dut.matrixEngine.systolicArr.row_loop[1].col_loop[1].mb.weightReg !== heldMatrixWeight[1][1])
                    $fatal(1, "matrix PE weight changed during update-wave stall");
            end

            @(negedge clk);
            release dut.matrixEngine.arrayAdvance;
            sawUpdateWaveStall = 1;
            wait(dut.matrixUpdateComplete);
        end
    endtask

    task wait_for_reduction_updates();
        integer watchdog;
        begin
            watchdog = 0;
            while ((reductionQueueCount != 0) && (watchdog < 100)) begin
                @(negedge clk);
                watchdog = watchdog + 1;
            end
            if (reductionQueueCount != 0)
                $fatal(1, "timed out advancing pending reduction boundaries");
        end
    endtask

    task wait_for_consumed(input integer expectedCount);
        integer watchdog;
        begin
            watchdog = 0;
            while ((consumedCount < expectedCount) && (watchdog < 200)) begin
                @(negedge clk);
                watchdog = watchdog + 1;
            end
            if (consumedCount != expectedCount)
                $fatal(1, "timed out with %0d results consumed, expected %0d",
                       consumedCount, expectedCount);
        end
    endtask

endmodule
