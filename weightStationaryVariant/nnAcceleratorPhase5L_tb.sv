`timescale 1ns / 1ps

// Phase 5L timing comparison. The two DUTs receive the same traffic and
// weights; only the training-enable value differs. Their control traces must
// remain identical while the training DUT carries overlapping matrix and
// reduction updates.
module nnAcceleratorPhase5L_tb;
    localparam int WIDTH = 8;
    localparam int N = 3;
    localparam int TARGET_WIDTH = 8;
    localparam int FRACTION_BITS = 0;
    localparam int REDUCTION_WEIGHT_WIDTH = 8;
    localparam int PREDICTION_WIDTH = 2*WIDTH + 2*$clog2(N);
    localparam int UPDATE_PIPE_STAGES = 2*N-2;
    localparam int REDUCTION_BOUNDARY_STAGES = 2*N-1;
    localparam int DEFAULT_INPUT_FIFO_DEPTH = 2*N;
    localparam int EXPECTED_SAMPLE_CONTEXT_DEPTH = 2*N+2;
    localparam int CONTINUOUS_SAMPLES = 32;
    localparam int BUBBLE_SAMPLES = 8;
    localparam int BACKPRESSURE_SAMPLES = 12;

    localparam int PH_IDLE = 0;
    localparam int PH_CONTINUOUS = 1;
    localparam int PH_BUBBLES = 2;
    localparam int PH_BACKPRESSURE = 3;
    localparam int PH_RESUME = 4;

    logic clk, rst_n;
    logic signed [WIDTH-1:0] weightData[N], activationData[N];
    logic signed [TARGET_WIDTH-1:0] targetData;
    logic weightValid, activationValid;
    logic trainingEnable;
    logic signed [REDUCTION_WEIGHT_WIDTH-1:0] reductionWeight[N];
    logic loadReductionWeights, reduceOutput, resultReady;
    logic reloadWeights, passThrough;

    logic weightReadyInference, weightReadyTraining;
    logic activationReadyInference, activationReadyTraining;
    logic signed [PREDICTION_WIDTH-1:0] inferenceResult[N], trainingResult[N];
    logic signed [TARGET_WIDTH-1:0] inferenceTarget, trainingTarget;
    logic signed [1:0] inferenceLearningDirection, trainingLearningDirection;
    logic signed [1:0] inferenceRowDirection[N], trainingRowDirection[N];
    logic signed [1:0] inferenceColumnDirection[N], trainingColumnDirection[N];
    logic inferenceMatrixUpdateValid, trainingMatrixUpdateValid;
    logic inferenceResultValid, trainingResultValid;
    logic inferenceResultLast, trainingResultLast;
    logic inferenceWeightsLoaded, trainingWeightsLoaded;
    logic inferenceReloadReady, trainingReloadReady;

    integer phase;
    integer cycleCount;
    integer acceptedInference, acceptedTraining;
    integer resultsInference, resultsTraining;
    integer firstAcceptInference, firstAcceptTraining;
    integer firstResultInference, firstResultTraining;
    integer lastResultCycleInference, lastResultCycleTraining;
    integer activationReadyMismatches, acceptanceMismatches;
    integer arrayAdvanceMismatches, resultValidMismatches;
    integer resultLastMismatches, resultHandshakeMismatches;
    integer trainingUpdateCount, reductionBoundaryApplyCount;
    integer lastUpdateCycle, lastBubbleAcceptCycle;
    integer lastAcceptTrainingCycle;
    integer continuousResultSpan;
    integer continuousAcceptSpan;
    bit continuousCadenceStarted;
    bit sawContinuousCadence;
    bit sawInputBubble;
    bit sawConsecutiveUpdates;
    bit sawOverlappingUpdates;
    bit sawBoundaryOnBubbleTraffic;

    nnAccelerator #(
        .WIDTH(WIDTH), .N(N), .TARGET_WIDTH(TARGET_WIDTH),
        .FRACTION_BITS(FRACTION_BITS),
        .REDUCTION_WEIGHT_WIDTH(REDUCTION_WEIGHT_WIDTH)
    ) inferenceDut (
        .clk(clk), .rst_n(rst_n),
        .weightData(weightData), .weightValid(weightValid),
        .weightReady(weightReadyInference),
        .activationData(activationData), .targetData(targetData),
        .trainingEnable(1'b0),
        .activationValid(activationValid),
        .activationReady(activationReadyInference),
        .reductionWeight(reductionWeight),
        .loadReductionWeights(loadReductionWeights),
        .reduceOutput(reduceOutput), .resultData(inferenceResult),
        .resultTargetData(inferenceTarget),
        .learningDirection(inferenceLearningDirection),
        .rowDirection(inferenceRowDirection),
        .columnDirection(inferenceColumnDirection),
        .matrixUpdateValid(inferenceMatrixUpdateValid),
        .resultValid(inferenceResultValid), .resultReady(resultReady),
        .resultLast(inferenceResultLast),
        .weightsLoaded(inferenceWeightsLoaded), .reloadWeights(reloadWeights),
        .reloadReady(inferenceReloadReady), .passThrough(passThrough)
    );

    nnAccelerator #(
        .WIDTH(WIDTH), .N(N), .TARGET_WIDTH(TARGET_WIDTH),
        .FRACTION_BITS(FRACTION_BITS),
        .REDUCTION_WEIGHT_WIDTH(REDUCTION_WEIGHT_WIDTH)
    ) trainingDut (
        .clk(clk), .rst_n(rst_n),
        .weightData(weightData), .weightValid(weightValid),
        .weightReady(weightReadyTraining),
        .activationData(activationData), .targetData(targetData),
        .trainingEnable(trainingEnable),
        .activationValid(activationValid),
        .activationReady(activationReadyTraining),
        .reductionWeight(reductionWeight),
        .loadReductionWeights(loadReductionWeights),
        .reduceOutput(reduceOutput), .resultData(trainingResult),
        .resultTargetData(trainingTarget),
        .learningDirection(trainingLearningDirection),
        .rowDirection(trainingRowDirection),
        .columnDirection(trainingColumnDirection),
        .matrixUpdateValid(trainingMatrixUpdateValid),
        .resultValid(trainingResultValid), .resultReady(resultReady),
        .resultLast(trainingResultLast),
        .weightsLoaded(trainingWeightsLoaded), .reloadWeights(reloadWeights),
        .reloadReady(trainingReloadReady), .passThrough(passThrough)
    );

    always #5 clk = ~clk;

    always @(posedge clk) begin
        integer liveUpdates;

        if (!rst_n) begin
            cycleCount = 0;
            acceptedInference = 0;
            acceptedTraining = 0;
            resultsInference = 0;
            resultsTraining = 0;
            firstAcceptInference = -1;
            firstAcceptTraining = -1;
            firstResultInference = -1;
            firstResultTraining = -1;
            lastResultCycleInference = -1;
            lastResultCycleTraining = -1;
            activationReadyMismatches = 0;
            acceptanceMismatches = 0;
            arrayAdvanceMismatches = 0;
            resultValidMismatches = 0;
            resultLastMismatches = 0;
            resultHandshakeMismatches = 0;
            trainingUpdateCount = 0;
            reductionBoundaryApplyCount = 0;
            lastUpdateCycle = -100;
            lastBubbleAcceptCycle = -100;
            lastAcceptTrainingCycle = -1;
            continuousAcceptSpan = 0;
            continuousCadenceStarted = 0;
            sawContinuousCadence = 0;
            sawInputBubble = 0;
            sawConsecutiveUpdates = 0;
            sawOverlappingUpdates = 0;
            sawBoundaryOnBubbleTraffic = 0;
        end else begin
            cycleCount = cycleCount + 1;

            // Hold valid continuously through the initial fill. Any ready
            // hole here would expose sample-context backpressure before the
            // first result retires.
            if ((phase == PH_CONTINUOUS) && activationValid &&
                !activationReadyTraining)
                $fatal(1, "default continuous acceptance stalled at cycle %0d before all samples were accepted", cycleCount);

            if (inferenceWeightsLoaded && trainingWeightsLoaded) begin
                if (activationReadyInference !== activationReadyTraining)
                    activationReadyMismatches = activationReadyMismatches + 1;
                if (inferenceDut.matrixDatapathAdvance !==
                    trainingDut.matrixDatapathAdvance)
                    arrayAdvanceMismatches = arrayAdvanceMismatches + 1;
                if (inferenceResultValid !== trainingResultValid)
                    resultValidMismatches = resultValidMismatches + 1;
                if (inferenceResultLast !== trainingResultLast)
                    resultLastMismatches = resultLastMismatches + 1;
                if ((inferenceResultValid && resultReady) !==
                    (trainingResultValid && resultReady))
                    resultHandshakeMismatches = resultHandshakeMismatches + 1;
            end

            if ((activationValid && activationReadyInference) !==
                (activationValid && activationReadyTraining))
                acceptanceMismatches = acceptanceMismatches + 1;

            if (activationValid && activationReadyInference) begin
                acceptedInference = acceptedInference + 1;
                if (firstAcceptInference < 0)
                    firstAcceptInference = cycleCount;
                if ((lastBubbleAcceptCycle >= 0) &&
                    (cycleCount > lastBubbleAcceptCycle + 1))
                    sawInputBubble = 1;
                lastBubbleAcceptCycle = cycleCount;
            end
            if (activationValid && activationReadyTraining) begin
                acceptedTraining = acceptedTraining + 1;
                if (firstAcceptTraining < 0)
                    firstAcceptTraining = cycleCount;
                lastAcceptTrainingCycle = cycleCount;
            end

            if (inferenceResultValid && resultReady) begin
                resultsInference = resultsInference + 1;
                if (firstResultInference < 0)
                    firstResultInference = cycleCount;
                lastResultCycleInference = cycleCount;
            end
            if (trainingResultValid && resultReady) begin
                resultsTraining = resultsTraining + 1;
                if (firstResultTraining < 0)
                    firstResultTraining = cycleCount;
                lastResultCycleTraining = cycleCount;
            end

            if (trainingMatrixUpdateValid) begin
                trainingUpdateCount = trainingUpdateCount + 1;
                if (lastUpdateCycle == cycleCount - 1)
                    sawConsecutiveUpdates = 1;
                lastUpdateCycle = cycleCount;
            end
            if (trainingDut.reductionBoundaryApply) begin
                if (!trainingDut.matrixDatapathAdvance)
                    $fatal(1, "reduction boundary applied without datapath advance");
                reductionBoundaryApplyCount = reductionBoundaryApplyCount + 1;
                if (phase == PH_BUBBLES)
                    sawBoundaryOnBubbleTraffic = 1;
            end

            liveUpdates = 0;
            for (int stage = 0; stage < UPDATE_PIPE_STAGES; stage++)
                if (trainingDut.matrixEngine.systolicArr.updateValidPipe[stage])
                    liveUpdates = liveUpdates + 1;
            if (liveUpdates >= 2)
                sawOverlappingUpdates = 1;

            // Once the first continuous result appears, every subsequent
            // result in this no-stall window must appear on the next cycle.
            if ((phase == PH_CONTINUOUS) &&
                (firstResultTraining >= 0) &&
                (resultsTraining < CONTINUOUS_SAMPLES)) begin
                continuousCadenceStarted = 1;
                if (!inferenceResultValid || !trainingResultValid)
                    $fatal(1, "continuous result cadence contains a learning-only gap");
            end
            if (continuousCadenceStarted &&
                (resultsTraining == CONTINUOUS_SAMPLES))
                sawContinuousCadence = 1;
        end
    end

    initial begin
        integer stalledResident[N];
        integer stalledPeWeight[N][N];
        logic stalledUpdateValid[UPDATE_PIPE_STAGES];
        logic signed [1:0] stalledUpdateRow[UPDATE_PIPE_STAGES][N];
        logic signed [1:0] stalledUpdateColumn[UPDATE_PIPE_STAGES][N];
        logic stalledReductionValid[REDUCTION_BOUNDARY_STAGES];
        logic signed [2*N-1:0]
            stalledReductionData[REDUCTION_BOUNDARY_STAGES];
        logic stalledSkewValid[N][N];
        logic signed [WIDTH-1:0] stalledSkewData[N][N];
        logic signed [PREDICTION_WIDTH-1:0] stalledResult[N];
        integer acceptedBeforeBackpressure;
        integer resultBeforeBackpressure;
        integer watchdog;

        clk = 0;
        rst_n = 0;
        phase = PH_IDLE;
        weightValid = 0;
        activationValid = 0;
        trainingEnable = 1;
        loadReductionWeights = 0;
        reduceOutput = 1;
        resultReady = 1;
        reloadWeights = 0;
        passThrough = 1;
        targetData = 127;
        activationData[0] = 1;
        activationData[1] = 2;
        activationData[2] = 3;
        reductionWeight[0] = 16;
        reductionWeight[1] = 24;
        reductionWeight[2] = 32;
        for (int lane = 0; lane < N; lane++)
            weightData[lane] = 0;

        repeat (3) @(posedge clk);
        @(negedge clk) begin
            rst_n = 1;
            loadReductionWeights = 1;
        end
        @(posedge clk);
        @(negedge clk) loadReductionWeights = 0;

        send_weight_row(7, 8, 9);
        send_weight_row(4, 5, 6);
        send_weight_row(1, 2, 3);
        wait(inferenceWeightsLoaded && trainingWeightsLoaded);

        // Continuous inference/training comparison. Both streams accept and
        // retire the same number of transactions, with one result per cycle
        // after the common first-result latency.
        phase = PH_CONTINUOUS;
        trainingEnable = 1;
        @(negedge clk) activationValid = 1;
        while (acceptedTraining < CONTINUOUS_SAMPLES) @(negedge clk);
        activationValid = 0;
        while (resultsTraining < CONTINUOUS_SAMPLES) @(negedge clk);
        phase = PH_IDLE;

        if (inferenceDut.INPUT_FIFO_DEPTH != DEFAULT_INPUT_FIFO_DEPTH ||
            trainingDut.INPUT_FIFO_DEPTH != DEFAULT_INPUT_FIFO_DEPTH)
            $fatal(1, "Phase 5M regression unexpectedly overrode INPUT_FIFO_DEPTH");
        if (inferenceDut.SAMPLE_CONTEXT_DEPTH != EXPECTED_SAMPLE_CONTEXT_DEPTH ||
            trainingDut.SAMPLE_CONTEXT_DEPTH != EXPECTED_SAMPLE_CONTEXT_DEPTH)
            $fatal(1, "N=3 SAMPLE_CONTEXT_DEPTH was not %0d",
                   EXPECTED_SAMPLE_CONTEXT_DEPTH);
        continuousAcceptSpan = lastAcceptTrainingCycle -
                                firstAcceptTraining + 1;
        if (continuousAcceptSpan != CONTINUOUS_SAMPLES)
            $fatal(1, "default continuous acceptance span was %0d cycles, expected %0d",
                   continuousAcceptSpan, CONTINUOUS_SAMPLES);
        if (!sawContinuousCadence)
            $fatal(1, "continuous result cadence was not one result per cycle");
        if (acceptedInference != CONTINUOUS_SAMPLES ||
            acceptedTraining != CONTINUOUS_SAMPLES ||
            resultsInference != CONTINUOUS_SAMPLES ||
            resultsTraining != CONTINUOUS_SAMPLES)
            $fatal(1, "continuous counts diverged: accepts %0d/%0d results %0d/%0d",
                   acceptedInference, acceptedTraining,
                   resultsInference, resultsTraining);
        if ((firstAcceptInference - firstResultInference) !=
            (firstAcceptTraining - firstResultTraining))
            $fatal(1, "inference/training first-result latency differs");
        continuousResultSpan = lastResultCycleTraining -
                               firstResultTraining + 1;
        if (continuousResultSpan != CONTINUOUS_SAMPLES)
            $fatal(1, "continuous training result span was %0d cycles, expected %0d",
                   continuousResultSpan, CONTINUOUS_SAMPLES);

        // Sparse traffic: the only result gaps permitted here are the same
        // gaps seen by the inference twin.
        phase = PH_BUBBLES;
        send_sample(1, 2, 3, 127, 1, 0);
        send_sample(2, 1, 3, 127, 1, 2);
        send_sample(3, 1, 2, 127, 1, 0);
        send_sample(1, 3, 2, 127, 1, 3);
        send_sample(2, 3, 1, 127, 1, 0);
        send_sample(3, 2, 1, 127, 1, 2);
        send_sample(1, 1, 1, 127, 1, 0);
        send_sample(2, 2, 1, 127, 1, 3);
        wait_for_results(40);
        phase = PH_IDLE;
        if (!sawInputBubble || !sawBoundaryOnBubbleTraffic)
            $fatal(1, "bubble phase did not exercise sparse traffic and a boundary");

        // Hold resultReady low until the ordinary output buffers fill. At
        // that point arrayAdvance is low and every state that can carry data,
        // matrix updates, or reduction updates must remain bit-for-bit fixed.
        phase = PH_BACKPRESSURE;
        resultReady = 0;
        activationValid = 1;
        acceptedBeforeBackpressure = acceptedTraining;
        resultBeforeBackpressure = resultsTraining;
        watchdog = 0;
        while (trainingDut.matrixDatapathAdvance && (watchdog < 200)) begin
            @(negedge clk);
            watchdog = watchdog + 1;
        end
        if (trainingDut.matrixDatapathAdvance ||
            inferenceDut.matrixDatapathAdvance)
            $fatal(1, "output backpressure did not reach the shared advance gate");

        for (int lane = 0; lane < N; lane++) begin
            stalledResident[lane] = trainingDut.residentReductionWeight[lane];
            for (int column = 0; column < N; column++) begin
                stalledSkewValid[lane][column] =
                    trainingDut.matrixEngine.skewValid[lane][column];
                stalledSkewData[lane][column] =
                    trainingDut.matrixEngine.skewData[lane][column];
            end
        end
        for (int lane = 0; lane < N; lane++)
            stalledResult[lane] = trainingResult[lane];
        stalledPeWeight[0][0] = trainingDut.matrixEngine.systolicArr.row_loop[0].col_loop[0].mb.weightReg;
        stalledPeWeight[0][1] = trainingDut.matrixEngine.systolicArr.row_loop[0].col_loop[1].mb.weightReg;
        stalledPeWeight[0][2] = trainingDut.matrixEngine.systolicArr.row_loop[0].col_loop[2].mb.weightReg;
        stalledPeWeight[1][0] = trainingDut.matrixEngine.systolicArr.row_loop[1].col_loop[0].mb.weightReg;
        stalledPeWeight[1][1] = trainingDut.matrixEngine.systolicArr.row_loop[1].col_loop[1].mb.weightReg;
        stalledPeWeight[1][2] = trainingDut.matrixEngine.systolicArr.row_loop[1].col_loop[2].mb.weightReg;
        stalledPeWeight[2][0] = trainingDut.matrixEngine.systolicArr.row_loop[2].col_loop[0].mb.weightReg;
        stalledPeWeight[2][1] = trainingDut.matrixEngine.systolicArr.row_loop[2].col_loop[1].mb.weightReg;
        stalledPeWeight[2][2] = trainingDut.matrixEngine.systolicArr.row_loop[2].col_loop[2].mb.weightReg;
        for (int stage = 0; stage < UPDATE_PIPE_STAGES; stage++) begin
            stalledUpdateValid[stage] =
                trainingDut.matrixEngine.systolicArr.updateValidPipe[stage];
            for (int lane = 0; lane < N; lane++) begin
                stalledUpdateRow[stage][lane] =
                    trainingDut.matrixEngine.systolicArr.updateRowPipe[stage][lane];
                stalledUpdateColumn[stage][lane] =
                    trainingDut.matrixEngine.systolicArr.updateColumnPipe[stage][lane];
            end
        end
        for (int stage = 0; stage < REDUCTION_BOUNDARY_STAGES; stage++) begin
            stalledReductionValid[stage] =
                trainingDut.reductionBoundaryValidPipe[stage];
            stalledReductionData[stage] =
                trainingDut.reductionBoundaryDataPipe[stage];
        end

        repeat (4) begin
            @(posedge clk);
            #1;
            if (trainingDut.matrixDatapathAdvance ||
                inferenceDut.matrixDatapathAdvance)
                $fatal(1, "arrayAdvance changed while output backpressure was active");
            for (int lane = 0; lane < N; lane++) begin
                if (trainingDut.residentReductionWeight[lane] !=
                    stalledResident[lane])
                    $fatal(1, "resident reduction weight advanced during backpressure");
                if (trainingResult[lane] != stalledResult[lane])
                    $fatal(1, "held result changed during backpressure");
                if ((trainingDut.matrixEngine.systolicArr.row_loop[0].col_loop[0].mb.weightReg != stalledPeWeight[0][0]) ||
                    (trainingDut.matrixEngine.systolicArr.row_loop[0].col_loop[1].mb.weightReg != stalledPeWeight[0][1]) ||
                    (trainingDut.matrixEngine.systolicArr.row_loop[0].col_loop[2].mb.weightReg != stalledPeWeight[0][2]) ||
                    (trainingDut.matrixEngine.systolicArr.row_loop[1].col_loop[0].mb.weightReg != stalledPeWeight[1][0]) ||
                    (trainingDut.matrixEngine.systolicArr.row_loop[1].col_loop[1].mb.weightReg != stalledPeWeight[1][1]) ||
                    (trainingDut.matrixEngine.systolicArr.row_loop[1].col_loop[2].mb.weightReg != stalledPeWeight[1][2]) ||
                    (trainingDut.matrixEngine.systolicArr.row_loop[2].col_loop[0].mb.weightReg != stalledPeWeight[2][0]) ||
                    (trainingDut.matrixEngine.systolicArr.row_loop[2].col_loop[1].mb.weightReg != stalledPeWeight[2][1]) ||
                    (trainingDut.matrixEngine.systolicArr.row_loop[2].col_loop[2].mb.weightReg != stalledPeWeight[2][2]))
                    $fatal(1, "PE weight changed during backpressure");
                for (int column = 0; column < N; column++) begin
                    if (trainingDut.matrixEngine.skewValid[lane][column] !=
                        stalledSkewValid[lane][column] ||
                        trainingDut.matrixEngine.skewData[lane][column] !=
                        stalledSkewData[lane][column])
                        $fatal(1, "data state changed during backpressure");
                end
            end
            for (int stage = 0; stage < UPDATE_PIPE_STAGES; stage++) begin
                if (trainingDut.matrixEngine.systolicArr.updateValidPipe[stage] !=
                    stalledUpdateValid[stage])
                    $fatal(1, "matrix update wave moved during backpressure");
                for (int lane = 0; lane < N; lane++)
                    if (trainingDut.matrixEngine.systolicArr.updateRowPipe[stage][lane] !=
                            stalledUpdateRow[stage][lane] ||
                        trainingDut.matrixEngine.systolicArr.updateColumnPipe[stage][lane] !=
                            stalledUpdateColumn[stage][lane])
                        $fatal(1, "matrix update package changed during backpressure");
            end
            for (int stage = 0; stage < REDUCTION_BOUNDARY_STAGES; stage++)
                if (trainingDut.reductionBoundaryValidPipe[stage] !=
                        stalledReductionValid[stage] ||
                    trainingDut.reductionBoundaryDataPipe[stage] !=
                        stalledReductionData[stage])
                    $fatal(1, "reduction update state moved during backpressure");
        end

        // Release the ordinary result backpressure. Keep the input valid until
        // more samples are accepted, then drain all ordinary results and the
        // reduction continuation with no special learning service cycle.
        phase = PH_RESUME;
        @(negedge clk) resultReady = 1;
        while (acceptedTraining < acceptedBeforeBackpressure +
               BACKPRESSURE_SAMPLES)
            @(negedge clk);
        activationValid = 0;
        wait_for_results(acceptedBeforeBackpressure + BACKPRESSURE_SAMPLES);
        wait_for_quiescence();

        if (acceptedInference != acceptedTraining ||
            resultsInference != resultsTraining)
            $fatal(1, "inference/training transaction counts diverged after resume");
        if (resultsTraining != acceptedTraining)
            $fatal(1, "not all accepted results drained after resume");
        if (!sawConsecutiveUpdates || !sawOverlappingUpdates)
            $fatal(1, "training did not produce consecutive overlapping updates");
        if (trainingUpdateCount != resultsTraining ||
            reductionBoundaryApplyCount != trainingUpdateCount)
            $fatal(1, "training update/boundary counts differ: updates=%0d boundaries=%0d results=%0d",
                   trainingUpdateCount, reductionBoundaryApplyCount, resultsTraining);
        if (activationReadyMismatches != 0 || acceptanceMismatches != 0 ||
            arrayAdvanceMismatches != 0 || resultValidMismatches != 0 ||
            resultLastMismatches != 0 || resultHandshakeMismatches != 0)
            $fatal(1, "inference/training timing mismatch: ready=%0d accept=%0d advance=%0d valid=%0d last=%0d handshake=%0d",
                   activationReadyMismatches, acceptanceMismatches,
                   arrayAdvanceMismatches, resultValidMismatches,
                   resultLastMismatches, resultHandshakeMismatches);

        $display("PASS: Phase 5L inference/training timing identical; continuous accepts/results=%0d/%0d, first latency=%0d cycles, sustained span=%0d cycles (1 result/cycle), updates=%0d, boundaries=%0d.",
                 CONTINUOUS_SAMPLES, CONTINUOUS_SAMPLES,
                 firstResultTraining - firstAcceptTraining,
                 continuousResultSpan, trainingUpdateCount,
                 reductionBoundaryApplyCount);
        $finish;
    end

    task send_weight_row(input integer w0, input integer w1, input integer w2);
        @(negedge clk);
        while (!weightReadyInference || !weightReadyTraining) @(negedge clk);
        weightData[0] = w0;
        weightData[1] = w1;
        weightData[2] = w2;
        weightValid = 1;
        @(posedge clk);
        @(negedge clk) weightValid = 0;
    endtask

    task send_sample(
        input integer x0,
        input integer x1,
        input integer x2,
        input integer target,
        input logic train,
        input integer bubbleCycles
    );
        activationValid = 0;
        repeat (bubbleCycles) @(negedge clk);
        activationData[0] = x0;
        activationData[1] = x1;
        activationData[2] = x2;
        targetData = target;
        trainingEnable = train;
        activationValid = 1;
        while (!activationReadyInference || !activationReadyTraining)
            @(negedge clk);
        @(posedge clk);
        @(negedge clk) activationValid = 0;
    endtask

    task wait_for_results(input integer expectedCount);
        integer localWatchdog;
        begin
            localWatchdog = 0;
            while ((resultsTraining < expectedCount) &&
                   (localWatchdog < 500)) begin
                @(negedge clk);
                localWatchdog = localWatchdog + 1;
            end
            if (resultsTraining != expectedCount ||
                resultsInference != expectedCount)
                $fatal(1, "timed out draining results: got %0d/%0d expected %0d",
                       resultsInference, resultsTraining, expectedCount);
        end
    endtask

    task wait_for_quiescence();
        integer localWatchdog;
        begin
            localWatchdog = 0;
            while ((trainingDut.matrixEngine.pipelineBusy ||
                    trainingDut.reductionBoundaryBusy ||
                    !trainingDut.matrixEngine.reloadReady) &&
                   (localWatchdog < 500)) begin
                @(negedge clk);
                localWatchdog = localWatchdog + 1;
            end
            if (trainingDut.matrixEngine.pipelineBusy ||
                trainingDut.reductionBoundaryBusy)
                $fatal(1, "training state did not drain after result resume");
        end
    endtask
endmodule
