`timescale 1ns / 1ps

// Integration-level directed verification for the Phase 5B resident reduction
// state and aligned SSLMS package. The scoreboard uses accepted ready/valid
// transactions only; it does not assume a pipeline latency.
module nnAccelerator_tb;
    localparam int WIDTH = 8;
    localparam int TARGET_WIDTH = 7;
    localparam int N = 2;
    localparam int FRACTION_BITS = 4;
    localparam int REDUCTION_FRACTION_BITS = WIDTH - 1;
    localparam int RESCALE_SHIFT = FRACTION_BITS
                                   + REDUCTION_FRACTION_BITS;
    localparam int SCALE = 1 << FRACTION_BITS;
    localparam int HALF = 1 << (REDUCTION_FRACTION_BITS - 1);
    localparam int PREDICTION_WIDTH = 2*WIDTH + 2*$clog2(N);
    localparam int SAMPLE_COUNT = 9;

    typedef logic signed [PREDICTION_WIDTH-1:0] result_t;
    typedef logic signed [TARGET_WIDTH-1:0] target_t;

    logic clk, rst_n;
    logic signed [WIDTH-1:0] weightData[N], activationData[N];
    target_t targetData, resultTargetData;
    logic weightValid, weightReady, activationValid, activationReady;
    logic signed [WIDTH-1:0] reductionWeight[N];
    result_t resultData[N];
    logic signed [1:0] learningDirection;
    logic signed [1:0] rowDirection[N], columnDirection[N];
    logic matrixUpdateValid, loadReductionWeights;
    logic reduceOutput, resultValid, resultReady, resultLast;
    logic weightsLoaded, reloadWeights, reloadReady, passThrough;

    integer cycleCount, acceptedCount, consumedCount;
    integer lastAcceptedCycle;
    target_t expectedTarget[0:SAMPLE_COUNT-1];
    logic signed [WIDTH-1:0] acceptedInput[0:SAMPLE_COUNT-1][N];
    logic signed [WIDTH-1:0] modelReductionWeight[N];
    bit sawBackToBack, sawBubble;
    bit sawPositive, sawZero, sawNegative;

    nnAccelerator #(
        .WIDTH(WIDTH), .N(N), .TARGET_WIDTH(TARGET_WIDTH),
        .FRACTION_BITS(FRACTION_BITS),
        .INPUT_FIFO_DEPTH(2), .OUTPUT_FIFO_DEPTH(2)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .weightData(weightData), .weightValid(weightValid), .weightReady(weightReady),
        .activationData(activationData), .targetData(targetData),
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

    // Loaded W is [[2, -1], [3, 4]], passThrough is enabled, and both
    // reduction coefficients are 0.5. The model keeps the matrix result and
    // complete weighted sum at full precision, then performs the one required
    // final rescale.
    function automatic result_t phase5_prediction(
        input logic signed [WIDTH-1:0] x0,
        input logic signed [WIDTH-1:0] x1,
        input logic signed [WIDTH-1:0] reduction0,
        input logic signed [WIDTH-1:0] reduction1,
        input logic applyPassThrough
    );
        integer xw0, xw1;
        integer fullReduction;
        begin
            xw0 = x0 * (2*SCALE) + x1 * (3*SCALE);
            xw1 = x0 * (-1*SCALE) + x1 * (4*SCALE);
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

        if (!rst_n) begin
            cycleCount        = 0;
            acceptedCount     = 0;
            consumedCount     = 0;
            lastAcceptedCycle = -2;
            sawBackToBack     = 0;
            sawBubble         = 0;
            sawPositive       = 0;
            sawZero           = 0;
            sawNegative       = 0;
            for (int lane = 0; lane < N; lane++)
                modelReductionWeight[lane] = 0;
        end else begin
            cycleCount = cycleCount + 1;

            if (dut.targetPush !== (activationValid && activationReady))
                $fatal(1, "target push did not equal the external sample handshake");
            if (dut.targetPush !==
                (dut.matrixActivationValid && dut.matrixActivationReady))
                $fatal(1, "activation and target were not accepted atomically");
            if (dut.targetPop !== (resultValid && resultReady))
                $fatal(1, "target pop did not equal the result handshake");
            if (dut.samplePush !== dut.targetPush ||
                dut.inputSignFifo.values !== dut.targetFifo.values)
                $fatal(1, "target and input-sign FIFOs lost alignment");
            if (matrixUpdateValid !== (resultValid && resultReady))
                $fatal(1, "matrix update valid did not equal result completion");
            for (int lane = 0; lane < N; lane++) begin
                if (dut.residentReductionWeight[lane] !==
                    modelReductionWeight[lane])
                    $fatal(1, "resident reduction weight %0d got %0d, expected %0d",
                           lane, dut.residentReductionWeight[lane],
                           modelReductionWeight[lane]);
            end

            if (loadReductionWeights) begin
                for (int lane = 0; lane < N; lane++)
                    modelReductionWeight[lane] = reductionWeight[lane];
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
                for (int lane = 0; lane < N; lane++)
                    acceptedInput[acceptedCount][lane] = activationData[lane];
                acceptedCount = acceptedCount + 1;
            end

            if (resultValid && resultReady) begin
                if (consumedCount >= acceptedCount)
                    $fatal(1, "result was consumed without an accepted sample");
                if (resultTargetData !== expectedTarget[consumedCount])
                    $fatal(1, "target %0d got %0d, expected %0d",
                           consumedCount, resultTargetData,
                           expectedTarget[consumedCount]);
                expectedPredictionNow = phase5_prediction(
                    acceptedInput[consumedCount][0],
                    acceptedInput[consumedCount][1],
                    modelReductionWeight[0], modelReductionWeight[1],
                    passThrough);
                if ($signed(expectedTarget[consumedCount]) >
                    $signed(expectedPredictionNow))
                    expectedDirectionNow = 2'sd1;
                else if ($signed(expectedTarget[consumedCount]) <
                         $signed(expectedPredictionNow))
                    expectedDirectionNow = -2'sd1;
                else
                    expectedDirectionNow = 2'sd0;

                rawResult[0] = acceptedInput[consumedCount][0] * (2*SCALE)
                               + acceptedInput[consumedCount][1] * (3*SCALE);
                rawResult[1] = acceptedInput[consumedCount][0] * (-1*SCALE)
                               + acceptedInput[consumedCount][1] * (4*SCALE);

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
                               ternary_sign(modelReductionWeight[lane]))
                         : 2'sd0))
                        $fatal(1, "column direction %0d:%0d got %0d",
                               consumedCount, lane, columnDirection[lane]);

                    case (ternary_product(
                              expectedDirectionNow,
                              ternary_sign(
                                  (passThrough || (rawResult[lane] > 0))
                                  ? rawResult[lane] : 0)))
                        2'sd1: if (modelReductionWeight[lane] != {1'b0, {(WIDTH-1){1'b1}}})
                            modelReductionWeight[lane] = modelReductionWeight[lane] + 1;
                        -2'sd1: if (modelReductionWeight[lane] != {1'b1, {(WIDTH-1){1'b0}}})
                            modelReductionWeight[lane] = modelReductionWeight[lane] - 1;
                        default: modelReductionWeight[lane] = modelReductionWeight[lane];
                    endcase
                end

                case (expectedDirectionNow)
                    2'sd1:  sawPositive = 1;
                    2'sd0:  sawZero = 1;
                    -2'sd1: sawNegative = 1;
                    default: $fatal(1, "non-ternary learning direction %0d",
                                    learningDirection);
                endcase
                consumedCount = consumedCount + 1;
            end
        end
    end

    initial begin
        target_t heldTarget;
        result_t heldPrediction;
        logic signed [1:0] heldDirection;

        clk = 0;
        rst_n = 0;
        weightValid = 0;
        activationValid = 0;
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

        // x0/x1/x2 produce stored predictions 0, -40, and -16 (real values
        // 0, -2.5, and -1) with stored targets 32, -48, and -16. Correct
        // directions are +1, -1, 0. Pairing targets one
        // position late instead yields -1, +1, +1.
        // The first two samples are accepted on consecutive cycles.
        @(negedge clk);
        activationData[0] = 7*SCALE;
        activationData[1] = -1*SCALE;
        targetData = 2*SCALE;
        activationValid = 1;
        if (!activationReady)
            $fatal(1, "first sample was unexpectedly backpressured");
        @(posedge clk);
        @(negedge clk);
        activationData[0] = 2*SCALE;
        activationData[1] = -1*SCALE;
        targetData = -3*SCALE;
        if (!activationReady)
            $fatal(1, "second back-to-back sample was unexpectedly backpressured");
        @(posedge clk);

        // Keep x2 asserted while the two-entry target FIFO is full. Even if
        // the matrix input can accept, neither half of this sample may move.
        @(negedge clk);
        activationData[0] = 5*SCALE;
        activationData[1] = -1*SCALE;
        targetData = -1*SCALE;
        wait(dut.targetFull && dut.matrixActivationReady);
        repeat (3) begin
            #1;
            if (activationReady || dut.matrixActivationValid ||
                dut.targetPush || acceptedCount != 2)
                $fatal(1, "target-full backpressure did not hold the complete sample");
            @(negedge clk);
        end

        // Hold the first output for several cycles. Prediction, selected FIFO
        // head, and comparator output must be one stable transaction.
        wait(resultValid);
        @(negedge clk);
        heldTarget = resultTargetData;
        heldPrediction = resultData[0];
        heldDirection = learningDirection;
        if (heldTarget !== 2*SCALE || heldPrediction !== 0 || heldDirection !== 2'sd1)
            $fatal(1, "unexpected first stalled tuple: prediction=%0d target=%0d direction=%0d",
                   heldPrediction, heldTarget, heldDirection);
        repeat (4) begin
            @(negedge clk);
            if (!resultValid || resultTargetData !== heldTarget ||
                resultData[0] !== heldPrediction ||
                learningDirection !== heldDirection || dut.targetPop)
                $fatal(1, "prediction, target, or direction changed while output was stalled");
        end

        // Permit exactly one result handshake. Full-FIFO lookahead accepts x2
        // on that edge. One target pops and one pushes, leaving target1 (-3),
        // not target2, selected next.
        resultReady = 1;
        @(posedge clk);
        @(negedge clk);
        resultReady = 0;
        activationValid = 0;
        if (acceptedCount != 3 || consumedCount != 1 ||
            dut.targetFifo.values != 2)
            $fatal(1, "simultaneous target pop/push accounting was not 3 accepted, 1 consumed, 2 queued");
        wait(resultValid);
        @(negedge clk);
        if (resultTargetData !== -3*SCALE)
            $fatal(1, "exactly-one advance failed: next target got %0d, expected %0d",
                   resultTargetData, -3*SCALE);

        // Drain x1/x2, then add intentional input bubbles. These samples also
        // retain negative predictions and repeat all comparator outcomes.
        resultReady = 1;
        wait_for_consumed(3);
        send_sample_with_bubbles(2*SCALE, 0, 2*SCALE, 2); // 1 -> +1
        send_sample_with_bubbles(2*SCALE, -1*SCALE, -3*SCALE, 3); // -2.5 -> -1
        send_sample_with_bubbles(0, -1*SCALE, -7*SCALE/2, 1);   // -3.5 -> 0
        wait_for_consumed(6);

        // Exercise both saturation endpoints without relying on arithmetic
        // overflow. The first sample requests a decrement of a resident MIN;
        // the second requests an increment of a freshly reloaded MAX.
        reductionWeight[0] = {1'b0, {(WIDTH-1){1'b1}}};
        reductionWeight[1] = {1'b1, {(WIDTH-1){1'b0}}};
        load_reduction_vector();
        send_sample_with_bubbles(2*SCALE, SCALE, -(1 << (TARGET_WIDTH-1)), 1);
        wait_for_consumed(7);
        @(negedge clk);
        if (dut.residentReductionWeight[1] !== {1'b1, {(WIDTH-1){1'b0}}})
            $fatal(1, "negative reduction-weight saturation failed");

        load_reduction_vector();
        send_sample_with_bubbles(0, SCALE, (1 << (TARGET_WIDTH-1))-1, 1);
        wait_for_consumed(8);
        @(negedge clk);
        if (dut.residentReductionWeight[0] !== {1'b0, {(WIDTH-1){1'b1}}})
            $fatal(1, "positive reduction-weight saturation failed");

        // ReLU blocks the negative second pre-activation from the column
        // package and from the reduction update, while row signs still report
        // the original +/− input vector.
        reductionWeight[0] = HALF;
        reductionWeight[1] = HALF;
        load_reduction_vector();
        passThrough = 0;
        send_sample_with_bubbles(7*SCALE, -1*SCALE,
                                 (1 << (TARGET_WIDTH-1))-1, 1);
        wait_for_consumed(SAMPLE_COUNT);
        @(negedge clk) resultReady = 0;

        if (acceptedCount != SAMPLE_COUNT || consumedCount != SAMPLE_COUNT)
            $fatal(1, "got %0d accepted samples and %0d consumed pairs, expected %0d/%0d",
                   acceptedCount, consumedCount, SAMPLE_COUNT, SAMPLE_COUNT);
        if (!sawBackToBack || !sawBubble)
            $fatal(1, "did not observe both consecutive and bubbled accepted samples");
        if (!sawPositive || !sawZero || !sawNegative)
            $fatal(1, "did not observe all three learning-direction outcomes");

        $display("PASS: aligned SSLMS packages, resident reduction updates, saturation, and backpressure.");
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
        input integer bubbleCycles
    );
        activationValid = 0;
        repeat (bubbleCycles) @(negedge clk);
        activationData[0] = lane0;
        activationData[1] = lane1;
        targetData = target;
        activationValid = 1;
        while (!activationReady) @(negedge clk);
        @(posedge clk);
        @(negedge clk) activationValid = 0;
    endtask

    task load_reduction_vector();
        @(negedge clk) loadReductionWeights = 1;
        @(posedge clk);
        @(negedge clk) loadReductionWeights = 0;
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
