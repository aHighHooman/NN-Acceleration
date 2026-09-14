module matrixMultiplierWeightStationary #(
    parameter int WIDTH = 16,
    parameter int N = 3,
    parameter int INPUT_FIFO_DEPTH = 2*N,
    parameter int OUTPUT_FIFO_DEPTH = 2*N
)(
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic signed [WIDTH-1:0]      weightData [N],
    input  logic                         weightValid,
    output logic                         weightReady,
    input  logic signed [WIDTH-1:0]      activationData [N],
    input  logic                         activationValid,
    output logic                         activationReady,
    input  logic signed [1:0]            rowDirection [N],
    input  logic signed [1:0]            columnDirection [N],
    input  logic                         matrixUpdateValid,
    output logic signed [2*WIDTH+$clog2(N)-1:0] resultData [N],
    output logic                         resultValid,
    input  logic                         resultReady,
    output logic                         resultLast,
    output logic                         weightsLoaded,
    input  logic                         reloadWeights,
    output logic                         reloadReady,
    output logic                         datapathAdvance,
    output logic                         resultEnqueue,
    output logic signed [2*WIDTH+$clog2(N)-1:0] resultEnqueueData [N]
);

    localparam int WEIGHT_COUNT_WIDTH   = $clog2(N+1);
    localparam int OUTPUT_ROW_WIDTH     = $clog2(N);
    localparam int RESULT_WIDTH         = $clog2(N) + 2*WIDTH;
    localparam int VECTOR_WIDTH         = N * WIDTH;
    localparam int RESULT_VECTOR_WIDTH  = N * RESULT_WIDTH;

    logic weightPush, consumePendingWeightRow;
    logic activationPush, activationPop;
    logic outputPop;
    logic [WEIGHT_COUNT_WIDTH-1:0] loadedWeightRows;
    logic [OUTPUT_ROW_WIDTH-1:0] outputRowIndex;

    logic signed [WIDTH-1:0] pendingWeightRow [N];
    logic pendingWeightValid;
    logic consumingFinalWeightRow;
    logic signed [VECTOR_WIDTH-1:0] activationVectorPushData;
    logic signed [VECTOR_WIDTH-1:0] activationVectorHead;
    logic signed [WIDTH-1:0] queuedActivation[N];
    logic activationFull, activationEmpty;
    logic signed [RESULT_VECTOR_WIDTH-1:0] outputVectorPushData;
    logic signed [RESULT_VECTOR_WIDTH-1:0] outputVectorHead;
    logic signed [RESULT_WIDTH-1:0] outputVectorLaneData[N];
    logic outputFull, outputEmpty;
    localparam int RESULT_ALIGN_STORAGE = (N > 1) ? N-1 : 1;
    logic signed [RESULT_WIDTH-1:0]
        resultAlignData[N][RESULT_ALIGN_STORAGE];
    logic resultAlignValid[N][RESULT_ALIGN_STORAGE];
    logic signed [RESULT_WIDTH-1:0] resultAlignedData[N];
    logic resultAlignedValid[N], resultAlignedAllValid, resultAlignBusy;
    logic signed [WIDTH-1:0] skewData[N][N];
    logic skewValid[N][N];
    logic signed [WIDTH-1:0] skewedActivation[N];
    logic skewedActivationValid[N];
    logic signed [RESULT_WIDTH-1:0] arrayResult[N];
    logic arrayResultValid[N];
    logic pipelineBusy, skewBusy, arrayAdvance, outputBlocked;

    always_comb begin
        activationVectorPushData = '0;
        outputVectorPushData     = '0;
        for (int lane = 0; lane < N; lane++) begin
            activationVectorPushData[lane*WIDTH +: WIDTH] = activationData[lane];
            outputVectorPushData[lane*RESULT_WIDTH +: RESULT_WIDTH] =
                resultAlignedData[lane];
        end
    end

    always_comb begin
        for (int lane = 0; lane < N; lane++) begin
            queuedActivation[lane] =
                activationVectorHead[lane*WIDTH +: WIDTH];
            outputVectorLaneData[lane] =
                outputVectorHead[lane*RESULT_WIDTH +: RESULT_WIDTH];
        end
    end

    // The systolic columns finish one cycle apart.  Delay the earlier columns
    // by the missing suffix of that fixed latency so the normal result FIFO
    // accepts complete vectors atomically.  These registers are ordinary
    // datapath state and use the same advance enable as the array.
    generate
        for (genvar alignLane = 0; alignLane < N; alignLane++) begin : result_alignment
            if (alignLane < N-1) begin : delayed_column
                localparam int ALIGN_DELAY = N-1-alignLane;
                assign resultAlignedData[alignLane] =
                    resultAlignData[alignLane][ALIGN_DELAY-1];
                assign resultAlignedValid[alignLane] =
                    resultAlignValid[alignLane][ALIGN_DELAY-1];
            end else begin : last_column
                assign resultAlignedData[alignLane] =
                    arrayResult[alignLane];
                assign resultAlignedValid[alignLane] =
                    arrayResultValid[alignLane];
            end
        end
    endgenerate

    always_comb begin
        resultAlignedAllValid = 1'b1;
        for (int lane = 0; lane < N; lane++)
            resultAlignedAllValid &= resultAlignedValid[lane];
    end

    always_comb begin
        resultAlignBusy = 1'b0;
        for (int lane = 0; lane < N; lane++)
            for (int stage = 0; stage < RESULT_ALIGN_STORAGE; stage++)
                if (stage < N-1-lane)
                    resultAlignBusy |= resultAlignValid[lane][stage];
    end

    always_comb begin
        skewBusy = 0;
        for (int i = 0; i < N; i++) begin
            // Lane i consumes only its diagonal stage, so stages 0..i are
            // the complete live skew path for that lane.
            for (int d = 0; d <= i; d++) begin
                skewBusy |= skewValid[i][d];
            end
        end
    end

    // The aligned result vector is the only transaction that can be blocked
    // by the output FIFO.  signedFifo permits a simultaneous pop when full,
    // matching the old lockstep lane FIFO behavior.
    assign outputBlocked    = resultAlignedAllValid && outputFull && !outputPop;

    // A pending row can be replaced on the same edge on which it is consumed,
    // except when that consumption completes the current N-row matrix.
    assign consumePendingWeightRow = !weightsLoaded && pendingWeightValid;
    assign consumingFinalWeightRow = consumePendingWeightRow &&
                                     (loadedWeightRows == N-1);
    assign weightReady      = !weightsLoaded &&
                              (!pendingWeightValid || !consumingFinalWeightRow);
    assign weightPush       = weightValid && weightReady;
    assign activationReady  = weightsLoaded && !activationFull;
    assign activationPush   = activationValid && activationReady;
    assign outputPop        = resultValid && resultReady;
    assign resultValid      = !outputEmpty;
    assign resultLast       = resultValid && (outputRowIndex == N-1);
    assign arrayAdvance     = !weightsLoaded ? consumePendingWeightRow : !outputBlocked;
    assign datapathAdvance = arrayAdvance;
    assign resultEnqueue = arrayAdvance && resultAlignedAllValid;
    // Expose the complete result vector at the same edge on which the normal
    // output FIFOs accept it. The accelerator uses this only to capture the
    // reduction metadata alongside the ordinary result; it is not a second
    // flow-control path.
    genvar enqueueLane;
    generate
        for (enqueueLane = 0; enqueueLane < N; enqueueLane = enqueueLane + 1) begin : enqueue_result_data
            assign resultEnqueueData[enqueueLane] = resultAlignedData[enqueueLane];
        end
    endgenerate
    assign activationPop    = weightsLoaded && !activationEmpty && arrayAdvance;
    // Stream quiescence is represented by the distributed empty/busy state.
    // Reload readiness is stricter: a complete N-row output frame must also
    // have retired, leaving the output frame position at row zero.
    assign reloadReady      = weightsLoaded && activationEmpty && !skewBusy &&
                              !pipelineBusy && !resultAlignBusy && outputEmpty &&
                              (outputRowIndex == 0);

    genvar resultLane;
    generate
        for (resultLane = 0; resultLane < N; resultLane = resultLane + 1) begin : output_lanes
            assign resultData[resultLane] = outputVectorLaneData[resultLane];
        end
    endgenerate

    signedFifo #(.WIDTH(VECTOR_WIDTH), .DEPTH(INPUT_FIFO_DEPTH)) activationVectorFifo (
        .clk(clk), .rst_n(rst_n), .push(activationPush),
        .pushData(activationVectorPushData), .pop(activationPop),
        .popData(activationVectorHead), .full(activationFull),
        .empty(activationEmpty), .values()
    );

    signedFifo #(.WIDTH(RESULT_VECTOR_WIDTH), .DEPTH(OUTPUT_FIFO_DEPTH)) outputVectorFifo (
        .clk(clk), .rst_n(rst_n),
        .push(arrayAdvance && resultAlignedAllValid),
        .pushData(outputVectorPushData), .pop(outputPop),
        .popData(outputVectorHead), .full(outputFull), .empty(outputEmpty),
        .values()
    );

    genvar laneIndex;
    generate
        for (laneIndex = 0; laneIndex < N; laneIndex = laneIndex + 1) begin : skew_outputs
            assign skewedActivation[laneIndex]      = skewData[laneIndex][laneIndex];
            assign skewedActivationValid[laneIndex] = skewValid[laneIndex][laneIndex];
        end
    endgenerate

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            weightsLoaded        <= 0;
            loadedWeightRows     <= 0;
            outputRowIndex       <= 0;
            pendingWeightValid   <= 0;

            for (int lane = 0; lane < N; lane++) begin
                pendingWeightRow[lane] <= 0;
                for (int stage = 0; stage < RESULT_ALIGN_STORAGE; stage++) begin
                    resultAlignData[lane][stage]  <= 0;
                    resultAlignValid[lane][stage] <= 0;
                end
                for (int d = 0; d < N; d++) begin
                    skewData[lane][d]  <= 0;
                    skewValid[lane][d] <= 0;
                end
            end
        end else begin
            if (weightPush) begin
                for (int lane = 0; lane < N; lane++)
                    pendingWeightRow[lane] <= weightData[lane];
                pendingWeightValid <= 1;
            end else if (consumePendingWeightRow) begin
                for (int lane = 0; lane < N; lane++)
                    pendingWeightRow[lane] <= 0;
                pendingWeightValid <= 0;
            end

            if (consumePendingWeightRow) begin
                if (loadedWeightRows == N-1) begin
                    loadedWeightRows <= 0;
                    weightsLoaded    <= 1;
                end else begin
                    loadedWeightRows <= loadedWeightRows + 1;
                end
            end

            if (outputPop) begin
                outputRowIndex <= (outputRowIndex == N-1) ? 0 : outputRowIndex + 1;
            end

            if (reloadWeights && reloadReady) begin
                weightsLoaded    <= 0;
                loadedWeightRows <= 0;
                pendingWeightValid <= 0;
                for (int lane = 0; lane < N; lane++)
                    pendingWeightRow[lane] <= 0;
            end

            if (weightsLoaded && arrayAdvance) begin
                for (int lane = 0; lane < N-1; lane++) begin
                    resultAlignData[lane][0] <= arrayResult[lane];
                    resultAlignValid[lane][0] <= arrayResultValid[lane];
                    for (int stage = 1; stage < N-1-lane; stage++) begin
                        resultAlignData[lane][stage] <=
                            resultAlignData[lane][stage-1];
                        resultAlignValid[lane][stage] <=
                            resultAlignValid[lane][stage-1];
                    end
                end
                for (int lane = 0; lane < N; lane++) begin
                    skewData[lane][0]  <= queuedActivation[lane];
                    skewValid[lane][0] <= activationPop;

                    for (int d = 1; d <= lane; d++) begin
                        skewData[lane][d]  <= skewData[lane][d-1];
                        skewValid[lane][d] <= skewValid[lane][d-1];
                    end
                end
            end
        end
    end

    systolicArrayWeightStationary #(.WIDTH(WIDTH), .N(N)) systolicArr (
        .clk(clk), .rst_n(rst_n), .advance(arrayAdvance), .loadWeight(consumePendingWeightRow),
        .rowDirection(rowDirection), .columnDirection(columnDirection),
        .updateValid(matrixUpdateValid),
        .row(skewedActivation), .rowValid(skewedActivationValid),
        .col(pendingWeightRow), .result(arrayResult),
        .resultValid(arrayResultValid),
        .updateComplete(),
        .pipelineBusy(pipelineBusy)
    );

endmodule
