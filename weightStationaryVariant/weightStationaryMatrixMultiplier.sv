module weightStationaryMatrixMultiplier #(
    parameter int WIDTH = 16,
    parameter int N = 3
)(
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic signed [WIDTH-1:0]      weightData [N],
    input  logic                         weightValid,
    output logic                         weightReady,
    input  logic signed [WIDTH-1:0]      inputData [N],
    input  logic                         inputValid,
    output logic                         inputReady,
    input  logic signed [1:0]            rowDirection [N],
    input  logic signed [1:0]            columnDirection [N],
    input  logic                         matrixUpdateValid,
    output logic signed [2*WIDTH+$clog2(N)-1:0] resultData [N],
    output logic                         resultValid,
    input  logic                         resultReady,
    output logic                         weightsLoaded,
    input  logic                         reloadWeights,
    output logic                         reloadReady,
    output logic                         arrayAdvance
);

    localparam int WEIGHT_COUNT_WIDTH   = $clog2(N+1);
    localparam int RESULT_WIDTH         = $clog2(N) + 2*WIDTH;
    localparam int VECTOR_WIDTH         = N * WIDTH;
    localparam int ACTIVATION_SKID_DEPTH = 1;

    initial begin
        if (WIDTH < 1 || N < 2)
            $fatal(1, "WIDTH>=1 and N>=2");
    end

    logic weightPush, consumePendingWeightRow;
    logic inputPush, inputPop;
    logic [WEIGHT_COUNT_WIDTH-1:0] loadedWeightRows;

    logic signed [WIDTH-1:0] pendingWeightRow [N];
    logic pendingWeightValid;
    logic consumingFinalWeightRow;
    logic signed [VECTOR_WIDTH-1:0] inputVectorPushData;
    logic signed [VECTOR_WIDTH-1:0] inputVectorHead;
    logic signed [WIDTH-1:0] queuedInput[N];
    logic inputFull, inputEmpty;
    localparam int RESULT_ALIGN_STORAGE = (N > 1) ? N-1 : 1;
    logic signed [RESULT_WIDTH-1:0]
        resultAlignData[N][RESULT_ALIGN_STORAGE];
    logic resultAlignValid[N][RESULT_ALIGN_STORAGE];
    logic signed [RESULT_WIDTH-1:0] resultAlignedData[N];
    logic resultAlignedValid[N], resultAlignedAllValid, resultAlignBusy;
    logic signed [WIDTH-1:0] skewData[N][N];
    logic skewValid[N][N];
    logic signed [WIDTH-1:0] skewedInput[N];
    logic skewedInputValid[N];
    logic signed [RESULT_WIDTH-1:0] arrayResult[N];
    logic arrayResultValid[N];
    logic pipelineBusy, skewBusy, outputBlocked;

    always_comb begin
        inputVectorPushData = '0;
        for (int lane = 0; lane < N; lane++) begin
            inputVectorPushData[lane*WIDTH +: WIDTH] = inputData[lane];
        end
    end

    always_comb begin
        for (int lane = 0; lane < N; lane++) begin
            queuedInput[lane] =
                inputVectorHead[lane*WIDTH +: WIDTH];
        end
    end

    // Delay earlier columns to align their staggered results into complete vectors.
    // Alignment registers share the array's advance enable.
    genvar alignLane;
    generate
        for (alignLane = 0; alignLane < N; alignLane = alignLane + 1) begin : result_alignment
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

    // nnAccelerator supplies resultReady from its result FIFO capacity.
    // A complete result waiting on !resultReady freezes the shared datapath.
    assign outputBlocked    = resultAlignedAllValid && !resultReady;

    // A pending row can be replaced on the same edge on which it is consumed,
    // except when that consumption completes the current N-row matrix.
    assign consumePendingWeightRow = !weightsLoaded && pendingWeightValid;
    assign consumingFinalWeightRow = consumePendingWeightRow &&
                                     (loadedWeightRows == N-1);
    assign weightReady      = !weightsLoaded &&
                              (!pendingWeightValid || !consumingFinalWeightRow);
    assign weightPush       = weightValid && weightReady;
    // The one-entry activation skid allows a simultaneous pop and replacement
    // push on advancing edges, sustaining one vector per cycle.
    assign inputReady  = weightsLoaded && (!inputFull || inputPop);
    assign inputPush   = inputValid && inputReady;
    assign resultValid      = resultAlignedAllValid;
    assign arrayAdvance     = !weightsLoaded ? consumePendingWeightRow : !outputBlocked;
    assign inputPop    = weightsLoaded && !inputEmpty && arrayAdvance;
    // The matrix engine reports only its own computation state.  Result
    // storage belongs to nnAccelerator.
    assign reloadReady      = weightsLoaded && inputEmpty && !skewBusy &&
                              !pipelineBusy && !resultAlignBusy && !resultValid;

    genvar resultLane;
    generate
        for (resultLane = 0; resultLane < N; resultLane = resultLane + 1) begin : output_lanes
            assign resultData[resultLane] = resultAlignedData[resultLane];
        end
    endgenerate

    signedFifo #(.WIDTH(VECTOR_WIDTH), .DEPTH(ACTIVATION_SKID_DEPTH)) inputVectorFifo (
        .clk(clk), .rst_n(rst_n), .push(inputPush),
        .pushData(inputVectorPushData), .pop(inputPop),
        .popData(inputVectorHead), .full(inputFull),
        .empty(inputEmpty)
    );

    genvar laneIndex;
    generate
        for (laneIndex = 0; laneIndex < N; laneIndex = laneIndex + 1) begin : skew_outputs
            assign skewedInput[laneIndex]      = skewData[laneIndex][laneIndex];
            assign skewedInputValid[laneIndex] = skewValid[laneIndex][laneIndex];
        end
    endgenerate

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            weightsLoaded        <= 0;
            loadedWeightRows     <= 0;
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
                    skewData[lane][0]  <= queuedInput[lane];
                    skewValid[lane][0] <= inputPop;

                    for (int d = 1; d <= lane; d++) begin
                        skewData[lane][d]  <= skewData[lane][d-1];
                        skewValid[lane][d] <= skewValid[lane][d-1];
                    end
                end
            end
        end
    end

    weightStationarySystolicArray #(.WIDTH(WIDTH), .N(N)) systolicArray (
        .clk(clk), .rst_n(rst_n), .advance(arrayAdvance), .loadWeight(consumePendingWeightRow),
        .rowDirection(rowDirection), .columnDirection(columnDirection),
        .updateValid(matrixUpdateValid),
        .row(skewedInput), .rowValid(skewedInputValid),
        .col(pendingWeightRow), .result(arrayResult),
        .resultValid(arrayResultValid),
        .pipelineBusy(pipelineBusy)
    );

endmodule
