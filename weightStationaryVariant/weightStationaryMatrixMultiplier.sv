module weightStationaryMatrixMultiplier #(
    parameter int WIDTH = 16,
    parameter int N = 3
)(
    input logic clk, rst_n,
    input logic signed [WIDTH-1:0] weightData[N],
    input logic weightValid,
    output logic weightReady,
    input logic signed [WIDTH-1:0] inputData[N],
    input logic inputValid,
    output logic inputReady,
    input logic signed [1:0] rowDirection[N], columnDirection[N],
    input logic matrixUpdateValid,
    output logic signed [2*WIDTH+$clog2(N)-1:0] resultData[N],
    output logic resultValid,
    input logic resultReady,
    output logic weightsLoaded,
    input logic reloadWeights,
    output logic reloadReady, arrayAdvance
);
    localparam int WEIGHT_COUNT_WIDTH = $clog2(N+1);
    localparam int RESULT_WIDTH = 2*WIDTH + $clog2(N);
    localparam int VECTOR_WIDTH = N*WIDTH;
    initial begin
        if (WIDTH < 1 || N < 2)
            $fatal(1, "WIDTH>=1 and N>=2");
    end

    // Preserve bottom-first host rows; gather once before two-ended shifting.
    logic signed [WIDTH-1:0] weightBuffer[N][N];
    logic [WEIGHT_COUNT_WIDTH-1:0] loadedWeightRows, loadStep;
    logic loadingWeights, captureWeights, weightPush;
    logic signed [WIDTH-1:0] topWeight[N], bottomWeight[N];
    logic signed [VECTOR_WIDTH-1:0] inputVectorData, inputVectorHead;
    logic inputVectorValid, inputPush, inputPop;
    logic signed [WIDTH-1:0] queuedInput[N];
    // Two taps share a sequence. Stage zero retains the registered skew entry.
    logic signed [WIDTH-1:0] skewData[N][N];
    logic skewValid[N][N];
    logic signed [WIDTH-1:0] skewedLeft[N], skewedRight[N];
    logic signed [RESULT_WIDTH-1:0] arrayResult[N];
    logic arrayResultValid[N];
    logic skewBusy, pipelineBusy, outputBlocked;

    assign weightReady = !weightsLoaded && !loadingWeights && !captureWeights;
    assign weightPush = weightValid && weightReady;
    assign inputVectorHead = inputVectorValid ? inputVectorData : '0;
    // An accepted reload starts a new weight epoch before any new sample.
    assign inputReady = weightsLoaded && !(reloadWeights && reloadReady) &&
                        (!inputVectorValid || inputPop);
    assign inputPush = inputValid && inputReady;
    assign outputBlocked = resultValid && !resultReady;
    assign arrayAdvance = weightsLoaded ? !outputBlocked :
                                         (loadingWeights || captureWeights);
    assign inputPop = weightsLoaded && inputVectorValid && arrayAdvance;
    assign reloadReady = weightsLoaded && !inputVectorValid && !skewBusy &&
                         !pipelineBusy && !resultValid;
    always_comb begin
        resultValid = 1'b1;
        for (int lane = 0; lane < N; lane++)
            resultValid &= arrayResultValid[lane];
    end
    always_comb begin
        for (int lane = 0; lane < N; lane++) begin
            queuedInput[lane] = inputVectorHead[lane*WIDTH +: WIDTH];
            topWeight[lane] = '0;
            bottomWeight[lane] = '0;
            if (loadingWeights) begin
                if (loadStep >= lane)
                    topWeight[lane] = weightBuffer[N-1-loadStep][lane];
                if (loadStep >= N-lane)
                    bottomWeight[lane] = weightBuffer[loadStep][lane];
            end
        end
    end
    always_comb begin
        skewBusy = 1'b0;
        for (int lane = 0; lane < N; lane++)
            for (int stage = 0; stage < N; stage++)
                if ((stage <= lane) || (stage <= N-1-lane && lane != 0))
                    skewBusy |= skewValid[lane][stage];
    end
    genvar lane;
    generate
        for (lane = 0; lane < N; lane++) begin : skew_outputs
            assign skewedLeft[lane] = skewData[lane][lane];
            // Row zero has no BR cells, so it needs no right-side injection.
            assign skewedRight[lane] = (lane == 0) ? '0 :
                                                     skewData[lane][N-1-lane];
            assign resultData[lane] = arrayResult[lane];
        end
    endgenerate
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            inputVectorValid <= 1'b0;
            inputVectorData <= '0;
        end else begin
            if (inputPush) begin
                for (int lane = 0; lane < N; lane++)
                    inputVectorData[lane*WIDTH +: WIDTH] <= inputData[lane];
                inputVectorValid <= 1'b1;
            end else if (inputPop) begin
                inputVectorValid <= 1'b0;
            end
        end
    end
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            weightsLoaded <= 1'b0;
            loadedWeightRows <= '0;
            loadStep <= '0;
            loadingWeights <= 1'b0;
            captureWeights <= 1'b0;
            for (int row = 0; row < N; row++) begin
                for (int col = 0; col < N; col++)
                    weightBuffer[row][col] <= '0;
                for (int stage = 0; stage < N; stage++) begin
                    skewData[row][stage] <= '0;
                    skewValid[row][stage] <= 1'b0;
                end
            end
        end else begin
            if (weightPush) begin
                for (int col = 0; col < N; col++)
                    weightBuffer[N-1-loadedWeightRows][col] <= weightData[col];
                loadedWeightRows <= loadedWeightRows + 1'b1;
                if (loadedWeightRows == N-1) begin
                    loadingWeights <= 1'b1;
                    loadStep <= '0;
                end
            end
            if (loadingWeights) begin
                if (loadStep == N-1) begin
                    loadingWeights <= 1'b0;
                    captureWeights <= 1'b1;
                    loadStep <= '0;
                end else begin
                    loadStep <= loadStep + 1'b1;
                end
            end
            if (captureWeights) begin
                captureWeights <= 1'b0;
                weightsLoaded <= 1'b1;
                loadedWeightRows <= '0;
            end
            if (reloadWeights && reloadReady) begin
                weightsLoaded <= 1'b0;
                loadedWeightRows <= '0;
                loadStep <= '0;
                loadingWeights <= 1'b0;
                captureWeights <= 1'b0;
                for (int row = 0; row < N; row++) begin
                    for (int col = 0; col < N; col++)
                        weightBuffer[row][col] <= '0;
                    for (int stage = 0; stage < N; stage++) begin
                        skewData[row][stage] <= '0;
                        skewValid[row][stage] <= 1'b0;
                    end
                end
            end else if (weightsLoaded && arrayAdvance) begin
                for (int row = 0; row < N; row++) begin
                    skewData[row][0] <= queuedInput[row];
                    skewValid[row][0] <= inputPop;
                    for (int stage = 1; stage < N; stage++) begin
                        if ((stage <= row) || (row != 0 && stage <= N-1-row)) begin
                            skewData[row][stage] <= skewData[row][stage-1];
                            skewValid[row][stage] <= skewValid[row][stage-1];
                        end
                    end
                end
            end
        end
    end
    weightStationarySystolicArray #(.WIDTH(WIDTH), .N(N)) systolicArray (
        .clk(clk), .rst_n(rst_n), .advance(arrayAdvance),
        .loadWeight(loadingWeights), .captureWeight(captureWeights),
        .rowDirection(rowDirection), .columnDirection(columnDirection),
        .updateValid(matrixUpdateValid),
        .rowLeft(skewedLeft), .rowRight(skewedRight), .actValid(skewValid[0][0]),
        .colTop(topWeight), .colBottom(bottomWeight),
        .result(arrayResult), .resultValid(arrayResultValid), .pipelineBusy(pipelineBusy)
    );
endmodule
