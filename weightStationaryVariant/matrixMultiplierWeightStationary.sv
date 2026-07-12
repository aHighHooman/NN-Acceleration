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
    output logic signed [2*WIDTH-1:0]    resultData [N],
    output logic                         resultValid,
    input  logic                         resultReady,
    output logic                         resultLast,
    output logic                         weightsLoaded,
    input  logic                         reloadWeights,
    output logic                         reloadReady
);

    localparam int WEIGHT_COUNT_WIDTH = (N <= 1) ? 1 : $clog2(N+1);
    localparam int ACT_ROW_WIDTH = (N <= 1) ? 1 : $clog2(N);

    logic weightPush, weightPop, allWeightValid, allWeightReady;
    logic activationPush, activationPop, allActivationValid, allActivationReady;
    logic outputPop, allOutputValid, allOutputReady, allOutputEmpty, allActivationEmpty;
    logic loadingWeights;
    logic [WEIGHT_COUNT_WIDTH-1:0] loadedWeightRows;
    logic [ACT_ROW_WIDTH-1:0] acceptedActivationRow;

    logic signed [WIDTH-1:0] weightHead [N];
    logic weightFull[N], weightEmpty[N];
    logic signed [WIDTH:0] activationPushWord[N], activationHead[N];
    logic activationFull[N], activationEmpty[N];
    logic signed [2*WIDTH:0] outputPushWord[N], outputHead[N];
    logic outputFull[N], outputEmpty[N];

    logic signed [WIDTH-1:0] skewData[N][N];
    logic skewValid[N][N], skewLast[N][N];
    logic signed [WIDTH-1:0] arrayRows[N];
    logic arrayRowValid[N], arrayRowLast[N];
    logic signed [2*WIDTH-1:0] arrayResults[N];
    logic arrayResultValid[N], arrayResultLast[N];
    logic pipelineBusy, skewBusy, arrayAdvance, outputBlocked;

    always_comb begin
        allWeightValid = 1'b1;
        allWeightReady = 1'b1;
        allActivationValid = 1'b1;
        allActivationReady = 1'b1;
        allOutputValid = 1'b1;
        allOutputReady = 1'b1;
        allOutputEmpty = 1'b1;
        allActivationEmpty = 1'b1;
        skewBusy = 1'b0;
        outputBlocked = 1'b0;

        for (int i = 0; i < N; i++) begin
            allWeightValid &= !weightEmpty[i];
            allWeightReady &= !weightFull[i];
            allActivationValid &= !activationEmpty[i];
            allActivationReady &= !activationFull[i];
            allActivationEmpty &= activationEmpty[i];
            allOutputValid &= !outputEmpty[i];
            allOutputEmpty &= outputEmpty[i];
            allOutputReady &= !outputFull[i] || outputPop;
            outputBlocked |= arrayResultValid[i] && outputFull[i] && !outputPop;
            for (int d = 0; d < N; d++) skewBusy |= skewValid[i][d];
        end
    end

    assign weightReady      = !weightsLoaded && allWeightReady;
    assign weightPush       = weightValid && weightReady;
    assign activationReady  = weightsLoaded && allActivationReady;
    assign activationPush   = activationValid && activationReady;
    assign outputPop        = resultValid && resultReady;
    assign resultValid      = allOutputValid;
    assign resultLast       = outputHead[0][2*WIDTH];
    assign arrayAdvance     = loadingWeights ? weightPop : !outputBlocked;
    assign activationPop    = weightsLoaded && allActivationValid && arrayAdvance;
    assign weightPop        = loadingWeights && allWeightValid;
    assign reloadReady      = weightsLoaded && allActivationEmpty && !skewBusy &&
                              !pipelineBusy && allOutputEmpty &&
                              (acceptedActivationRow == 0);

    generate
        for (genvar i = 0; i < N; i++) begin : fifo_banks
            assign activationPushWord[i] = {acceptedActivationRow == N-1, activationData[i]};
            assign outputPushWord[i] = {arrayResultLast[i], arrayResults[i]};
            assign resultData[i] = outputHead[i][2*WIDTH-1:0];

            signedFifo #(.WIDTH(WIDTH), .DEPTH(N)) weightFifo (
                .clk(clk), .rst_n(rst_n), .push(weightPush), .pushData(weightData[i]),
                .pop(weightPop), .popData(weightHead[i]), .full(weightFull[i]),
                .empty(weightEmpty[i]), .count()
            );
            signedFifo #(.WIDTH(WIDTH+1), .DEPTH(INPUT_FIFO_DEPTH)) activationFifo (
                .clk(clk), .rst_n(rst_n), .push(activationPush), .pushData(activationPushWord[i]),
                .pop(activationPop), .popData(activationHead[i]), .full(activationFull[i]),
                .empty(activationEmpty[i]), .count()
            );
            signedFifo #(.WIDTH(2*WIDTH+1), .DEPTH(OUTPUT_FIFO_DEPTH)) outputFifo (
                .clk(clk), .rst_n(rst_n),
                .push(arrayAdvance && arrayResultValid[i]), .pushData(outputPushWord[i]),
                .pop(outputPop), .popData(outputHead[i]), .full(outputFull[i]),
                .empty(outputEmpty[i]), .count()
            );
        end
    endgenerate

    for (genvar lane = 0; lane < N; lane++) begin : skew_outputs
        assign arrayRows[lane]     = skewData[lane][lane];
        assign arrayRowValid[lane] = skewValid[lane][lane];
        assign arrayRowLast[lane]  = skewLast[lane][lane];
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            weightsLoaded        <= 1'b0;
            loadingWeights       <= 1'b1;
            loadedWeightRows     <= '0;
            acceptedActivationRow<= '0;
            for (int lane = 0; lane < N; lane++) begin
                for (int d = 0; d < N; d++) begin
                    skewData[lane][d]  <= '0;
                    skewValid[lane][d] <= 1'b0;
                    skewLast[lane][d]  <= 1'b0;
                end
            end
        end else begin
            if (weightPop) begin
                if (loadedWeightRows == N-1) begin
                    loadedWeightRows <= '0;
                    loadingWeights   <= 1'b0;
                    weightsLoaded    <= 1'b1;
                end else begin
                    loadedWeightRows <= loadedWeightRows + 1'b1;
                end
            end

            if (activationPush) begin
                acceptedActivationRow <= (acceptedActivationRow == N-1) ? '0 :
                                         acceptedActivationRow + 1'b1;
            end

            if (reloadWeights && reloadReady) begin
                weightsLoaded    <= 1'b0;
                loadingWeights   <= 1'b1;
                loadedWeightRows <= '0;
            end

            if (weightsLoaded && arrayAdvance) begin
                for (int lane = 0; lane < N; lane++) begin
                    skewData[lane][0]  <= activationHead[lane][WIDTH-1:0];
                    skewValid[lane][0] <= activationPop;
                    skewLast[lane][0]  <= activationPop && activationHead[lane][WIDTH];
                    for (int d = 1; d < N; d++) begin
                        skewData[lane][d]  <= skewData[lane][d-1];
                        skewValid[lane][d] <= skewValid[lane][d-1];
                        skewLast[lane][d]  <= skewLast[lane][d-1];
                    end
                end
            end
        end
    end

    systolicArrayWeightStationary #(.WIDTH(WIDTH), .N(N)) systolicArr (
        .clk(clk), .rst_n(rst_n), .advance(arrayAdvance), .loadWeight(weightPop),
        .row(arrayRows), .rowValid(arrayRowValid), .rowLast(arrayRowLast),
        .col(weightHead), .result(arrayResults), .resultValid(arrayResultValid),
        .resultLast(arrayResultLast), .pipelineBusy(pipelineBusy)
    );

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (rst_n && resultValid)
            for (int i = 1; i < N; i++)
                assert (outputHead[i][2*WIDTH] == resultLast)
                    else $error("Misaligned result-row last metadata");
    end
`endif

endmodule
