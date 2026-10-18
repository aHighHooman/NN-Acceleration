// Fountain source, dedicated layer chain, row loader, and final collector.
// Internal layers stay staggered. Only the external vector boundary is aligned.
module weightStationaryMatrixMultiplier #(
    parameter int WIDTH=16, N=3, LAYERS=1, TAG_WIDTH=16, G=0, J=1,
    parameter int RESULT_WIDTH=WIDTH+LAYERS*(WIDTH+$clog2(N))
)(
    input logic clk, rst_n,
    input logic signed [WIDTH-1:0] weightData[N],
    input logic weightValid,
    output logic weightReady,
    input logic signed [WIDTH-1:0] inputData[N],
    input logic inputValid,
    input logic [TAG_WIDTH-1:0] inputTag,
    output logic inputReady,
    input logic passThrough,
    input logic signed [1:0] rowDirection[LAYERS][N], columnDirection[LAYERS][N],
    input logic matrixUpdateValid,
    output logic signed [RESULT_WIDTH-1:0] resultData[N],
    output logic resultValid,
    output logic [TAG_WIDTH-1:0] resultTag,
    input logic resultReady,
    output logic weightsLoaded,
    input logic reloadWeights,
    output logic reloadReady, arrayAdvance,
    output logic signed [WIDTH-1:0] weights[LAYERS][N][N],
    output logic tapValid[LAYERS][N], tapGate[LAYERS][N],
    output logic [TAG_WIDTH-1:0] tapTag[LAYERS][N],
    output logic signed [1:0] tapSign[LAYERS][N]
);
    localparam int H=(N-1)/2;
    localparam int STEP=WIDTH+$clog2(N);
    localparam int INDEX_WIDTH=(N>1)?$clog2(N):1;
    logic signed [WIDTH-1:0] weightBuffer[LAYERS][N][N];
    integer hostLayer, loadedWeightRows, loadLayer, loadStep;
    logic loadingWeights, captureWeights, weightPush;
    logic signed [WIDTH-1:0] inputVectorData[N];
    logic inputVectorValid;
    logic [TAG_WIDTH-1:0] inputVectorTag;
    logic inputPush, inputPop, outputBlocked;
    logic cfgWe[LAYERS];
    logic signed [WIDTH-1:0] cfgData[N];
    wire signed [RESULT_WIDTH-1:0] chainData[LAYERS+1][N];
    wire chainValid[LAYERS+1][N];
    wire [TAG_WIDTH-1:0] chainTag[LAYERS+1][N];
    wire sourceBusy[N], collectorBusy[N], arrayBusy[LAYERS], arrayLoaded[LAYERS];
    wire collectedValid[N];
    wire [TAG_WIDTH-1:0] collectedTag[N];
    logic skewBusy, pipelineBusy, collectorActive, allLoaded;
    initial begin
        if (WIDTH<1 || N<3 || N%2==0 || LAYERS<1 || TAG_WIDTH<1 || G<0 || J<1 ||
            RESULT_WIDTH!=WIDTH+LAYERS*STEP)
            $fatal(1, "Fountain requires WIDTH>=1, odd N>=3, LAYERS>=1 and exact widths");
    end
    assign weightReady=rst_n && !weightsLoaded && !loadingWeights && !captureWeights;
    assign weightPush=weightValid && weightReady;
    assign inputReady=weightsLoaded && !(reloadWeights && reloadReady) &&
                      !matrixUpdateValid && (!inputVectorValid || inputPop);
    assign inputPush=inputValid && inputReady;
    assign outputBlocked=resultValid && !resultReady;
    assign arrayAdvance=weightsLoaded && !outputBlocked;
    assign inputPop=inputVectorValid && arrayAdvance;
    assign reloadReady=weightsLoaded && !inputVectorValid && !skewBusy &&
                       !pipelineBusy && !collectorActive && !resultValid && !matrixUpdateValid;
    assign resultTag=collectedTag[0];
    always_comb begin
        resultValid=1;
        skewBusy=0;
        pipelineBusy=0;
        collectorActive=0;
        allLoaded=1;
        for (int l=0; l<LAYERS; l++) begin
            pipelineBusy |= arrayBusy[l];
            allLoaded &= arrayLoaded[l];
            cfgWe[l]=loadingWeights && loadLayer==l;
        end
        for (int j=0; j<N; j++) begin
            resultValid &= collectedValid[j] && collectedTag[j]==collectedTag[0];
            skewBusy |= sourceBusy[j];
            collectorActive |= collectorBusy[j];
            cfgData[j]=weightBuffer[loadLayer][loadStep][j];
        end
    end
    genvar i,l,j;
    generate
    for (i=0; i<N; i++) begin : source
        localparam int A=(i<H)?H-i:i-H;
        wire signed [WIDTH-1:0] laneData;
        weightStationaryLaneDelay #(.WIDTH(WIDTH), .TAG_WIDTH(TAG_WIDTH), .DEPTH(A)) skew (
            .clk, .rst_n, .advance(arrayAdvance),
            .in_data(inputVectorData[i]), .in_valid(inputVectorValid), .in_tag(inputVectorTag),
            .out_data(laneData), .out_valid(chainValid[0][i]), .out_tag(chainTag[0][i]),
            .busy(sourceBusy[i])
        );
        assign chainData[0][i]={{(RESULT_WIDTH-WIDTH){laneData[WIDTH-1]}},laneData};
    end
    for (l=0; l<LAYERS; l++) begin : layers
        localparam int XW=WIDTH+l*STEP;
        localparam int AW=XW+STEP;
        wire signed [XW-1:0] laneInput[N];
        wire signed [AW-1:0] raw[N];
        wire rawValid[N];
        wire [TAG_WIDTH-1:0] rawTag[N];
        for (j=0; j<N; j++) begin : lanes
            wire signed [AW-1:0] activated;
            assign laneInput[j]=chainData[l][j][XW-1:0];
            assign activated=(passThrough || !raw[j][AW-1])?raw[j]:AW'('0);
            assign chainData[l+1][j]={{(RESULT_WIDTH-AW){activated[AW-1]}},activated};
            assign chainValid[l+1][j]=rawValid[j];
            assign chainTag[l+1][j]=rawTag[j];
            assign tapValid[l][j]=rawValid[j];
            assign tapTag[l][j]=rawTag[j];
            assign tapGate[l][j]=passThrough || raw[j]>0;
            assign tapSign[l][j]=(activated==0)?2'sd0:(activated[AW-1]?-2'sd1:2'sd1);
        end
        weightStationarySystolicArray #(.N(N), .WIDTH(WIDTH), .INPUT_WIDTH(XW),
            .RESULT_WIDTH(AW), .TAG_WIDTH(TAG_WIDTH), .G(G), .J(J)) systolicArray (
            .clk, .rst_n, .advance(arrayAdvance), .cfg_we(cfgWe[l]),
            .cfg_row(INDEX_WIDTH'(loadStep)), .cfg_data(cfgData),
            .update_valid(matrixUpdateValid), .row_direction(rowDirection[l]),
            .column_direction(columnDirection[l]),
            .in_data(laneInput), .in_valid(chainValid[l]), .in_tag(chainTag[l]),
            .result(raw), .resultValid(rawValid), .resultTag(rawTag),
            .weights(weights[l]), .loaded(arrayLoaded[l]), .busy(arrayBusy[l])
        );
    end
    for (j=0; j<N; j++) begin : collector
        localparam int Q=(j<H)?H-j:j-H;
        // Clamp only to let unsupported even dimensions elaborate far enough
        // to report the parameter contract instead of an illegal array size.
        weightStationaryLaneDelay #(.WIDTH(RESULT_WIDTH), .TAG_WIDTH(TAG_WIDTH), .DEPTH((H>=Q)?H-Q:0)) align (
            .clk, .rst_n, .advance(arrayAdvance),
            .in_data(chainData[LAYERS][j]), .in_valid(chainValid[LAYERS][j]),
            .in_tag(chainTag[LAYERS][j]), .out_data(resultData[j]),
            .out_valid(collectedValid[j]), .out_tag(collectedTag[j]), .busy(collectorBusy[j])
        );
    end
    endgenerate
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            inputVectorValid<=0;
            inputVectorTag<='0;
            for (int i=0; i<N; i++) inputVectorData[i]<='0;
        end else begin
            if (inputPush) begin
                inputVectorValid<=1;
                inputVectorTag<=inputTag;
                for (int i=0; i<N; i++) inputVectorData[i]<=inputData[i];
            end else if (inputPop) inputVectorValid<=0;
        end
    end
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            weightsLoaded<=0;
            hostLayer<=0;
            loadedWeightRows<=0;
            loadLayer<=0;
            loadStep<=0;
            loadingWeights<=0;
            captureWeights<=0;
        end else begin
            if (weightPush) begin
                for (int j=0; j<N; j++)
                    weightBuffer[hostLayer][N-1-loadedWeightRows][j]<=weightData[j];
                if (loadedWeightRows==N-1) begin
                    loadedWeightRows<=0;
                    if (hostLayer==LAYERS-1) begin
                        loadingWeights<=1;
                        loadLayer<=0;
                        loadStep<=0;
                    end else hostLayer<=hostLayer+1;
                end else loadedWeightRows<=loadedWeightRows+1;
            end
            if (loadingWeights) begin
                if (loadStep==N-1) begin
                    loadStep<=0;
                    if (loadLayer==LAYERS-1) begin
                        loadingWeights<=0;
                        captureWeights<=1;
                    end else loadLayer<=loadLayer+1;
                end else loadStep<=loadStep+1;
            end
            if (captureWeights) begin
                captureWeights<=0;
                weightsLoaded<=1;
                hostLayer<=0;
            end
            if (reloadWeights && reloadReady) begin
                weightsLoaded<=0;
                hostLayer<=0;
                loadedWeightRows<=0;
            end
            // synthesis translate_off
            if (captureWeights) assert (allLoaded)
                else $fatal(1, "Fountain loader committed incomplete weights");
            if (rst_n && arrayAdvance) for (int j=1; j<N; j++) begin
                assert (collectedValid[j]==collectedValid[0])
                    else $fatal(1, "Fountain collector valid mismatch");
                if (collectedValid[0] && collectedValid[j])
                    assert (collectedTag[j]==collectedTag[0])
                        else $fatal(1, "Fountain collector tag mismatch");
            end
            // synthesis translate_on
        end
    end
endmodule
