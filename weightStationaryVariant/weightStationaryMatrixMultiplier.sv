// Fountain source, dedicated layer chain and row loader. Every layer's lanes,
// including the outputs, stay on the staggered center-first front. While
// backward is held, the readout weights run through arrays LAYERS-1..1 in reverse
// and each layer's gradient vector settles on its lanes.
module weightStationaryMatrixMultiplier #(
    parameter int WIDTH=16, N=3, LAYERS=1, REDUCTION_WEIGHT_WIDTH=8,
    parameter int RESULT_WIDTH=WIDTH+LAYERS*(WIDTH+$clog2(N))
)(
    input logic clk, rst_n, backward,
    input logic signed [WIDTH-1:0] weightData[N],
    input logic weightValid,
    output logic weightReady,
    input logic signed [WIDTH-1:0] inputData[N],
    input logic inputValid,
    output logic inputReady,
    input logic passThrough,
    input logic signed [REDUCTION_WEIGHT_WIDTH-1:0] reductionWeight[N],
    input logic activationGates[LAYERS][N],
    input logic signed [1:0] rowDirection[LAYERS][N], columnDirection[LAYERS][N],
    input logic matrixUpdateValid,
    output logic signed [RESULT_WIDTH-1:0] resultData[N],
    output logic resultValid[N],
    input logic outputBlocked,
    output logic weightsLoaded,
    input logic reloadWeights,
    output logic reloadReady, arrayAdvance,
    output logic signed [WIDTH-1:0] weights[LAYERS][N][N],
    output logic tapValid[LAYERS][N], tapGate[LAYERS][N],
    output logic signed [1:0] tapSign[LAYERS][N], tapGradientSign[LAYERS][N]
);
    localparam int H=(N-1)/2;
    localparam int STEP=WIDTH+$clog2(N);
    localparam int INDEX_WIDTH=(N>1)?$clog2(N):1;
    localparam int GRADIENT_WIDTH=REDUCTION_WEIGHT_WIDTH+(LAYERS-1)*STEP;
    logic signed [WIDTH-1:0] weightBuffer[LAYERS][N][N];
    integer hostLayer, loadedWeightRows, loadLayer, loadStep;
    logic loadingWeights, captureWeights, weightPush;
    logic signed [WIDTH-1:0] inputVectorData[N];
    logic inputVectorValid;
    logic inputPush, inputPop;
    logic cfgWe[LAYERS];
    logic signed [WIDTH-1:0] cfgData[N];
    wire signed [RESULT_WIDTH-1:0] chainData[LAYERS+1][N];
    wire chainValid[LAYERS+1][N];
    // Array l's backward output, before layer l-1's zeroed lanes are applied.
    // Index LAYERS holds the readout weights, which start the pass.
    wire signed [GRADIENT_WIDTH-1:0] gradientChain[LAYERS+1][N];
    wire sourceBusy[N], arrayBusy[LAYERS], arrayLoaded[LAYERS];
    logic skewBusy, pipelineBusy, allLoaded;
    initial begin
        if (WIDTH<1 || N<3 || N%2==0 || LAYERS<1 || REDUCTION_WEIGHT_WIDTH<1 ||
            RESULT_WIDTH!=WIDTH+LAYERS*STEP)
            $fatal(1, "Fountain requires WIDTH>=1, odd N>=3, LAYERS>=1 and exact widths");
    end
    assign weightReady=rst_n && !weightsLoaded && !loadingWeights && !captureWeights;
    assign weightPush=weightValid && weightReady;
    assign inputReady=weightsLoaded && !(reloadWeights && reloadReady) &&
                      !matrixUpdateValid && (!inputVectorValid || inputPop);
    assign inputPush=inputValid && inputReady;
    assign arrayAdvance=weightsLoaded && !outputBlocked;
    assign inputPop=inputVectorValid && arrayAdvance;
    assign reloadReady=weightsLoaded && !inputVectorValid && !skewBusy &&
                       !pipelineBusy && !matrixUpdateValid;
    always_comb begin
        skewBusy=0;
        pipelineBusy=0;
        allLoaded=1;
        for (int l=0; l<LAYERS; l++) begin
            pipelineBusy |= arrayBusy[l];
            allLoaded &= arrayLoaded[l];
            cfgWe[l]=loadingWeights && loadLayer==l;
        end
        for (int j=0; j<N; j++) begin
            skewBusy |= sourceBusy[j];
            cfgData[j]=weightBuffer[loadLayer][loadStep][j];
            resultData[j]=chainData[LAYERS][j];
            resultValid[j]=chainValid[LAYERS][j];
        end
    end
    genvar i,l,j;
    generate
    for (i=0; i<N; i++) begin : source
        localparam int A=(i<H)?H-i:i-H;
        wire signed [WIDTH-1:0] laneData;
        weightStationaryLaneDelay #(.WIDTH(WIDTH), .DEPTH(A)) skew (
            .clk, .rst_n, .advance(arrayAdvance),
            .in_data(inputVectorData[i]), .in_valid(inputVectorValid),
            .out_data(laneData), .out_valid(chainValid[0][i]), .busy(sourceBusy[i])
        );
        assign chainData[0][i]={{(RESULT_WIDTH-WIDTH){laneData[WIDTH-1]}},laneData};
        assign gradientChain[LAYERS][i]=reductionWeight[i];
    end
    for (l=0; l<LAYERS; l++) begin : layers
        localparam int XW=WIDTH+l*STEP;
        localparam int AW=XW+STEP;
        // The gradient vector shrinks going forward as activations grow. Array 0
        // never runs backward; the others hold whichever vector is wider.
        localparam int GW=REDUCTION_WEIGHT_WIDTH+(LAYERS-1-l)*STEP;
        localparam int PE_INPUT_WIDTH=(l==0 || XW>GW)?XW:GW;
        wire signed [PE_INPUT_WIDTH-1:0] laneInput[N], gradientInput[N];
        wire signed [PE_INPUT_WIDTH+STEP-1:0] raw[N], gradientOutput[N];
        wire rawValid[N];
        for (j=0; j<N; j++) begin : lanes
            wire signed [AW-1:0] activated;
            wire signed [GRADIENT_WIDTH-1:0] gradient;
            assign laneInput[j]=chainData[l][j];
            assign activated=(passThrough || !raw[j][AW-1])?raw[j]:AW'('0);
            assign chainData[l+1][j]={{(RESULT_WIDTH-AW){activated[AW-1]}},activated};
            assign chainValid[l+1][j]=rawValid[j];
            assign tapValid[l][j]=rawValid[j];
            assign tapGate[l][j]=passThrough || raw[j]>0;
            assign tapSign[l][j]=(activated==0)?2'sd0:(activated[AW-1]?-2'sd1:2'sd1);
            // The gradient vector at this layer's output has 0 wherever ReLU zeroed the output.
            assign gradient=activationGates[l][j]?gradientChain[l+1][j]:0;
            assign gradientInput[j]=gradient;
            assign gradientChain[l][j]=gradientOutput[j];
            assign tapGradientSign[l][j]=(gradient==0)?2'sd0:(gradient[GRADIENT_WIDTH-1]?-2'sd1:2'sd1);
        end
        weightStationarySystolicArray #(.N(N), .WIDTH(WIDTH), .INPUT_WIDTH(PE_INPUT_WIDTH),
            .RESULT_WIDTH(PE_INPUT_WIDTH+STEP)) systolicArray (
            .clk, .rst_n, .advance(arrayAdvance), .backward(l>0 && backward), .cfg_we(cfgWe[l]),
            .cfg_row(INDEX_WIDTH'(loadStep)), .cfg_data(cfgData),
            .update_valid(matrixUpdateValid), .row_direction(rowDirection[l]),
            .column_direction(columnDirection[l]),
            .in_data(laneInput), .gradient_in(gradientInput), .in_valid(chainValid[l]),
            .result(raw), .gradient_out(gradientOutput), .resultValid(rawValid),
            .weights(weights[l]), .loaded(arrayLoaded[l]), .busy(arrayBusy[l])
        );
    end
    endgenerate
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            inputVectorValid<=0;
            for (int i=0; i<N; i++) inputVectorData[i]<='0;
        end else begin
            if (inputPush) begin
                inputVectorValid<=1;
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
            // synthesis translate_on
        end
    end
endmodule
