// Fountain source, dedicated layer chain and row loader. Every layer's lanes,
// including the outputs, stay on the staggered center-first front and in one
// ACTIVATION_WIDTH format with FRACTION_BITS fraction bits. Arrays 1..LAYERS-1
// read their lanes from a register after the lane mux, which adds one step per
// array to the front. While backward is held, those arrays use their transposed
// weights and take their input from the next array's output, and the readout
// weights settle into every layer's gradient vector.
module weightStationaryMatrixMultiplier #(
    parameter int WIDTH=16, ACTIVATION_WIDTH=19, FRACTION_BITS=8,
    parameter int N=3, LAYERS=1, REDUCTION_WEIGHT_WIDTH=8
)(
    input logic clk, rst_n, backward,
    input logic signed [WIDTH-1:0] weightData[N],
    input logic weightValid,
    output logic weightReady,
    input logic signed [ACTIVATION_WIDTH-1:0] inputData[N],
    input logic inputValid,
    output logic inputReady,
    input logic passThrough,
    input logic signed [REDUCTION_WEIGHT_WIDTH-1:0] reductionWeight[N],
    input logic activationGates[LAYERS][N],
    input logic signed [1:0] rowDirection[LAYERS][N], columnDirection[LAYERS][N],
    input logic matrixUpdateValid,
    output logic signed [ACTIVATION_WIDTH-1:0] resultData[N],
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
    localparam int INDEX_WIDTH=(N>1)?$clog2(N):1;
    localparam int SUM_WIDTH=ACTIVATION_WIDTH+WIDTH+$clog2(N);
    localparam logic signed [ACTIVATION_WIDTH-1:0] MAX_ACTIVATION={1'b0,{(ACTIVATION_WIDTH-1){1'b1}}};
    localparam logic signed [ACTIVATION_WIDTH-1:0] MIN_ACTIVATION={1'b1,{(ACTIVATION_WIDTH-1){1'b0}}};
    logic signed [WIDTH-1:0] weightBuffer[LAYERS][N][N];
    integer hostLayer, loadedWeightRows, loadLayer, loadStep;
    logic loadingWeights, captureWeights, weightPush;
    logic signed [ACTIVATION_WIDTH-1:0] inputVectorData[N];
    logic inputVectorValid;
    logic inputPush, inputPop;
    logic cfgWe[LAYERS];
    logic signed [WIDTH-1:0] cfgData[N];
    wire signed [ACTIVATION_WIDTH-1:0] chainData[LAYERS+1][N];
    wire chainValid[LAYERS+1][N];
    // Array l's requantized output during the backward pass, before layer l-1's
    // zeroed lanes are applied. Index LAYERS holds the readout weights.
    wire signed [ACTIVATION_WIDTH-1:0] gradientChain[LAYERS+1][N];
    // The gradient vector at each layer's output, with 0 wherever ReLU zeroed it.
    wire signed [ACTIVATION_WIDTH-1:0] laneGradient[LAYERS][N];
    // Array l's lanes for l>0, held after the lane mux. The backward select is
    // registered per array so it doesn't fan out from the controller in the
    // multiply's cycle. Index 0 is unused: array 0 reads the source directly.
    logic signed [ACTIVATION_WIDTH-1:0] heldLanes[LAYERS][N];
    logic heldValid[LAYERS][N], arrayBackward[LAYERS];
    wire sourceBusy[N], arrayBusy[LAYERS], arrayLoaded[LAYERS];
    logic skewBusy, pipelineBusy, allLoaded;
    initial begin
        if (WIDTH<1 || N<3 || N%2==0 || LAYERS<1 || FRACTION_BITS<0 ||
            REDUCTION_WEIGHT_WIDTH<1 || ACTIVATION_WIDTH<REDUCTION_WEIGHT_WIDTH)
            $fatal(1, "Fountain requires WIDTH>=1, odd N>=3, LAYERS>=1 and readout weights that fit an activation");
    end
    // A product has twice the shared fraction bits. Shift once to restore the
    // activation format, then cap instead of wrapping.
    function automatic logic signed [ACTIVATION_WIDTH-1:0] requantize(input logic signed [SUM_WIDTH-1:0] value);
        logic signed [SUM_WIDTH-1:0] shifted;
        shifted=value>>>FRACTION_BITS;
        if (shifted>MAX_ACTIVATION) return MAX_ACTIVATION;
        if (shifted<MIN_ACTIVATION) return MIN_ACTIVATION;
        return shifted[ACTIVATION_WIDTH-1:0];
    endfunction
    function automatic logic signed [1:0] sign_of(input logic signed [ACTIVATION_WIDTH-1:0] value);
        return (value==0) ? 2'sd0 : ((value<0) ? -2'sd1 : 2'sd1);
    endfunction
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
        for (int l=1; l<LAYERS; l++)
            for (int j=0; j<N; j++) pipelineBusy |= heldValid[l][j];
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
        weightStationaryLaneDelay #(.WIDTH(ACTIVATION_WIDTH), .DEPTH(A)) skew (
            .clk, .rst_n, .advance(arrayAdvance),
            .in_data(inputVectorData[i]), .in_valid(inputVectorValid),
            .out_data(chainData[0][i]), .out_valid(chainValid[0][i]), .busy(sourceBusy[i])
        );
        assign gradientChain[LAYERS][i]=reductionWeight[i];
    end
    for (l=0; l<LAYERS; l++) begin : layers
        wire signed [ACTIVATION_WIDTH-1:0] laneInput[N];
        wire laneValid[N];
        wire signed [SUM_WIDTH-1:0] raw[N];
        wire rawValid[N];
        for (j=0; j<N; j++) begin : lanes
            assign laneInput[j]=(l==0)?chainData[0][j]:heldLanes[l][j];
            assign laneValid[j]=(l==0)?chainValid[0][j]:heldValid[l][j];
            assign chainData[l+1][j]=requantize((passThrough || !raw[j][SUM_WIDTH-1])?raw[j]:0);
            assign chainValid[l+1][j]=rawValid[j];
            assign tapValid[l][j]=rawValid[j];
            assign tapGate[l][j]=passThrough || raw[j]>0;
            assign tapSign[l][j]=sign_of(chainData[l+1][j]);
            assign laneGradient[l][j]=activationGates[l][j]?gradientChain[l+1][j]:0;
            assign gradientChain[l][j]=requantize(raw[j]);
            assign tapGradientSign[l][j]=sign_of(laneGradient[l][j]);
        end
        weightStationarySystolicArray #(.N(N), .WIDTH(WIDTH), .INPUT_WIDTH(ACTIVATION_WIDTH),
            .RESULT_WIDTH(SUM_WIDTH)) systolicArray (
            .clk, .rst_n, .advance(arrayAdvance), .backward(arrayBackward[l]), .cfg_we(cfgWe[l]),
            .cfg_row(INDEX_WIDTH'(loadStep)), .cfg_data(cfgData),
            .update_valid(matrixUpdateValid), .row_direction(rowDirection[l]),
            .column_direction(columnDirection[l]),
            .in_data(laneInput), .in_valid(laneValid),
            .result(raw), .resultValid(rawValid),
            .weights(weights[l]), .loaded(arrayLoaded[l]), .busy(arrayBusy[l])
        );
    end
    endgenerate
    // The lane mux picks the previous layer's output, or during backward this
    // layer's gradient vector.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int l=0; l<LAYERS; l++) begin
                arrayBackward[l]<=0;
                for (int j=0; j<N; j++) begin heldLanes[l][j]<='0; heldValid[l][j]<=0; end
            end
        end else for (int l=1; l<LAYERS; l++) begin
            arrayBackward[l]<=backward;
            if (arrayAdvance) for (int j=0; j<N; j++) begin
                heldLanes[l][j]<=arrayBackward[l]?laneGradient[l][j]:chainData[l][j];
                heldValid[l][j]<=chainValid[l][j];
            end
        end
    end
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
