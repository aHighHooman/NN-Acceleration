// Fountain accelerator: matrix stages, a scalar readout and a result FIFO.
// Inference is pipelined; training drains behind one accepted training sample,
// runs the gradient vectors back through the arrays, and commits one-LSB updates
// atomically.
module nnAccelerator #(
    parameter int WIDTH=16, ACTIVATION_WIDTH=19, N=3, LAYERS=3, FRACTION_BITS=8,
    parameter int TARGET_WIDTH=WIDTH, REDUCTION_WEIGHT_WIDTH=8,
    parameter int OUTPUT_FIFO_DEPTH=2*N,
    parameter int IN_FLIGHT_DEPTH=LAYERS*(N+2)+OUTPUT_FIFO_DEPTH+2
)(
    input logic clk, rst_n,
    input logic signed [WIDTH-1:0] weightData[N],
    input logic weightValid,
    output logic weightReady,
    input logic signed [ACTIVATION_WIDTH-1:0] inputData[N],
    input logic signed [TARGET_WIDTH-1:0] targetData,
    input logic trainingEnable, inputValid,
    output logic inputReady,
    input logic signed [REDUCTION_WEIGHT_WIDTH-1:0] reductionWeight[N],
    input logic loadReductionWeights, passThrough,
    output logic signed [ACTIVATION_WIDTH+$clog2(N)-1:0] resultData,
    output logic resultValid,
    input logic resultReady,
    output logic weightsLoaded,
    input logic reloadWeights,
    output logic reloadReady
);
    localparam int LG=$clog2(N);
    localparam int PREDICTION_WIDTH=ACTIVATION_WIDTH+LG;
    // The arrays switch to their transposed weights one edge after backward
    // rises. From the held readout weights to the last lane, the longest register
    // path then crosses LAYERS-1 lane registers and (LAYERS-1)*((N-1)/2+1)+(N-1)/2
    // PE registers, and one more edge captures the signs.
    localparam int BACKWARD_CYCLES=LAYERS*((N-1)/2+2);
    localparam int COUNT_WIDTH=$clog2(IN_FLIGHT_DEPTH+1);
    typedef enum logic [1:0] {IDLE, BACKWARD, APPLY} training_state_t;
    training_state_t trainingState;
    wire signed [REDUCTION_WEIGHT_WIDTH-1:0] residentReductionWeight[N];
    wire signed [WIDTH-1:0] residentWeights[LAYERS][N][N];
    wire signed [ACTIVATION_WIDTH-1:0] activatedMatrixResultData[N];
    wire matrixResultValid[N];
    // The readout reads the last layer's lanes from a register, like arrays 1..LAYERS-1.
    logic signed [ACTIVATION_WIDTH-1:0] readoutInput[N];
    logic readoutValid[N];
    wire matrixInputReady, matrixReloadReady, arrayAdvance, predictionValid;
    logic matrixInputValid, matrixUpdateValid, samplePush;
    logic resultFifoPop, resultFifoCanAccept;
    wire resultFifoFull, resultFifoEmpty;
    wire signed [PREDICTION_WIDTH-1:0] prediction;
    logic [COUNT_WIDTH-1:0] outstanding;
    logic trainingPending, configurationError;
    logic modePassThrough, quiescent, reloadAccepted, sampleCanAccept;
    logic signed [TARGET_WIDTH-1:0] savedTarget;
    logic signed [1:0] inputSigns[LAYERS][N], finalSigns[N];
    logic activationGates[LAYERS][N];
    wire tapValid[LAYERS][N], tapGate[LAYERS][N];
    wire signed [1:0] tapSign[LAYERS][N], tapGradientSign[LAYERS][N];
    logic signed [1:0] gradientSign[LAYERS][N];
    logic signed [1:0] learningDirection, columnDirection[LAYERS][N];
    logic [$clog2(BACKWARD_CYCLES)-1:0] backwardCount;

    initial begin
        if (WIDTH<1 || N<3 || N%2==0 || LAYERS<1 || FRACTION_BITS<0 ||
            TARGET_WIDTH<1 || REDUCTION_WEIGHT_WIDTH<1 || ACTIVATION_WIDTH<REDUCTION_WEIGHT_WIDTH ||
            IN_FLIGHT_DEPTH<1 || OUTPUT_FIFO_DEPTH<1)
            $fatal(1, "Fountain requires positive widths/depths and odd N>=3");
    end
    assign quiescent=outstanding==0 && !trainingPending && (!weightsLoaded || matrixReloadReady);
    assign reloadReady=weightsLoaded && matrixReloadReady && quiescent && resultFifoEmpty;
    assign reloadAccepted=reloadWeights && reloadReady;
    assign sampleCanAccept=(outstanding<IN_FLIGHT_DEPTH || resultFifoPop) &&
        !trainingPending && trainingState==IDLE && !loadReductionWeights && !reloadAccepted;
    assign inputReady=matrixInputReady && sampleCanAccept;
    assign matrixInputValid=inputValid && sampleCanAccept;
    assign samplePush=inputValid && inputReady;
    assign resultValid=!resultFifoEmpty;
    assign resultFifoPop=resultValid && resultReady;
    assign resultFifoCanAccept=!resultFifoFull || resultFifoPop;
    assign matrixUpdateValid=trainingState==APPLY;
    always_comb begin
        for (int l=0; l<LAYERS; l++) for (int j=0; j<N; j++) begin
            if (gradientSign[l][j]==0) columnDirection[l][j]=0;
            else columnDirection[l][j]=(gradientSign[l][j]<0)?-learningDirection:learningDirection;
        end
    end

    weightStationaryMatrixMultiplier #(.WIDTH(WIDTH), .ACTIVATION_WIDTH(ACTIVATION_WIDTH),
        .FRACTION_BITS(FRACTION_BITS), .N(N), .LAYERS(LAYERS),
        .REDUCTION_WEIGHT_WIDTH(REDUCTION_WEIGHT_WIDTH)) matrixEngine (
        .clk, .rst_n, .backward(trainingState==BACKWARD), .weightData, .weightValid, .weightReady,
        .inputData, .inputValid(matrixInputValid), .inputReady(matrixInputReady),
        .passThrough(modePassThrough), .reductionWeight(residentReductionWeight), .activationGates,
        .rowDirection(inputSigns), .columnDirection, .matrixUpdateValid,
        .resultData(activatedMatrixResultData), .resultValid(matrixResultValid),
        .outputBlocked(predictionValid && !resultFifoCanAccept), .weightsLoaded,
        .reloadWeights(reloadAccepted),
        .reloadReady(matrixReloadReady), .arrayAdvance, .weights(residentWeights),
        .tapValid, .tapGate, .tapSign, .tapGradientSign
    );
    always_ff @(posedge clk) begin
        if (!rst_n) for (int j=0; j<N; j++) begin readoutInput[j]<='0; readoutValid[j]<=0; end
        else if (arrayAdvance) for (int j=0; j<N; j++) begin
            readoutInput[j]<=activatedMatrixResultData[j];
            readoutValid[j]<=matrixResultValid[j];
        end
    end
    // One scalar readout on the last stage's lanes; it advances with the array.
    weightedVectorReduction #(.ACTIVATION_WIDTH(ACTIVATION_WIDTH),
        .REDUCTION_WEIGHT_WIDTH(REDUCTION_WEIGHT_WIDTH), .N(N),
        .RESCALE_SHIFT(REDUCTION_WEIGHT_WIDTH-1)) reductionColumn (
        .clk, .rst_n, .advance(arrayAdvance),
        .cfg_we(loadReductionWeights && quiescent), .cfg_data(reductionWeight),
        .update_valid(matrixUpdateValid), .row_direction(finalSigns), .column_direction(learningDirection),
        .in_data(readoutInput), .in_valid(readoutValid),
        .prediction, .predictionValid, .weights(residentReductionWeight)
    );
    signedFifo #(.WIDTH(PREDICTION_WIDTH), .DEPTH(OUTPUT_FIFO_DEPTH)) resultFifo (
        .clk, .rst_n, .push(predictionValid), .pushData(prediction),
        .pop(resultFifoPop), .popData(resultData), .full(resultFifoFull), .empty(resultFifoEmpty)
    );

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            trainingState<=IDLE;
            trainingPending<=0;
            outstanding<='0;
            savedTarget<='0;
            learningDirection<=0;
            modePassThrough<=0;
            configurationError<=0;
            backwardCount<='0;
            for (int j=0; j<N; j++) begin
                finalSigns[j]<=0;
                for (int l=0; l<LAYERS; l++) begin
                    inputSigns[l][j]<=0;
                    activationGates[l][j]<=0;
                    gradientSign[l][j]<=0;
                end
            end
        end else begin
            // Free-running: only the edge into APPLY matters, and the lanes have
            // settled by then.
            for (int l=0; l<LAYERS; l++) for (int j=0; j<N; j++)
                gradientSign[l][j]<=tapGradientSign[l][j];
            if (quiescent) modePassThrough<=passThrough;
            else if (passThrough!=modePassThrough || loadReductionWeights) configurationError<=1;
            case ({samplePush,resultFifoPop})
                2'b10: outstanding<=outstanding+1'b1;
                2'b01: outstanding<=outstanding-1'b1;
                default: ;
            endcase
            if (samplePush && trainingEnable) begin
                trainingPending<=1;
                savedTarget<=targetData;
                for (int j=0; j<N; j++) inputSigns[0][j]<=(inputData[j]==0)?2'sd0:
                    (inputData[j][ACTIVATION_WIDTH-1]?-2'sd1:2'sd1);
            end
            // A pending training sample is the last one admitted, so the last
            // valid value at each tap is its own; earlier samples pass first.
            if (trainingPending && arrayAdvance)
                for (int l=0; l<LAYERS; l++) for (int j=0; j<N; j++)
                    if (tapValid[l][j]) begin
                        activationGates[l][j]<=tapGate[l][j];
                        if (l<LAYERS-1) inputSigns[l+1][j]<=tapSign[l][j];
                        if (l==LAYERS-1) finalSigns[j]<=tapSign[l][j];
                    end
            case (trainingState)
                IDLE: if (resultFifoPop && trainingPending && outstanding==1) begin
                    if (savedTarget>resultData) learningDirection<=2'sd1;
                    else if (savedTarget<resultData) learningDirection<=-2'sd1;
                    else learningDirection<=0;
                    backwardCount<='0;
                    if (LAYERS==1 || savedTarget==resultData) trainingState<=APPLY;
                    else trainingState<=BACKWARD;
                end
                BACKWARD: if (arrayAdvance) begin
                    backwardCount<=backwardCount+1'b1;
                    if (backwardCount==BACKWARD_CYCLES-1) trainingState<=APPLY;
                end
                APPLY: if (arrayAdvance) begin
                    trainingPending<=0;
                    trainingState<=IDLE;
                end
                default: trainingState<=IDLE;
            endcase
            // synthesis translate_off
            if (resultFifoPop) assert (outstanding!=0) else $fatal(1, "Unsolicited result");
            if (trainingState!=IDLE) assert (outstanding==0 && resultFifoEmpty &&
                !matrixEngine.inputVectorValid && !matrixEngine.skewBusy &&
                !matrixEngine.pipelineBusy && !readoutValid.or() && reductionColumn.activationValid=='0)
                else $fatal(1, "Learning barrier did not drain");
            // synthesis translate_on
        end
    end
endmodule
