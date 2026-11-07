// Fountain replaces the former inward datapath in the existing hierarchy.
// Inference is pipelined; training drains behind one accepted training sample,
// computes full backward sensitivities, and commits one-LSB updates atomically.
module nnAccelerator #(
    parameter int WIDTH=16, N=3, LAYERS=3, FRACTION_BITS=4,
    parameter int TARGET_WIDTH=WIDTH, REDUCTION_WEIGHT_WIDTH=8,
    parameter int OUTPUT_FIFO_DEPTH=2*N,
    parameter int IN_FLIGHT_DEPTH=LAYERS*(N+2)+OUTPUT_FIFO_DEPTH+2
)(
    input logic clk, rst_n,
    input logic signed [WIDTH-1:0] weightData[N],
    input logic weightValid,
    output logic weightReady,
    input logic signed [WIDTH-1:0] inputData[N],
    input logic signed [TARGET_WIDTH-1:0] targetData,
    input logic trainingEnable, inputValid,
    output logic inputReady,
    input logic signed [REDUCTION_WEIGHT_WIDTH-1:0] reductionWeight[N],
    input logic loadReductionWeights, passThrough,
    output logic signed [WIDTH+LAYERS*(WIDTH+$clog2(N))+$clog2(N)-1:0] resultData,
    output logic resultValid,
    input logic resultReady,
    output logic weightsLoaded,
    input logic reloadWeights,
    output logic reloadReady
);
    localparam int LG=$clog2(N);
    localparam int MATRIX_RESULT_WIDTH=WIDTH+LAYERS*(WIDTH+LG);
    localparam int PREDICTION_WIDTH=MATRIX_RESULT_WIDTH+LG;
    localparam int SENS_WIDTH=REDUCTION_WEIGHT_WIDTH+(LAYERS-1)*(WIDTH+LG);
    localparam int BACK_PRODUCT_WIDTH=WIDTH+SENS_WIDTH;
    localparam int BACK_SUM_WIDTH=BACK_PRODUCT_WIDTH+LG;
    localparam int COUNT_WIDTH=$clog2(IN_FLIGHT_DEPTH+1);
    localparam int COMPARE_WIDTH=(PREDICTION_WIDTH>TARGET_WIDTH)?PREDICTION_WIDTH:TARGET_WIDTH;
    typedef enum logic [1:0] {IDLE, BACKWARD, APPLY} training_state_t;
    training_state_t trainingState;
    wire signed [REDUCTION_WEIGHT_WIDTH-1:0] residentReductionWeight[N];
    wire signed [WIDTH-1:0] residentWeights[LAYERS][N][N];
    wire signed [MATRIX_RESULT_WIDTH-1:0] activatedMatrixResultData[N];
    wire matrixResultValid[N];
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
    wire signed [1:0] tapSign[LAYERS][N];
    logic signed [SENS_WIDTH-1:0] sensitivity[LAYERS][N];
    logic signed [1:0] learningDirection, rowDirection[LAYERS][N], columnDirection[LAYERS][N];
    logic signed [COMPARE_WIDTH-1:0] compareTarget, comparePrediction;
    integer backwardLayer, backwardRow, backwardColumn;
    // Reserve DSP capacity for the II=1 forward stages on the Cyclone V.
    // The single time-shared backward multiplier is implemented in logic.
    (* multstyle = "logic" *) logic signed [BACK_PRODUCT_WIDTH-1:0] backwardProduct;
    logic signed [BACK_SUM_WIDTH-1:0] backwardAccumulator, backwardSum;

    initial begin
        if (WIDTH<1 || N<3 || N%2==0 || LAYERS<1 || FRACTION_BITS<0 ||
            TARGET_WIDTH<1 || REDUCTION_WEIGHT_WIDTH<1 ||
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
    assign compareTarget=savedTarget;
    assign comparePrediction=resultData;
    always_comb begin
        for (int l=0; l<LAYERS; l++) for (int j=0; j<N; j++) begin
            rowDirection[l][j]=inputSigns[l][j];
            if (learningDirection==0 || sensitivity[l][j]==0) columnDirection[l][j]=0;
            else columnDirection[l][j]=sensitivity[l][j][SENS_WIDTH-1]?-learningDirection:learningDirection;
        end
    end
    generate
    if (LAYERS>1) begin : backward_multiplier
        assign backwardProduct=residentWeights[backwardLayer+1][backwardRow][backwardColumn]*
                               sensitivity[backwardLayer+1][backwardColumn];
    end else begin : single_layer
        assign backwardProduct='0;
    end
    endgenerate
    assign backwardSum=((backwardColumn==0)?BACK_SUM_WIDTH'('0):backwardAccumulator)+
        {{LG{backwardProduct[BACK_PRODUCT_WIDTH-1]}},backwardProduct};

    weightStationaryMatrixMultiplier #(.WIDTH(WIDTH), .N(N), .LAYERS(LAYERS)) matrixEngine (
        .clk, .rst_n, .weightData, .weightValid, .weightReady,
        .inputData, .inputValid(matrixInputValid), .inputReady(matrixInputReady),
        .passThrough(modePassThrough), .rowDirection, .columnDirection, .matrixUpdateValid,
        .resultData(activatedMatrixResultData), .resultValid(matrixResultValid),
        .outputBlocked(predictionValid && !resultFifoCanAccept), .weightsLoaded,
        .reloadWeights(reloadAccepted),
        .reloadReady(matrixReloadReady), .arrayAdvance, .weights(residentWeights),
        .tapValid, .tapGate, .tapSign
    );
    // One scalar readout on the last stage's lanes; it advances with the array.
    weightedVectorReduction #(.MATRIX_RESULT_WIDTH(MATRIX_RESULT_WIDTH),
        .REDUCTION_WEIGHT_WIDTH(REDUCTION_WEIGHT_WIDTH), .N(N),
        .RESCALE_SHIFT(LAYERS*FRACTION_BITS+REDUCTION_WEIGHT_WIDTH-1)) reductionColumn (
        .clk, .rst_n, .advance(arrayAdvance),
        .cfg_we(loadReductionWeights && quiescent), .cfg_data(reductionWeight),
        .update_valid(matrixUpdateValid), .row_direction(finalSigns), .column_direction(learningDirection),
        .in_data(activatedMatrixResultData), .in_valid(matrixResultValid),
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
            backwardLayer<=0;
            backwardRow<=0;
            backwardColumn<=0;
            backwardAccumulator<='0;
            for (int j=0; j<N; j++) begin
                finalSigns[j]<=0;
                for (int l=0; l<LAYERS; l++) begin
                    inputSigns[l][j]<=0;
                    activationGates[l][j]<=0;
                    sensitivity[l][j]<='0;
                end
            end
        end else begin
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
                    (inputData[j][WIDTH-1]?-2'sd1:2'sd1);
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
                    if (compareTarget>comparePrediction) learningDirection<=2'sd1;
                    else if (compareTarget<comparePrediction) learningDirection<=-2'sd1;
                    else learningDirection<=0;
                    for (int j=0; j<N; j++) sensitivity[LAYERS-1][j]<=activationGates[LAYERS-1][j] ?
                        {{(SENS_WIDTH-REDUCTION_WEIGHT_WIDTH){residentReductionWeight[j][REDUCTION_WEIGHT_WIDTH-1]}},residentReductionWeight[j]} : SENS_WIDTH'('0);
                    backwardLayer<=(LAYERS>1)?LAYERS-2:0;
                    backwardRow<=0;
                    backwardColumn<=0;
                    backwardAccumulator<='0;
                    if (LAYERS==1 || compareTarget==comparePrediction) trainingState<=APPLY;
                    else trainingState<=BACKWARD;
                end
                BACKWARD: if (arrayAdvance) begin
                    if (backwardColumn==N-1) begin
                        sensitivity[backwardLayer][backwardRow]<=activationGates[backwardLayer][backwardRow] ?
                            backwardSum[SENS_WIDTH-1:0] : SENS_WIDTH'('0);
                        backwardColumn<=0;
                        backwardAccumulator<='0;
                        if (backwardRow==N-1) begin
                            backwardRow<=0;
                            if (backwardLayer==0) trainingState<=APPLY;
                            else backwardLayer<=backwardLayer-1;
                        end else backwardRow<=backwardRow+1;
                    end else begin
                        backwardAccumulator<=backwardSum;
                        backwardColumn<=backwardColumn+1;
                    end
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
                !matrixEngine.pipelineBusy && reductionColumn.activationValid=='0)
                else $fatal(1, "Learning barrier did not drain");
            if (trainingState==BACKWARD && backwardColumn==N-1)
                assert (backwardSum=={{(BACK_SUM_WIDTH-SENS_WIDTH){backwardSum[SENS_WIDTH-1]}},backwardSum[SENS_WIDTH-1:0]})
                    else $fatal(1, "Backward sensitivity overflow");
            // synthesis translate_on
        end
    end
endmodule
