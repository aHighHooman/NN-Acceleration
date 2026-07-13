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
    localparam int RESULT_ROW_WIDTH = (N <= 1) ? 1 : $clog2(N);

    logic weightPush, weightPop, allWeightValid, allWeightReady;
    logic activationPush, activationPop, allActivationValid, allActivationReady;
    logic outputPop, allOutputValid, allOutputReady, allOutputEmpty, allActivationEmpty;
    logic [WEIGHT_COUNT_WIDTH-1:0] loadedWeightRows;
    logic [ACT_ROW_WIDTH-1:0] acceptedActivationRow;
    logic [RESULT_ROW_WIDTH-1:0] transmittedResultRow;

    logic signed [WIDTH-1:0] weightData_FifoToLoader [N];
    logic weightFull[N], weightEmpty[N];
    logic signed [WIDTH-1:0] activationData_FifoToOrch[N];
    logic activationFull[N], activationEmpty[N];
    logic signed [2*WIDTH-1:0] resultData_FifoToOutput[N];
    logic outputFull[N], outputEmpty[N];

    logic signed [WIDTH-1:0] skewData[N][N];
    logic skewValid[N][N];
    logic signed [WIDTH-1:0] rowData_OrchToSyst[N];
    logic validData_OrchToSyst[N];
    logic signed [2*WIDTH-1:0] resultData_SystToFifo[N];
    logic validData_SystToFifo[N];
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
            outputBlocked |= validData_SystToFifo[i] && outputFull[i] && !outputPop;
            for (int d = 0; d < N; d++) skewBusy |= skewValid[i][d];
        end
    end

    assign weightReady      = !weightsLoaded && allWeightReady;
    assign weightPush       = weightValid && weightReady;
    assign activationReady  = weightsLoaded && allActivationReady;
    assign activationPush   = activationValid && activationReady;
    assign outputPop        = resultValid && resultReady;
    assign resultValid      = allOutputValid;
    assign resultLast       = resultValid && (transmittedResultRow == N-1);
    assign arrayAdvance     = !weightsLoaded ? weightPop : !outputBlocked;
    assign activationPop    = weightsLoaded && allActivationValid && arrayAdvance;
    assign weightPop        = !weightsLoaded && allWeightValid;
    assign reloadReady      = weightsLoaded && allActivationEmpty && !skewBusy &&
                              !pipelineBusy && allOutputEmpty &&
                              (acceptedActivationRow == 0);

    genvar fifoIndex;
    generate
        for (fifoIndex = 0; fifoIndex < N; fifoIndex = fifoIndex + 1) begin : fifo_banks
            assign resultData[fifoIndex] = resultData_FifoToOutput[fifoIndex];

            signedFifo #(.WIDTH(WIDTH), .DEPTH(N)) weightFifo (
                .clk(clk), .rst_n(rst_n), .push(weightPush), .pushData(weightData[fifoIndex]),
                .pop(weightPop), .popData(weightData_FifoToLoader[fifoIndex]), .full(weightFull[fifoIndex]),
                .empty(weightEmpty[fifoIndex]), .count()
            );
            signedFifo #(.WIDTH(WIDTH), .DEPTH(INPUT_FIFO_DEPTH)) activationFifo (
                .clk(clk), .rst_n(rst_n), .push(activationPush), .pushData(activationData[fifoIndex]),
                .pop(activationPop), .popData(activationData_FifoToOrch[fifoIndex]), .full(activationFull[fifoIndex]),
                .empty(activationEmpty[fifoIndex]), .count()
            );
            signedFifo #(.WIDTH(2*WIDTH), .DEPTH(OUTPUT_FIFO_DEPTH)) outputFifo (
                .clk(clk), .rst_n(rst_n),
                .push(arrayAdvance && validData_SystToFifo[fifoIndex]), .pushData(resultData_SystToFifo[fifoIndex]),
                .pop(outputPop), .popData(resultData_FifoToOutput[fifoIndex]), .full(outputFull[fifoIndex]),
                .empty(outputEmpty[fifoIndex]), .count()
            );
        end
    endgenerate

    genvar laneIndex;
    generate
        for (laneIndex = 0; laneIndex < N; laneIndex = laneIndex + 1) begin : skew_outputs
            assign rowData_OrchToSyst[laneIndex]   = skewData[laneIndex][laneIndex];
            assign validData_OrchToSyst[laneIndex] = skewValid[laneIndex][laneIndex];
        end
    endgenerate

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            weightsLoaded        <= 1'b0;
            loadedWeightRows     <= '0;
            acceptedActivationRow<= '0;
            transmittedResultRow <= '0;
            for (int lane = 0; lane < N; lane++) begin
                for (int d = 0; d < N; d++) begin
                    skewData[lane][d]  <= '0;
                    skewValid[lane][d] <= 1'b0;
                end
            end
        end else begin
            if (weightPop) begin
                if (loadedWeightRows == N-1) begin
                    loadedWeightRows <= '0;
                    weightsLoaded    <= 1'b1;
                end else begin
                    loadedWeightRows <= loadedWeightRows + 1'b1;
                end
            end

            if (activationPush) begin
                acceptedActivationRow <= (acceptedActivationRow == N-1) ? '0 :
                                         acceptedActivationRow + 1'b1;
            end

            if (outputPop) begin
                transmittedResultRow <= (transmittedResultRow == N-1) ? '0 :
                                        transmittedResultRow + 1'b1;
            end

            if (reloadWeights && reloadReady) begin
                weightsLoaded    <= 1'b0;
                loadedWeightRows <= '0;
            end

            if (weightsLoaded && arrayAdvance) begin
                for (int lane = 0; lane < N; lane++) begin
                    skewData[lane][0]  <= activationData_FifoToOrch[lane][WIDTH-1:0];
                    skewValid[lane][0] <= activationPop;
                    for (int d = 1; d < N; d++) begin
                        skewData[lane][d]  <= skewData[lane][d-1];
                        skewValid[lane][d] <= skewValid[lane][d-1];
                    end
                end
            end
        end
    end

    systolicArrayWeightStationary #(.WIDTH(WIDTH), .N(N)) systolicArr (
        .clk(clk), .rst_n(rst_n), .advance(arrayAdvance), .loadWeight(weightPop),
        .row(rowData_OrchToSyst), .rowValid(validData_OrchToSyst),
        .col(weightData_FifoToLoader), .result(resultData_SystToFifo),
        .resultValid(validData_SystToFifo),
        .pipelineBusy(pipelineBusy)
    );

endmodule
