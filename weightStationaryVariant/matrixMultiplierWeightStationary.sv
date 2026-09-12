module matrixMultiplierWeightStationary #(
    parameter int WIDTH = 16,
    parameter int N = 3,
    parameter int INPUT_FIFO_DEPTH = 2*N,
    parameter int OUTPUT_FIFO_DEPTH = 2*N,
    parameter int MATRIX_VERSION_WIDTH = $clog2(INPUT_FIFO_DEPTH+2)
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
    output logic [MATRIX_VERSION_WIDTH-1:0] matrixResultVersion,
    output logic                         resultValid,
    input  logic                         resultReady,
    output logic                         resultLast,
    output logic                         weightsLoaded,
    input  logic                         reloadWeights,
    output logic                         reloadReady,
    output logic                         matrixUpdateAccepted,
    output logic                         matrixUpdateComplete
);

    localparam int WEIGHT_COUNT_WIDTH   = $clog2(N+1);
    localparam int ACT_ROW_WIDTH        = $clog2(N);
    localparam int RESULT_ROW_WIDTH     = $clog2(N);
    localparam int RESULT_WIDTH         = $clog2(N) + 2*WIDTH;

    logic weightPush, weightPop, allWeightValid, allWeightReady;
    logic activationPush, activationPop, allActivationValid, allActivationReady;
    logic outputPop, allOutputValid, allOutputEmpty, allActivationEmpty;
    logic [WEIGHT_COUNT_WIDTH-1:0] loadedWeightRows;
    logic [ACT_ROW_WIDTH-1:0] acceptedActivationRow;
    logic [RESULT_ROW_WIDTH-1:0] transmittedResultRow;

    logic signed [WIDTH-1:0] weightData_FifoToLoader [N];
    logic weightFull[N], weightEmpty[N];
    logic signed [WIDTH-1:0] activationData_FifoToOrch[N];
    logic activationFull[N], activationEmpty[N];
    logic signed [RESULT_WIDTH-1:0] resultData_FifoToOutput[N];
    logic outputFull[N], outputEmpty[N];
    logic signed [MATRIX_VERSION_WIDTH-1:0] versionFifoHead;
    logic versionFifoFull, versionFifoEmpty;

    logic signed [WIDTH-1:0] skewData[N][N];
    logic skewValid[N][N];
    logic signed [WIDTH-1:0] rowData_OrchToSyst[N];
    logic validData_OrchToSyst[N];
    logic signed [RESULT_WIDTH-1:0] resultData_SystToFifo[N];
    logic validData_SystToFifo[N];
    logic pipelineBusy, skewBusy, arrayAdvance, outputBlocked;
    logic [MATRIX_VERSION_WIDTH-1:0] matrixWeightVersion;
    logic [MATRIX_VERSION_WIDTH-1:0] resultVersionPipe[N];
    logic updateBoundaryValid;

    always_comb begin
        allWeightValid      = 1;
        allWeightReady      = 1;
        allActivationValid  = 1;
        allActivationReady  = 1;
        allOutputValid      = 1;
        allOutputEmpty      = 1;
        allActivationEmpty  = 1;
        skewBusy            = 0;
        outputBlocked       = 0;

        for (int i = 0; i < N; i++) begin
            allWeightValid      &= !weightEmpty[i];
            allWeightReady      &= !weightFull[i];
            allActivationValid  &= !activationEmpty[i];
            allActivationReady  &= !activationFull[i];
            allActivationEmpty  &= activationEmpty[i];
            allOutputValid      &= !outputEmpty[i];
            allOutputEmpty      &= outputEmpty[i];
            outputBlocked       |= validData_SystToFifo[i] && outputFull[i] && !outputPop;

            for (int d = 0; d < N; d++) begin
                skewBusy |= skewValid[i][d];
            end
        end

        outputBlocked |= validData_SystToFifo[0] && versionFifoFull &&
                         !outputPop;
    end

    assign weightReady      = !weightsLoaded && allWeightReady;
    assign weightPush       = weightValid && weightReady;
    assign activationReady  = weightsLoaded && allActivationReady;
    assign activationPush   = activationValid && activationReady;
    assign outputPop        = resultValid && resultReady;
    assign resultValid      = allOutputValid && !versionFifoEmpty;
    assign matrixResultVersion = versionFifoHead;
    assign resultLast       = resultValid && (transmittedResultRow == N-1);
    assign arrayAdvance     = !weightsLoaded ? weightPop : !outputBlocked;
    assign matrixUpdateAccepted = matrixUpdateValid && arrayAdvance;
    assign activationPop    = weightsLoaded && allActivationValid && arrayAdvance;
    assign weightPop        = !weightsLoaded && allWeightValid;
    assign reloadReady      = weightsLoaded && allActivationEmpty && !skewBusy &&
                              !pipelineBusy && allOutputEmpty &&
                              (acceptedActivationRow == 0);

    // A sample entering on the edge that applies update stage zero still
    // multiplies by the old PE value.  Increment after that edge, so the next
    // sample behind the wave is the first one tagged with the new version.
    // The scalar tag advances under the same enable as the systolic data.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            matrixWeightVersion <= '0;
            updateBoundaryValid <= 1'b0;
            for (int stage = 0; stage < N; stage++)
                resultVersionPipe[stage] <= '0;
        end else if (arrayAdvance) begin
            for (int stage = N-1; stage > 0; stage--)
                resultVersionPipe[stage] <= resultVersionPipe[stage-1];
            resultVersionPipe[0] <= matrixWeightVersion;
            if (updateBoundaryValid)
                matrixWeightVersion <= matrixWeightVersion + 1'b1;
            updateBoundaryValid <= matrixUpdateValid;
        end
    end

    signedFifo #(
        .WIDTH(MATRIX_VERSION_WIDTH), .DEPTH(OUTPUT_FIFO_DEPTH)
    ) outputVersionFifo (
        .clk(clk), .rst_n(rst_n),
        .push(arrayAdvance && validData_SystToFifo[0]),
        .pushData($signed(resultVersionPipe[N-1])),
        .pop(outputPop), .popData(versionFifoHead),
        .full(versionFifoFull), .empty(versionFifoEmpty), .values()
    );

    genvar fifoIndex;
    generate
        for (fifoIndex = 0; fifoIndex < N; fifoIndex = fifoIndex + 1) begin : fifo_banks
            assign resultData[fifoIndex] = resultData_FifoToOutput[fifoIndex];

            signedFifo #(.WIDTH(WIDTH), .DEPTH(N)) weightFifo (
                .clk(clk), .rst_n(rst_n), .push(weightPush), .pushData(weightData[fifoIndex]),
                .pop(weightPop), .popData(weightData_FifoToLoader[fifoIndex]), .full(weightFull[fifoIndex]),
                .empty(weightEmpty[fifoIndex]), .values()
            );
            signedFifo #(.WIDTH(WIDTH), .DEPTH(INPUT_FIFO_DEPTH)) activationFifo (
                .clk(clk), .rst_n(rst_n), .push(activationPush), .pushData(activationData[fifoIndex]),
                .pop(activationPop), .popData(activationData_FifoToOrch[fifoIndex]), .full(activationFull[fifoIndex]),
                .empty(activationEmpty[fifoIndex]), .values()
            );
            signedFifo #(.WIDTH(RESULT_WIDTH), .DEPTH(OUTPUT_FIFO_DEPTH)) outputFifo (
                .clk(clk), .rst_n(rst_n),
                .push(arrayAdvance && validData_SystToFifo[fifoIndex]), .pushData(resultData_SystToFifo[fifoIndex]),
                .pop(outputPop), .popData(resultData_FifoToOutput[fifoIndex]), .full(outputFull[fifoIndex]),
                .empty(outputEmpty[fifoIndex]), .values()
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
            weightsLoaded        <= 0;
            loadedWeightRows     <= 0;
            acceptedActivationRow<= 0;
            transmittedResultRow <= 0;

            for (int lane = 0; lane < N; lane++) begin
                for (int d = 0; d < N; d++) begin
                    skewData[lane][d]  <= 0;
                    skewValid[lane][d] <= 0;
                end
            end
        end else begin
            if (weightPop) begin
                if (loadedWeightRows == N-1) begin
                    loadedWeightRows <= 0;
                    weightsLoaded    <= 1;
                end else begin
                    loadedWeightRows <= loadedWeightRows + 1;
                end
            end

            if (activationPush) begin
                acceptedActivationRow <= (acceptedActivationRow == N-1) ? 0 : acceptedActivationRow + 1;
            end

            if (outputPop) begin
                transmittedResultRow <= (transmittedResultRow == N-1) ? 0 : transmittedResultRow + 1;
            end

            if (reloadWeights && reloadReady) begin
                weightsLoaded    <= 0;
                loadedWeightRows <= 0;
            end

            if (weightsLoaded && arrayAdvance) begin
                for (int lane = 0; lane < N; lane++) begin
                    skewData[lane][0]  <= activationData_FifoToOrch[lane];
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
        .rowDirection(rowDirection), .columnDirection(columnDirection),
        .updateValid(matrixUpdateValid),
        .row(rowData_OrchToSyst), .rowValid(validData_OrchToSyst),
        .col(weightData_FifoToLoader), .result(resultData_SystToFifo),
        .resultValid(validData_SystToFifo),
        .updateComplete(matrixUpdateComplete),
        .pipelineBusy(pipelineBusy)
    );

endmodule
