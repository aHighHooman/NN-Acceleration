module matrixMultiplierWeightStationarySPI #(
    parameter int WIDTH = 16,
    parameter int N = 3,
    parameter int FRACTION_BITS = 4,
    parameter int REDUCTION_WEIGHT_WIDTH = 8,
    parameter int INPUT_FIFO_DEPTH = 2*N,
    parameter int OUTPUT_FIFO_DEPTH = 2*N
)(
    input  logic                    clk,
    input  logic                    rst_n,
    output logic                    weightReady,
    output logic                    inputReady,
    input  logic                    passThrough,
    input  logic                    reduceOutput,
    input  logic                    trainingEnable,
    input  logic signed [REDUCTION_WEIGHT_WIDTH-1:0] reductionWeight [N],
    input  logic                    loadReductionWeights,
    output logic                    weightsLoaded,
    input  logic                    reloadWeights,
    output logic                    reloadReady,
    input  logic                    sclk,
    input  logic                    cs_n [N],
    output logic                    miso [N],
    output logic                    misoValid [N],
    input  logic                    weightCs_n [N],
    input  logic                    weightMosi [N],
    input  logic                    inputCs_n [N],
    input  logic                    inputMosi [N]
);

    localparam int MATRIX_RESULT_WIDTH = 2*WIDTH + $clog2(N);
    localparam int PREDICTION_WIDTH = MATRIX_RESULT_WIDTH + $clog2(N);

    logic signed [WIDTH-1:0] weightData[N], inputData[N];
    logic weightValid, inputValid;
    logic weightFifoReady, inputFifoReady;
    logic weightDataValid[N], inputDataValid[N];
    logic allWeightDataValid, allInputDataValid, allWeightSpiReady, allInputSpiReady;
    logic weightValidSync, weightValidSyncDelay, weightSent;
    logic inputValidSync, inputValidSyncDelay, inputSent;
    logic weightAccepted, weightAcceptedSync, weightAcceptedSyncDelay, weightAcceptedSeen;
    logic inputAccepted, inputAcceptedSync;
    logic inputAcceptedSyncDelay, inputAcceptedSeen;
    logic signed [PREDICTION_WIDTH-1:0] resultData[N], spiData[N];
    logic resultValid, resultReady;
    logic spiReady[N], allSpiReady;
    logic request, requestSync, requestSyncDelay, acknowledge;
    logic acknowledgeSync, acknowledgeSyncDelay;

    assign resultReady      = resultValid && (request == acknowledgeSyncDelay);
    assign weightValid      = weightValidSyncDelay && !weightSent;
    assign inputValid      = inputValidSyncDelay && !inputSent;
    assign weightReady      = weightFifoReady && allWeightSpiReady;
    assign inputReady      = inputFifoReady && allInputSpiReady;

    always_comb begin
        allSpiReady             = 1;
        allWeightDataValid      = 1;
        allInputDataValid       = 1;
        allWeightSpiReady       = 1;
        allInputSpiReady        = 1;

        for (int i = 0; i < N; i++) begin
            allSpiReady             &= spiReady[i];
            allWeightDataValid      &= weightDataValid[i];
            allInputDataValid       &= inputDataValid[i];
            allWeightSpiReady       &= !weightDataValid[i];
            allInputSpiReady        &= !inputDataValid[i];
        end

    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            request                     <= 0;
            acknowledgeSync             <= 0;
            acknowledgeSyncDelay        <= 0;
            weightValidSync             <= 0;
            weightValidSyncDelay        <= 0;
            inputValidSync             <= 0;
            inputValidSyncDelay        <= 0;
            weightAccepted              <= 0;
            inputAccepted              <= 0;
            weightSent                  <= 0;
            inputSent                  <= 0;

            for (int i = 0; i < N; i++) begin
                spiData[i] <= 0;
            end
        end else begin
            acknowledgeSync          <= acknowledge;
            acknowledgeSyncDelay     <= acknowledgeSync;
            weightValidSync          <= allWeightDataValid;
            weightValidSyncDelay     <= weightValidSync;
            inputValidSync          <= allInputDataValid;
            inputValidSyncDelay     <= inputValidSync;

            if (resultReady) begin
                request <= ~request;

                for (int i = 0; i < N; i++) begin
                    spiData[i] <= resultData[i];
                end
            end

            if (weightValid && weightFifoReady) begin
                weightAccepted  <= ~weightAccepted;
                weightSent      <= 1;
            end else if (!weightValidSyncDelay) begin
                weightSent      <= 0;
            end

            if (inputValid && inputFifoReady) begin
                inputAccepted  <= ~inputAccepted;
                inputSent      <= 1;
            end else if (!inputValidSyncDelay) begin
                inputSent      <= 0;
            end
        end
    end

    always_ff @(posedge sclk) begin
        if (!rst_n) begin
            requestSync                     <= 0;
            requestSyncDelay                <= 0;
            acknowledge                     <= 0;
            weightAcceptedSync              <= 0;
            weightAcceptedSyncDelay         <= 0;
            weightAcceptedSeen              <= 0;
            inputAcceptedSync          <= 0;
            inputAcceptedSyncDelay     <= 0;
            inputAcceptedSeen          <= 0;
        end else begin
            requestSync                     <= request;
            requestSyncDelay                <= requestSync;
            weightAcceptedSync              <= weightAccepted;
            weightAcceptedSyncDelay         <= weightAcceptedSync;
            inputAcceptedSync          <= inputAccepted;
            inputAcceptedSyncDelay     <= inputAcceptedSync;

            if (requestSyncDelay != acknowledge && allSpiReady) begin
                acknowledge <= requestSyncDelay;
            end

            if (weightAcceptedSyncDelay != weightAcceptedSeen) begin
                weightAcceptedSeen <= weightAcceptedSyncDelay;
            end

            if (inputAcceptedSyncDelay != inputAcceptedSeen) begin
                inputAcceptedSeen <= inputAcceptedSyncDelay;
            end
        end
    end

    nnAccelerator #(
        .WIDTH(WIDTH), .N(N),
        .FRACTION_BITS(FRACTION_BITS),
        .REDUCTION_WEIGHT_WIDTH(REDUCTION_WEIGHT_WIDTH),
        .INPUT_FIFO_DEPTH(INPUT_FIFO_DEPTH),
        .OUTPUT_FIFO_DEPTH(OUTPUT_FIFO_DEPTH)
    ) accelerator (
        .clk(clk), .rst_n(rst_n),
        .weightData(weightData), .weightValid(weightValid), .weightReady(weightFifoReady),
        .inputData(inputData), .targetData('0),
        .trainingEnable(trainingEnable), .inputValid(inputValid),
        .inputReady(inputFifoReady), .resultData(resultData),
        .resultValid(resultValid), .resultReady(resultReady), .passThrough(passThrough),
        .reduceOutput(reduceOutput), .reductionWeight(reductionWeight),
        .loadReductionWeights(loadReductionWeights),
        .weightsLoaded(weightsLoaded),
        .reloadWeights(reloadWeights), .reloadReady(reloadReady)
    );

    genvar spiIndex;
    generate
        for (spiIndex = 0; spiIndex < N; spiIndex = spiIndex + 1) begin : spi_outputs
            SPI_Slave_Output_Module #(.WIDTH(PREDICTION_WIDTH)) spi (
                .rst_n(rst_n), .data_in(spiData[spiIndex]),
                .data_valid(requestSyncDelay != acknowledge),
                .cs_n(cs_n[spiIndex]), .sclk(sclk),
                .miso(miso[spiIndex]), .miso_valid(misoValid[spiIndex]),
                .ready(spiReady[spiIndex])
            );
            SPI_Slave_Input_Module #(.WIDTH(WIDTH)) weightSpi (
                .rst_n(rst_n), .mosi(weightMosi[spiIndex]),
                .cs_n(weightCs_n[spiIndex]), .sclk(sclk),
                .data_out(weightData[spiIndex]), .data_valid(weightDataValid[spiIndex]),
                .ready(weightAcceptedSyncDelay != weightAcceptedSeen)
            );
            SPI_Slave_Input_Module #(.WIDTH(WIDTH)) inputSpi (
                .rst_n(rst_n), .mosi(inputMosi[spiIndex]),
                .cs_n(inputCs_n[spiIndex]), .sclk(sclk),
                .data_out(inputData[spiIndex]), .data_valid(inputDataValid[spiIndex]),
                .ready(inputAcceptedSyncDelay != inputAcceptedSeen)
            );
        end
    endgenerate

endmodule
