module matrixMultiplierWeightStationarySPI #(
    parameter int WIDTH = 16,
    parameter int N = 3,
    parameter int INPUT_FIFO_DEPTH = 2*N,
    parameter int OUTPUT_FIFO_DEPTH = 2*N
)(
    input  logic                    clk,
    input  logic                    rst_n,
    output logic                    weightReady,
    output logic                    activationReady,
    input  logic                    passThrough,
    output logic                    weightsLoaded,
    input  logic                    reloadWeights,
    output logic                    reloadReady,
    input  logic                    sclk,
    input  logic                    cs_n [N],
    output logic                    miso [N],
    output logic                    misoValid [N],
    input  logic                    weightCs_n [N],
    input  logic                    weightMosi [N],
    input  logic                    activationCs_n [N],
    input  logic                    activationMosi [N]
);

    localparam int RESULT_WIDTH = 2*WIDTH + $clog2(N);

    logic signed [WIDTH-1:0] weightData[N], activationData[N];
    logic weightValid, activationValid;
    logic weightFifoReady, activationFifoReady;
    logic weightDataValid[N], activationDataValid[N];
    logic allWeightDataValid, allActivationDataValid, allWeightSpiReady, allActivationSpiReady;
    logic weightValidSync, weightValidSyncDelay, weightSent;
    logic activationValidSync, activationValidSyncDelay, activationSent;
    logic weightAccepted, weightAcceptedSync, weightAcceptedSyncDelay, weightAcceptedSeen;
    logic activationAccepted, activationAcceptedSync, activationAcceptedSyncDelay, activationAcceptedSeen;
    logic signed [RESULT_WIDTH-1:0] resultData[N], spiData[N];
    logic resultValid, resultReady;
    logic spiReady[N], allSpiReady;
    logic request, requestSync, requestSyncDelay, acknowledge;
    logic acknowledgeSync, acknowledgeSyncDelay;

    assign resultReady      = resultValid && (request == acknowledgeSyncDelay);
    assign weightValid      = weightValidSyncDelay && !weightSent;
    assign activationValid  = activationValidSyncDelay && !activationSent;
    assign weightReady      = weightFifoReady && allWeightSpiReady;
    assign activationReady  = activationFifoReady && allActivationSpiReady;

    always_comb begin
        allSpiReady             = 1;
        allWeightDataValid      = 1;
        allActivationDataValid  = 1;
        allWeightSpiReady       = 1;
        allActivationSpiReady   = 1;

        for (int i = 0; i < N; i++) begin
            allSpiReady             &= spiReady[i];
            allWeightDataValid      &= weightDataValid[i];
            allActivationDataValid  &= activationDataValid[i];
            allWeightSpiReady       &= !weightDataValid[i];
            allActivationSpiReady   &= !activationDataValid[i];
        end

    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            request                     <= 0;
            acknowledgeSync             <= 0;
            acknowledgeSyncDelay        <= 0;
            weightValidSync             <= 0;
            weightValidSyncDelay        <= 0;
            activationValidSync         <= 0;
            activationValidSyncDelay    <= 0;
            weightAccepted              <= 0;
            activationAccepted          <= 0;
            weightSent                  <= 0;
            activationSent              <= 0;

            for (int i = 0; i < N; i++) begin
                spiData[i] <= 0;
            end
        end else begin
            acknowledgeSync          <= acknowledge;
            acknowledgeSyncDelay     <= acknowledgeSync;
            weightValidSync          <= allWeightDataValid;
            weightValidSyncDelay     <= weightValidSync;
            activationValidSync      <= allActivationDataValid;
            activationValidSyncDelay <= activationValidSync;

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

            if (activationValid && activationFifoReady) begin
                activationAccepted  <= ~activationAccepted;
                activationSent      <= 1;
            end else if (!activationValidSyncDelay) begin
                activationSent      <= 0;
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
            activationAcceptedSync          <= 0;
            activationAcceptedSyncDelay     <= 0;
            activationAcceptedSeen          <= 0;
        end else begin
            requestSync                     <= request;
            requestSyncDelay                <= requestSync;
            weightAcceptedSync              <= weightAccepted;
            weightAcceptedSyncDelay         <= weightAcceptedSync;
            activationAcceptedSync          <= activationAccepted;
            activationAcceptedSyncDelay     <= activationAcceptedSync;

            if (requestSyncDelay != acknowledge && allSpiReady) begin
                acknowledge <= requestSyncDelay;
            end

            if (weightAcceptedSyncDelay != weightAcceptedSeen) begin
                weightAcceptedSeen <= weightAcceptedSyncDelay;
            end

            if (activationAcceptedSyncDelay != activationAcceptedSeen) begin
                activationAcceptedSeen <= activationAcceptedSyncDelay;
            end
        end
    end

    matrixMultiplierWeightStationary #(
        .WIDTH(WIDTH), .N(N),
        .INPUT_FIFO_DEPTH(INPUT_FIFO_DEPTH),
        .OUTPUT_FIFO_DEPTH(OUTPUT_FIFO_DEPTH)
    ) accelerator (
        .clk(clk), .rst_n(rst_n),
        .weightData(weightData), .weightValid(weightValid), .weightReady(weightFifoReady),
        .activationData(activationData), .activationValid(activationValid),
        .activationReady(activationFifoReady), .resultData(resultData),
        .resultValid(resultValid), .resultReady(resultReady), .passThrough(passThrough),
        .resultLast(), .weightsLoaded(weightsLoaded),
        .reloadWeights(reloadWeights), .reloadReady(reloadReady)
    );

    genvar spiIndex;
    generate
        for (spiIndex = 0; spiIndex < N; spiIndex = spiIndex + 1) begin : spi_outputs
            SPI_Slave_Output_Module #(.WIDTH(RESULT_WIDTH)) spi (
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
            SPI_Slave_Input_Module #(.WIDTH(WIDTH)) activationSpi (
                .rst_n(rst_n), .mosi(activationMosi[spiIndex]),
                .cs_n(activationCs_n[spiIndex]), .sclk(sclk),
                .data_out(activationData[spiIndex]), .data_valid(activationDataValid[spiIndex]),
                .ready(activationAcceptedSyncDelay != activationAcceptedSeen)
            );
        end
    endgenerate

endmodule
