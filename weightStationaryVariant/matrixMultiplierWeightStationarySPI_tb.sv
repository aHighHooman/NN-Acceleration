`timescale 1ns / 1ps

// SPI owns serial framing, clock-domain crossing, ordering, and backpressure.
// The single identity-matrix transaction is only a composition smoke check;
// accelerator arithmetic is covered by the Python golden references.
module matrixMultiplierWeightStationarySPI_tb;
    localparam int WIDTH = 8;
    localparam int N = 2;
    localparam int FRACTION_BITS = 4;
    localparam int SCALE = 1 << FRACTION_BITS;
    localparam int PREDICTION_WIDTH = 2*WIDTH + 2*$clog2(N);

    typedef logic signed [WIDTH-1:0] data_t;
    typedef logic signed [PREDICTION_WIDTH-1:0] result_t;

    logic clk, sclk, rst_n;
    logic weightReady, activationReady, weightsLoaded;
    logic reloadReady;
    logic cs_n[N], miso[N], misoValid[N];
    logic weightCs_n[N], weightMosi[N];
    logic activationCs_n[N], activationMosi[N];
    logic signed [7:0] reductionWeight[N];

    matrixMultiplierWeightStationarySPI #(
        .WIDTH(WIDTH), .N(N), .FRACTION_BITS(FRACTION_BITS)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .weightReady(weightReady), .activationReady(activationReady),
        .passThrough(1'b1), .reduceOutput(1'b0), .trainingEnable(1'b0),
        .reductionWeight(reductionWeight), .loadReductionWeights(1'b0),
        .weightsLoaded(weightsLoaded), .reloadWeights(1'b0),
        .reloadReady(reloadReady),
        .sclk(sclk), .cs_n(cs_n), .miso(miso), .misoValid(misoValid),
        .weightCs_n(weightCs_n), .weightMosi(weightMosi),
        .activationCs_n(activationCs_n), .activationMosi(activationMosi)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    // Deliberately asynchronous to clk.
    initial begin
        sclk = 1'b0;
        #2;
        forever #7 sclk = ~sclk;
    end

    initial begin
        data_t vector[N];

        rst_n = 1'b0;
        for (int lane = 0; lane < N; lane++) begin
            reductionWeight[lane] = '0;
            cs_n[lane] = 1'b1;
            weightCs_n[lane] = 1'b1;
            activationCs_n[lane] = 1'b1;
            weightMosi[lane] = 1'b0;
            activationMosi[lane] = 1'b0;
        end

        repeat (3) @(posedge sclk);
        repeat (3) @(posedge clk);
        rst_n = 1'b1;

        // The core loads rows in reverse host order. These serialized frames
        // install a Q4 identity matrix.
        vector[0] = 0;
        vector[1] = SCALE;
        send_weight_vector(vector);
        vector[0] = SCALE;
        vector[1] = 0;
        send_weight_vector(vector);
        wait(weightsLoaded);

        // Leave the first result unread so the second accepted vector must
        // remain ordered and stable behind the SPI output shifter.
        vector[0] = 2;
        vector[1] = -3;
        send_activation_vector(vector);
        vector[0] = 4;
        vector[1] = 5;
        send_activation_vector(vector);

        check_output_backpressure(4*SCALE, 5*SCALE);
        expect_serialized_row("identity smoke row 0", 2*SCALE, -3*SCALE);
        expect_serialized_row("identity smoke row 1", 4*SCALE, 5*SCALE);

        $display("PASS: asynchronous SPI serialization, CDC, ordering, backpressure, and numerical smoke check completed.");
        $finish;
    end

    initial begin
        #100_000;
        $fatal(1, "FAIL: asynchronous-clock SPI smoke test timed out.");
    end

    task send_weight_vector(input data_t vector[N]);
        wait(weightReady);
        @(negedge sclk);
        for (int lane = 0; lane < N; lane++)
            weightCs_n[lane] = 1'b0;
        for (int bitIndex = WIDTH-1; bitIndex >= 0; bitIndex--) begin
            for (int lane = 0; lane < N; lane++)
                weightMosi[lane] = vector[lane][bitIndex];
            @(posedge sclk);
            @(negedge sclk);
        end
        for (int lane = 0; lane < N; lane++)
            weightCs_n[lane] = 1'b1;
    endtask

    task send_activation_vector(input data_t vector[N]);
        wait(activationReady);
        @(negedge sclk);
        for (int lane = 0; lane < N; lane++)
            activationCs_n[lane] = 1'b0;
        for (int bitIndex = WIDTH-1; bitIndex >= 0; bitIndex--) begin
            for (int lane = 0; lane < N; lane++)
                activationMosi[lane] = vector[lane][bitIndex];
            @(posedge sclk);
            @(negedge sclk);
        end
        for (int lane = 0; lane < N; lane++)
            activationCs_n[lane] = 1'b1;
    endtask

    task check_output_backpressure(input result_t expected0,
                                   input result_t expected1);
        wait(dut.request != dut.acknowledgeSyncDelay);
        if (dut.spiData[0] !== expected0 || dut.spiData[1] !== expected1)
            $fatal(1, "FAIL: buffered row got [%0d,%0d], expected [%0d,%0d].",
                   dut.spiData[0], dut.spiData[1], expected0, expected1);
        repeat (5) begin
            @(posedge clk);
            if (dut.request == dut.acknowledgeSyncDelay ||
                dut.spiData[0] !== expected0 || dut.spiData[1] !== expected1)
                $fatal(1, "FAIL: buffered result changed or escaped under SPI backpressure.");
        end
    endtask

    task expect_serialized_row(input string label,
                               input result_t expected0,
                               input result_t expected1);
        result_t actual[N];
        bit available;

        available = 1'b0;
        while (!available) begin
            @(negedge sclk);
            for (int lane = 0; lane < N; lane++)
                cs_n[lane] = 1'b0;
            #1;
            available = 1'b1;
            for (int lane = 0; lane < N; lane++)
                available &= misoValid[lane];
            if (!available)
                for (int lane = 0; lane < N; lane++)
                    cs_n[lane] = 1'b1;
        end

        for (int bitIndex = PREDICTION_WIDTH-1; bitIndex >= 0; bitIndex--) begin
            @(posedge sclk);
            for (int lane = 0; lane < N; lane++) begin
                if (!misoValid[lane])
                    $fatal(1, "FAIL: MISO lane %0d lost framing.", lane);
                actual[lane][bitIndex] = miso[lane];
            end
        end
        @(negedge sclk);
        for (int lane = 0; lane < N; lane++)
            cs_n[lane] = 1'b1;

        if (actual[0] !== expected0 || actual[1] !== expected1)
            $fatal(1, "FAIL: %s got [%0d,%0d], expected [%0d,%0d].",
                   label, actual[0], actual[1], expected0, expected1);
        $display("PASS: %s = [%0d,%0d]", label, actual[0], actual[1]);
    endtask
endmodule
