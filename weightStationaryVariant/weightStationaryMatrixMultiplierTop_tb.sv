`timescale 1ns / 1ps

// Check SPI framing, CDC, ordering, and backpressure with an identity-matrix
// smoke test; Python references cover accelerator arithmetic.
module weightStationaryMatrixMultiplierTop_tb;
    localparam int WIDTH = 8;
    localparam int N = 2;
    localparam int FRACTION_BITS = 4;
    localparam int SCALE = 1 << FRACTION_BITS;
    localparam int PREDICTION_WIDTH = 2*WIDTH + 2*$clog2(N);

    typedef logic signed [WIDTH-1:0] data_t;
    typedef logic signed [PREDICTION_WIDTH-1:0] result_t;

    logic clk, sclk, rst_n;
    logic weightReady, inputReady, weightsLoaded;
    logic reloadReady;
    logic cs_n[N], miso[N], misoValid[N];
    logic weightCs_n[N], weightMosi[N];
    logic inputCs_n[N], inputMosi[N];
    logic signed [7:0] reductionWeight[N];

    weightStationaryMatrixMultiplierTop #(
        .WIDTH(WIDTH), .N(N), .FRACTION_BITS(FRACTION_BITS)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .weightReady(weightReady), .inputReady(inputReady),
        .passThrough(1'b1), .reduceOutput(1'b0), .trainingEnable(1'b0),
        .reductionWeight(reductionWeight), .loadReductionWeights(1'b0),
        .weightsLoaded(weightsLoaded), .reloadWeights(1'b0),
        .reloadReady(reloadReady),
        .sclk(sclk), .cs_n(cs_n), .miso(miso), .misoValid(misoValid),
        .weightCs_n(weightCs_n), .weightMosi(weightMosi),
        .inputCs_n(inputCs_n), .inputMosi(inputMosi)
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
            inputCs_n[lane] = 1'b1;
            weightMosi[lane] = 1'b0;
            inputMosi[lane] = 1'b0;
        end

        repeat (3) @(posedge sclk);
        repeat (3) @(posedge clk);
        rst_n = 1'b1;

        // The core loads rows in reverse host order. These serialized frames
        // install a Q4 identity matrix.
        vector[0] = 0;
        vector[1] = SCALE;
        send_partial_weight_frame(vector);
        reset_during_weight_frame(vector);
        send_weight_vector_with_extra_clocks(vector, 1);
        vector[0] = SCALE;
        vector[1] = 0;
        send_weight_vector(vector);
        wait(weightsLoaded);

        // Leave the first result unread so the second accepted vector must
        // remain ordered and stable behind the SPI output shifter.
        vector[0] = 2;
        vector[1] = -3;
        send_input_vector(vector);
        vector[0] = 4;
        vector[1] = 5;
        send_input_vector(vector);

        check_output_backpressure(4*SCALE, 5*SCALE);
        expect_serialized_row("identity smoke row 0", 2*SCALE, -3*SCALE, 3);
        expect_serialized_row("identity smoke row 1", 4*SCALE, 5*SCALE, -1);

        $display("PASS: asynchronous SPI framing, reset recovery, CDC, ordering, backpressure, and numerical smoke check completed.");
        $finish;
    end

    initial begin
        #100_000;
        $fatal(1, "FAIL: asynchronous-clock SPI smoke test timed out.");
    end

    task send_weight_vector(input data_t vector[N]);
        wait(weightReady);
        send_serialized_input(vector, 1'b1, 0);
    endtask

    task send_weight_vector_with_extra_clocks(input data_t vector[N],
                                               input int extraClocks);
        wait(weightReady);
        send_serialized_input(vector, 1'b1, extraClocks);
    endtask

    task send_input_vector(input data_t vector[N]);
        wait(inputReady);
        send_serialized_input(vector, 1'b0, 0);
    endtask

    task send_partial_weight_frame(input data_t vector[N]);
        wait(weightReady);
        @(negedge sclk);
        for (int lane = 0; lane < N; lane++)
            weightCs_n[lane] = 1'b0;
        for (int bitIndex = WIDTH-1; bitIndex >= WIDTH/2; bitIndex--) begin
            for (int lane = 0; lane < N; lane++)
                weightMosi[lane] = vector[lane][bitIndex];
            @(posedge sclk);
            @(negedge sclk);
        end
        for (int lane = 0; lane < N; lane++)
            weightCs_n[lane] = 1'b1;
    endtask

    task reset_during_weight_frame(input data_t vector[N]);
        wait(weightReady);
        @(negedge sclk);
        for (int lane = 0; lane < N; lane++)
            weightCs_n[lane] = 1'b0;
        for (int bitIndex = WIDTH-1; bitIndex >= WIDTH/2; bitIndex--) begin
            for (int lane = 0; lane < N; lane++)
                weightMosi[lane] = vector[lane][bitIndex];
            @(posedge sclk);
            @(negedge sclk);
        end
        rst_n = 1'b0;
        repeat (2) @(posedge sclk);
        repeat (2) @(posedge clk);
        for (int lane = 0; lane < N; lane++)
            weightCs_n[lane] = 1'b1;
        @(negedge sclk) rst_n = 1'b1;
    endtask

    task send_serialized_input(input data_t vector[N], input bit isWeight,
                               input int extraClocks);
        @(negedge sclk);
        for (int lane = 0; lane < N; lane++) begin
            if (isWeight)
                weightCs_n[lane] = 1'b0;
            else
                inputCs_n[lane] = 1'b0;
        end
        for (int bitIndex = WIDTH-1; bitIndex >= 0; bitIndex--) begin
            for (int lane = 0; lane < N; lane++) begin
                if (isWeight)
                    weightMosi[lane] = vector[lane][bitIndex];
                else
                    inputMosi[lane] = vector[lane][bitIndex];
            end
            @(posedge sclk);
            @(negedge sclk);
        end
        repeat (extraClocks) begin
            for (int lane = 0; lane < N; lane++) begin
                if (isWeight)
                    weightMosi[lane] = 1'b0;
                else
                    inputMosi[lane] = 1'b0;
            end
            @(posedge sclk);
            @(negedge sclk);
        end
        for (int lane = 0; lane < N; lane++) begin
            if (isWeight)
                weightCs_n[lane] = 1'b1;
            else
                inputCs_n[lane] = 1'b1;
        end
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
                               input result_t expected1,
                               input int pauseAfterBits);
        result_t actual[N];
        bit available;
        int bitsReceived;

        available = 1'b0;
        bitsReceived = 0;
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
            bitsReceived++;
            if (bitsReceived == pauseAfterBits) begin
                @(negedge sclk);
                for (int lane = 0; lane < N; lane++)
                    cs_n[lane] = 1'b1;
                repeat (2) @(posedge sclk);
                @(negedge sclk);
                for (int lane = 0; lane < N; lane++)
                    cs_n[lane] = 1'b0;
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
