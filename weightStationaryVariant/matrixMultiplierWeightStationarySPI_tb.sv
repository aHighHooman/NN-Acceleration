`timescale 1ns / 1ps

module matrixMultiplierWeightStationarySPI_tb;
    localparam int WIDTH = 8;
    localparam int N = 2;
    localparam int RESULT_WIDTH = 2*WIDTH + $clog2(N);

    typedef logic signed [WIDTH-1:0] data_t;
    typedef logic signed [RESULT_WIDTH-1:0] result_t;

    logic clk, sclk, rst_n;
    logic weightReady, activationReady, passThrough;
    logic weightsLoaded, reloadWeights, reloadReady;
    logic cs_n[N], miso[N], misoValid[N];
    logic weightCs_n[N], weightMosi[N];
    logic activationCs_n[N], activationMosi[N];

    matrixMultiplierWeightStationarySPI #(
        .WIDTH(WIDTH),
        .N(N)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .weightReady(weightReady), .activationReady(activationReady),
        .passThrough(passThrough), .weightsLoaded(weightsLoaded),
        .reloadWeights(reloadWeights), .reloadReady(reloadReady),
        .sclk(sclk), .cs_n(cs_n), .miso(miso), .misoValid(misoValid),
        .weightCs_n(weightCs_n), .weightMosi(weightMosi),
        .activationCs_n(activationCs_n), .activationMosi(activationMosi)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        sclk = 1'b0;
        #2;
        forever #7 sclk = ~sclk;
    end

    initial begin
        data_t weight_vector[N], activation_vector[N];

        rst_n = 1'b0;
        passThrough = 1'b1;
        reloadWeights = 1'b0;
        for (int lane = 0; lane < N; lane++) begin
            cs_n[lane] = 1'b1;
            weightCs_n[lane] = 1'b1;
            activationCs_n[lane] = 1'b1;
            weightMosi[lane] = 1'b0;
            activationMosi[lane] = 1'b0;
        end

        repeat (3) @(posedge sclk);
        repeat (3) @(posedge clk);
        rst_n = 1'b1;

        // Identity weights are loaded from the final row to the first row.
        weight_vector[0] = 0;
        weight_vector[1] = 1;
        send_weight_vector(weight_vector);

        weight_vector[0] = 1;
        weight_vector[1] = 0;
        send_weight_vector(weight_vector);
        wait(weightsLoaded);

        activation_vector[0] = 2;
        activation_vector[1] = -3;
        send_activation_vector(activation_vector);

        activation_vector[0] = 4;
        activation_vector[1] = 5;
        send_activation_vector(activation_vector);

        expect_result_row("row 0", 2, -3);
        expect_result_row("row 1", 4, 5);

        passThrough = 1'b0;

        activation_vector[0] = -6;
        activation_vector[1] = 7;
        send_activation_vector(activation_vector);

        activation_vector[0] = 8;
        activation_vector[1] = -9;
        send_activation_vector(activation_vector);

        expect_result_row("ReLU row 0", 0, 7);
        expect_result_row("ReLU row 1", 8, 0);

        $display("PASS: asynchronous-clock SPI integration test completed.");
        $finish;
    end

    initial begin
        #100_000;
        $fatal(1, "FAIL: asynchronous-clock SPI integration test timed out.");
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

    task expect_result_row(
        input string label,
        input result_t expected0,
        input result_t expected1
    );
        result_t actual[N];
        bit result_available;

        result_available = 1'b0;

        // Poll only through top-level signals. Failed polls raise CS again before
        // the next SCLK edge, so no result bit is consumed.
        while (!result_available) begin
            @(negedge sclk);
            for (int lane = 0; lane < N; lane++) cs_n[lane] = 1'b0;
            #1;
            result_available = 1'b1;
            for (int lane = 0; lane < N; lane++)
                result_available &= misoValid[lane];
            if (!result_available)
                for (int lane = 0; lane < N; lane++) cs_n[lane] = 1'b1;
        end

        for (int bitIndex = RESULT_WIDTH-1; bitIndex >= 0; bitIndex--) begin
            @(posedge sclk);
            for (int lane = 0; lane < N; lane++) begin
                assert(misoValid[lane])
                    else $fatal(1, "FAIL: MISO lane %0d was not valid.", lane);
                actual[lane][bitIndex] = miso[lane];
            end
        end

        @(negedge sclk);
        for (int lane = 0; lane < N; lane++) cs_n[lane] = 1'b1;

        if (actual[0] !== expected0 || actual[1] !== expected1)
            $fatal(1, "FAIL: %s got [%0d, %0d], expected [%0d, %0d].",
                   label, actual[0], actual[1], expected0, expected1);
        $display("PASS: %s = [%0d, %0d]", label, actual[0], actual[1]);
    endtask

endmodule
