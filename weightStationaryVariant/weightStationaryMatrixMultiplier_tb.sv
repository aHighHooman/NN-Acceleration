`timescale 1ns / 1ps

// Independent dot-product scoreboard checks every accepted vector. Post-edge
// timing checks distinguish the AD result register from an extra output register.
module weightStationaryMatrixMultiplier_testcase #(
    parameter int WIDTH = 16,
    parameter int N = 3
) (output logic done);
    localparam int RESULT_WIDTH = 2*WIDTH + $clog2(N);
    localparam int MAX_SAMPLES = 1024;
    localparam int RANDOM_SAMPLES = 64;
`ifdef CONVENTIONAL_BASELINE
    localparam int RAW_LATENCY = 2*N;
`else
    localparam int RAW_LATENCY = N+1;
`endif
    typedef logic signed [WIDTH-1:0] data_t;
    typedef logic signed [RESULT_WIDTH-1:0] result_t;
    typedef data_t matrix_t[N][N];
    logic clk = 0, rst_n;
    data_t weightData[N], inputData[N];
    logic weightValid, weightReady, inputValid, inputReady;
    logic signed [1:0] rowDirection[N], columnDirection[N];
    result_t resultData[N];
    logic resultValid, resultReady, weightsLoaded, reloadWeights, reloadReady;
    logic latencyChecking;
    matrix_t currentWeights;
    data_t randomInputs[RANDOM_SAMPLES][N];
    wire signed [WIDTH-1:0] residentWeights[N][N];
    result_t expected[MAX_SAMPLES][N];
    int acceptanceCycle[MAX_SAMPLES];
    int cycle, acceptedCount, retiredCount;

    weightStationaryMatrixMultiplier #(.WIDTH(WIDTH), .N(N)) dut (
        .clk(clk), .rst_n(rst_n),
        .weightData(weightData), .weightValid(weightValid), .weightReady(weightReady),
        .inputData(inputData), .inputValid(inputValid), .inputReady(inputReady),
        .rowDirection(rowDirection), .columnDirection(columnDirection),
        .matrixUpdateValid(1'b0), .arrayAdvance(),
        .resultData(resultData), .resultValid(resultValid), .resultReady(resultReady),
        .weightsLoaded(weightsLoaded), .reloadWeights(reloadWeights), .reloadReady(reloadReady)
    );
    always #5 clk = ~clk;

    for (genvar r = 0; r < N; r++) begin : inspect_row
        for (genvar c = 0; c < N; c++) begin : inspect_col
            assign residentWeights[r][c] = dut.systolicArray.row_loop[r].col_loop[c].pe.weightReg;
            // Shift registers stage weights; capture commits all PEs together.
`ifndef CONVENTIONAL_BASELINE
            always @(posedge clk) begin : check_load_capture
                data_t oldWeight;
                if (rst_n && dut.loadingWeights && !dut.captureWeights) begin
                    oldWeight = residentWeights[r][c];
                    #1;
                    if (residentWeights[r][c] !== oldWeight)
                        $fatal(1, "N=%0d PE[%0d][%0d] weight changed before capture", N, r, c);
                end
            end
`endif
        end
    end

    always @(posedge clk) begin : scoreboard
        bit blockedBefore;
        result_t heldResult[N];
        cycle = cycle + 1;
        blockedBefore = rst_n && resultValid && !resultReady;
        for (int lane = 0; lane < N; lane++) heldResult[lane] = resultData[lane];
        if (!rst_n) begin
            acceptedCount = 0;
            retiredCount = 0;
        end else begin
            if (resultValid && resultReady) begin
                if (retiredCount >= acceptedCount)
                    $fatal(1, "N=%0d unsolicited result at cycle %0d", N, cycle);
                for (int lane = 0; lane < N; lane++)
                    if (resultData[lane] !== expected[retiredCount][lane])
                        $fatal(1, "N=%0d sample%0d lane%0d got%0d expected%0d",
                            N, retiredCount, lane, resultData[lane], expected[retiredCount][lane]);
                retiredCount++;
            end
            if (inputValid && inputReady) begin
                if (acceptedCount >= MAX_SAMPLES) $fatal(1, "test scoreboard overflow");
                for (int col = 0; col < N; col++) begin
                    longint signed sum;
                    sum = 0;
                    for (int row = 0; row < N; row++)
                        sum += $signed(inputData[row]) * $signed(currentWeights[row][col]);
                    expected[acceptedCount][col] = result_t'(sum);
                end
                acceptanceCycle[acceptedCount] = cycle;
                acceptedCount++;
            end
            #1;
            if (!weightsLoaded && inputReady)
                $fatal(1, "N=%0d input ready before the new weight bank captured", N);
            if (blockedBefore) begin
                if (!resultValid) $fatal(1, "N=%0d valid dropped during backpressure", N);
                for (int lane = 0; lane < N; lane++)
                    if (resultData[lane] !== heldResult[lane])
                        $fatal(1, "N=%0d stalled result changed lane%0d", N, lane);
            end
            if (latencyChecking && resultValid) begin
                if (retiredCount >= acceptedCount || cycle - acceptanceCycle[retiredCount] != RAW_LATENCY)
                    $fatal(1, "N=%0d result latency got%0d expected%0d clocks",
                        N, cycle - acceptanceCycle[retiredCount], RAW_LATENCY);
            end
        end
    end

    task automatic load_weights(input matrix_t weights, input bit bubbles);
        int guard, finalHostCycle;
        currentWeights = weights;
        for (int row = N-1; row >= 0; row--) begin
            @(negedge clk);
            guard = 0;
            while (!weightReady) begin
                @(negedge clk);
                if (++guard > 100) $fatal(1, "N=%0d weight ready timeout", N);
            end
            weightValid = 1;
            for (int col = 0; col < N; col++) weightData[col] = weights[row][col];
            @(posedge clk); #1;
            if (row == 0) finalHostCycle = cycle;
            if (bubbles && row != 0) begin
                @(negedge clk) weightValid = 0;
                repeat (2) @(posedge clk);
            end
        end
        @(negedge clk) weightValid = 0;
        guard = 0;
        while (!weightsLoaded) begin
            @(posedge clk); #1;
            if (++guard > N+3) $fatal(1, "N=%0d capture timeout", N);
        end
`ifndef CONVENTIONAL_BASELINE
        if (cycle - finalHostCycle != N+1)
            $fatal(1, "N=%0d loader used%0d clocks after final host row; expected%0d",
                N, cycle-finalHostCycle, N+1);
`endif
        for (int row = 0; row < N; row++)
            for (int col = 0; col < N; col++)
                if (residentWeights[row][col] !== weights[row][col])
                    $fatal(1, "N=%0d two-ended load PE[%0d][%0d] got%0d expected%0d",
                        N, row, col, residentWeights[row][col], weights[row][col]);
    endtask

    task automatic reload();
        int guard, acceptedBefore;
        guard = 0;
        @(negedge clk);
        while (!reloadReady) begin
            @(negedge clk);
            if (++guard > 100) $fatal(1, "N=%0d reload never became ready", N);
        end
        acceptedBefore = acceptedCount;
        // A reload is an epoch barrier. If idle input and reload are offered
        // together, no old-bank sample may sneak through on the reload edge.
        reloadWeights = 1;
`ifndef CONVENTIONAL_BASELINE
        inputValid = 1;
        #1;
        if (inputReady) $fatal(1, "N=%0d reload did not take priority over input", N);
`endif
        @(posedge clk); #1;
`ifndef CONVENTIONAL_BASELINE
        if (acceptedCount != acceptedBefore)
            $fatal(1, "N=%0d accepted a sample on the reload edge", N);
`endif
        if (weightsLoaded) $fatal(1, "N=%0d reload did not clear loaded flag", N);
        @(negedge clk) begin reloadWeights = 0; inputValid = 0; end
    endtask

    // Modes: continuous, bubbles, backpressure, signed-minimum inputs, seeded
    // random inputs with simultaneous bubbles and backpressure.
    // A blocked input remains stable until acceptance.
    task automatic run_stream(input int count, input int mode);
        int sent, targetCount, phase, guard;
        bit accepted, holdInput;
        sent = 0; phase = 0; guard = 0;
        holdInput = 0;
        targetCount = acceptedCount + count;
        latencyChecking = mode != 2 && mode != 4;
        while (sent < count) begin
            @(negedge clk);
            resultReady = (mode != 2 && mode != 4) || ((phase % (N+5)) >= 4);
            inputValid = holdInput || !((mode == 1 && phase % 3 == 1) ||
                                        (mode == 4 && phase % 4 == 1));
            for (int lane = 0; lane < N; lane++)
                inputData[lane] = (mode == 4) ? randomInputs[sent][lane] :
                    (mode == 3) ? data_t'({1'b1, {(WIDTH-1){1'b0}}}) :
                    data_t'((sent+1)*(lane+2) * ((sent+lane)%2 ? -1 : 1));
            #1;
            accepted = inputValid && inputReady;
            @(posedge clk); #2;
            holdInput = inputValid && !accepted;
            if (accepted) sent++;
            phase++;
            if (++guard > 20*count+100) $fatal(1, "N=%0d stream acceptance timeout", N);
        end
        @(negedge clk);
        inputValid = 0;
        resultReady = 1;
        guard = 0;
        while (retiredCount != targetCount) begin
            @(posedge clk); #2;
            if (++guard > 4*N+20) $fatal(1, "N=%0d stream drain timeout", N);
        end
        @(negedge clk);
        latencyChecking = 0;
        repeat (N+3) begin
            @(posedge clk); #2;
            if (resultValid) $fatal(1, "N=%0d duplicate output after drain", N);
        end
    endtask

    // Each parameter instance has a local PRNG, independent of scheduling and
    // the simulator's global random state. Baseline and inward runs replay it.
    function automatic logic [31:0] random_next(input logic [31:0] state);
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    endfunction

    initial begin : scenarios
        matrix_t weights;
        logic [31:0] randomState;
        done = 0; rst_n = 0; cycle = 0;
        acceptedCount = 0; retiredCount = 0;
        latencyChecking = 0; weightValid = 0; inputValid = 0;
        resultReady = 1; reloadWeights = 0;
        for (int lane = 0; lane < N; lane++) begin
            weightData[lane] = 0; inputData[lane] = 0;
            rowDirection[lane] = 0; columnDirection[lane] = 0;
        end
        repeat (3) @(posedge clk);
        @(negedge clk) rst_n = 1;
        // Distinct entries expose permutations hidden by uniform/identity loads.
        for (int row = 0; row < N; row++)
            for (int col = 0; col < N; col++)
                weights[row][col] = data_t'((1+row*N+col) * ((row+col)%2 ? -1 : 1));
        load_weights(weights, 1);
        run_stream(4*N+3, 0);
        run_stream(3*N+1, 1);
        run_stream(4*N+5, 2);
        reload();
        randomState = 32'h9e37_79b9 ^ N;
        for (int row = 0; row < N; row++)
            for (int col = 0; col < N; col++) begin
                randomState = random_next(randomState);
                weights[row][col] = data_t'(randomState);
            end
        for (int sampleIndex = 0; sampleIndex < RANDOM_SAMPLES; sampleIndex++)
            for (int lane = 0; lane < N; lane++) begin
                randomState = random_next(randomState);
                randomInputs[sampleIndex][lane] = data_t'(randomState);
            end
        load_weights(weights, 1);
        run_stream(RANDOM_SAMPLES, 4);
        reload();
        for (int row = 0; row < N; row++)
            for (int col = 0; col < N; col++) weights[row][col] = {1'b1, {(WIDTH-1){1'b0}}};
        load_weights(weights, 0);
        run_stream(3*N+1, 3);
        reload();
        for (int row = 0; row < N; row++)
            for (int col = 0; col < N; col++) weights[row][col] = {1'b0, {(WIDTH-1){1'b1}}};
        load_weights(weights, 1);
        run_stream(3*N+1, 3);
        $display("PASS: N=%0d loading, %0d-clock latency, streaming, seeded random, bubbles, stalls, reload, signed edges (%0d samples)", N, RAW_LATENCY, retiredCount);
        done = 1;
    end
endmodule

module weightStationaryMatrixMultiplier_tb;
    logic done2, done3, done4, done5, done8;
    weightStationaryMatrixMultiplier_testcase #(.N(2)) test_2x2(.done(done2));
    weightStationaryMatrixMultiplier_testcase #(.N(3)) test_3x3(.done(done3));
    weightStationaryMatrixMultiplier_testcase #(.N(4)) test_4x4(.done(done4));
    weightStationaryMatrixMultiplier_testcase #(.N(5)) test_5x5(.done(done5));
    weightStationaryMatrixMultiplier_testcase #(.N(8)) test_8x8(.done(done8));
    initial begin
        wait(done2 && done3 && done4 && done5 && done8);
        $display("PASS: N=2,3,4,5,8 matrix-engine suites completed.");
        $finish;
    end
    initial begin
        #100000;
        $fatal(1, "matrix-engine suite timeout");
    end
endmodule
