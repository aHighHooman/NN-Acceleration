`timescale 1ns / 1ps

// Exercise all PE roles, including the AD merge and loading suppression.
module multiplierBlockWeightUpdate_testcase #(
    parameter int ROLE = 0
) (input logic clk, output logic done);
    localparam int WIDTH = 4;
    localparam int RESULT_WIDTH = 2*WIDTH+2;
    logic rst_n, advance, loadWeight, captureWeight, updateWeight;
    logic signed [1:0] updateDirection;
    logic signed [WIDTH-1:0] activationIn, activationOut;
    logic signed [RESULT_WIDTH-1:0] psumIn, lowerPsumIn, psumOut;
    weightStationaryProcessingElement #(.WIDTH(WIDTH), .RESULT_WIDTH(RESULT_WIDTH), .ROLE(ROLE)) dut (
        .clk(clk), .rst_n(rst_n), .advance(advance), .loadWeight(loadWeight),
        .captureWeight(captureWeight), .updateWeight(updateWeight), .updateDirection(updateDirection),
        .activationIn(activationIn), .psumIn(psumIn), .lowerPsumIn(lowerPsumIn),
        .activationOut(activationOut), .psumOut(psumOut)
    );

    task automatic load_weight(input int value);
        @(negedge clk);
        advance = 1; loadWeight = 1; captureWeight = 0; updateWeight = 1;
        updateDirection = 1; psumIn = value;
        // Array boundaries force activation zero during loading. The AD lower
        // chain must still be excluded from its downward weight shift.
        activationIn = 0; lowerPsumIn = 123;
        @(posedge clk); #1;
        if (psumOut !== value || activationOut !== 0)
            $fatal(1, "ROLE=%0d MAC load path did not suppress computation", ROLE);
        @(negedge clk);
        loadWeight = 0; captureWeight = 1; psumIn = 0;
        @(posedge clk); #1;
        if (dut.weightReg !== value) $fatal(1, "ROLE=%0d capture failed", ROLE);
        @(negedge clk) begin captureWeight = 0; updateWeight = 0; lowerPsumIn = 0; end
    endtask

    task automatic apply_update(input int direction, input int expectedWeight);
        @(negedge clk); updateWeight = 1; updateDirection = direction;
        @(posedge clk); #1;
        if (dut.weightReg !== expectedWeight)
            $fatal(1, "ROLE=%0d saturated update got%0d expected%0d", ROLE, dut.weightReg, expectedWeight);
        @(negedge clk) updateWeight = 0;
    endtask

    initial begin : test_pe
        logic signed [WIDTH-1:0] heldAct, heldWeight;
        logic signed [RESULT_WIDTH-1:0] heldPsum;
        done = 0; rst_n = 0; advance = 1;
        loadWeight = 0; captureWeight = 0; updateWeight = 0;
        updateDirection = 0; activationIn = 0; psumIn = 0; lowerPsumIn = 0;
        repeat (3) @(posedge clk);
        @(negedge clk) rst_n = 1;
        load_weight(0);
        apply_update(1, 1); apply_update(0, 1); apply_update(-1, 0);
        load_weight(7); apply_update(1, 7);
        load_weight(-8); apply_update(-1, -8);
        load_weight(3);
        @(negedge clk);
        activationIn = -2; psumIn = 5; lowerPsumIn = -7;
        updateWeight = 1; updateDirection = 1;
        @(posedge clk); #1;
        if (psumOut !== ((ROLE == 1) ? -8 : -1) || dut.weightReg !== 4)
            $fatal(1, "ROLE=%0d update edge did not compute with old weight", ROLE);
        if (activationOut !== ((ROLE == 1) ? 0 : -2))
            $fatal(1, "ROLE=%0d activation forwarding incorrect", ROLE);
        @(negedge clk) updateWeight = 0;
        @(posedge clk); #1;
        if (psumOut !== ((ROLE == 1) ? -10 : -3))
            $fatal(1, "ROLE=%0d next sample did not use updated weight", ROLE);
        heldAct = activationOut; heldPsum = psumOut; heldWeight = dut.weightReg;
        @(negedge clk);
        advance = 0; captureWeight = 1; updateWeight = 1;
        activationIn = 7; psumIn = 79; lowerPsumIn = 91;
        repeat (3) begin
            @(posedge clk); #1;
            if (activationOut !== heldAct || psumOut !== heldPsum || dut.weightReg !== heldWeight)
                $fatal(1, "ROLE=%0d state changed while stalled", ROLE);
        end
        $display("PASS: PE ROLE=%0d loading, capture, signed MAC, saturation, old-weight edge, freeze", ROLE);
        done = 1;
    end
endmodule

module systolicWeightUpdateWave_testcase #(
    parameter int N = 3
) (input logic clk, output logic done);
    localparam int WIDTH = 8;
    localparam int RESULT_WIDTH = 2*WIDTH+$clog2(N);
    logic rst_n, advance, loadWeight, captureWeight, updateValid, actValid;
    logic signed [WIDTH-1:0] rowLeft[N], rowRight[N], colTop[N], colBottom[N];
    logic signed [1:0] rowDirection[N], columnDirection[N];
    logic signed [RESULT_WIDTH-1:0] result[N];
    logic resultValid[N], pipelineBusy;
    wire signed [WIDTH-1:0] resident[N][N];
    int expected[N][N];
    bit historyValid[N-1];
    int historyRow[N-1][N], historyCol[N-1][N];
    weightStationarySystolicArray #(.WIDTH(WIDTH), .N(N)) dut (
        .clk(clk), .rst_n(rst_n), .advance(advance), .loadWeight(loadWeight),
        .captureWeight(captureWeight), .rowDirection(rowDirection),
        .columnDirection(columnDirection), .updateValid(updateValid),
        .rowLeft(rowLeft), .rowRight(rowRight), .actValid(actValid),
        .colTop(colTop), .colBottom(colBottom), .result(result),
        .resultValid(resultValid), .pipelineBusy(pipelineBusy)
    );
    for (genvar r = 0; r < N; r++) begin : read_row
        for (genvar c = 0; c < N; c++) begin : read_col
            assign resident[r][c] = dut.row_loop[r].col_loop[c].pe.weightReg;
        end
    end

    task automatic load_zero();
        @(negedge clk);
        advance = 1; updateValid = 0; actValid = 0; loadWeight = 1;
        for (int lane = 0; lane < N; lane++) begin
            colTop[lane] = 0; colBottom[lane] = 0;
            rowLeft[lane] = 23; rowRight[lane] = -31;
        end
        repeat (N) @(posedge clk);
        @(negedge clk); loadWeight = 0; captureWeight = 1;
        @(posedge clk); #1;
        for (int r = 0; r < N; r++)
            for (int c = 0; c < N; c++) begin
                expected[r][c] = 0;
                if (resident[r][c] !== 0) $fatal(1, "N=%0d zero load corrupted", N);
            end
        for (int stage = 0; stage < N-1; stage++) historyValid[stage] = 0;
        @(negedge clk); captureWeight = 0;
    endtask

    // Independently model the inward update phase and ternary outer product.
    task automatic update_edge(input bit stepAdvance, input bit acceptPackage);
        @(negedge clk); advance = stepAdvance; updateValid = acceptPackage;
        if (stepAdvance) begin
            for (int r = 0; r < N; r++)
                for (int c = 0; c < N; c++) begin
                    int phase, rd, cd;
                    bit live;
                    phase = (r+c < N) ? r+c : 2*N-2-r-c;
                    live = phase == 0 ? acceptPackage : historyValid[phase-1];
                    rd = phase == 0 ? int'($signed(rowDirection[r])) : historyRow[phase-1][r];
                    cd = phase == 0 ? int'($signed(columnDirection[c])) : historyCol[phase-1][c];
                    if (live) expected[r][c] += rd * cd;
                end
            for (int stage = N-2; stage > 0; stage--) begin
                historyValid[stage] = historyValid[stage-1];
                for (int lane = 0; lane < N; lane++) begin
                    historyRow[stage][lane] = historyRow[stage-1][lane];
                    historyCol[stage][lane] = historyCol[stage-1][lane];
                end
            end
            historyValid[0] = acceptPackage;
            for (int lane = 0; lane < N; lane++) begin
                historyRow[0][lane] = $signed(rowDirection[lane]);
                historyCol[0][lane] = $signed(columnDirection[lane]);
            end
        end
        @(posedge clk); #1;
        for (int r = 0; r < N; r++)
            for (int c = 0; c < N; c++)
                if (resident[r][c] !== expected[r][c])
                    $fatal(1, "N=%0d inward update PE[%0d][%0d] got%0d expected%0d",
                        N, r, c, resident[r][c], expected[r][c]);
    endtask

    function automatic int sample_lane(input int sampleIndex, input int lane);
        return $signed(WIDTH'((sampleIndex+1)*(lane+1)*((sampleIndex+lane)%2 ? -1 : 1)));
    endfunction

    initial begin : test_array
        int sampleIndex, streams, sum, version;
        done = 0; rst_n = 0; advance = 1;
        loadWeight = 0; captureWeight = 0; updateValid = 0; actValid = 0;
        for (int lane = 0; lane < N; lane++) begin
            rowLeft[lane] = 0; rowRight[lane] = 0; colTop[lane] = 0; colBottom[lane] = 0;
            rowDirection[lane] = 0; columnDirection[lane] = 0;
        end
        repeat (3) @(posedge clk);
        @(negedge clk) rst_n = 1;
        load_zero();
        for (int lane = 0; lane < N; lane++) begin
            rowDirection[lane] = (lane%3)-1;
            columnDirection[lane] = ((lane+1)%3)-1;
        end
        update_edge(1, 1);
        for (int lane = 0; lane < N; lane++) begin rowDirection[lane] = 1; columnDirection[lane] = 1; end
        update_edge(1, 1);
        update_edge(0, 1); update_edge(0, 0);
        repeat (N) update_edge(1, 0);
        if (pipelineBusy) $fatal(1, "N=%0d update tail did not drain", N);

        // Every sample must use one generation, even while adjacent packages
        // update both corners and converge at the AD. Include stalls mid-wave.
        load_zero();
        streams = 3*N+2;
        for (int tick = 0; tick < streams+N; tick++) begin
            @(negedge clk);
            advance = 1; updateValid = tick < 2; actValid = tick < streams;
            for (int lane = 0; lane < N; lane++) begin
                rowDirection[lane] = 1; columnDirection[lane] = 1;
                rowLeft[lane] = tick >= lane && tick-lane < streams ? sample_lane(tick-lane, lane) : 0;
                rowRight[lane] = tick >= N-1-lane && tick-(N-1-lane) < streams ?
                    sample_lane(tick-(N-1-lane), lane) : 0;
            end
            @(posedge clk); #1;
            sampleIndex = tick-(N-1);
            for (int col = 0; col < N; col++) begin
                if (resultValid[col] !== (sampleIndex >= 0 && sampleIndex < streams))
                    $fatal(1, "N=%0d aligned valid wrong at advance%0d lane%0d", N, tick, col);
                if (resultValid[col]) begin
                    version = sampleIndex < 2 ? sampleIndex : 2;
                    sum = 0;
                    for (int row = 0; row < N; row++) sum += version * sample_lane(sampleIndex, row);
                    if (result[col] !== sum)
                        $fatal(1, "N=%0d sample%0d lane%0d mixed generations got%0d expected%0d",
                            N, sampleIndex, col, result[col], sum);
                end
            end
            if (tick == 1 || tick == N) begin
                @(negedge clk) advance = 0;
                repeat (2) @(posedge clk);
            end
        end
        $display("PASS: N=%0d inward phase, signed/zero directions, overlapping packages, stalls, coherent sample generations", N);
        done = 1;
    end
endmodule

module matrixWeightUpdateWave_tb;
    logic clk = 0;
    logic pe0, pe1, pe2, array2, array3, array4, array5, array8;
    always #5 clk = ~clk;
    multiplierBlockWeightUpdate_testcase #(.ROLE(0)) tl(.clk(clk), .done(pe0));
    multiplierBlockWeightUpdate_testcase #(.ROLE(1)) ad(.clk(clk), .done(pe1));
    multiplierBlockWeightUpdate_testcase #(.ROLE(2)) br(.clk(clk), .done(pe2));
    systolicWeightUpdateWave_testcase #(.N(2)) n2(.clk(clk), .done(array2));
    systolicWeightUpdateWave_testcase #(.N(3)) n3(.clk(clk), .done(array3));
    systolicWeightUpdateWave_testcase #(.N(4)) n4(.clk(clk), .done(array4));
    systolicWeightUpdateWave_testcase #(.N(5)) n5(.clk(clk), .done(array5));
    systolicWeightUpdateWave_testcase #(.N(8)) n8(.clk(clk), .done(array8));
    initial begin
        wait(pe0 && pe1 && pe2 && array2 && array3 && array4 && array5 && array8);
        $display("PASS: all PE roles and N=2,3,4,5,8 inward update-wave suites completed.");
        $finish;
    end
    initial begin #100000; $fatal(1, "PE/update suite timeout"); end
endmodule
