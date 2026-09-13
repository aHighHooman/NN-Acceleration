`timescale 1ns / 1ps

module multiplierBlockWeightUpdate_testcase(
    input  logic clk,
    output logic done
);
    localparam int WIDTH = 4;
    localparam int RESULT_WIDTH = 2*WIDTH + 1;

    logic rst_n, advance, loadWeight, updateWeight;
    logic signed [1:0] updateDirection;
    logic signed [WIDTH-1:0] leftIn, rightOut;
    logic leftValid, rightValid;
    logic signed [RESULT_WIDTH-1:0] topIn, bottomOut;
    logic topValid, bottomValid;

    multiplierBlockWeightStationary #(
        .WIDTH(WIDTH), .RESULT_WIDTH(RESULT_WIDTH)
    ) dut (
        .clk(clk), .rst_n(rst_n), .advance(advance),
        .loadWeight(loadWeight), .updateWeight(updateWeight),
        .updateDirection(updateDirection),
        .leftIn(leftIn), .leftValid(leftValid),
        .topIn(topIn), .topValid(topValid),
        .rightOut(rightOut), .rightValid(rightValid),
        .bottomOut(bottomOut), .bottomValid(bottomValid)
    );

    initial begin
        done = 1'b0;
        rst_n = 1'b0;
        advance = 1'b1;
        loadWeight = 1'b0;
        updateWeight = 1'b0;
        updateDirection = 2'sd0;
        leftIn = '0;
        leftValid = 1'b0;
        topIn = '0;
        topValid = 1'b0;

        repeat (3) @(posedge clk);
        @(negedge clk) rst_n = 1'b1;

        load_pe_weight(0);
        apply_pe_update(2'sd1, 1);
        apply_pe_update(2'sd0, 1);
        apply_pe_update(-2'sd1, 0);

        load_pe_weight(7);
        apply_pe_update(2'sd1, 7);
        load_pe_weight(-8);
        apply_pe_update(-2'sd1, -8);

        // No state, forwarded data, or valid bit may move while stalled.
        load_pe_weight(2);
        @(negedge clk);
        advance = 1'b0;
        updateWeight = 1'b1;
        updateDirection = 2'sd1;
        leftIn = 3;
        leftValid = 1'b1;
        topIn = 4;
        topValid = 1'b1;
        repeat (2) begin
            @(posedge clk); #1;
            if (dut.weightReg !== 2 || rightValid || bottomValid)
                $fatal(1, "PE state advanced while advance was low");
        end

        // Loading wins over a simultaneous update request.
        @(negedge clk);
        advance = 1'b1;
        loadWeight = 1'b1;
        topIn = 3;
        updateDirection = -2'sd1;
        @(posedge clk); #1;
        if (dut.weightReg !== 3)
            $fatal(1, "loadWeight did not have priority over updateWeight");

        // The multiply on the update edge sees 3; the following edge sees 4.
        @(negedge clk);
        loadWeight = 1'b0;
        updateDirection = 2'sd1;
        leftIn = 2;
        leftValid = 1'b1;
        topIn = 0;
        topValid = 1'b1;
        @(posedge clk); #1;
        if (bottomOut !== 6 || dut.weightReg !== 4)
            $fatal(1, "update edge did not multiply with the old weight");
        @(negedge clk) updateWeight = 1'b0;
        @(posedge clk); #1;
        if (bottomOut !== 8)
            $fatal(1, "sample after the update did not use the new weight");

        $display("PASS: PE ternary updates, saturation, stalls, and weight-version edge.");
        done = 1'b1;
    end

    task load_pe_weight(input integer value);
        @(negedge clk);
        advance = 1'b1;
        loadWeight = 1'b1;
        updateWeight = 1'b0;
        updateDirection = 2'sd0;
        topIn = value;
        leftValid = 1'b0;
        topValid = 1'b0;
        @(posedge clk); #1;
        if (dut.weightReg !== value)
            $fatal(1, "PE load got %0d, expected %0d", dut.weightReg, value);
        @(negedge clk) loadWeight = 1'b0;
    endtask

    task apply_pe_update(
        input logic signed [1:0] direction,
        input integer expected
    );
        @(negedge clk);
        advance = 1'b1;
        updateWeight = 1'b1;
        updateDirection = direction;
        @(posedge clk); #1;
        if (dut.weightReg !== expected)
            $fatal(1, "PE direction %0d got %0d, expected %0d",
                   direction, dut.weightReg, expected);
        @(negedge clk) updateWeight = 1'b0;
    endtask
endmodule

module systolicWeightUpdateWave_testcase(
    input  logic clk,
    output logic done
);
    localparam int WIDTH = 8;
    localparam int N = 3;
    localparam int RESULT_WIDTH = 2*WIDTH + $clog2(N);

    logic rst_n, advance, loadWeight, updateValid;
    logic signed [WIDTH-1:0] row[N], col[N];
    logic rowValid[N];
    logic signed [1:0] rowDirection[N], columnDirection[N];
    logic signed [RESULT_WIDTH-1:0] result[N];
    logic resultValid[N], updateComplete, pipelineBusy;
    integer resultCount[N];
    integer acceptedUpdateCount, completedUpdateCount;

    systolicArrayWeightStationary #(.WIDTH(WIDTH), .N(N)) dut (
        .clk(clk), .rst_n(rst_n), .advance(advance),
        .loadWeight(loadWeight),
        .rowDirection(rowDirection), .columnDirection(columnDirection),
        .updateValid(updateValid),
        .row(row), .rowValid(rowValid), .col(col),
        .result(result), .resultValid(resultValid),
        .updateComplete(updateComplete),
        .pipelineBusy(pipelineBusy)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            acceptedUpdateCount = 0;
            completedUpdateCount = 0;
        end else begin
            if (advance && updateValid)
                acceptedUpdateCount = acceptedUpdateCount + 1;
            if (updateComplete)
                completedUpdateCount = completedUpdateCount + 1;
            if (updateComplete && !advance)
                $fatal(1, "update completion asserted while the array was stalled");
            if (completedUpdateCount > acceptedUpdateCount)
                $fatal(1, "more update completions than accepted packages");
        end
    end

    initial begin
        done = 1'b0;
        rst_n = 1'b0;
        advance = 1'b1;
        loadWeight = 1'b0;
        updateValid = 1'b0;
        for (int lane = 0; lane < N; lane++) begin
            row[lane] = '0;
            rowValid[lane] = 1'b0;
            col[lane] = '0;
            rowDirection[lane] = 2'sd0;
            columnDirection[lane] = 2'sd0;
            resultCount[lane] = 0;
        end

        repeat (3) @(posedge clk);
        @(negedge clk) rst_n = 1'b1;
        load_uniform_weights(0);

        // P0 produces +, -, and zero local directions. P1 is accepted on the
        // following advance, proving packages can occupy adjacent stages.
        set_directions(1, 0, -1, 1, -1, 0);
        pulse_package();
        set_directions(1, 1, 1, 1, 1, 1);
        advance_package(1'b1);
        check_weights(2, -1, 0, 0, 0, 0, 0, 0, 0,
                      "overlapping packages at diagonal zero");

        // Freeze while both a data register and two update stages are live.
        @(negedge clk);
        advance = 1'b0;
        updateValid = 1'b0;
        row[0] = 5;
        rowValid[0] = 1'b1;
        repeat (2) begin
            @(posedge clk); #1;
            if (!dut.updateValidPipe[0] || !dut.updateValidPipe[1] ||
                dut.row_loop[0].col_loop[0].mb.weightReg !== 2 ||
                dut.row_loop[0].col_loop[0].mb.rightValid !== 1'b0)
                $fatal(1, "array stall did not freeze data and update waves");
        end

        rowValid[0] = 1'b0;
        advance_package(1'b0);
        check_weights(2, 0, 0, 1, 0, 0, -1, 0, 0,
                      "overlapping anti-diagonal one/two");
        advance_package(1'b0);
        check_weights(2, 0, 1, 1, 1, 0, 0, 1, 0,
                      "overlapping anti-diagonal two/three");
        advance_package(1'b0);
        check_weights(2, 0, 1, 1, 1, 1, 0, 2, 0,
                      "overlapping anti-diagonal three/four");
        advance_package(1'b0);
        check_weights(2, 0, 1, 1, 1, 1, 0, 2, 1,
                      "second package completion");
        if (pipelineBusy)
            $fatal(1, "update stages remained busy after both packages drained");
        if (completedUpdateCount != 2)
            $fatal(1, "overlapping packages produced %0d completions, expected 2",
                   completedUpdateCount);

        // A zero-row package must traverse without changing any PE.
        set_directions(0, 0, 0, 1, 1, 1);
        pulse_package();
        repeat (4) advance_package(1'b0);
        if (completedUpdateCount != 3)
            $fatal(1, "zero-direction package did not produce one completion");
        check_weights(2, 0, 1, 1, 1, 1, 0, 2, 1,
                      "zero update package");

        // Stream four all-one samples while an all-positive package follows
        // the first sample wave. Sample 0 uses the old weights on the update
        // edge; samples 1 through 3 use the incremented weights.
        load_uniform_weights(2);
        for (int lane = 0; lane < N; lane++) resultCount[lane] = 0;
        for (int waveCycle = 0; waveCycle < 10; waveCycle++) begin
            @(negedge clk);
            advance = 1'b1;
            updateValid = (waveCycle == 0);
            for (int lane = 0; lane < N; lane++) begin
                rowDirection[lane] = 2'sd1;
                columnDirection[lane] = 2'sd1;
                row[lane] = 1;
                rowValid[lane] = ((waveCycle-lane) >= 0) &&
                                 ((waveCycle-lane) < 4);
            end
            @(posedge clk); #1;
            check_stream_results();

            if (waveCycle == 1) begin
                @(negedge clk) advance = 1'b0;
                repeat (2) begin
                    @(posedge clk); #1;
                    if (!dut.updateValidPipe[1])
                        $fatal(1, "update wave moved during stream stall");
                end
            end
        end

        for (int lane = 0; lane < N; lane++)
            if (resultCount[lane] != 4)
                $fatal(1, "column %0d produced %0d versioned samples, expected 4",
                       lane, resultCount[lane]);
        if (completedUpdateCount != acceptedUpdateCount)
            $fatal(1, "accepted/completed update counts differ: %0d/%0d",
                   acceptedUpdateCount, completedUpdateCount);

        $display("PASS: anti-diagonal ordering, overlapping packages, stalls, and coherent versions.");
        done = 1'b1;
    end

    task set_directions(
        input integer r0, input integer r1, input integer r2,
        input integer c0, input integer c1, input integer c2
    );
        rowDirection[0] = r0;
        rowDirection[1] = r1;
        rowDirection[2] = r2;
        columnDirection[0] = c0;
        columnDirection[1] = c1;
        columnDirection[2] = c2;
    endtask

    task load_uniform_weights(input integer value);
        @(negedge clk);
        updateValid = 1'b0;
        loadWeight = 1'b1;
        advance = 1'b1;
        for (int lane = 0; lane < N; lane++) begin
            col[lane] = value;
            rowValid[lane] = 1'b0;
        end
        repeat (N) @(posedge clk);
        #1;
        @(negedge clk) loadWeight = 1'b0;
        check_weights(value, value, value, value, value, value,
                      value, value, value, "uniform load");
    endtask

    task pulse_package();
        @(negedge clk) updateValid = 1'b1;
        @(posedge clk); #1;
    endtask

    task advance_package(input logic acceptPackage);
        @(negedge clk);
        advance = 1'b1;
        updateValid = acceptPackage;
        @(posedge clk); #1;
    endtask

    task check_weights(
        input integer w00, input integer w01, input integer w02,
        input integer w10, input integer w11, input integer w12,
        input integer w20, input integer w21, input integer w22,
        input string label
    );
        if (dut.row_loop[0].col_loop[0].mb.weightReg !== w00 ||
            dut.row_loop[0].col_loop[1].mb.weightReg !== w01 ||
            dut.row_loop[0].col_loop[2].mb.weightReg !== w02 ||
            dut.row_loop[1].col_loop[0].mb.weightReg !== w10 ||
            dut.row_loop[1].col_loop[1].mb.weightReg !== w11 ||
            dut.row_loop[1].col_loop[2].mb.weightReg !== w12 ||
            dut.row_loop[2].col_loop[0].mb.weightReg !== w20 ||
            dut.row_loop[2].col_loop[1].mb.weightReg !== w21 ||
            dut.row_loop[2].col_loop[2].mb.weightReg !== w22)
            $fatal(1, "%s: matrix weights did not match expected diagonal state", label);
    endtask

    task check_stream_results();
        integer expected;
        for (int lane = 0; lane < N; lane++) begin
            if (resultValid[lane]) begin
                expected = (resultCount[lane] < 1) ? 6 : 9;
                if (result[lane] !== expected)
                    $fatal(1, "column %0d sample %0d got %0d, expected %0d",
                           lane, resultCount[lane], result[lane], expected);
                resultCount[lane] = resultCount[lane] + 1;
            end
        end
    endtask
endmodule

module matrixWeightUpdateWave_tb;
    logic clk;
    logic peDone, arrayDone;

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    multiplierBlockWeightUpdate_testcase peTest(.clk(clk), .done(peDone));
    systolicWeightUpdateWave_testcase arrayTest(.clk(clk), .done(arrayDone));

    initial begin
        wait(peDone && arrayDone);
        $display("PASS: Phase 5K live-entry matrix update-wave overlap, stall, completion-order, and version-boundary tests completed.");
        $finish;
    end
endmodule
