`timescale 1ns/1ps
// The old inward update wave is replaced by drained atomic updates. Pin the
// PE arithmetic, saturating step, same-edge old-weight read, and common enable.
module multiplierBlockWeightUpdate_testcase #(parameter bit SEED=0)(output logic done);
    logic clk=0, rst_n, advance, cfg_we, update_valid;
    logic signed [3:0] cfg_data, activation_in, activation_out, weight;
    logic signed [1:0] update_direction;
    logic activation_valid, sum_valid, activation_out_valid, sum_out_valid;
    logic [3:0] activation_tag, sum_tag, activation_out_tag, sum_out_tag;
    logic signed [9:0] sum_in, sum_out;
    always #5 clk=~clk;
    weightStationaryProcessingElement #(.INPUT_WIDTH(4),.WIDTH(4),.RESULT_WIDTH(10),.TAG_WIDTH(4),.SEED(SEED)) dut(.*);
    task automatic load(input int value);
        @(negedge clk); cfg_we=1; cfg_data=4'(value); advance=0; update_valid=1; update_direction=1;
        @(posedge clk); #1;
        if (weight!==4'(value)) $fatal(1,"PE configuration priority/disabled-load failed");
        @(negedge clk); cfg_we=0; update_valid=0; advance=1;
    endtask
    task automatic step_weight(input int direction, expected);
        @(negedge clk); update_valid=1; update_direction=2'(direction);
        @(posedge clk); #1;
        if (weight!==4'(expected)) $fatal(1,"PE saturating update failed SEED=%0d",SEED);
        @(negedge clk); update_valid=0;
    endtask
    initial begin
        done=0; rst_n=0; advance=1; cfg_we=0; update_valid=0; update_direction=0;
        cfg_data=0; activation_in=0; activation_valid=0; activation_tag=7;
        sum_in=0; sum_valid=0; sum_tag=7;
        repeat(2) @(posedge clk);
        @(negedge clk); rst_n=1;
        load(0); step_weight(1,1); step_weight(0,1); step_weight(-1,0);
        load(7); step_weight(1,7); load(-8); step_weight(-1,-8);
        load(3);
        @(negedge clk); activation_in=-2; activation_valid=1; sum_in=5; sum_valid=1;
        update_valid=1; update_direction=1;
        @(posedge clk); #1;
        if (sum_out!==(SEED?-6:-1) || weight!==4 || !sum_out_valid || sum_out_tag!=7)
            $fatal(1,"PE same-edge read-before-write failed SEED=%0d",SEED);
        @(negedge clk); update_valid=0;
        @(posedge clk); #1;
        if (sum_out!==(SEED?-8:-3)) $fatal(1,"PE next sample did not use new weight");
        @(negedge clk); advance=0; activation_in=3; sum_in=-9; update_valid=1;
        repeat(3) begin @(posedge clk); #1;
            if (sum_out!==(SEED?-8:-3) || activation_out!==-2 || weight!==4 || !sum_out_valid)
                $fatal(1,"PE freeze failed");
        end
        @(negedge clk); rst_n=0;
        @(posedge clk); #1;
        if (sum_out_valid || activation_out_valid || weight!=0) $fatal(1,"PE reset priority failed");
        done=1;
    end
endmodule

module matrixWeightUpdateWave_tb;
    wire [1:0] done;
    multiplierBlockWeightUpdate_testcase #(.SEED(0)) a(done[0]);
    multiplierBlockWeightUpdate_testcase #(.SEED(1)) b(done[1]);
    initial begin wait(&done); $display("PASS: seed/non-seed PE, configuration, saturated atomic updates, old-weight reads, enable/reset"); $finish; end
    initial begin #10000; $fatal(1,"PE regression timeout"); end
endmodule

// Expected failures are run separately by run_modelsim.ps1; a pass-through
// simulation ending without the requested assertion is itself a failure.
module matrixProtocolFault_tb #(parameter int FAULT=0);
    localparam int N=3;
    logic clk=0, rst_n=0, advance=1, cfg_we=0, update_valid=0;
    logic [1:0] cfg_row=0;
    logic signed [7:0] cfg_data[N], in_data[N];
    logic signed [1:0] row_direction[N], column_direction[N];
    logic in_valid[N];
    logic [3:0] in_tag[N];
    wire signed [17:0] result[N];
    wire resultValid[N], loaded, busy;
    wire [3:0] resultTag[N];
    wire signed [7:0] weights[N][N];
    always #5 clk=~clk;
    weightStationarySystolicArray #(.N(N),.WIDTH(8),.INPUT_WIDTH(8),.RESULT_WIDTH(18),.TAG_WIDTH(4)) dut(.*);
    initial begin
        for (int i=0; i<N; i++) begin
            cfg_data[i]=1; in_data[i]=1; in_valid[i]=0; in_tag[i]=1;
            row_direction[i]=0; column_direction[i]=0;
        end
        repeat(2) @(posedge clk);
        @(negedge clk); rst_n=1;
        if (FAULT!=4) begin
            for (int row=0; row<N; row++) begin
                cfg_we=1; cfg_row=2'(row);
                @(posedge clk); #1;
                @(negedge clk);
            end
            cfg_we=0;
        end
        in_valid[1]=1;
        @(posedge clk); #1;
        @(negedge clk); in_valid[1]=0;
        if (FAULT==0) begin in_valid[0]=1; in_valid[2]=0; end
        else begin in_valid[0]=1; in_valid[2]=1; end
        if (FAULT==1) in_tag[2]=2;
        if (FAULT==2) in_tag[0]=2;
        if (FAULT==3) cfg_we=1;
        @(posedge clk); #1;
        @(negedge clk); for (int i=0; i<N; i++) in_valid[i]=0;
        repeat(6) @(posedge clk);
        $fatal(1,"EXPECTED_PROTOCOL_ASSERTION_MISSING FAULT=%0d",FAULT);
    end
endmodule
