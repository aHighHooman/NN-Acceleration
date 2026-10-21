`timescale 1ns/1ps
// Seed and non-seed PE arithmetic, configuration, saturated updates, the common
// enable and reset. Training updates arrive only behind the drained barrier, so
// they never coincide with configuration or with a live MAC.
module weightStationaryProcessingElement_testcase #(parameter bit SEED=0)(output logic done);
    logic clk=0, rst_n, advance, cfg_we, update_valid;
    logic signed [3:0] cfg_data, activation_in, activation_out, weight;
    logic signed [1:0] update_direction;
    logic activation_valid, activation_out_valid;
    logic signed [9:0] sum_in, sum_out;
    always #5 clk=~clk;
    weightStationaryProcessingElement #(.INPUT_WIDTH(4),.WIDTH(4),.RESULT_WIDTH(10),.SEED(SEED)) dut(.*);
    task automatic load(input int value);
        @(negedge clk); cfg_we=1; cfg_data=4'(value); advance=0;
        @(posedge clk); #1;
        if (weight!==4'(value)) $fatal(1,"PE configuration while disabled failed SEED=%0d",SEED);
        @(negedge clk); cfg_we=0; advance=1;
    endtask
    task automatic step_weight(input int direction, expected);
        @(negedge clk); update_valid=1; update_direction=2'(direction);
        @(posedge clk); #1;
        if (weight!==4'(expected)) $fatal(1,"PE saturating update failed SEED=%0d",SEED);
        @(negedge clk); update_valid=0;
    endtask
    task automatic mac(input int a, input bit valid, input int partial, input int expected);
        @(negedge clk); activation_in=4'(a); activation_valid=valid; sum_in=10'(partial);
        @(posedge clk); #1;
        if (activation_out!==4'(a) || activation_out_valid!==valid || (valid && sum_out!==10'(expected)))
            $fatal(1,"PE MAC failed SEED=%0d a=%0d partial=%0d got=%0d expected=%0d",SEED,a,partial,sum_out,expected);
    endtask
    initial begin
        done=0; rst_n=0; advance=1; cfg_we=0; update_valid=0; update_direction=0;
        cfg_data=0; activation_in=0; activation_valid=0; sum_in=0;
        repeat(2) @(posedge clk);
        @(negedge clk); rst_n=1;
        load(0); step_weight(1,1); step_weight(0,1); step_weight(-1,0);
        load(7); step_weight(1,7); step_weight(-1,6);
        load(-8); step_weight(-1,-8); step_weight(1,-7);
        load(3);
        mac(-2,1,5,SEED?-6:-1);
        load(-8);
        mac(-8,1,-100,SEED?64:-36);
        mac(7,1,300,SEED?-56:244);
        mac(1,0,0,0);
        // A low enable freezes data and validity together.
        mac(5,1,1,SEED?-40:-39);
        @(negedge clk); advance=0; activation_in=3; activation_valid=0; sum_in=-9;
        repeat(3) begin @(posedge clk); #1;
            if (sum_out!==10'(SEED?-40:-39) || activation_out!==5 || !activation_out_valid)
                $fatal(1,"PE freeze failed SEED=%0d",SEED);
        end
        @(negedge clk); rst_n=0;
        @(posedge clk); #1;
        if (activation_out_valid || weight!=0) $fatal(1,"PE reset priority failed SEED=%0d",SEED);
        done=1;
    end
endmodule

module weightStationaryProcessingElement_tb;
    wire [1:0] done;
    weightStationaryProcessingElement_testcase #(.SEED(0)) a(done[0]);
    weightStationaryProcessingElement_testcase #(.SEED(1)) b(done[1]);
    initial begin wait(&done); $display("PASS: seed/non-seed PE arithmetic, configuration, saturated updates, common enable and reset"); $finish; end
    initial begin #10000; $fatal(1,"PE regression timeout"); end
endmodule
