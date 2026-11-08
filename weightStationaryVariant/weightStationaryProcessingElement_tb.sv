`timescale 1ns/1ps
// PE arithmetic with and without an incoming sum, both weight copies'
// configuration, selection and saturated updates, the common enable and reset. Training updates arrive only behind the
// drained barrier, so they never coincide with configuration or with a live MAC.
module weightStationaryProcessingElement_tb;
    logic clk=0, rst_n, advance, backward, cfg_we, cfg_transposed_we, update_valid;
    logic signed [3:0] cfg_data, cfg_transposed_data, activation_in, activation_out, weight;
    logic signed [1:0] update_direction, transposed_update_direction;
    logic activation_valid, activation_out_valid;
    logic signed [9:0] sum_in, sum_out;
    always #5 clk=~clk;
    weightStationaryProcessingElement #(.INPUT_WIDTH(4),.WIDTH(4),.RESULT_WIDTH(10)) dut(.*);
    task automatic load(input int value);
        @(negedge clk); cfg_we=1; cfg_data=4'(value); advance=0;
        @(posedge clk); #1;
        if (weight!==4'(value)) $fatal(1,"PE configuration while disabled failed");
        @(negedge clk); cfg_we=0; advance=1;
    endtask
    task automatic step_weight(input int direction, expected);
        @(negedge clk); update_valid=1; update_direction=2'(direction);
        @(posedge clk); #1;
        if (weight!==4'(expected)) $fatal(1,"PE saturating update failed");
        @(negedge clk); update_valid=0;
    endtask
    task automatic load_transposed(input int value);
        logic signed [3:0] forward;
        forward=weight;
        @(negedge clk); cfg_transposed_we=1; cfg_transposed_data=4'(value); advance=0;
        @(posedge clk); #1;
        if (dut.transposed_weight!==4'(value) || weight!==forward) $fatal(1,"PE transposed configuration failed");
        @(negedge clk); cfg_transposed_we=0; advance=1;
    endtask
    task automatic step_transposed(input int direction, expected);
        logic signed [3:0] forward;
        forward=weight;
        @(negedge clk); update_valid=1; update_direction=0; transposed_update_direction=2'(direction);
        @(posedge clk); #1;
        if (dut.transposed_weight!==4'(expected) || weight!==forward) $fatal(1,"PE transposed saturating update failed");
        @(negedge clk); update_valid=0; transposed_update_direction=0;
    endtask
    task automatic mac(input int a, input bit valid, input int partial, input int expected);
        @(negedge clk); activation_in=4'(a); activation_valid=valid; sum_in=10'(partial);
        @(posedge clk); #1;
        if (activation_out!==4'(a) || activation_out_valid!==valid || (valid && sum_out!==10'(expected)))
            $fatal(1,"PE MAC failed a=%0d partial=%0d got=%0d expected=%0d",a,partial,sum_out,expected);
    endtask
    initial begin
        rst_n=0; advance=1; backward=0; cfg_we=0; cfg_transposed_we=0; update_valid=0;
        update_direction=0; transposed_update_direction=0;
        cfg_data=0; cfg_transposed_data=0; activation_in=0; activation_valid=0; sum_in=0;
        repeat(2) @(posedge clk);
        @(negedge clk); rst_n=1;
        load(0); step_weight(1,1); step_weight(0,1); step_weight(-1,0);
        load(7); step_weight(1,7); step_weight(-1,6);
        load(-8); step_weight(-1,-8); step_weight(1,-7);
        load_transposed(7); step_transposed(1,7); step_transposed(-1,6);
        load_transposed(-8); step_transposed(-1,-8); step_transposed(1,-7);
        load(3);
        load_transposed(-5);
        mac(-2,1,0,-6);
        mac(-2,1,5,-1);
        // The backward select multiplies by the transposed copy instead.
        backward=1;
        mac(-2,1,0,10);
        mac(-2,1,5,15);
        backward=0;
        load(-8);
        mac(-8,1,0,64);
        mac(-8,1,-100,-36);
        mac(7,1,300,244);
        mac(1,0,0,0);
        // A low enable freezes data and validity together.
        mac(5,1,1,-39);
        @(negedge clk); advance=0; activation_in=3; activation_valid=0; sum_in=-9;
        repeat(3) begin @(posedge clk); #1;
            if (sum_out!==10'(-39) || activation_out!==5 || !activation_out_valid)
                $fatal(1,"PE freeze failed");
        end
        @(negedge clk); rst_n=0;
        @(posedge clk); #1;
        if (activation_out_valid || weight!=0 || dut.transposed_weight!=0) $fatal(1,"PE reset priority failed");
        $display("PASS: PE arithmetic, both weight copies, configuration, saturated updates, common enable and reset");
        $finish;
    end
    initial begin #10000; $fatal(1,"PE regression timeout"); end
endmodule
