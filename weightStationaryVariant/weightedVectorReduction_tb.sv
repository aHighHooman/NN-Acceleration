`timescale 1ns / 1ps
// Drives the column through the array's center-first lane delays and checks each
// scalar against an independent dense sum, plus timing, stalls, loads and updates.
module weightedVectorReduction_testcase #(
    parameter int N = 5,
    parameter int MATRIX_RESULT_WIDTH = 8,
    parameter int REDUCTION_WEIGHT_WIDTH = 8,
    parameter int RESCALE_SHIFT = REDUCTION_WEIGHT_WIDTH-1
)(output logic done);
    localparam int H = (N-1)/2;
    localparam int PREDICTION_WIDTH = MATRIX_RESULT_WIDTH+$clog2(N);
    localparam int LIMIT = 512;
    typedef logic signed [MATRIX_RESULT_WIDTH-1:0] data_t;
    typedef logic signed [REDUCTION_WEIGHT_WIDTH-1:0] weight_t;
    typedef logic signed [PREDICTION_WIDTH-1:0] prediction_t;
    localparam data_t DATA_MIN = {1'b1, {(MATRIX_RESULT_WIDTH-1){1'b0}}};
    localparam data_t DATA_MAX = {1'b0, {(MATRIX_RESULT_WIDTH-1){1'b1}}};
    localparam weight_t WEIGHT_MIN = {1'b1, {(REDUCTION_WEIGHT_WIDTH-1){1'b0}}};
    localparam weight_t WEIGHT_MAX = {1'b0, {(REDUCTION_WEIGHT_WIDTH-1){1'b1}}};

    logic clk=0, rst_n, advance, cfg_we, update_valid, vectorValid;
    data_t vectorData[N];
    weight_t cfg_data[N];
    logic signed [1:0] row_direction[N], column_direction;
    wire signed [MATRIX_RESULT_WIDTH-1:0] laneData[N];
    wire laneValid[N];
    wire signed [PREDICTION_WIDTH-1:0] prediction;
    wire predictionValid;
    wire signed [REDUCTION_WEIGHT_WIDTH-1:0] weights[N];
    prediction_t expected[LIMIT];
    int acceptedStep[LIMIT], acceptedCount, checkedCount, enabledStep;
    int unsigned randomState;

    always #5 clk=~clk;

    for (genvar i=0; i<N; i++) begin : source
        weightStationaryLaneDelay #(.WIDTH(MATRIX_RESULT_WIDTH), .DEPTH((i<H)?H-i:i-H)) skew (
            .clk, .rst_n, .advance, .in_data(vectorData[i]), .in_valid(vectorValid),
            .out_data(laneData[i]), .out_valid(laneValid[i]), .busy()
        );
    end
    weightedVectorReduction #(.MATRIX_RESULT_WIDTH(MATRIX_RESULT_WIDTH),
        .REDUCTION_WEIGHT_WIDTH(REDUCTION_WEIGHT_WIDTH), .N(N), .RESCALE_SHIFT(RESCALE_SHIFT)) dut (
        .clk, .rst_n, .advance, .cfg_we, .cfg_data, .update_valid, .row_direction, .column_direction,
        .in_data(laneData), .in_valid(laneValid), .prediction, .predictionValid, .weights
    );

    function automatic int unsigned random_word();
        randomState=1664525*randomState+1013904223;
        return randomState;
    endfunction

    always @(posedge clk) begin : check
        prediction_t held;
        bit heldValid, blocked, stepped;
        longint sum;
        blocked=rst_n && !advance;
        stepped=rst_n && advance;
        held=prediction;
        heldValid=predictionValid;
        if (!rst_n) begin
            acceptedCount=0; checkedCount=0; enabledStep=0;
        end else if (advance) begin
            enabledStep++;
            if (vectorValid) begin
                if (acceptedCount>=LIMIT) $fatal(1, "Reduction test sample limit");
                sum=0;
                for (int i=0; i<N; i++) sum+=longint'(vectorData[i])*longint'(weights[i]);
                expected[acceptedCount]=prediction_t'(sum>>>RESCALE_SHIFT);
                acceptedStep[acceptedCount]=enabledStep;
                acceptedCount++;
            end
        end
        #1;
        if (blocked && (predictionValid!==heldValid || prediction!==held))
            $fatal(1, "Reduction output changed during a stall N=%0d", N);
        // Center row captures on the accepting step, the boundary rows H steps later.
        if (stepped && predictionValid) begin
            if (checkedCount>=acceptedCount || enabledStep-acceptedStep[checkedCount]!=H)
                $fatal(1, "Reduction timing N=%0d sample=%0d", N, checkedCount);
            if (prediction!==expected[checkedCount])
                $fatal(1, "Reduction result N=%0d sample=%0d got=%0d expected=%0d",
                    N, checkedCount, prediction, expected[checkedCount]);
            checkedCount++;
        end
    end

    task automatic reset_dut();
        @(negedge clk); rst_n=0; vectorValid=0; cfg_we=0; update_valid=0; advance=1;
        repeat(2) @(posedge clk);
        #1;
        if (predictionValid!==0) $fatal(1, "Reset left a valid prediction");
        for (int i=0; i<N; i++) if (weights[i]!==0) $fatal(1, "Reset left a reduction weight");
        @(negedge clk); rst_n=1;
    endtask

    // Loading ignores advance, so hold the column stalled to prove it.
    task automatic load_weights(input int kind);
        @(negedge clk); cfg_we=1; advance=0;
        for (int i=0; i<N; i++) case (kind)
            0: cfg_data[i]=weight_t'(random_word()>>8);
            1: cfg_data[i]=WEIGHT_MIN;
            2: cfg_data[i]=WEIGHT_MAX;
            default: cfg_data[i]=(i%2) ? WEIGHT_MAX : WEIGHT_MIN;
        endcase
        @(posedge clk); #1;
        for (int i=0; i<N; i++) if (weights[i]!==cfg_data[i]) $fatal(1, "Reduction load row %0d", i);
        @(negedge clk); cfg_we=0; advance=1;
    endtask

    // Modes: 0 back-to-back, 1 bubbles, 2 stalls, 3 all minimum, 4 alternating extremes.
    task automatic stream(input int count, mode);
        int sent, guard;
        sent=0; guard=0;
        while (sent<count) begin
            @(negedge clk);
            advance=(mode!=2 || (random_word()>>16)%3!=0);
            vectorValid=(mode!=1 || (random_word()>>16)%3!=0);
            for (int i=0; i<N; i++) case (mode)
                3: vectorData[i]=DATA_MIN;
                4: vectorData[i]=((sent+i)%2) ? DATA_MAX : DATA_MIN;
                default: vectorData[i]=data_t'({random_word(), random_word()}>>4);
            endcase
            @(posedge clk);
            if (vectorValid && advance) sent++;
            if (++guard>3000) $fatal(1, "Reduction input timeout");
        end
        @(negedge clk); vectorValid=0; advance=1;
        guard=0;
        while (checkedCount<acceptedCount) begin
            @(posedge clk); #2;
            if (++guard>N+4) $fatal(1, "Reduction drain timeout");
        end
    endtask

    // Each row sees every start weight with every direction and enable, and the
    // offsets keep neighbouring rows different so a crossed row cannot pass.
    task automatic check_updates();
        weight_t starts[5];
        int next[N];
        starts[0]=WEIGHT_MIN; starts[1]=WEIGHT_MIN+weight_t'(1); starts[2]=weight_t'(1);
        starts[3]=WEIGHT_MAX-weight_t'(1); starts[4]=WEIGHT_MAX;
        for (int first=0; first<5; first++) for (int rowOffset=0; rowOffset<3; rowOffset++)
        for (int column=-1; column<=1; column++) for (int enables=0; enables<4; enables++) begin
            @(negedge clk);
            cfg_we=1;
            for (int i=0; i<N; i++) begin
                cfg_data[i]=starts[(first+i)%5];
                row_direction[i]=2'((rowOffset+i)%3-1);
            end
            @(negedge clk);
            cfg_we=0; column_direction=2'(column); advance=enables[0]; update_valid=enables[1];
            for (int i=0; i<N; i++) begin
                next[i]=int'(cfg_data[i]);
                if (enables==3 && row_direction[i]!=0 && column!=0)
                    next[i]+=(int'(row_direction[i])==column) ? 1 : -1;
                if (next[i]>int'(WEIGHT_MAX)) next[i]=int'(WEIGHT_MAX);
                if (next[i]<int'(WEIGHT_MIN)) next[i]=int'(WEIGHT_MIN);
            end
            @(posedge clk); #1;
            for (int i=0; i<N; i++) if (int'(weights[i])!=next[i])
                $fatal(1, "Reduction update N=%0d row=%0d start=%0d direction=%0d column=%0d enables=%0d got=%0d expected=%0d",
                    N, i, cfg_data[i], row_direction[i], column, enables, weights[i], next[i]);
        end
        // A load beats a simultaneous update.
        @(negedge clk); cfg_we=1; update_valid=1; advance=1; column_direction=1;
        for (int i=0; i<N; i++) begin row_direction[i]=1; cfg_data[i]=weight_t'(2); end
        @(posedge clk); #1;
        for (int i=0; i<N; i++) if (weights[i]!==weight_t'(2)) $fatal(1, "Reduction load priority");
        @(negedge clk); cfg_we=0; update_valid=0;
    endtask

    initial begin
        done=0; rst_n=0; advance=1; cfg_we=0; update_valid=0; vectorValid=0; column_direction=0;
        randomState=32'(N*131+MATRIX_RESULT_WIDTH*17+RESCALE_SHIFT);
        for (int i=0; i<N; i++) begin vectorData[i]=0; cfg_data[i]=0; row_direction[i]=0; end
        reset_dut();
        if ($bits(dut.sumOut[0]) != MATRIX_RESULT_WIDTH+REDUCTION_WEIGHT_WIDTH+$clog2(N))
            $fatal(1, "Reduction accumulator width changed");
        load_weights(0); stream(60, 0); stream(40, 1); stream(40, 2);
        load_weights(1); stream(4, 3);   // Largest positive sum uses every accumulator bit.
        load_weights(2); stream(4, 3);   // Most negative sum needs an arithmetic rescale.
        load_weights(3); stream(8, 4);   // Opposing extremes cancel between chains.
        check_updates();
        // Reset discards samples already inside the column.
        load_weights(0);
        @(negedge clk); vectorValid=1;
        repeat(H+1) @(posedge clk);
        reset_dut();
        repeat(N) @(posedge clk);
        #1;
        if (checkedCount!=0 || predictionValid!==0) $fatal(1, "Reduction sample survived reset");
        $display("PASS: Reduction column N=%0d WIDTH=%0d RWIDTH=%0d SHIFT=%0d arithmetic, lane timing, II=1, stalls, loads, updates, reset",
            N, MATRIX_RESULT_WIDTH, REDUCTION_WEIGHT_WIDTH, RESCALE_SHIFT);
        done=1;
    end
endmodule

module weightedVectorReduction_tb;
    wire [4:0] done;
    weightedVectorReduction_testcase #(.N(3)) a(done[0]);
    weightedVectorReduction_testcase #(.N(5)) b(done[1]);
    weightedVectorReduction_testcase #(.N(9)) c(done[2]);
    weightedVectorReduction_testcase #(.N(3), .RESCALE_SHIFT(9)) d(done[3]);
    // Quartus-profile widths: a 19-bit lane, 30-bit sum and 22-bit prediction.
    weightedVectorReduction_testcase #(.N(5), .MATRIX_RESULT_WIDTH(19)) e(done[4]);
    initial begin wait(&done); $display("PASS: Reduction column regression"); $finish; end
    initial begin #1000000; $fatal(1, "Reduction column regression timeout"); end
endmodule
