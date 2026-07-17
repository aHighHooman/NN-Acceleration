`timescale 1ns / 1ps

module matrixMultiplierWeightStationaryAxisOut_testcase #(
    parameter int WIDTH = 8,
    parameter int N = 3
) (
    output logic done
);
    localparam int CLK_PERIOD = 10;
    localparam int RESULT_WIDTH = 2*WIDTH + $clog2(N);
    localparam int AXIS_LANE_WIDTH = 8*((RESULT_WIDTH + 7) / 8);

    typedef logic signed [WIDTH-1:0] data_t;
    typedef logic signed [RESULT_WIDTH-1:0] result_t;

    logic clk, rst_n;
    data_t weightData[N], activationData[N];
    logic weightValid, weightReady, activationValid, activationReady;
    logic passThrough, weightsLoaded, reloadWeights, reloadReady;
    logic output_start_valid, output_start_ready, result_frame_available;
    logic [N*AXIS_LANE_WIDTH-1:0] m_axis_tdata;
    logic m_axis_tvalid, m_axis_tready, m_axis_tlast;
    data_t inputMatrix[4][N][N];

    matrixMultiplierWeightStationaryAxisOut #(
        .WIDTH(WIDTH), .N(N), .OUTPUT_FIFO_DEPTH(2*N),
        .AXIS_LANE_WIDTH(AXIS_LANE_WIDTH)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .weightData(weightData), .weightValid(weightValid), .weightReady(weightReady),
        .activationData(activationData), .activationValid(activationValid),
        .activationReady(activationReady), .passThrough(passThrough),
        .weightsLoaded(weightsLoaded), .reloadWeights(reloadWeights), .reloadReady(reloadReady),
        .output_start_valid(output_start_valid), .output_start_ready(output_start_ready),
        .result_frame_available(result_frame_available),
        .m_axis_tdata(m_axis_tdata), .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready), .m_axis_tlast(m_axis_tlast)
    );

    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    initial begin
        build_input_matrices();
        initialize_signals();
        apply_reset();
        load_identity_weights();

        // A held command issued before computation must wait for a complete frame.
        @(negedge clk) output_start_valid = 1'b1;
        assert(!output_start_ready) else $error("Output start was ready before a frame existed");
        send_activation_matrix(0);
        wait_for_start_accept();
        @(negedge clk) output_start_valid = 1'b0;
        receive_and_check_frame(0, 1'b1);

        // Hold two complete frames in the existing output FIFO bank before reading either.
        send_activation_matrix(1);
        send_activation_matrix(2);
        wait_for_buffered_frames(2);
        assert(result_frame_available) else $error("First buffered frame was not reported available");
        for (int lane = 0; lane < N; lane++)
            assert(dut.core.outputFull[lane]) else $error("Output FIFO did not fill while two frames were held");
        assert(!m_axis_tvalid) else $error("AXI output started without a start command");
        request_and_receive_frame(1, 1'b1);
        assert(!m_axis_tvalid) else $error("AXI output remained active after TLAST transfer");
        request_and_receive_frame(2, 1'b0);

        // Reset must abort a stalled transaction and discard the buffered frame.
        send_activation_matrix(3);
        wait (result_frame_available);
        @(negedge clk) begin
            output_start_valid = 1'b1;
            m_axis_tready = 1'b0;
        end
        wait_for_start_accept();
        @(negedge clk) output_start_valid = 1'b0;
        wait (m_axis_tvalid);
        @(negedge clk) rst_n = 1'b0;
        @(posedge clk);
        #1;
        assert(!m_axis_tvalid && !m_axis_tlast) else $error("Reset did not abort AXI transmission");
        @(negedge clk) rst_n = 1'b1;

        done = 1'b1;
    end

    task automatic initialize_signals();
        begin
            rst_n = 1'b0;
            weightValid = 1'b0;
            activationValid = 1'b0;
            passThrough = 1'b1;
            reloadWeights = 1'b0;
            output_start_valid = 1'b0;
            m_axis_tready = 1'b0;
            done = 1'b0;
            for (int lane = 0; lane < N; lane++) begin
                weightData[lane] = '0;
                activationData[lane] = '0;
            end
        end
    endtask

    task automatic apply_reset();
        begin
            @(negedge clk) rst_n = 1'b0;
            repeat (2) @(posedge clk);
            @(negedge clk) rst_n = 1'b1;
            @(posedge clk);
        end
    endtask

    task automatic build_input_matrices();
        begin
            for (int frame = 0; frame < 4; frame++) begin
                for (int row = 0; row < N; row++) begin
                    for (int col = 0; col < N; col++) begin
                        inputMatrix[frame][row][col] = frame*(N*N) + row*N + col - N;
                    end
                end
            end
        end
    endtask

    task automatic load_identity_weights();
        begin
            // Weight values shift downward through the array, so load bottom row first.
            for (int row = N-1; row >= 0; row--) begin
                @(negedge clk);
                for (int col = 0; col < N; col++)
                    weightData[col] = (row == col) ? 1 : 0;
                weightValid = 1'b1;
                do @(posedge clk); while (!weightReady);
                @(negedge clk) weightValid = 1'b0;
            end
            wait (weightsLoaded);
        end
    endtask

    task automatic send_activation_matrix(input int frame);
        begin
            for (int row = 0; row < N; row++) begin
                @(negedge clk);
                for (int col = 0; col < N; col++)
                    activationData[col] = inputMatrix[frame][row][col];
                activationValid = 1'b1;
                do @(posedge clk); while (!activationReady);
                @(negedge clk) activationValid = 1'b0;
                if ((row % 2) == 0) @(posedge clk); // legal input bubble
            end
        end
    endtask

    task automatic wait_for_start_accept();
        begin
            while (!output_start_ready) @(posedge clk);
            @(posedge clk);
        end
    endtask

    task automatic wait_for_buffered_frames(input int frameCount);
        bit allBuffered;
        begin
            do begin
                allBuffered = 1'b1;
                for (int lane = 0; lane < N; lane++)
                    allBuffered &= (dut.core.outputCount[lane] >= frameCount*N);
                if (!allBuffered) @(posedge clk);
            end while (!allBuffered);
        end
    endtask

    task automatic request_and_receive_frame(input int frame, input bit add_backpressure);
        begin
            @(negedge clk) output_start_valid = 1'b1;
            wait_for_start_accept();
            @(negedge clk) output_start_valid = 1'b0;
            receive_and_check_frame(frame, add_backpressure);
        end
    endtask

    task automatic receive_and_check_frame(input int frame, input bit add_backpressure);
        int row, cycle;
        logic [N*AXIS_LANE_WIDTH-1:0] heldData;
        logic heldLast, wasStalled;
        begin
            row = 0;
            cycle = 0;
            wasStalled = 1'b0;
            while (row < N) begin
                @(negedge clk);
                m_axis_tready = !add_backpressure || ((cycle % 4) != 1);
                @(posedge clk);

                if (wasStalled) begin
                    assert(m_axis_tvalid) else $error("TVALID dropped during output backpressure");
                    assert(m_axis_tdata == heldData) else $error("TDATA changed during output backpressure");
                    assert(m_axis_tlast == heldLast) else $error("TLAST changed during output backpressure");
                end

                if (m_axis_tvalid && m_axis_tready) begin
                    for (int col = 0; col < N; col++) begin
                        assert($signed(m_axis_tdata[col*AXIS_LANE_WIDTH +: RESULT_WIDTH]) == inputMatrix[frame][row][col])
                            else $error("Frame %0d result mismatch at row %0d, column %0d: got %0d expected %0d",
                                        frame, row, col,
                                        $signed(m_axis_tdata[col*AXIS_LANE_WIDTH +: RESULT_WIDTH]),
                                        inputMatrix[frame][row][col]);
                        assert(m_axis_tdata[col*AXIS_LANE_WIDTH + RESULT_WIDTH +: AXIS_LANE_WIDTH-RESULT_WIDTH] ==
                               {(AXIS_LANE_WIDTH-RESULT_WIDTH){inputMatrix[frame][row][col][WIDTH-1]}})
                            else $error("Frame %0d sign extension mismatch at row %0d, column %0d", frame, row, col);
                    end
                    assert(m_axis_tlast == (row == N-1))
                        else $error("TLAST mismatch for frame %0d row %0d", frame, row);
                    row++;
                end

                wasStalled = m_axis_tvalid && !m_axis_tready;
                heldData = m_axis_tdata;
                heldLast = m_axis_tlast;
                cycle++;
            end
            @(negedge clk) m_axis_tready = 1'b0;
        end
    endtask

endmodule

module matrixMultiplierWeightStationaryAxisOut_tb;
    logic done2, done3, done4;

    matrixMultiplierWeightStationaryAxisOut_testcase #(.N(2)) test_2x2 (.done(done2));
    matrixMultiplierWeightStationaryAxisOut_testcase #(.N(3)) test_3x3 (.done(done3));
    matrixMultiplierWeightStationaryAxisOut_testcase #(.N(4)) test_4x4 (.done(done4));

    initial begin
        wait (done2 && done3 && done4);
        $display("All AXI output wrapper tests passed.");
        $finish;
    end
endmodule
