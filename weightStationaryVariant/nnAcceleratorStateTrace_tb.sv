`timescale 1ns/1ps

// Drive one external input bundle per clock and write semantic RTL state.
// All expected mathematics and timing live in Python.
module nnAcceleratorStateTrace_tb;
    localparam int N = 3;
    localparam int WIDTH = 8;
    localparam int TARGET_WIDTH = 8;
    localparam int REDUCTION_WEIGHT_WIDTH = 8;
    localparam int FRACTION_BITS = 0;
    localparam int INPUT_FIFO_DEPTH = 2*N;
    localparam int OUTPUT_FIFO_DEPTH = 2*N;
    localparam int MATRIX_RESULT_WIDTH = 2*WIDTH + $clog2(N);
    localparam int PREDICTION_WIDTH = MATRIX_RESULT_WIDTH + $clog2(N);
    localparam int SAMPLE_CONTEXT_WIDTH = TARGET_WIDTH + 2*N + 1;
    localparam int SAMPLE_CONTEXT_DEPTH = (INPUT_FIFO_DEPTH > 2*N+2) ? INPUT_FIFO_DEPTH : 2*N+2;

    logic clk = 0;
    logic rst_n, weightValid, activationValid, trainingEnable, resultReady;
    logic loadReductionWeights, reloadWeights, passThrough, reduceOutput;
    logic signed [WIDTH-1:0] weightData[N], activationData[N];
    logic signed [TARGET_WIDTH-1:0] targetData;
    logic signed [REDUCTION_WEIGHT_WIDTH-1:0] reductionWeight[N];
    logic weightReady, activationReady, resultValid, resultLast, weightsLoaded, reloadReady;
    logic signed [PREDICTION_WIDTH-1:0] resultData[N];
    logic signed [TARGET_WIDTH-1:0] resultTargetData;
    logic signed [1:0] learningDirection, rowDirection[N], columnDirection[N];
    logic matrixUpdateValid;

    integer stimulus_fd, trace_fd, status, cycle_count, input_cycle;
    integer scanned_reset_n, scanned_weight_valid;
    integer scanned_weight_lane0, scanned_weight_lane1, scanned_weight_lane2;
    integer scanned_activation_valid;
    integer scanned_activation_lane0, scanned_activation_lane1, scanned_activation_lane2;
    integer scanned_target, scanned_training_enable, scanned_result_ready;
    integer scanned_reduction_lane0, scanned_reduction_lane1, scanned_reduction_lane2;
    integer scanned_load_reduction_weights, scanned_reload_weights;
    integer scanned_pass_through, scanned_reduce_output;
    integer entry, lane, index;
    integer retired, retired_last, retired_prediction, retired_direction, retired_target;
    integer retired_raw[0:N-1], retired_activated[0:N-1], retired_result[0:N-1];
    string stimulus_path, trace_path;
    // Verification-only contract state. Stream quiescence describes only
    // accepted samples, buffered results, and learning updates. In particular,
    // the output frame position used by resultLast/reloadReady is not
    // outstanding work; reloadReady additionally requires that position to be
    // row zero.
    logic streamQuiescent;
    logic configurationActive;
    logic configuredPassThrough, configuredReduceOutput;
    logic trace_datapath_advance, trace_output_blocked;
    logic trace_reduction_update_busy, trace_pipeline_busy, trace_result_align_busy;
    integer trace_matrix_wave_mask, trace_reduction_pipe_mask;
    integer trace_skew_valid_mask, trace_align_valid_mask;

    always #5ns clk = ~clk;

    nnAccelerator #(.WIDTH(WIDTH), .N(N), .FRACTION_BITS(FRACTION_BITS),
        .TARGET_WIDTH(TARGET_WIDTH), .REDUCTION_WEIGHT_WIDTH(REDUCTION_WEIGHT_WIDTH),
        .INPUT_FIFO_DEPTH(INPUT_FIFO_DEPTH), .OUTPUT_FIFO_DEPTH(OUTPUT_FIFO_DEPTH)) dut (
        .clk(clk), .rst_n(rst_n), .weightData(weightData), .weightValid(weightValid),
        .weightReady(weightReady), .activationData(activationData), .targetData(targetData),
        .trainingEnable(trainingEnable), .activationValid(activationValid),
        .activationReady(activationReady), .reductionWeight(reductionWeight),
        .loadReductionWeights(loadReductionWeights), .reduceOutput(reduceOutput),
        .resultData(resultData), .resultTargetData(resultTargetData),
        .learningDirection(learningDirection), .rowDirection(rowDirection),
        .columnDirection(columnDirection), .matrixUpdateValid(matrixUpdateValid),
        .resultValid(resultValid), .resultReady(resultReady), .resultLast(resultLast),
        .weightsLoaded(weightsLoaded), .reloadWeights(reloadWeights),
        .reloadReady(reloadReady), .passThrough(passThrough)
    );

    assign streamQuiescent =
        dut.matrixEngine.activationEmpty &&
        !dut.matrixEngine.skewBusy &&
        !dut.matrixEngine.pipelineBusy &&
        !dut.matrixEngine.resultAlignBusy &&
        dut.matrixEngine.outputEmpty &&
        dut.sampleContextEmpty &&
        dut.resultReadoutFifo.empty &&
        !dut.reductionUpdateBusy;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            configurationActive <= 1'b0;
            configuredPassThrough <= passThrough;
            configuredReduceOutput <= reduceOutput;
        end else begin
            if (configurationActive && !streamQuiescent) begin
                if (passThrough !== configuredPassThrough)
                    $fatal(1, "passThrough changed while accelerator work was outstanding");
                if (reduceOutput !== configuredReduceOutput)
                    $fatal(1, "reduceOutput changed while accelerator work was outstanding");
            end

            if (activationValid && activationReady) begin
                if (!configurationActive || streamQuiescent) begin
                    configuredPassThrough <= passThrough;
                    configuredReduceOutput <= reduceOutput;
                end
                configurationActive <= 1'b1;
            end else if (configurationActive && streamQuiescent) begin
                configurationActive <= 1'b0;
            end
        end
    end

    task automatic dump_snapshot(input integer c);
        begin
            if (dut.matrixEngine.outputVectorFifo.values !==
                dut.resultReadoutFifo.values)
                $fatal(1, "output and readout FIFO occupancies diverged at cycle %0d", c);
            $fwrite(trace_fd, "C %0d\n", c);
            $fwrite(trace_fd, "W %0d %0d %0d %0d %0d %0d %0d %0d %0d\n",
                $signed(dut.matrixEngine.systolicArr.row_loop[0].col_loop[0].mb.weightReg),
                $signed(dut.matrixEngine.systolicArr.row_loop[0].col_loop[1].mb.weightReg),
                $signed(dut.matrixEngine.systolicArr.row_loop[0].col_loop[2].mb.weightReg),
                $signed(dut.matrixEngine.systolicArr.row_loop[1].col_loop[0].mb.weightReg),
                $signed(dut.matrixEngine.systolicArr.row_loop[1].col_loop[1].mb.weightReg),
                $signed(dut.matrixEngine.systolicArr.row_loop[1].col_loop[2].mb.weightReg),
                $signed(dut.matrixEngine.systolicArr.row_loop[2].col_loop[0].mb.weightReg),
                $signed(dut.matrixEngine.systolicArr.row_loop[2].col_loop[1].mb.weightReg),
                $signed(dut.matrixEngine.systolicArr.row_loop[2].col_loop[2].mb.weightReg));
            $fwrite(trace_fd, "R");
            for (lane = 0; lane < N; lane++) $fwrite(trace_fd, " %0d", $signed(dut.residentReductionWeight[lane]));
            $fwrite(trace_fd, "\nWF %0d", dut.matrixEngine.weightVectorFifo.values);
            for (entry = 0; entry < dut.matrixEngine.weightVectorFifo.values; entry++) begin
                index = dut.matrixEngine.weightVectorFifo.readPtr + entry;
                if (index >= N) index = index - N;
                for (lane = 0; lane < N; lane++)
                    $fwrite(trace_fd, " %0d", $signed(dut.matrixEngine.weightVectorFifo.data[index][lane*WIDTH +: WIDTH]));
            end
            $fwrite(trace_fd, "\nAF %0d", dut.matrixEngine.activationVectorFifo.values);
            for (entry = 0; entry < dut.matrixEngine.activationVectorFifo.values; entry++) begin
                index = dut.matrixEngine.activationVectorFifo.readPtr + entry;
                if (index >= INPUT_FIFO_DEPTH) index = index - INPUT_FIFO_DEPTH;
                for (lane = 0; lane < N; lane++)
                    $fwrite(trace_fd, " %0d", $signed(dut.matrixEngine.activationVectorFifo.data[index][lane*WIDTH +: WIDTH]));
            end
            $fwrite(trace_fd, "\nSF %0d", dut.sampleContextFifo.values);
            for (entry = 0; entry < dut.sampleContextFifo.values; entry++) begin
                index = dut.sampleContextFifo.readPtr + entry;
                if (index >= SAMPLE_CONTEXT_DEPTH) index = index - SAMPLE_CONTEXT_DEPTH;
                $fwrite(trace_fd, " %0d", $signed(dut.sampleContextFifo.data[index][SAMPLE_CONTEXT_WIDTH-1 -: TARGET_WIDTH]));
                for (lane = 0; lane < N; lane++)
                    $fwrite(trace_fd, " %0d", $signed(dut.sampleContextFifo.data[index][2*lane+1 +: 2]));
                $fwrite(trace_fd, " %0d", dut.sampleContextFifo.data[index][0]);
            end
            $fwrite(trace_fd, "\nOF %0d", dut.matrixEngine.outputVectorFifo.values);
            for (entry = 0; entry < dut.matrixEngine.outputVectorFifo.values; entry++) begin
                index = dut.matrixEngine.outputVectorFifo.readPtr + entry;
                if (index >= OUTPUT_FIFO_DEPTH) index = index - OUTPUT_FIFO_DEPTH;
                for (lane = 0; lane < N; lane++)
                    $fwrite(trace_fd, " %0d", $signed(dut.matrixEngine.outputVectorFifo.data[index][lane*MATRIX_RESULT_WIDTH +: MATRIX_RESULT_WIDTH]));
            end
            $fwrite(trace_fd, "\nRF %0d", dut.resultReadoutFifo.values);
            for (entry = 0; entry < dut.resultReadoutFifo.values; entry++) begin
                index = dut.resultReadoutFifo.readPtr + entry;
                if (index >= OUTPUT_FIFO_DEPTH) index = index - OUTPUT_FIFO_DEPTH;
                $fwrite(trace_fd, " %0d", $signed(dut.resultReadoutFifo.data[index][PREDICTION_WIDTH+2*N-1:2*N]));
                for (lane = 0; lane < N; lane++)
                    $fwrite(trace_fd, " %0d", $signed(dut.resultReadoutFifo.data[index][2*lane +: 2]));
            end
            $fwrite(trace_fd, "\n");
        end
    endtask

    initial begin
        if (!$value$plusargs("STIMULUS=%s", stimulus_path)) $fatal(1, "missing +STIMULUS path");
        if (!$value$plusargs("TRACE=%s", trace_path)) $fatal(1, "missing +TRACE path");
        stimulus_fd = $fopen(stimulus_path, "r");
        trace_fd = $fopen(trace_path, "w");
        if (!stimulus_fd || !trace_fd) $fatal(1, "cannot open stimulus or trace file");
        status = $fscanf(stimulus_fd, "%d\n", cycle_count);
        if (status != 1) $fatal(1, "bad stimulus header");
        rst_n = 0; weightValid = 0; activationValid = 0; trainingEnable = 0;
        resultReady = 0; loadReductionWeights = 0; reloadWeights = 0;
        passThrough = 1; reduceOutput = 0; targetData = 0;
        for (lane = 0; lane < N; lane++) begin weightData[lane] = 0; activationData[lane] = 0; reductionWeight[lane] = 0; end

        for (integer c = 0; c < cycle_count; c++) begin
            @(negedge clk);
            status = $fscanf(stimulus_fd,
                "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",
                input_cycle, scanned_reset_n, scanned_weight_valid,
                scanned_weight_lane0, scanned_weight_lane1, scanned_weight_lane2,
                scanned_activation_valid, scanned_activation_lane0,
                scanned_activation_lane1, scanned_activation_lane2, scanned_target,
                scanned_training_enable, scanned_result_ready,
                scanned_reduction_lane0, scanned_reduction_lane1,
                scanned_reduction_lane2, scanned_load_reduction_weights,
                scanned_reload_weights, scanned_pass_through, scanned_reduce_output);
            if (status != 20 || input_cycle != c) $fatal(1, "bad stimulus at cycle %0d", c);
            rst_n = scanned_reset_n;
            weightValid = scanned_weight_valid;
            weightData[0] = scanned_weight_lane0;
            weightData[1] = scanned_weight_lane1;
            weightData[2] = scanned_weight_lane2;
            activationValid = scanned_activation_valid;
            activationData[0] = scanned_activation_lane0;
            activationData[1] = scanned_activation_lane1;
            activationData[2] = scanned_activation_lane2;
            targetData = scanned_target;
            trainingEnable = scanned_training_enable;
            resultReady = scanned_result_ready;
            reductionWeight[0] = scanned_reduction_lane0;
            reductionWeight[1] = scanned_reduction_lane1;
            reductionWeight[2] = scanned_reduction_lane2;
            loadReductionWeights = scanned_load_reduction_weights;
            reloadWeights = scanned_reload_weights;
            passThrough = scanned_pass_through;
            reduceOutput = scanned_reduce_output;
            // Sample the combinational contract after the stimulus is applied
            // but before this cycle's rising edge.  The post-edge snapshot
            // alone describes the next edge after FIFO state may have moved.
            #1ps;
            trace_datapath_advance = dut.matrixEngine.datapathAdvance;
            trace_output_blocked = dut.matrixEngine.outputBlocked;
            trace_reduction_update_busy = dut.reductionUpdateBusy;
            trace_pipeline_busy = dut.matrixEngine.pipelineBusy;
            trace_result_align_busy = dut.matrixEngine.resultAlignBusy;
            trace_matrix_wave_mask = 0;
            for (entry = 0; entry < 2*N-2; entry++)
                if (dut.matrixEngine.systolicArr.updateValidPipe[entry])
                    trace_matrix_wave_mask |= (1 << entry);
            trace_reduction_pipe_mask = 0;
            for (entry = 0; entry < 2*N-1; entry++)
                if (dut.reductionUpdateValidPipe[entry])
                    trace_reduction_pipe_mask |= (1 << entry);
            trace_skew_valid_mask = 0;
            for (lane = 0; lane < N; lane++)
                for (index = 0; index < N; index++)
                    if (dut.matrixEngine.skewValid[lane][index])
                        trace_skew_valid_mask |= (1 << (lane*N + index));
            trace_align_valid_mask = 0;
            for (lane = 0; lane < N; lane++)
                for (index = 0; index < N-1; index++)
                    if (dut.matrixEngine.resultAlignValid[lane][index])
                        trace_align_valid_mask |= (1 << (lane*(N-1) + index));
            @(posedge clk);
            retired = resultValid && resultReady;
            retired_last = resultLast; retired_prediction = $signed(dut.prediction);
            retired_direction = $signed(learningDirection); retired_target = $signed(resultTargetData);
            for (lane = 0; lane < N; lane++) begin
                retired_raw[lane] = $signed(dut.rawResultData[lane]);
                retired_activated[lane] = $signed(dut.activatedData[lane]);
                retired_result[lane] = $signed(resultData[lane]);
            end
            #1ps;
            dump_snapshot(c);
            // Verification-only hierarchical evidence.  These are internal
            // combinational states, not synthesizable accelerator ports.
            $fwrite(trace_fd, "P %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d\n", c,
                trace_datapath_advance, trace_output_blocked,
                trace_reduction_update_busy, trace_pipeline_busy,
                trace_result_align_busy, trace_matrix_wave_mask,
                trace_reduction_pipe_mask, trace_skew_valid_mask,
                trace_align_valid_mask);
            if (retired) $fwrite(trace_fd,
                "RT %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d\n",
                c, retired_raw[0], retired_raw[1], retired_raw[2],
                retired_activated[0], retired_activated[1], retired_activated[2],
                retired_prediction, retired_direction, retired_target,
                retired_result[0], retired_result[1], retired_result[2], retired_last);
        end
        $fclose(stimulus_fd); $fclose(trace_fd);
        $display("PASS: wrote %0d post-edge snapshots", cycle_count);
        $finish;
    end
endmodule
