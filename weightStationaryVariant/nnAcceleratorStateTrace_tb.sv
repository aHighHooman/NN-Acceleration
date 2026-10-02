`timescale 1ns/1ps

// Drive one external input bundle per clock and write semantic RTL state.
// All expected mathematics and timing live in Python.
module nnAcceleratorStateTrace_tb;
    localparam int N = 3;
    localparam int WIDTH = 8;
    localparam int TARGET_WIDTH = 8;
    localparam int REDUCTION_WEIGHT_WIDTH = 8;
    localparam int FRACTION_BITS = 0;
    localparam int IN_FLIGHT_DEPTH = 2*N+2;
    localparam int OUTPUT_FIFO_DEPTH = 2*N;
    localparam int MATRIX_RESULT_WIDTH = 2*WIDTH + $clog2(N);
    localparam int PREDICTION_WIDTH = MATRIX_RESULT_WIDTH + $clog2(N);
    localparam int SAMPLE_CONTEXT_WIDTH = TARGET_WIDTH + 2*N + 1;

    logic clk = 0;
    logic rst_n, weightValid, inputValid, trainingEnable, resultReady;
    logic loadReductionWeights, reloadWeights, passThrough, reduceOutput;
    logic signed [WIDTH-1:0] weightData[N], inputData[N];
    logic signed [TARGET_WIDTH-1:0] targetData;
    logic signed [REDUCTION_WEIGHT_WIDTH-1:0] reductionWeight[N];
    logic weightReady, inputReady, resultValid, weightsLoaded, reloadReady;
    logic signed [PREDICTION_WIDTH-1:0] resultData[N];

    integer stimulus_fd, trace_fd, status, cycle_count, input_cycle;
    integer scanned_reset_n, scanned_weight_valid;
    integer scanned_weight_lane0, scanned_weight_lane1, scanned_weight_lane2;
    integer scanned_input_valid;
    integer scanned_input_lane0, scanned_input_lane1, scanned_input_lane2;
    integer scanned_target, scanned_training_enable, scanned_result_ready;
    integer scanned_reduction_lane0, scanned_reduction_lane1, scanned_reduction_lane2;
    integer scanned_load_reduction_weights, scanned_reload_weights;
    integer scanned_pass_through, scanned_reduce_output;
    integer entry, lane, index;
    integer accepted, retired, retired_prediction, completed_direction, completed_target;
    integer completed_prediction, completed_training;
    integer accepted_count, completed_count, retired_count;
    logic last_training_completed;
    integer retired_activated[0:N-1], retired_result[0:N-1];
    integer enqueued_raw[0:N-1];
    integer completed_activated[0:N-1];
    string stimulus_path, trace_path;
    // Stream quiescence covers accepted samples, buffered results, and learning
    // updates; weight loading is outside this verification boundary.
    logic streamQuiescent;
    logic configurationActive;
    logic configuredPassThrough, configuredReduceOutput;
    logic trace_stalled, trace_reset_stalled, trace_matrix_update_pending;
    logic trace_matrix_result_handshake;
    logic trace_reduction_update_busy;
    logic held_update_valid[N-1], held_reduction_valid[N];
    logic signed [1:0] held_update_row[N-1][N], held_update_column[N-1][N];
    logic signed [2*N-1:0] held_reduction_direction[N];
    logic signed [WIDTH-1:0] held_skew_data[N][N];
    logic held_skew_valid[N][N];
    logic held_array_valid[N];

    always #5ns clk = ~clk;

    nnAccelerator #(.WIDTH(WIDTH), .N(N), .FRACTION_BITS(FRACTION_BITS),
        .TARGET_WIDTH(TARGET_WIDTH), .REDUCTION_WEIGHT_WIDTH(REDUCTION_WEIGHT_WIDTH),
        .IN_FLIGHT_DEPTH(IN_FLIGHT_DEPTH), .OUTPUT_FIFO_DEPTH(OUTPUT_FIFO_DEPTH)) dut (
        .clk(clk), .rst_n(rst_n), .weightData(weightData), .weightValid(weightValid),
        .weightReady(weightReady), .inputData(inputData), .targetData(targetData),
        .trainingEnable(trainingEnable), .inputValid(inputValid),
        .inputReady(inputReady), .reductionWeight(reductionWeight),
        .loadReductionWeights(loadReductionWeights), .reduceOutput(reduceOutput),
        .resultData(resultData),
        .resultValid(resultValid), .resultReady(resultReady),
        .weightsLoaded(weightsLoaded), .reloadWeights(reloadWeights),
        .reloadReady(reloadReady), .passThrough(passThrough)
    );

    // Check PE registers on the edge itself, including the data registers that
    // are absent from the architectural trace. The output-stall event is logged
    // separately so Python can check that the directed case reaches this path.
    for (genvar row = 0; row < N; row++) begin : freeze_row
        for (genvar col = 0; col < N; col++) begin : freeze_col
            always @(posedge clk) begin : check_pe_freeze
                logic signed [WIDTH-1:0] weight_before, activation_before;
                logic signed [MATRIX_RESULT_WIDTH-1:0] psum_before;
                if (rst_n && dut.matrixEngine.outputBlocked && !dut.matrixEngine.arrayAdvance) begin
                    weight_before = dut.matrixEngine.systolicArray.row_loop[row].col_loop[col].pe.weightReg;
                    activation_before = dut.matrixEngine.systolicArray.row_loop[row].col_loop[col].pe.activationOut;
                    psum_before = dut.matrixEngine.systolicArray.row_loop[row].col_loop[col].pe.psumOut;
                    #1ps;
                    if (weight_before !== dut.matrixEngine.systolicArray.row_loop[row].col_loop[col].pe.weightReg ||
                        activation_before !== dut.matrixEngine.systolicArray.row_loop[row].col_loop[col].pe.activationOut ||
                        psum_before !== dut.matrixEngine.systolicArray.row_loop[row].col_loop[col].pe.psumOut)
                        $fatal(1, "PE (%0d,%0d) advanced during output stall", row, col);
                end
            end
        end
    end

    assign streamQuiescent =
        !dut.matrixEngine.inputVectorValid &&
        !dut.matrixEngine.skewBusy &&
        !dut.matrixEngine.pipelineBusy &&
        !dut.matrixEngine.resultValid &&
        dut.sampleContextEmpty &&
        dut.resultFifo.empty &&
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

            if (inputValid && inputReady) begin
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
            $fwrite(trace_fd, "C %0d\n", c);
            $fwrite(trace_fd, "W %0d %0d %0d %0d %0d %0d %0d %0d %0d\n",
                $signed(dut.matrixEngine.systolicArray.row_loop[0].col_loop[0].pe.weightReg),
                $signed(dut.matrixEngine.systolicArray.row_loop[0].col_loop[1].pe.weightReg),
                $signed(dut.matrixEngine.systolicArray.row_loop[0].col_loop[2].pe.weightReg),
                $signed(dut.matrixEngine.systolicArray.row_loop[1].col_loop[0].pe.weightReg),
                $signed(dut.matrixEngine.systolicArray.row_loop[1].col_loop[1].pe.weightReg),
                $signed(dut.matrixEngine.systolicArray.row_loop[1].col_loop[2].pe.weightReg),
                $signed(dut.matrixEngine.systolicArray.row_loop[2].col_loop[0].pe.weightReg),
                $signed(dut.matrixEngine.systolicArray.row_loop[2].col_loop[1].pe.weightReg),
                $signed(dut.matrixEngine.systolicArray.row_loop[2].col_loop[2].pe.weightReg));
            $fwrite(trace_fd, "R");
            for (lane = 0; lane < N; lane++) $fwrite(trace_fd, " %0d", $signed(dut.residentReductionWeight[lane]));
            $fwrite(trace_fd, "\nWB");
            for (int row = 0; row < N; row++)
                for (int col = 0; col < N; col++)
                    $fwrite(trace_fd, " %0d", $signed(dut.matrixEngine.weightBuffer[row][col]));
            $fwrite(trace_fd, "\nWL %0d %0d %0d %0d %0d",
                dut.matrixEngine.weightsLoaded, dut.matrixEngine.loadedWeightRows,
                dut.matrixEngine.loadingWeights, dut.matrixEngine.loadStep,
                dut.matrixEngine.captureWeights);
            $fwrite(trace_fd, "\nIS %0d", dut.matrixEngine.inputVectorValid);
            if (dut.matrixEngine.inputVectorValid) begin
                for (lane = 0; lane < N; lane++)
                    $fwrite(trace_fd, " %0d", $signed(dut.matrixEngine.inputVectorData[lane*WIDTH +: WIDTH]));
            end
            $fwrite(trace_fd, "\nSF %0d", dut.sampleContextFifo.values);
            for (entry = 0; entry < dut.sampleContextFifo.values; entry++) begin
                index = dut.sampleContextFifo.readPtr + entry;
                if (index >= IN_FLIGHT_DEPTH) index = index - IN_FLIGHT_DEPTH;
                $fwrite(trace_fd, " %0d", $signed(dut.sampleContextFifo.data[index][SAMPLE_CONTEXT_WIDTH-1 -: TARGET_WIDTH]));
                for (lane = 0; lane < N; lane++)
                    $fwrite(trace_fd, " %0d", $signed(dut.sampleContextFifo.data[index][2*lane+1 +: 2]));
                $fwrite(trace_fd, " %0d", dut.sampleContextFifo.data[index][0]);
            end
            $fwrite(trace_fd, "\nRF %0d", dut.resultFifo.values);
            for (entry = 0; entry < dut.resultFifo.values; entry++) begin
                index = dut.resultFifo.readPtr + entry;
                if (index >= OUTPUT_FIFO_DEPTH) index = index - OUTPUT_FIFO_DEPTH;
                for (lane = 0; lane < N; lane++)
                    $fwrite(trace_fd, " %0d", $signed(dut.resultFifo.data[index][lane*MATRIX_RESULT_WIDTH +: MATRIX_RESULT_WIDTH]));
                $fwrite(trace_fd, " %0d", $signed(dut.resultFifo.data[index][N*MATRIX_RESULT_WIDTH +: PREDICTION_WIDTH]));
            end
            $fwrite(trace_fd, "\n");
        end
    endtask

    task automatic capture_stalled_pipelines;
        for (int stage = 0; stage < N-1; stage++) begin
            held_update_valid[stage] = dut.matrixEngine.systolicArray.updateValidPipe[stage];
            for (int wire_lane = 0; wire_lane < N; wire_lane++) begin
                held_update_row[stage][wire_lane] = dut.matrixEngine.systolicArray.updateRowPipe[stage][wire_lane];
                held_update_column[stage][wire_lane] = dut.matrixEngine.systolicArray.updateColumnPipe[stage][wire_lane];
            end
        end
        for (int stage = 0; stage < N; stage++) begin
            held_reduction_valid[stage] = dut.reductionUpdateValidPipe[stage];
            held_reduction_direction[stage] = dut.reductionUpdateDirectionPipe[stage];
        end
        for (int wire_lane = 0; wire_lane < N; wire_lane++) begin
            for (int stage = 0; stage < N; stage++) begin
                held_skew_data[wire_lane][stage] = dut.matrixEngine.skewData[wire_lane][stage];
                held_skew_valid[wire_lane][stage] = dut.matrixEngine.skewValid[wire_lane][stage];
            end
        end
        for (int stage = 0; stage < N; stage++)
            held_array_valid[stage] = dut.matrixEngine.systolicArray.validPipe[stage];
    endtask

    task automatic check_stalled_pipelines(input integer c);
        for (int stage = 0; stage < N-1; stage++) begin
            if (held_update_valid[stage] !== dut.matrixEngine.systolicArray.updateValidPipe[stage])
                $fatal(1, "matrix update valid advanced during stall at cycle %0d", c);
            for (int wire_lane = 0; wire_lane < N; wire_lane++)
                if (held_update_row[stage][wire_lane] !== dut.matrixEngine.systolicArray.updateRowPipe[stage][wire_lane] ||
                    held_update_column[stage][wire_lane] !== dut.matrixEngine.systolicArray.updateColumnPipe[stage][wire_lane])
                    $fatal(1, "matrix update payload advanced during stall at cycle %0d", c);
        end
        for (int stage = 0; stage < N; stage++)
            if (held_reduction_valid[stage] !== dut.reductionUpdateValidPipe[stage] ||
                held_reduction_direction[stage] !== dut.reductionUpdateDirectionPipe[stage])
                $fatal(1, "reduction update pipe advanced during stall at cycle %0d", c);
        for (int wire_lane = 0; wire_lane < N; wire_lane++) begin
            for (int stage = 0; stage < N; stage++)
                if (held_skew_data[wire_lane][stage] !== dut.matrixEngine.skewData[wire_lane][stage] ||
                    held_skew_valid[wire_lane][stage] !== dut.matrixEngine.skewValid[wire_lane][stage])
                    $fatal(1, "input skew advanced during stall at cycle %0d", c);
        end
        for (int stage = 0; stage < N; stage++)
            if (held_array_valid[stage] !== dut.matrixEngine.systolicArray.validPipe[stage])
                $fatal(1, "array valid advanced during stall at cycle %0d", c);
    endtask

    initial begin
        if (!$value$plusargs("STIMULUS=%s", stimulus_path)) $fatal(1, "missing +STIMULUS path");
        if (!$value$plusargs("TRACE=%s", trace_path)) $fatal(1, "missing +TRACE path");
        stimulus_fd = $fopen(stimulus_path, "r");
        trace_fd = $fopen(trace_path, "w");
        if (!stimulus_fd || !trace_fd) $fatal(1, "cannot open stimulus or trace file");
        status = $fscanf(stimulus_fd, "%d\n", cycle_count);
        if (status != 1) $fatal(1, "bad stimulus header");
        rst_n = 0; weightValid = 0; inputValid = 0; trainingEnable = 0;
        resultReady = 0; loadReductionWeights = 0; reloadWeights = 0;
        passThrough = 1; reduceOutput = 0; targetData = 0;
        for (lane = 0; lane < N; lane++) begin weightData[lane] = 0; inputData[lane] = 0; reductionWeight[lane] = 0; end
        accepted_count = 0; completed_count = 0; retired_count = 0;

        for (integer c = 0; c < cycle_count; c++) begin
            @(negedge clk);
            status = $fscanf(stimulus_fd,
                "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",
                input_cycle, scanned_reset_n, scanned_weight_valid,
                scanned_weight_lane0, scanned_weight_lane1, scanned_weight_lane2,
                scanned_input_valid, scanned_input_lane0,
                scanned_input_lane1, scanned_input_lane2, scanned_target,
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
            inputValid = scanned_input_valid;
            inputData[0] = scanned_input_lane0;
            inputData[1] = scanned_input_lane1;
            inputData[2] = scanned_input_lane2;
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
            // Sample after stimulus but before the rising edge; the post-edge snapshot
            // reflects FIFO changes and describes the next edge's contract.
            #1ps;
            accepted = rst_n && inputValid && inputReady;
            retired = rst_n && resultValid && resultReady;
            trace_matrix_result_handshake = rst_n && dut.matrixResultValid && dut.matrixResultReady;
            if (trace_matrix_result_handshake && dut.sampleContextEmpty)
                $fatal(1, "result completed without a context at cycle %0d", c);
            last_training_completed = trace_matrix_result_handshake &&
                dut.sampleContextFifo.values == 1 && dut.trainingEnableHead;
            retired_prediction = $signed(dut.prediction);
            completed_prediction = $signed(dut.enqueuePrediction);
            completed_direction = $signed(dut.learningDirection);
            completed_target = $signed(dut.resultTargetData);
            completed_training = dut.matrixUpdateValid;
            for (lane = 0; lane < N; lane++) begin
                retired_activated[lane] = $signed(dut.activatedData[lane]);
                retired_result[lane] = $signed(resultData[lane]);
                completed_activated[lane] = $signed(dut.activatedMatrixResultData[lane]);
            end
            for (lane = 0; lane < N; lane++)
                enqueued_raw[lane] = $signed(dut.rawResultData[lane]);
            trace_stalled = rst_n && dut.matrixEngine.outputBlocked && !dut.matrixEngine.arrayAdvance;
            trace_reset_stalled = !rst_n && dut.matrixEngine.outputBlocked && !dut.matrixEngine.arrayAdvance;
            trace_reduction_update_busy = dut.reductionUpdateBusy;
            trace_matrix_update_pending = 0;
            for (entry = 0; entry < N-1; entry++)
                if (dut.matrixEngine.systolicArray.updateValidPipe[entry])
                    trace_matrix_update_pending = 1;
            if (trace_stalled)
                capture_stalled_pipelines();
            @(posedge clk);
            #1ps;
            if (!rst_n) begin
                accepted_count = 0;
                completed_count = 0;
                retired_count = 0;
            end else begin
                accepted_count += accepted;
                completed_count += trace_matrix_result_handshake;
                retired_count += retired;
            end
            if (dut.sampleContextFifo.values !== accepted_count - completed_count)
                $fatal(1, "context occupancy differs from accepted minus completed at cycle %0d", c);
            if (dut.resultFifo.values !== completed_count - retired_count)
                $fatal(1, "output occupancy differs from completed minus retired at cycle %0d", c);
            if (!dut.resultFifoEmpty && reloadReady)
                $fatal(1, "reload allowed with buffered output at cycle %0d", c);
            if (dut.sampleContextEmpty) begin
                if (dut.matrixEngine.inputVectorValid || dut.matrixEngine.skewBusy ||
                    dut.matrixEngine.resultValid)
                    $fatal(1, "sample datapath live with empty context at cycle %0d", c);
                for (int stage = 0; stage < N; stage++)
                    if (dut.matrixEngine.systolicArray.validPipe[stage])
                        $fatal(1, "array sample live with empty context at cycle %0d", c);
            end
            if (!rst_n) begin
                for (int stage = 0; stage < N-1; stage++)
                    if (dut.matrixEngine.systolicArray.updateValidPipe[stage])
                        $fatal(1, "reset retained matrix update at cycle %0d", c);
                for (int stage = 0; stage < N; stage++)
                    if (dut.reductionUpdateValidPipe[stage])
                        $fatal(1, "reset retained reduction update at cycle %0d", c);
            end
            if (last_training_completed && dut.sampleContextEmpty) begin
                if (!dut.matrixEngine.systolicArray.updateValidPipe[0] ||
                    !dut.reductionUpdateValidPipe[0] ||
                    dut.matrixReloadReady || reloadReady)
                    $fatal(1, "last training completion lost update-tail interlock at cycle %0d", c);
            end
            if (trace_stalled)
                check_stalled_pipelines(c);
            dump_snapshot(c);
            if (trace_stalled)
                $fwrite(trace_fd, "STALL %0d %0d %0d\n", c,
                    trace_matrix_update_pending, trace_reduction_update_busy);
            if (trace_reset_stalled)
                $fwrite(trace_fd, "RESET_STALL %0d\n", c);
            if (trace_matrix_result_handshake) $fwrite(trace_fd,
                "ENQ %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d\n", c,
                enqueued_raw[0], enqueued_raw[1], enqueued_raw[2],
                completed_activated[0], completed_activated[1], completed_activated[2],
                completed_prediction, completed_direction, completed_target, completed_training);
            if (retired) $fwrite(trace_fd,
                "RT %0d %0d %0d %0d %0d %0d %0d %0d\n",
                c, retired_activated[0], retired_activated[1], retired_activated[2],
                retired_prediction,
                retired_result[0], retired_result[1], retired_result[2]);
        end
        $fclose(stimulus_fd); $fclose(trace_fd);
        $display("PASS: wrote %0d post-edge snapshots", cycle_count);
        $finish;
    end
endmodule
