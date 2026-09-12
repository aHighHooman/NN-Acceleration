    // ------------------------------------------------------------------
    // Phase 5F accelerator-level training test
    // ------------------------------------------------------------------

    class nn_uvm_training_test extends uvm_test;
        `uvm_component_utils(nn_uvm_training_test)

        localparam int TARGET_WIDTH = WIDTH;
        localparam int REDUCTION_WEIGHT_WIDTH = 8;
        localparam int FRACTION_BITS = 4;
        localparam int REDUCTION_FRACTION_BITS = REDUCTION_WEIGHT_WIDTH - 1;
        localparam int TRAIN_PREDICTION_WIDTH = 2*WIDTH + 2*$clog2(N);
        localparam int UPDATE_STAGES = 2*N - 1;
        localparam int SCALE = 1 << FRACTION_BITS;
        localparam int HALF = 1 << (REDUCTION_FRACTION_BITS - 1);

        typedef logic signed [TRAIN_PREDICTION_WIDTH-1:0] train_result_t;
        virtual nn_training_if #(
            WIDTH, N, TARGET_WIDTH, REDUCTION_WEIGHT_WIDTH
        ) vif;

        function new(string name, uvm_component parent);
            super.new(name, parent);
        endfunction

        function void build_phase(uvm_phase phase);
            super.build_phase(phase);
            if (!uvm_config_db #(virtual nn_training_if #(
                    WIDTH, N, TARGET_WIDTH, REDUCTION_WEIGHT_WIDTH
                ))::get(this, "", "training_vif", vif))
                `uvm_fatal("NO_TRAIN_VIF", "training test did not receive nn_training_if")
        endfunction

        function automatic logic signed [1:0] sign_of(input data_t value);
            if (value > 0)
                return 2'sd1;
            if (value < 0)
                return -2'sd1;
            return 2'sd0;
        endfunction

        function automatic train_result_t predict(
            input data_t sample[N],
            input data_t matrix_weight[N][N],
            input logic signed [REDUCTION_WEIGHT_WIDTH-1:0]
                reduction_weight[N]
        );
            longint signed matrix_sum;
            longint signed reduction_sum;
            reduction_sum = 0;
            for (int col = 0; col < N; col++) begin
                matrix_sum = 0;
                for (int row = 0; row < N; row++)
                    matrix_sum += $signed(sample[row]) *
                                  $signed(matrix_weight[row][col]);
                reduction_sum += matrix_sum * $signed(reduction_weight[col]);
            end
            return train_result_t'(
                reduction_sum >>>
                (FRACTION_BITS + REDUCTION_FRACTION_BITS));
        endfunction

        task reset_dut();
            @(negedge vif.clk);
            vif.rst_n = 1'b0;
            vif.weightValid = 1'b0;
            vif.activationValid = 1'b0;
            vif.resultReady = 1'b0;
            vif.reloadWeights = 1'b0;
            vif.loadReductionWeights = 1'b0;
            repeat (3) @(posedge vif.clk);
            @(negedge vif.clk);
            vif.rst_n = 1'b1;
        endtask

        task load_initial_weights();
            @(negedge vif.clk);
            for (int lane = 0; lane < N; lane++)
                vif.reductionWeight[lane] = HALF;
            vif.loadReductionWeights = 1'b1;
            @(posedge vif.clk);
            @(negedge vif.clk);
            vif.loadReductionWeights = 1'b0;

            // Identity matrix in the core's reverse-row loading order.
            for (int row_offset = 0; row_offset < N; row_offset++) begin
                for (int lane = 0; lane < N; lane++)
                    vif.weightData[lane] =
                        ((N-1-row_offset) == lane) ? SCALE : 0;
                vif.weightValid = 1'b1;
                do @(posedge vif.clk);
                while (!(vif.weightValid && vif.weightReady));
                @(negedge vif.clk);
                vif.weightValid = 1'b0;
            end
            wait(vif.weightsLoaded === 1'b1);
        endtask

        task send_sample(
            input data_t sample[N],
            input integer target,
            input bit train,
            input int unsigned bubbles
        );
            vif.activationValid = 1'b0;
            repeat (bubbles) @(negedge vif.clk);
            for (int lane = 0; lane < N; lane++)
                vif.activationData[lane] = sample[lane];
            vif.targetData = target;
            vif.trainingEnable = train;
            vif.activationValid = 1'b1;
            do @(posedge vif.clk);
            while (!(vif.activationValid && vif.activationReady));
            @(negedge vif.clk);
            vif.activationValid = 1'b0;
        endtask

        task check_and_consume(
            input int unsigned sample_index,
            input data_t sample[N],
            input integer expected_target,
            input train_result_t expected_prediction,
            input bit expected_train
        );
            while (!vif.resultValid) @(negedge vif.clk);
            if ($signed(vif.resultTargetData) !== expected_target ||
                $signed(vif.resultData[0]) !== expected_prediction)
                `uvm_error("TRAIN_RESULT", $sformatf(
                    "sample %0d got target/prediction %0d/%0d expected %0d/%0d",
                    sample_index, vif.resultTargetData, vif.resultData[0],
                    expected_target, expected_prediction))
            if (vif.bufferedTrainingEnable !== expected_train)
                `uvm_error("TRAIN_ALIGN", $sformatf(
                    "sample %0d buffered trainingEnable=%0b expected %0b",
                    sample_index, vif.bufferedTrainingEnable, expected_train))
            if (vif.matrixUpdateValid !== expected_train)
                `uvm_error("TRAIN_UPDATE", $sformatf(
                    "sample %0d matrixUpdateValid=%0b expected %0b",
                    sample_index, vif.matrixUpdateValid, expected_train))
            for (int lane = 0; lane < N; lane++) begin
                if (vif.rowDirection[lane] !== sign_of(sample[lane]))
                    `uvm_error("TRAIN_SIGNS", $sformatf(
                        "sample %0d lane %0d row sign %0d expected %0d",
                        sample_index, lane, vif.rowDirection[lane],
                        sign_of(sample[lane])))
                if (lane != 0 && vif.resultData[lane] !== '0)
                    `uvm_error("TRAIN_REDUCTION", $sformatf(
                        "sample %0d nonzero reduced lane %0d", sample_index, lane))
            end
            @(posedge vif.clk);
            @(negedge vif.clk);
        endtask

        task run_phase(uvm_phase phase);
            data_t samples[7][N];
            train_result_t expected_prediction[7];
            integer expected_target[7];
            bit expected_train[7];
            data_t inference_matrix_before[N][N];
            logic signed [REDUCTION_WEIGHT_WIDTH-1:0]
                inference_reduction_before[N];
            int unsigned watchdog;

            phase.raise_objection(this);
            reset_dut();
            load_initial_weights();

            samples[0] = '{data_t'(SCALE), data_t'(0), data_t'(0)};
            samples[1] = '{data_t'(2*SCALE), data_t'(0), data_t'(0)};
            samples[2] = '{data_t'(-SCALE), data_t'(SCALE), data_t'(0)};
            samples[3] = '{data_t'(SCALE), data_t'(SCALE), data_t'(0)};
            expected_target[0] = SCALE;
            expected_target[1] = 0;
            expected_target[2] = 0;
            expected_target[3] = 2*SCALE;
            expected_prediction[0] = SCALE/2;
            expected_prediction[1] = SCALE;
            expected_prediction[2] = 0;
            expected_prediction[3] = SCALE;
            expected_train[0] = 0;
            expected_train[1] = 1;
            expected_train[2] = 0;
            expected_train[3] = 1;

            // Four consecutive samples are accepted before any result is
            // released.  The live control intentionally disagrees with each
            // earlier buffered value as the results are inspected.
            vif.resultReady = 1'b0;
            for (int sample_index = 0; sample_index < 4; sample_index++)
                send_sample(samples[sample_index], expected_target[sample_index],
                            expected_train[sample_index], 0);
            vif.trainingEnable = 1'b1;
            wait(vif.resultValid);
            if (vif.bufferedTrainingEnable !== 1'b0 ||
                vif.trainingEnable !== 1'b1)
                `uvm_error("TRAIN_LIVE", "sample 0 did not retain trainingEnable=0 while live input was 1")
            repeat (3) @(posedge vif.clk);

            @(negedge vif.clk);
            vif.resultReady = 1'b1;
            for (int sample_index = 0; sample_index < 4; sample_index++) begin
                vif.trainingEnable = ~expected_train[sample_index];
                check_and_consume(sample_index, samples[sample_index],
                                  expected_target[sample_index],
                                  expected_prediction[sample_index],
                                  expected_train[sample_index]);
            end

            watchdog = 0;
            while ((!vif.reductionUpdateEmpty ||
                    (vif.updateStageValid != '0)) && watchdog < 1000) begin
                @(negedge vif.clk);
                watchdog++;
            end
            if (watchdog == 1000)
                `uvm_fatal("TRAIN_DRAIN", "learning update pipeline did not drain")

            for (int row = 0; row < N; row++)
                for (int col = 0; col < N; col++)
                    inference_matrix_before[row][col] =
                        vif.residentMatrixWeight[row][col];
            for (int lane = 0; lane < N; lane++)
                inference_reduction_before[lane] =
                    vif.residentReductionWeight[lane];

            samples[4] = '{data_t'(SCALE), data_t'(0), data_t'(0)};
            samples[5] = '{data_t'(0), data_t'(SCALE), data_t'(0)};
            samples[6] = '{data_t'(-SCALE), data_t'(SCALE), data_t'(0)};
            for (int sample_index = 4; sample_index < 7; sample_index++) begin
                expected_target[sample_index] = 0;
                expected_train[sample_index] = 0;
                expected_prediction[sample_index] = predict(
                    samples[sample_index], inference_matrix_before,
                    inference_reduction_before);
            end

            // Inference samples include input bubbles and a held output.  The
            // predictions must complete without creating either kind of update.
            vif.resultReady = 1'b0;
            send_sample(samples[4], 0, 0, 1);
            send_sample(samples[5], 0, 0, 2);
            send_sample(samples[6], 0, 0, 1);
            wait(vif.resultValid);
            repeat (3) @(posedge vif.clk);
            @(negedge vif.clk);
            vif.resultReady = 1'b1;
            for (int sample_index = 4; sample_index < 7; sample_index++)
                check_and_consume(sample_index, samples[sample_index], 0,
                                  expected_prediction[sample_index], 0);

            repeat (UPDATE_STAGES + N + 2) @(posedge vif.clk);
            for (int row = 0; row < N; row++)
                for (int col = 0; col < N; col++)
                    if (vif.residentMatrixWeight[row][col] !==
                        inference_matrix_before[row][col])
                        `uvm_error("INFERENCE_STABILITY", $sformatf(
                            "matrix weight [%0d][%0d] changed during inference",
                            row, col))
            for (int lane = 0; lane < N; lane++)
                if (vif.residentReductionWeight[lane] !==
                    inference_reduction_before[lane])
                    `uvm_error("INFERENCE_STABILITY", $sformatf(
                        "reduction weight %0d changed during inference", lane))

            `uvm_info("PHASE5F", "Verified 0,1,0,1 per-sample training alignment and inference weight stability", UVM_NONE)
            phase.drop_objection(this);
        endtask
    endclass
