    // ------------------------------------------------------------------
    // Passive public-interface monitors
    // ------------------------------------------------------------------

    class nn_core_input_monitor extends uvm_monitor;
        `uvm_component_utils(nn_core_input_monitor)

        virtual nn_core_if #(WIDTH, N, TARGET_WIDTH,
                             REDUCTION_WEIGHT_WIDTH) vif;
        uvm_analysis_port #(nn_core_sample_item) sample_ap;

        data_t resident_weights[N][N];
        data_t pending_weights[N][N];
        reduction_t resident_reduction_weights[N];
        int unsigned pending_weight_rows;
        bit weights_valid;
        bit in_reset;
        bit released_once;

        int unsigned samples_observed;
        int unsigned input_backpressure_cycles;
        int unsigned reload_count;
        int unsigned reset_count;

        function new(string name, uvm_component parent);
            super.new(name, parent);
            sample_ap = new("sample_ap", this);
            pending_weight_rows = 0;
            weights_valid = 1'b0;
            in_reset = 1'b0;
            released_once = 1'b0;
            samples_observed = 0;
            input_backpressure_cycles = 0;
            reload_count = 0;
            reset_count = 0;
            clear_configuration();
        endfunction

        function void build_phase(uvm_phase phase);
            super.build_phase(phase);
            if (!uvm_config_db #(virtual nn_core_if #(WIDTH, N,
                                                       TARGET_WIDTH,
                                                       REDUCTION_WEIGHT_WIDTH))::get(
                    this, "", "vif", vif))
                `uvm_fatal("NO_VIF", "nn_core_input_monitor did not receive nn_core_if")
        endfunction

        function void clear_configuration();
            for (int row = 0; row < N; row++) begin
                for (int lane = 0; lane < N; lane++) begin
                    resident_weights[row][lane] = '0;
                    pending_weights[row][lane] = '0;
                end
                resident_reduction_weights[row] = '0;
            end
            pending_weight_rows = 0;
            weights_valid = 1'b0;
        endfunction

        task run_phase(uvm_phase phase);
            forever begin
                @(vif.monitor_cb);

                if (!vif.monitor_cb.rst_n) begin
                    if (!in_reset && released_once)
                        reset_count++;
                    in_reset = 1'b1;
                    clear_configuration();
                    continue;
                end

                released_once = 1'b1;
                in_reset = 1'b0;

                // These events update only the monitor's local configuration
                // model; neither event becomes an analysis transaction.
                if (vif.monitor_cb.reloadWeights &&
                    vif.monitor_cb.reloadReady) begin
                    reload_count++;
                    clear_configuration();
                end

                if (vif.monitor_cb.loadReductionWeights)
                    for (int lane = 0; lane < N; lane++)
                        resident_reduction_weights[lane] =
                            vif.monitor_cb.reductionWeight[lane];

                if (vif.monitor_cb.weightValid && vif.monitor_cb.weightReady) begin
                    int row_index;
                    row_index = N-1-pending_weight_rows;
                    for (int lane = 0; lane < N; lane++)
                        pending_weights[row_index][lane] =
                            vif.monitor_cb.weightData[lane];

                    if (pending_weight_rows == N-1) begin
                        for (int row = 0; row < N; row++)
                            for (int lane = 0; lane < N; lane++)
                                resident_weights[row][lane] =
                                    pending_weights[row][lane];
                        for (int lane = 0; lane < N; lane++)
                            resident_weights[row_index][lane] =
                                vif.monitor_cb.weightData[lane];
                        pending_weight_rows = 0;
                        weights_valid = 1'b1;
                    end else
                        pending_weight_rows++;
                end

                if (vif.monitor_cb.inputValid &&
                    !vif.monitor_cb.inputReady)
                    input_backpressure_cycles++;

                if (vif.monitor_cb.inputValid &&
                    vif.monitor_cb.inputReady) begin
                    nn_core_sample_item sample;
                    sample = nn_core_sample_item::type_id::create(
                        "accepted_sample");
                    for (int lane = 0; lane < N; lane++) begin
                        sample.input_vector[lane] = vif.monitor_cb.inputData[lane];
                        sample.reduction_weights[lane] =
                            resident_reduction_weights[lane];
                    end
                    for (int row = 0; row < N; row++)
                        for (int lane = 0; lane < N; lane++)
                            sample.weights[row][lane] = resident_weights[row][lane];
                    sample.target = vif.monitor_cb.targetData;
                    sample.training_enable = vif.monitor_cb.trainingEnable;
                    sample.weights_valid = weights_valid;
                    sample.pass_through = vif.monitor_cb.passThrough;
                    sample.reduce_output = vif.monitor_cb.reduceOutput;
                    samples_observed++;
                    sample_ap.write(sample);
                end
            end
        endtask
    endclass

    class nn_core_result_monitor extends uvm_monitor;
        `uvm_component_utils(nn_core_result_monitor)

        virtual nn_core_if #(WIDTH, N, TARGET_WIDTH,
                             REDUCTION_WEIGHT_WIDTH) vif;
        uvm_analysis_port #(nn_core_result_transaction) result_ap;
        int unsigned results_observed;
        int unsigned output_stall_cycles;

        function new(string name, uvm_component parent);
            super.new(name, parent);
            result_ap = new("result_ap", this);
            results_observed = 0;
            output_stall_cycles = 0;
        endfunction

        function void build_phase(uvm_phase phase);
            super.build_phase(phase);
            if (!uvm_config_db #(virtual nn_core_if #(WIDTH, N,
                                                       TARGET_WIDTH,
                                                       REDUCTION_WEIGHT_WIDTH))::get(
                    this, "", "vif", vif))
                `uvm_fatal("NO_VIF", "nn_core_result_monitor did not receive nn_core_if")
        endfunction

        task run_phase(uvm_phase phase);
            forever begin
                @(vif.monitor_cb);
                if (!vif.monitor_cb.rst_n)
                    continue;
                if (vif.monitor_cb.resultValid && !vif.monitor_cb.resultReady)
                    output_stall_cycles++;
                if (vif.monitor_cb.resultValid && vif.monitor_cb.resultReady) begin
                    nn_core_result_transaction result;
                    result = nn_core_result_transaction::type_id::create(
                        "retired_result");
                    for (int lane = 0; lane < N; lane++)
                        result.data[lane] = vif.monitor_cb.resultData[lane];
                    results_observed++;
                    result_ap.write(result);
                end
            end
        endtask
    endclass
