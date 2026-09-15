    // ------------------------------------------------------------------
    // Passive public-interface monitors
    // ------------------------------------------------------------------

    class nn_core_input_monitor extends uvm_monitor;
        `uvm_component_utils(nn_core_input_monitor)

        virtual nn_core_if #(WIDTH, N, TARGET_WIDTH,
                             REDUCTION_WEIGHT_WIDTH) vif;
        uvm_analysis_port #(nn_core_sample_transaction) sample_ap;
        uvm_analysis_port #(nn_core_weight_row_transaction) weight_row_ap;
        uvm_analysis_port #(nn_core_reduction_load_transaction) reduction_ap;
        uvm_analysis_port #(nn_core_reload_transaction) reload_ap;
        uvm_analysis_port #(nn_core_reset_transaction) reset_ap;

        int unsigned samples_observed;
        int unsigned weight_rows_observed;
        int unsigned activation_backpressure_cycles;
        int unsigned reload_count;
        int unsigned reset_count;
        int unsigned reset_generation;
        bit in_reset;
        bit released_once;

        function new(string name, uvm_component parent);
            super.new(name, parent);
            sample_ap = new("sample_ap", this);
            weight_row_ap = new("weight_row_ap", this);
            reduction_ap = new("reduction_ap", this);
            reload_ap = new("reload_ap", this);
            reset_ap = new("reset_ap", this);
            samples_observed = 0;
            weight_rows_observed = 0;
            activation_backpressure_cycles = 0;
            reload_count = 0;
            reset_count = 0;
            reset_generation = 0;
            in_reset = 1'b0;
            released_once = 1'b0;
        endfunction

        function void build_phase(uvm_phase phase);
            super.build_phase(phase);
            if (!uvm_config_db #(virtual nn_core_if #(WIDTH, N,
                                                       TARGET_WIDTH,
                                                       REDUCTION_WEIGHT_WIDTH))::get(
                    this, "", "vif", vif))
                `uvm_fatal("NO_VIF", "nn_core_input_monitor did not receive nn_core_if")
        endfunction

        task publish_reset();
            nn_core_reset_transaction reset_event;
            reset_generation++;
            weight_rows_observed = 0;
            reset_event = nn_core_reset_transaction::type_id::create(
                "observed_reset");
            reset_event.generation = reset_generation;
            reset_ap.write(reset_event);
        endtask

        task run_phase(uvm_phase phase);
            forever begin
                @(vif.monitor_cb);

                if (!vif.monitor_cb.rst_n) begin
                    if (!in_reset) begin
                        if (released_once)
                            reset_count++;
                        publish_reset();
                    end
                    in_reset = 1'b1;
                    continue;
                end

                released_once = 1'b1;
                in_reset = 1'b0;

                if (vif.monitor_cb.reloadWeights &&
                    vif.monitor_cb.reloadReady) begin
                    nn_core_reload_transaction reload_event;
                    reload_event = nn_core_reload_transaction::type_id::create(
                        "observed_reload");
                    reload_count++;
                    weight_rows_observed = 0;
                    reload_ap.write(reload_event);
                end

                if (vif.monitor_cb.loadReductionWeights) begin
                    nn_core_reduction_load_transaction reduction_event;
                    reduction_event =
                        nn_core_reduction_load_transaction::type_id::create(
                            "observed_reduction_load");
                    for (int lane = 0; lane < N; lane++)
                        reduction_event.data[lane] =
                            vif.monitor_cb.reductionWeight[lane];
                    reduction_ap.write(reduction_event);
                end

                if (vif.monitor_cb.weightValid && vif.monitor_cb.weightReady) begin
                    nn_core_weight_row_transaction weight_row;
                    weight_row = nn_core_weight_row_transaction::type_id::create(
                        "observed_weight_row");
                    weight_row.row_index = N-1-weight_rows_observed;
                    weight_row.completes_load = (weight_rows_observed == N-1);
                    for (int lane = 0; lane < N; lane++)
                        weight_row.data[lane] = vif.monitor_cb.weightData[lane];
                    weight_row_ap.write(weight_row);

                    if (weight_rows_observed == N-1)
                        weight_rows_observed = 0;
                    else
                        weight_rows_observed++;
                end

                if (vif.monitor_cb.activationValid &&
                    !vif.monitor_cb.activationReady)
                    activation_backpressure_cycles++;

                if (vif.monitor_cb.activationValid &&
                    vif.monitor_cb.activationReady) begin
                    nn_core_sample_transaction sample;
                    sample = nn_core_sample_transaction::type_id::create(
                        "accepted_sample");
                    for (int lane = 0; lane < N; lane++)
                        sample.activation[lane] =
                            vif.monitor_cb.activationData[lane];
                    sample.target = vif.monitor_cb.targetData;
                    sample.training_enable = vif.monitor_cb.trainingEnable;
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

        // Every result handshake is one independent vector transaction.
        // Lanes are the elements of that result, not a stream position.
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
                        "accepted_result");
                    for (int lane = 0; lane < N; lane++)
                        result.data[lane] = vif.monitor_cb.resultData[lane];
                    results_observed++;
                    result_ap.write(result);
                end
            end
        endtask
    endclass
