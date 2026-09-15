    // ------------------------------------------------------------------
    // Active public-interface driver
    // ------------------------------------------------------------------

    class nn_core_driver extends uvm_driver #(uvm_sequence_item);
        `uvm_component_utils(nn_core_driver)

        virtual nn_core_if #(WIDTH, N, TARGET_WIDTH,
                             REDUCTION_WEIGHT_WIDTH) vif;
        int unsigned random_state;
        int unsigned active_result_stall_percent;
        int unsigned ready_cycle;
        int unsigned stall_run;
        bit reset_active;
        bit hold_result_for_input_pressure;

        // Scenario counters are deliberately local to the driver.  They are
        // used by the test's explicit assertions, not a coverage subsystem.
        int unsigned weight_bubbles_injected;
        int unsigned activation_bubbles_injected;

        function new(string name, uvm_component parent);
            super.new(name, parent);
            random_state = DEFAULT_SEED ^ 32'hd1a5_7001;
            active_result_stall_percent = 0;
            ready_cycle = 0;
            stall_run = 0;
            reset_active = 1'b0;
            hold_result_for_input_pressure = 1'b0;
            weight_bubbles_injected = 0;
            activation_bubbles_injected = 0;
        endfunction

        function void build_phase(uvm_phase phase);
            int unsigned configured_seed;
            super.build_phase(phase);
            if (!uvm_config_db #(virtual nn_core_if #(WIDTH, N,
                                                       TARGET_WIDTH,
                                                       REDUCTION_WEIGHT_WIDTH))::get(
                    this, "", "vif", vif))
                `uvm_fatal("NO_VIF", "nn_core_driver did not receive nn_core_if")
            if (uvm_config_db #(int unsigned)::get(this, "", "seed",
                                                   configured_seed))
                random_state = configured_seed ^ 32'hd1a5_7001;
        endfunction

        function int unsigned next_random();
            random_state = $urandom(random_state);
            return random_state;
        endfunction

        task run_phase(uvm_phase phase);
            reset_dut();
            fork
                drive_result_ready();
                drive_items();
            join
        endtask

        task drive_items();
            uvm_sequence_item item;
            nn_core_sample_item sample;
            nn_core_config_item config_item;

            forever begin
                seq_item_port.get_next_item(item);
                if ($cast(sample, item))
                    drive_sample(sample);
                else if ($cast(config_item, item))
                    drive_configuration(config_item);
                else
                    `uvm_fatal("ITEM_TYPE", $sformatf(
                        "unsupported sequence item type %s", item.get_type_name()))
                seq_item_port.item_done();
            end
        endtask

        task drive_sample(nn_core_sample_item item);
            active_result_stall_percent = (item.result_stall_percent > 100) ?
                                          100 : item.result_stall_percent;
            if (item.hold_result_until_activation_backpressure)
                hold_result_for_input_pressure = 1'b1;

            `uvm_info("DRV", {"Driving sample ", item.convert2string()}, UVM_MEDIUM)
            if (vif.weightsLoaded !== 1'b1)
                `uvm_fatal("WEIGHT_STATE",
                           "sample arrived without a loaded weight configuration")

            drive_activation_vector(item);
            if (item.reset_after_accept)
                reset_dut();
        endtask

        task drive_configuration(nn_core_config_item item);
            active_result_stall_percent = 0;
            hold_result_for_input_pressure = 1'b0;
            `uvm_info("DRV", {"Driving configuration ",
                               item.convert2string()}, UVM_MEDIUM)

            if (item.reload_before)
                request_weight_reload();
            else if (vif.weightsLoaded === 1'b1)
                `uvm_fatal("WEIGHT_STATE",
                           "new weights require an explicit drained reload")

            for (int row_offset = 0; row_offset < N; row_offset++)
                drive_weight_vector(item.weights[N-1-row_offset],
                                    item.weight_bubble);

            while (vif.weightsLoaded !== 1'b1)
                @(posedge vif.clk);

            if (item.load_reduction_weights)
                load_reduction_weights(item.reduction_weights);
        endtask

        task drive_result_ready();
            forever begin
                @(negedge vif.clk);
                ready_cycle++;
                if (!vif.rst_n || reset_active) begin
                    vif.resultReady = 1'b0;
                    stall_run = 0;
                end else if (hold_result_for_input_pressure) begin
                    // Hold the result FIFO until the public input path shows
                    // backpressure, proving the two ready/valid paths meet.
                    if (vif.activationValid && !vif.activationReady) begin
                        vif.resultReady = 1'b1;
                        hold_result_for_input_pressure = 1'b0;
                        stall_run = 0;
                    end else begin
                        vif.resultReady = 1'b0;
                        stall_run++;
                    end
                end else if (active_result_stall_percent == 0) begin
                    vif.resultReady = 1'b1;
                    stall_run = 0;
                end else if (stall_run >= 4 || (ready_cycle % 5) == 0) begin
                    // A bounded release makes a seeded run deterministic and
                    // keeps a stress test from becoming a false deadlock.
                    vif.resultReady = 1'b1;
                    stall_run = 0;
                end else begin
                    vif.resultReady = (next_random() % 100) >=
                                      active_result_stall_percent;
                    if (vif.resultReady)
                        stall_run = 0;
                    else
                        stall_run++;
                end
            end
        endtask

        task reset_dut();
            reset_active = 1'b1;
            @(negedge vif.clk);
            vif.rst_n = 1'b0;
            vif.weightValid = 1'b0;
            vif.activationValid = 1'b0;
            vif.targetData = '0;
            vif.trainingEnable = 1'b0;
            vif.loadReductionWeights = 1'b0;
            vif.passThrough = 1'b1;
            vif.reduceOutput = 1'b0;
            vif.reloadWeights = 1'b0;
            vif.resultReady = 1'b0;
            for (int lane = 0; lane < N; lane++) begin
                vif.weightData[lane] = '0;
                vif.activationData[lane] = '0;
                vif.reductionWeight[lane] = '0;
            end
            repeat (3) @(posedge vif.clk);
            @(negedge vif.clk);
            vif.rst_n = 1'b1;
            reset_active = 1'b0;
        endtask

        task request_weight_reload();
            int unsigned cycles;
            cycles = 0;
            while (vif.reloadReady !== 1'b1) begin
                @(posedge vif.clk);
                if (++cycles > 100000)
                    `uvm_fatal("RELOAD_TIMEOUT", "reloadReady never asserted")
            end
            @(negedge vif.clk) vif.reloadWeights = 1'b1;
            @(posedge vif.clk);
            @(negedge vif.clk) vif.reloadWeights = 1'b0;
            cycles = 0;
            while (vif.weightsLoaded !== 1'b0) begin
                @(posedge vif.clk);
                if (++cycles > 100000)
                    `uvm_fatal("RELOAD_TIMEOUT",
                               "weightsLoaded did not clear after reload")
            end
        endtask

        task load_reduction_weights(input reduction_t values[N]);
            @(negedge vif.clk);
            for (int lane = 0; lane < N; lane++)
                vif.reductionWeight[lane] = values[lane];
            vif.loadReductionWeights = 1'b1;
            @(posedge vif.clk);
            @(negedge vif.clk) vif.loadReductionWeights = 1'b0;
        endtask

        task drive_weight_vector(input data_t vector[N], input bit insert_bubble);
            if (insert_bubble) begin
                vif.weightValid = 1'b0;
                weight_bubbles_injected++;
                repeat (1 + (next_random() % 2)) @(negedge vif.clk);
            end
            for (int lane = 0; lane < N; lane++)
                vif.weightData[lane] = vector[lane];
            vif.weightValid = 1'b1;
            do @(posedge vif.clk);
            while (vif.weightValid !== 1'b1 || vif.weightReady !== 1'b1);
            @(negedge vif.clk) vif.weightValid = 1'b0;
        endtask

        task drive_activation_vector(nn_core_sample_item item);
            if (item.activation_bubble) begin
                vif.activationValid = 1'b0;
                activation_bubbles_injected++;
                repeat (1 + (next_random() % 2)) @(negedge vif.clk);
            end
            for (int lane = 0; lane < N; lane++)
                vif.activationData[lane] = item.activation[lane];
            vif.targetData = item.target;
            vif.trainingEnable = item.training_enable;
            vif.activationValid = 1'b1;
            do @(posedge vif.clk);
            while (vif.activationValid !== 1'b1 || vif.activationReady !== 1'b1);
            @(negedge vif.clk) vif.activationValid = 1'b0;
        endtask
    endclass
