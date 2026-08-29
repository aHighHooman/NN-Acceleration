    // ------------------------------------------------------------------
    // Active ready/valid driver
    // ------------------------------------------------------------------

    class nn_core_driver extends uvm_driver #(nn_core_matrix_item);
        `uvm_component_utils(nn_core_driver)

        virtual nn_core_if #(WIDTH, N) vif;
        int unsigned random_state;
        int unsigned active_stall_percent;
        int unsigned ready_cycle;
        int unsigned stall_run;
        bit reset_active;
        bit current_mode;
        bit current_mode_valid;

        function new(string name, uvm_component parent);
            super.new(name, parent);
            random_state = DEFAULT_SEED ^ 32'hd1a5_7001;
            active_stall_percent = 0;
            ready_cycle = 0;
            stall_run = 0;
            reset_active = 1'b0;
            current_mode = 1'b1;
            current_mode_valid = 1'b0;
        endfunction

        function void build_phase(uvm_phase phase);
            int unsigned configured_seed;
            super.build_phase(phase);
            if (!uvm_config_db #(virtual nn_core_if #(WIDTH, N))::get(
                    this, "", "vif", vif))
                `uvm_fatal("NO_VIF", "nn_core_driver did not receive nn_core_if")

            if (uvm_config_db #(int unsigned)::get(this, "", "seed", configured_seed))
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
            forever begin
                seq_item_port.get_next_item(req);
                active_stall_percent = (req.stall_percent > 100) ?
                                       100 : req.stall_percent;
                `uvm_info("DRV", {"Driving ", req.convert2string()}, UVM_MEDIUM)

                if (req.reload_before)
                    request_weight_reload();

                if (req.reset_phase == NN_RESET_DURING_WEIGHT_LOAD) begin
                    if (!req.load_weights)
                        `uvm_fatal("RESET_CONFIG", "weight reset item must load weights")
                    drive_weight_rows(req);
                    seq_item_port.item_done();
                    continue;
                end

                if (req.load_weights) begin
                    if (vif.weightsLoaded === 1'b1)
                        `uvm_fatal("WEIGHT_STATE", "load_weights requested while weightsLoaded is high")
                    drive_weight_rows(req);
                end else if (vif.weightsLoaded !== 1'b1) begin
                    `uvm_fatal("WEIGHT_STATE", "activation item arrived without a loaded weight matrix")
                end

                if (req.reset_phase == NN_RESET_DURING_ACTIVATION) begin
                    if (req.wait_for_drain)
                        wait_for_idle();
                    set_pass_through(req.pass_through);
                    drive_activation_rows(req, req.reset_after_rows);
                    reset_dut();
                end else begin
                    if (req.wait_for_drain)
                        wait_for_idle();
                    set_pass_through(req.pass_through);
                    drive_activation_rows(req, N);
                end

                seq_item_port.item_done();
            end
        endtask

        task drive_result_ready();
            forever begin
                @(negedge vif.clk);
                ready_cycle++;

                if (!vif.rst_n || reset_active) begin
                    vif.resultReady = 1'b0;
                    stall_run = 0;
                end else if (active_stall_percent == 0) begin
                    vif.resultReady = 1'b1;
                    stall_run = 0;
                end else begin
                    // Periodic releases keep backpressure coverage deterministic and bounded.
                    if (stall_run >= 4 || (ready_cycle % 5) == 0) begin
                        vif.resultReady = 1'b1;
                        stall_run = 0;
                    end else begin
                        vif.resultReady = (next_random() % 100) >=
                                          active_stall_percent;
                        if (vif.resultReady)
                            stall_run = 0;
                        else
                            stall_run++;
                    end
                end
            end
        endtask

        task reset_dut();
            reset_active = 1'b1;
            @(negedge vif.clk);
            vif.rst_n = 1'b0;
            vif.weightValid = 1'b0;
            vif.activationValid = 1'b0;
            vif.reloadWeights = 1'b0;
            vif.resultReady = 1'b0;
            for (int lane = 0; lane < N; lane++) begin
                vif.weightData[lane] = '0;
                vif.activationData[lane] = '0;
            end

            repeat (3) @(posedge vif.clk);
            @(negedge vif.clk);
            vif.rst_n = 1'b1;
            reset_active = 1'b0;
            current_mode_valid = 1'b0;
        endtask

        task request_weight_reload();
            int unsigned cycles;
            cycles = 0;
            while (vif.reloadReady !== 1'b1) begin
                @(posedge vif.clk);
                cycles++;
                if (cycles > 100000)
                    `uvm_fatal("RELOAD_TIMEOUT", "reloadReady never asserted")
            end

            @(negedge vif.clk);
            vif.reloadWeights = 1'b1;
            @(posedge vif.clk);
            @(negedge vif.clk);
            vif.reloadWeights = 1'b0;

            cycles = 0;
            while (vif.weightsLoaded !== 1'b0) begin
                @(posedge vif.clk);
                cycles++;
                if (cycles > 100000)
                    `uvm_fatal("RELOAD_TIMEOUT", "weightsLoaded did not clear after reload")
            end
        endtask

        task wait_for_idle();
            int unsigned cycles;
            cycles = 0;
            while (vif.reloadReady !== 1'b1) begin
                @(posedge vif.clk);
                cycles++;
                if (cycles > 100000)
                    `uvm_fatal("DRAIN_TIMEOUT", "core did not drain before a mode/reset boundary")
            end
        endtask

        task set_pass_through(input bit pass_through_value);
            if (current_mode_valid && current_mode != pass_through_value)
                wait_for_idle();
            vif.passThrough = pass_through_value;
            current_mode = pass_through_value;
            current_mode_valid = 1'b1;
        endtask

        task drive_weight_rows(nn_core_matrix_item item);
            int unsigned row_limit;
            row_limit = N;
            if (item.reset_phase == NN_RESET_DURING_WEIGHT_LOAD) begin
                if (item.reset_after_rows == 0 || item.reset_after_rows >= N)
                    `uvm_fatal("RESET_CONFIG", "weight reset must interrupt a partial weight frame")
                row_limit = item.reset_after_rows;
            end

            // The core consumes the stationary matrix from row N-1 to row 0.
            for (int unsigned row_offset = 0; row_offset < row_limit; row_offset++)
                drive_vector(item.weights[N-1-row_offset], 1'b1, item.weight_bubbles);

            if (item.reset_phase == NN_RESET_DURING_WEIGHT_LOAD) begin
                reset_dut();
                return;
            end

            while (vif.weightsLoaded !== 1'b1)
                @(posedge vif.clk);
        endtask

        task drive_activation_rows(nn_core_matrix_item item, input int unsigned row_limit);
            if (row_limit == 0 || row_limit > N)
                `uvm_fatal("RESET_CONFIG", "activation row count must be between 1 and N")

            for (int row = 0; row < row_limit; row++)
                drive_vector(item.activations[row], 1'b0, item.activation_bubbles);
        endtask

        task drive_vector(input data_t vector[N], input bit is_weight,
                          input bit insert_bubble);
            if (insert_bubble) begin
                if (is_weight)
                    vif.weightValid = 1'b0;
                else
                    vif.activationValid = 1'b0;
                repeat (1 + (next_random() % 2)) @(negedge vif.clk);
            end

            if (is_weight) begin
                for (int lane = 0; lane < N; lane++)
                    vif.weightData[lane] = vector[lane];
                vif.weightValid = 1'b1;
            end else begin
                for (int lane = 0; lane < N; lane++)
                    vif.activationData[lane] = vector[lane];
                vif.activationValid = 1'b1;
            end

            // Hold valid and data through stalls to exercise ready/valid stability.
            if (is_weight) begin
                do begin
                    @(posedge vif.clk);
                end while (vif.weightValid !== 1'b1 ||
                           vif.weightReady !== 1'b1);
            end else begin
                do begin
                    @(posedge vif.clk);
                end while (vif.activationValid !== 1'b1 ||
                           vif.activationReady !== 1'b1);
            end

            @(negedge vif.clk);
            if (is_weight)
                vif.weightValid = 1'b0;
            else
                vif.activationValid = 1'b0;
        endtask
    endclass
