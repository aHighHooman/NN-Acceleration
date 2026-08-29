    // ------------------------------------------------------------------
    // Passive monitors
    // ------------------------------------------------------------------

    class nn_core_input_monitor extends uvm_monitor;
        `uvm_component_utils(nn_core_input_monitor)

        virtual nn_core_if #(WIDTH, N) vif;
        uvm_analysis_port #(nn_core_matrix_item) matrix_ap;

        data_t weight_matrix[N][N];
        data_t activation_matrix[N][N];
        int unsigned weight_rows_seen;
        int unsigned activation_rows_seen;
        int unsigned weight_generation;
        bit have_weights;
        bit matrix_mode_valid;
        bit matrix_mode;
        bit in_reset;

        function new(string name, uvm_component parent);
            super.new(name, parent);
            matrix_ap = new("matrix_ap", this);
            weight_rows_seen = 0;
            activation_rows_seen = 0;
            weight_generation = 0;
            have_weights = 1'b0;
            matrix_mode_valid = 1'b0;
            matrix_mode = 1'b1;
            in_reset = 1'b0;
        endfunction

        function void build_phase(uvm_phase phase);
            super.build_phase(phase);
            if (!uvm_config_db #(virtual nn_core_if #(WIDTH, N))::get(
                    this, "", "vif", vif))
                `uvm_fatal("NO_VIF", "nn_core_input_monitor did not receive nn_core_if")
        endfunction

        function void clear_frame_state();
            weight_rows_seen = 0;
            activation_rows_seen = 0;
            have_weights = 1'b0;
            matrix_mode_valid = 1'b0;
        endfunction

        function void publish_matrix();
            nn_core_matrix_item item;
            item = nn_core_matrix_item::type_id::create("accepted_matrix");
            for (int row = 0; row < N; row++) begin
                for (int col = 0; col < N; col++) begin
                    item.weights[row][col] = weight_matrix[row][col];
                    item.activations[row][col] = activation_matrix[row][col];
                end
            end
            item.pass_through = matrix_mode;
            item.weight_generation = weight_generation;
            matrix_ap.write(item);
        endfunction

        task run_phase(uvm_phase phase);
            forever begin
                @(vif.monitor_cb);

                if (!vif.monitor_cb.rst_n) begin
                    if (!in_reset)
                        weight_generation++;
                    in_reset = 1'b1;
                    clear_frame_state();
                end else begin
                    in_reset = 1'b0;

                    if (vif.monitor_cb.reloadWeights && vif.monitor_cb.reloadReady) begin
                        weight_generation++;
                        clear_frame_state();
                    end

                    if (vif.monitor_cb.weightValid && vif.monitor_cb.weightReady) begin
                        if (weight_rows_seen >= N) begin
                            `uvm_error("INPUT_FRAME", "accepted more than N weight rows without a frame boundary")
                        end else begin
                            for (int lane = 0; lane < N; lane++)
                                weight_matrix[N-1-weight_rows_seen][lane] =
                                    vif.monitor_cb.weightData[lane];
                            weight_rows_seen++;
                            if (weight_rows_seen == N) begin
                                weight_rows_seen = 0;
                                have_weights = 1'b1;
                            end
                        end
                    end

                    if (vif.monitor_cb.activationValid && vif.monitor_cb.activationReady) begin
                        if (!have_weights) begin
                            `uvm_error("INPUT_FRAME", "accepted activation row before a complete weight frame")
                        end else begin
                            for (int lane = 0; lane < N; lane++)
                                activation_matrix[activation_rows_seen][lane] =
                                    vif.monitor_cb.activationData[lane];

                            if (!matrix_mode_valid) begin
                                matrix_mode = vif.monitor_cb.passThrough;
                                matrix_mode_valid = 1'b1;
                            end else if (matrix_mode != vif.monitor_cb.passThrough) begin
                                `uvm_error("INPUT_FRAME", "passThrough changed inside an accepted activation matrix")
                            end

                            activation_rows_seen++;
                            if (activation_rows_seen == N) begin
                                publish_matrix();
                                activation_rows_seen = 0;
                                matrix_mode_valid = 1'b0;
                            end
                        end
                    end
                end
            end
        endtask
    endclass

    class nn_core_result_monitor extends uvm_monitor;
        `uvm_component_utils(nn_core_result_monitor)

        virtual nn_core_if #(WIDTH, N) vif;
        uvm_analysis_port #(nn_core_result_row) result_ap;
        int unsigned next_row;
        int unsigned rows_observed;
        int unsigned framing_errors;

        function new(string name, uvm_component parent);
            super.new(name, parent);
            result_ap = new("result_ap", this);
            next_row = 0;
            rows_observed = 0;
            framing_errors = 0;
        endfunction

        function void build_phase(uvm_phase phase);
            super.build_phase(phase);
            if (!uvm_config_db #(virtual nn_core_if #(WIDTH, N))::get(
                    this, "", "vif", vif))
                `uvm_fatal("NO_VIF", "nn_core_result_monitor did not receive nn_core_if")
        endfunction

        task run_phase(uvm_phase phase);
            forever begin
                @(vif.monitor_cb);
                if (!vif.monitor_cb.rst_n) begin
                    next_row = 0;
                end else if (vif.monitor_cb.resultValid && vif.monitor_cb.resultReady) begin
                    nn_core_result_row row;
                    row = nn_core_result_row::type_id::create("accepted_result_row");
                    row.row_index = next_row;
                    row.last = vif.monitor_cb.resultLast;
                    for (int lane = 0; lane < N; lane++)
                        row.data[lane] = vif.monitor_cb.resultData[lane];

                    if (row.last !== (next_row == N-1)) begin
                        framing_errors++;
                        `uvm_error("FRAMING", $sformatf(
                            "resultLast mismatch at accepted row %0d", next_row))
                    end

                    if (row.last)
                        next_row = 0;
                    else
                        next_row++;
                    rows_observed++;
                    result_ap.write(row);
                end
            end
        endtask
    endclass
