    // ------------------------------------------------------------------
    // Independent reference model and scoreboard
    // ------------------------------------------------------------------

    class nn_core_scoreboard extends uvm_scoreboard;
        `uvm_component_utils(nn_core_scoreboard)

        uvm_tlm_analysis_fifo #(nn_core_matrix_item) expected_fifo;
        uvm_tlm_analysis_fifo #(nn_core_result_row) actual_fifo;
        virtual nn_core_if #(WIDTH, N) vif;
        int unsigned matrices_checked;

        function new(string name, uvm_component parent);
            super.new(name, parent);
            matrices_checked = 0;
        endfunction

        function void build_phase(uvm_phase phase);
            super.build_phase(phase);
            expected_fifo = new("expected_fifo", this);
            actual_fifo = new("actual_fifo", this);
            if (!uvm_config_db #(virtual nn_core_if #(WIDTH, N))::get(
                    this, "", "vif", vif))
                `uvm_fatal("NO_VIF", "nn_core_scoreboard did not receive nn_core_if")
        endfunction

        // This model starts from passive accepted input traffic.  It uses a
        // wider signed accumulator, then applies the DUT's result-width
        // truncation.
        function void predict(nn_core_matrix_item item,
                              output result_matrix_t expected);
            longint signed sum;
            longint signed activation_value;
            longint signed weight_value;

            for (int row = 0; row < N; row++) begin
                for (int col = 0; col < N; col++) begin
                    sum = 0;
                    for (int k = 0; k < N; k++) begin
                        activation_value = $signed(item.activations[row][k]);
                        weight_value = $signed(item.weights[k][col]);
                        sum += activation_value * weight_value;
                    end
                    expected[row][col] = result_t'(sum);
                end
            end
        endfunction

        task run_phase(uvm_phase phase);
            nn_core_matrix_item item;
            nn_core_result_row actual;
            result_matrix_t expected;

            forever begin
                expected_fifo.get(item);
                predict(item, expected);

                for (int row = 0; row < N; row++) begin
                    actual_fifo.get(actual);
                    if (actual.row_index != row) begin
                        `uvm_error("FRAMING", $sformatf(
                            "scoreboard saw row index %0d while expecting %0d",
                            actual.row_index, row))
                    end
                    if (actual.last !== (row == N-1)) begin
                        `uvm_error("FRAMING", $sformatf(
                            "scoreboard saw resultLast=%0b on row %0d",
                            actual.last, row))
                    end
                    for (int col = 0; col < N; col++) begin
                        if (actual.data[col] !== expected[row][col]) begin
                            `uvm_error("MISMATCH", $sformatf(
                                "C[%0d][%0d] got %0d (0x%0h) expected %0d (0x%0h)",
                                row, col, actual.data[col], actual.data[col],
                                expected[row][col], expected[row][col]))
                        end
                    end
                end

                matrices_checked++;
                `uvm_info("SCOREBOARD", $sformatf(
                    "Checked accepted matrix %0d: %s",
                    matrices_checked, item.convert2string()), UVM_LOW)
            end
        endtask

        task wait_for_matrices(input int unsigned expected_count,
                               input int unsigned max_cycles = 100000);
            int unsigned cycles;
            cycles = 0;
            while (matrices_checked < expected_count) begin
                @(posedge vif.clk);
                cycles++;
                if (cycles > max_cycles)
                    `uvm_fatal("SCOREBOARD_TIMEOUT", $sformatf(
                        "checked %0d of %0d expected matrices",
                        matrices_checked, expected_count))
            end
        endtask
    endclass

    // Counter-based bins keep the regression runnable without a coverage
    // feature license; the broad test checks the required bins explicitly.
    class nn_core_coverage extends uvm_component;
        `uvm_component_utils(nn_core_coverage)

        virtual nn_core_if #(WIDTH, N) vif;
        uvm_analysis_imp_matrix #(nn_core_matrix_item, nn_core_coverage) matrix_imp;
        uvm_analysis_imp_result #(nn_core_result_row, nn_core_coverage) result_imp;

        int unsigned matrix_count;
        int unsigned negative_operand_count;
        int unsigned repeated_weight_matrix_count;
        bit result_last_seen;
        int unsigned weight_bubble_cycles;
        int unsigned activation_bubble_cycles;
        int unsigned activation_stall_cycles;
        int unsigned output_stall_cycles;
        int unsigned reload_count;
        int unsigned reset_count;

        int unsigned last_weight_generation;
        int unsigned weight_rows_in_frame;
        int unsigned activation_rows_in_frame;
        bit have_last_generation;
        bit released_once;
        bit in_reset;

        function new(string name, uvm_component parent);
            super.new(name, parent);
            matrix_imp = new("matrix_imp", this);
            result_imp = new("result_imp", this);
            matrix_count = 0;
            negative_operand_count = 0;
            repeated_weight_matrix_count = 0;
            result_last_seen = 1'b0;
            weight_bubble_cycles = 0;
            activation_bubble_cycles = 0;
            activation_stall_cycles = 0;
            output_stall_cycles = 0;
            reload_count = 0;
            reset_count = 0;
            last_weight_generation = 0;
            weight_rows_in_frame = 0;
            activation_rows_in_frame = 0;
            have_last_generation = 1'b0;
            released_once = 1'b0;
            in_reset = 1'b0;
        endfunction

        function void build_phase(uvm_phase phase);
            super.build_phase(phase);
            if (!uvm_config_db #(virtual nn_core_if #(WIDTH, N))::get(
                    this, "", "vif", vif))
                `uvm_fatal("NO_VIF", "nn_core_coverage did not receive nn_core_if")
        endfunction

        function void write_matrix(nn_core_matrix_item item);
            bit has_negative;
            matrix_count++;

            has_negative = 1'b0;
            for (int row = 0; row < N; row++) begin
                for (int col = 0; col < N; col++) begin
                    has_negative |= item.activations[row][col][WIDTH-1];
                    has_negative |= item.weights[row][col][WIDTH-1];
                end
            end
            if (has_negative)
                negative_operand_count++;

            if (have_last_generation &&
                item.weight_generation == last_weight_generation)
                repeated_weight_matrix_count++;
            last_weight_generation = item.weight_generation;
            have_last_generation = 1'b1;
        endfunction

        function void write_result(nn_core_result_row row);
            if (row.last)
                result_last_seen = 1'b1;
        endfunction

        task run_phase(uvm_phase phase);
            forever begin
                @(vif.monitor_cb);
                if (!vif.monitor_cb.rst_n) begin
                    if (!in_reset && released_once)
                        reset_count++;
                    in_reset = 1'b1;
                end else begin
                    released_once = 1'b1;
                    in_reset = 1'b0;

                    if (!vif.monitor_cb.weightsLoaded) begin
                        if (weight_rows_in_frame != 0 &&
                            vif.monitor_cb.weightReady &&
                            !vif.monitor_cb.weightValid)
                            weight_bubble_cycles++;
                        if (vif.monitor_cb.weightValid && vif.monitor_cb.weightReady) begin
                            if (weight_rows_in_frame == N-1)
                                weight_rows_in_frame = 0;
                            else
                                weight_rows_in_frame++;
                        end
                    end else begin
                        weight_rows_in_frame = 0;
                    end

                    if (vif.monitor_cb.weightsLoaded) begin
                        if (activation_rows_in_frame != 0 &&
                            vif.monitor_cb.activationReady &&
                            !vif.monitor_cb.activationValid)
                            activation_bubble_cycles++;
                        if (vif.monitor_cb.activationValid &&
                            !vif.monitor_cb.activationReady)
                            activation_stall_cycles++;
                        if (vif.monitor_cb.activationValid && vif.monitor_cb.activationReady) begin
                            if (activation_rows_in_frame == N-1)
                                activation_rows_in_frame = 0;
                            else
                                activation_rows_in_frame++;
                        end
                    end else begin
                        activation_rows_in_frame = 0;
                    end

                    if (vif.monitor_cb.resultValid && !vif.monitor_cb.resultReady)
                        output_stall_cycles++;
                    if (vif.monitor_cb.reloadWeights && vif.monitor_cb.reloadReady)
                        reload_count++;
                end
            end
        endtask

        function void check_broad_coverage();
            if (matrix_count == 0)
                `uvm_error("COVERAGE", "no accepted activation matrices were observed")
            if (negative_operand_count == 0)
                `uvm_error("COVERAGE", "negative operand bin was not observed")
            if (repeated_weight_matrix_count == 0)
                `uvm_error("COVERAGE", "repeated matrix under stationary weights was not observed")
            if (weight_bubble_cycles == 0)
                `uvm_error("COVERAGE", "weight input bubble bin was not observed")
            if (activation_bubble_cycles == 0)
                `uvm_error("COVERAGE", "activation input bubble bin was not observed")
            if (activation_stall_cycles == 0)
                `uvm_error("COVERAGE", "activation input backpressure bin was not observed")
            if (output_stall_cycles == 0)
                `uvm_error("COVERAGE", "output backpressure bin was not observed")
            if (reload_count == 0)
                `uvm_error("COVERAGE", "weight reload bin was not observed")
            if (reset_count == 0)
                `uvm_error("COVERAGE", "injected reset bin was not observed")
            if (!result_last_seen)
                `uvm_error("COVERAGE", "resultLast bin was not observed")
        endfunction

        function void report_phase(uvm_phase phase);
            `uvm_info("COVERAGE", $sformatf(
                "matrices=%0d negativeOperands=%0d repeatedUnderWeights=%0d",
                matrix_count,
                negative_operand_count, repeated_weight_matrix_count), UVM_NONE)
            `uvm_info("COVERAGE", $sformatf(
                "weightBubbles=%0d activationBubbles=%0d activationStallCycles=%0d outputStallCycles=%0d reloads=%0d injectedResets=%0d",
                weight_bubble_cycles, activation_bubble_cycles,
                activation_stall_cycles, output_stall_cycles,
                reload_count, reset_count), UVM_NONE)
        endfunction
    endclass
