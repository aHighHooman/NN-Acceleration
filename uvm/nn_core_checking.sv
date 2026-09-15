    // ------------------------------------------------------------------
    // Compact ordered scoreboard
    // ------------------------------------------------------------------

    class nn_core_expected_result extends uvm_object;
        result_t data[N];
        bit check_numeric;

        `uvm_object_utils(nn_core_expected_result)

        function new(string name = "nn_core_expected_result");
            super.new(name);
            check_numeric = 1'b0;
            for (int lane = 0; lane < N; lane++)
                data[lane] = '0;
        endfunction
    endclass

    class nn_core_scoreboard extends uvm_scoreboard;
        `uvm_component_utils(nn_core_scoreboard)

        uvm_analysis_imp_sample #(nn_core_sample_transaction,
                                  nn_core_scoreboard) sample_imp;
        uvm_analysis_imp_result #(nn_core_result_transaction,
                                  nn_core_scoreboard) result_imp;
        uvm_analysis_imp_weight_row #(nn_core_weight_row_transaction,
                                      nn_core_scoreboard) weight_row_imp;
        uvm_analysis_imp_reduction #(nn_core_reduction_load_transaction,
                                     nn_core_scoreboard) reduction_imp;
        uvm_analysis_imp_reload #(nn_core_reload_transaction,
                                  nn_core_scoreboard) reload_imp;
        uvm_analysis_imp_reset #(nn_core_reset_transaction,
                                 nn_core_scoreboard) reset_imp;

        virtual nn_core_if #(WIDTH, N, TARGET_WIDTH,
                             REDUCTION_WEIGHT_WIDTH) vif;

        nn_core_expected_result expected_samples[$];
        data_t resident_weights[N][N];
        data_t pending_weights[N][N];
        reduction_t resident_reduction_weights[N];
        int unsigned pending_weight_rows;
        bit weights_valid;

        int unsigned accepted_sample_count;
        int unsigned retired_result_count;
        int unsigned samples_discarded_on_reset;
        int unsigned reloads_observed;
        int unsigned numeric_mismatch_count;
        int unsigned result_without_sample_count;

        function new(string name, uvm_component parent);
            super.new(name, parent);
            sample_imp = new("sample_imp", this);
            result_imp = new("result_imp", this);
            weight_row_imp = new("weight_row_imp", this);
            reduction_imp = new("reduction_imp", this);
            reload_imp = new("reload_imp", this);
            reset_imp = new("reset_imp", this);
            pending_weight_rows = 0;
            weights_valid = 1'b0;
            accepted_sample_count = 0;
            retired_result_count = 0;
            samples_discarded_on_reset = 0;
            reloads_observed = 0;
            numeric_mismatch_count = 0;
            result_without_sample_count = 0;
            clear_model();
        endfunction

        function void build_phase(uvm_phase phase);
            super.build_phase(phase);
            if (!uvm_config_db #(virtual nn_core_if #(WIDTH, N,
                                                       TARGET_WIDTH,
                                                       REDUCTION_WEIGHT_WIDTH))::get(
                    this, "", "vif", vif))
                `uvm_fatal("NO_VIF", "nn_core_scoreboard did not receive nn_core_if")
        endfunction

        function void clear_model();
            for (int row = 0; row < N; row++) begin
                resident_weights[row] = '{default: '0};
                pending_weights[row] = '{default: '0};
                resident_reduction_weights[row] = '0;
            end
            pending_weight_rows = 0;
            weights_valid = 1'b0;
        endfunction

        function matrix_result_t narrow_matrix(input longint signed value);
            narrow_matrix = matrix_result_t'(value);
        endfunction

        function result_t narrow_result(input longint signed value);
            narrow_result = result_t'(value);
        endfunction

        function void predict_sample(
            input nn_core_sample_transaction sample,
            output nn_core_expected_result expected);
            matrix_result_t raw_value[N];
            matrix_result_t activated_value[N];
            longint signed sum;
            longint signed reduction_sum;

            expected = nn_core_expected_result::type_id::create(
                "expected_sample_result");
            expected.check_numeric = !sample.training_enable;

            for (int col = 0; col < N; col++) begin
                sum = 0;
                for (int k = 0; k < N; k++)
                    sum += $signed(sample.activation[k]) *
                           $signed(resident_weights[k][col]);
                raw_value[col] = narrow_matrix(sum);

                if (vif.passThrough || raw_value[col] >= 0)
                    activated_value[col] = raw_value[col];
                else
                    activated_value[col] = '0;
            end

            if (vif.reduceOutput) begin
                reduction_sum = 0;
                for (int lane = 0; lane < N; lane++)
                    reduction_sum += $signed(activated_value[lane]) *
                                     $signed(resident_reduction_weights[lane]);
                reduction_sum = reduction_sum >>>
                    (FRACTION_BITS + REDUCTION_WEIGHT_WIDTH - 1);
                expected.data[0] = narrow_result(reduction_sum);
                for (int lane = 1; lane < N; lane++)
                    expected.data[lane] = '0;
            end else begin
                for (int lane = 0; lane < N; lane++)
                    expected.data[lane] = result_t'(activated_value[lane]);
            end
        endfunction

        function void write_sample(nn_core_sample_transaction sample);
            nn_core_expected_result expected;

            if (!weights_valid)
                `uvm_error("SAMPLE_STATE",
                           "accepted sample observed without resident weights")

            predict_sample(sample, expected);
            expected_samples.push_back(expected);
            accepted_sample_count++;
        endfunction

        function void write_result(nn_core_result_transaction actual);
            nn_core_expected_result expected;

            retired_result_count++;
            if (expected_samples.size() == 0) begin
                result_without_sample_count++;
                `uvm_error("RESULT_ORDER",
                           "result handshake observed with no outstanding sample")
                return;
            end

            expected = expected_samples.pop_front();
            if (!expected.check_numeric)
                return;

            for (int lane = 0; lane < N; lane++) begin
                if (actual.data[lane] !== expected.data[lane]) begin
                    numeric_mismatch_count++;
                    `uvm_error("RESULT_MISMATCH", $sformatf(
                        "sample result lane %0d got %0d (0x%0h) expected %0d (0x%0h)",
                        lane, actual.data[lane], actual.data[lane],
                        expected.data[lane], expected.data[lane]))
                end
            end
        endfunction

        function void write_weight_row(nn_core_weight_row_transaction row);
            if (row.row_index >= N) begin
                `uvm_error("WEIGHT_CONFIG", "observed invalid weight row index")
                return;
            end

            for (int lane = 0; lane < N; lane++)
                pending_weights[row.row_index][lane] = row.data[lane];

            pending_weight_rows++;
            if (row.completes_load) begin
                if (pending_weight_rows != N)
                    `uvm_error("WEIGHT_CONFIG", $sformatf(
                        "weight load completed after %0d rows", pending_weight_rows))
                for (int row_index = 0; row_index < N; row_index++)
                    for (int lane = 0; lane < N; lane++)
                        resident_weights[row_index][lane] =
                            pending_weights[row_index][lane];
                weights_valid = 1'b1;
                pending_weight_rows = 0;
            end
        endfunction

        function void write_reduction(nn_core_reduction_load_transaction load);
            for (int lane = 0; lane < N; lane++)
                resident_reduction_weights[lane] = load.data[lane];
        endfunction

        function void write_reload(nn_core_reload_transaction reload);
            reloads_observed++;
            if (expected_samples.size() != 0)
                `uvm_error("RELOAD_ORDER",
                           "reload observed while samples were outstanding")
            for (int row = 0; row < N; row++)
                resident_weights[row] = '{default: '0};
            pending_weight_rows = 0;
            weights_valid = 1'b0;
        endfunction

        function void write_reset(nn_core_reset_transaction reset_event);
            samples_discarded_on_reset += expected_samples.size();
            expected_samples.delete();
            clear_model();
        endfunction

        task wait_for_completion(input int unsigned expected_results,
                                 input int unsigned max_cycles = 100000);
            int unsigned cycles;
            cycles = 0;
            while (retired_result_count < expected_results ||
                   expected_samples.size() != 0) begin
                @(posedge vif.clk);
                cycles++;
                if (cycles > max_cycles)
                    `uvm_fatal("SCOREBOARD_TIMEOUT", $sformatf(
                        "retired %0d of %0d results with %0d samples outstanding; accepted=%0d resultValid/Ready=%0b/%0b reloadReady=%0b weightsLoaded=%0b",
                        retired_result_count, expected_results,
                        expected_samples.size(), accepted_sample_count,
                        vif.resultValid, vif.resultReady,
                        vif.reloadReady, vif.weightsLoaded))
            end

            if (retired_result_count != expected_results)
                `uvm_error("RESULT_COUNT", $sformatf(
                    "retired %0d results, expected %0d",
                    retired_result_count, expected_results))
            if (result_without_sample_count != 0)
                `uvm_error("RESULT_DUPLICATION", $sformatf(
                    "%0d results had no matching accepted sample",
                    result_without_sample_count))
        endtask

        function void report_phase(uvm_phase phase);
            `uvm_info("SCOREBOARD", $sformatf(
                "acceptedSamples=%0d retiredResults=%0d discardedOnReset=%0d reloads=%0d mismatches=%0d",
                accepted_sample_count, retired_result_count,
                samples_discarded_on_reset, reloads_observed,
                numeric_mismatch_count), UVM_NONE)
        endfunction
    endclass
