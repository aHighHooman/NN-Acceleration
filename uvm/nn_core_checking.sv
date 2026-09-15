    // ------------------------------------------------------------------
    // Ordered black-box scoreboard
    // ------------------------------------------------------------------

    class nn_core_scoreboard extends uvm_scoreboard;
        `uvm_component_utils(nn_core_scoreboard)

        uvm_analysis_imp_sample #(nn_core_sample_item,
                                  nn_core_scoreboard) sample_imp;
        uvm_analysis_imp_result #(nn_core_result_transaction,
                                  nn_core_scoreboard) result_imp;
        virtual nn_core_if #(WIDTH, N, TARGET_WIDTH,
                             REDUCTION_WEIGHT_WIDTH) vif;

        nn_core_sample_item expected_samples[$];
        int unsigned accepted_sample_count;
        int unsigned retired_result_count;
        int unsigned samples_discarded_on_reset;
        int unsigned numeric_mismatch_count;
        int unsigned result_without_sample_count;
        bit reset_seen;

        function new(string name, uvm_component parent);
            super.new(name, parent);
            sample_imp = new("sample_imp", this);
            result_imp = new("result_imp", this);
            accepted_sample_count = 0;
            retired_result_count = 0;
            samples_discarded_on_reset = 0;
            numeric_mismatch_count = 0;
            result_without_sample_count = 0;
            reset_seen = 1'b0;
        endfunction

        function void build_phase(uvm_phase phase);
            super.build_phase(phase);
            if (!uvm_config_db #(virtual nn_core_if #(WIDTH, N,
                                                       TARGET_WIDTH,
                                                       REDUCTION_WEIGHT_WIDTH))::get(
                    this, "", "vif", vif))
                `uvm_fatal("NO_VIF", "nn_core_scoreboard did not receive nn_core_if")
        endfunction

        // Reset is observed directly because it invalidates queued samples;
        // it is intentionally not a third analysis stream.
        task run_phase(uvm_phase phase);
            forever begin
                @(posedge vif.clk);
                if (!vif.rst_n) begin
                    if (!reset_seen) begin
                        samples_discarded_on_reset += expected_samples.size();
                        expected_samples.delete();
                    end
                    reset_seen = 1'b1;
                end else
                    reset_seen = 1'b0;
            end
        endtask

        function matrix_result_t narrow_matrix(input longint signed value);
            narrow_matrix = matrix_result_t'(value);
        endfunction

        function result_t narrow_result(input longint signed value);
            narrow_result = result_t'(value);
        endfunction

        function void check_inference(
            input nn_core_sample_item sample,
            input nn_core_result_transaction actual);
            matrix_result_t raw_value[N];
            matrix_result_t activated_value[N];
            result_t expected[N];
            longint signed sum;
            longint signed reduction_sum;

            for (int col = 0; col < N; col++) begin
                sum = 0;
                for (int row = 0; row < N; row++)
                    sum += $signed(sample.activation[row]) *
                           $signed(sample.weights[row][col]);
                raw_value[col] = narrow_matrix(sum);
                if (sample.pass_through || raw_value[col] >= 0)
                    activated_value[col] = raw_value[col];
                else
                    activated_value[col] = '0;
            end

            if (sample.reduce_output) begin
                reduction_sum = 0;
                for (int lane = 0; lane < N; lane++)
                    reduction_sum += $signed(activated_value[lane]) *
                                     $signed(sample.reduction_weights[lane]);
                reduction_sum = reduction_sum >>>
                    (FRACTION_BITS + REDUCTION_WEIGHT_WIDTH - 1);
                expected[0] = narrow_result(reduction_sum);
                for (int lane = 1; lane < N; lane++)
                    expected[lane] = '0;
            end else begin
                for (int lane = 0; lane < N; lane++)
                    expected[lane] = result_t'(activated_value[lane]);
            end

            for (int lane = 0; lane < N; lane++)
                if (actual.data[lane] !== expected[lane]) begin
                    numeric_mismatch_count++;
                    `uvm_error("RESULT_MISMATCH", $sformatf(
                        "lane %0d got %0d (0x%0h), expected %0d (0x%0h)",
                        lane, actual.data[lane], actual.data[lane],
                        expected[lane], expected[lane]))
                end
        endfunction

        function void write_sample(nn_core_sample_item sample);
            if (!sample.weights_valid)
                `uvm_error("SAMPLE_STATE",
                           "accepted sample observed without resident weights")
            expected_samples.push_back(sample);
            accepted_sample_count++;
        endfunction

        function void write_result(nn_core_result_transaction actual);
            nn_core_sample_item sample;
            retired_result_count++;
            if (expected_samples.size() == 0) begin
                result_without_sample_count++;
                `uvm_error("RESULT_ORDER",
                           "retired result observed with no accepted sample")
                return;
            end
            sample = expected_samples.pop_front();
            // Training is a liveness/order check only.  The references own
            // the learning state and update-wave semantics.
            if (!sample.training_enable)
                check_inference(sample, actual);
        endfunction

        task wait_for_completion(input int unsigned expected_results,
                                 input int unsigned max_cycles = 100000);
            int unsigned cycles;
            cycles = 0;
            while (retired_result_count < expected_results ||
                   expected_samples.size() != 0) begin
                @(posedge vif.clk);
                if (++cycles > max_cycles)
                    `uvm_fatal("SCOREBOARD_TIMEOUT", $sformatf(
                        "retired %0d of %0d results with %0d samples outstanding",
                        retired_result_count, expected_results,
                        expected_samples.size()))
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
                "accepted=%0d retired=%0d discardedOnReset=%0d mismatches=%0d unexpected=%0d",
                accepted_sample_count, retired_result_count,
                samples_discarded_on_reset, numeric_mismatch_count,
                result_without_sample_count), UVM_NONE)
        endfunction
    endclass
