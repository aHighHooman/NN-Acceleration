    // ------------------------------------------------------------------
    // Environment
    // ------------------------------------------------------------------

    class nn_core_env extends uvm_env;
        `uvm_component_utils(nn_core_env)

        virtual nn_core_if #(WIDTH, N, TARGET_WIDTH,
                             REDUCTION_WEIGHT_WIDTH) vif;
        nn_core_sequencer sequencer;
        nn_core_driver driver;
        nn_core_input_monitor input_monitor;
        nn_core_result_monitor result_monitor;
        nn_core_scoreboard scoreboard;

        function new(string name, uvm_component parent);
            super.new(name, parent);
        endfunction

        function void build_phase(uvm_phase phase);
            super.build_phase(phase);
            if (!uvm_config_db #(virtual nn_core_if #(WIDTH, N,
                                                       TARGET_WIDTH,
                                                       REDUCTION_WEIGHT_WIDTH))::get(
                    this, "", "vif", vif))
                `uvm_fatal("NO_VIF", "nn_core_env did not receive nn_core_if")

            sequencer = nn_core_sequencer::type_id::create("sequencer", this);
            driver = nn_core_driver::type_id::create("driver", this);
            input_monitor = nn_core_input_monitor::type_id::create(
                "input_monitor", this);
            result_monitor = nn_core_result_monitor::type_id::create(
                "result_monitor", this);
            scoreboard = nn_core_scoreboard::type_id::create("scoreboard", this);
        endfunction

        function void connect_phase(uvm_phase phase);
            super.connect_phase(phase);
            driver.seq_item_port.connect(sequencer.seq_item_export);
            input_monitor.sample_ap.connect(scoreboard.sample_imp);
            input_monitor.weight_row_ap.connect(scoreboard.weight_row_imp);
            input_monitor.reduction_ap.connect(scoreboard.reduction_imp);
            input_monitor.reload_ap.connect(scoreboard.reload_imp);
            input_monitor.reset_ap.connect(scoreboard.reset_imp);
            result_monitor.result_ap.connect(scoreboard.result_imp);
        endfunction

        task wait_for_completion(input int unsigned expected_results);
            int unsigned drain_cycles;

            scoreboard.wait_for_completion(expected_results);
            drain_cycles = 0;
            while (vif.reloadReady !== 1'b1) begin
                @(posedge vif.clk);
                drain_cycles++;
                if (drain_cycles > 100000)
                    `uvm_fatal("DRAIN_TIMEOUT", $sformatf(
                        "core did not become reloadable after checking %0d results",
                        expected_results))
            end
        endtask
    endclass

    // ------------------------------------------------------------------
    // Sample and configuration sequences
    // ------------------------------------------------------------------

    class nn_core_sequence_base extends uvm_sequence #(uvm_sequence_item);
        int unsigned random_seed;
        int unsigned random_state;
        int unsigned expected_results;
        int unsigned sent_samples;

        `uvm_object_utils(nn_core_sequence_base)

        function new(string name = "nn_core_sequence_base");
            super.new(name);
            random_seed = DEFAULT_SEED;
            random_state = DEFAULT_SEED;
            expected_results = 0;
            sent_samples = 0;
        endfunction

        function int unsigned next_random();
            random_state = $urandom(random_state);
            return random_state;
        endfunction

        function data_t random_data();
            return data_t'(next_random());
        endfunction

        function data_t min_data();
            return data_t'({1'b1, {(WIDTH-1){1'b0}}});
        endfunction

        function data_t max_data();
            return data_t'({1'b0, {(WIDTH-1){1'b1}}});
        endfunction

        function void fill_identity_weights(nn_core_weight_load_item item);
            for (int row = 0; row < N; row++)
                for (int col = 0; col < N; col++)
                    item.weights[row][col] = (row == col) ?
                                             data_t'(1) : data_t'(0);
        endfunction

        function void fill_signed_weights(nn_core_weight_load_item item);
            for (int row = 0; row < N; row++)
                for (int col = 0; col < N; col++)
                    item.weights[row][col] = (row == col) ?
                                             data_t'(1) :
                                             (((row + col) % 2) ?
                                              data_t'(-1) : data_t'(2));
        endfunction

        function void fill_edge_weights(nn_core_weight_load_item item);
            for (int row = 0; row < N; row++)
                for (int col = 0; col < N; col++)
                    item.weights[row][col] = (row == col) ?
                                             data_t'(1) :
                                             (((row + col) % 2) ?
                                              min_data() : max_data());
        endfunction

        function void fill_random_weights(nn_core_weight_load_item item);
            for (int row = 0; row < N; row++)
                for (int col = 0; col < N; col++)
                    item.weights[row][col] = random_data();
            item.weights[0][0] = data_t'(-3);
        endfunction

        function void fill_signed_sample(nn_core_sample_item item);
            for (int lane = 0; lane < N; lane++) begin
                if (lane == 0)
                    item.activation[lane] = data_t'(2);
                else if (lane == 1)
                    item.activation[lane] = data_t'(-3);
                else
                    item.activation[lane] = data_t'(lane + 1);
            end
            item.target = target_t'(1);
        endfunction

        function void fill_zero_sample(nn_core_sample_item item);
            for (int lane = 0; lane < N; lane++)
                item.activation[lane] = '0;
            item.target = '0;
        endfunction

        function void fill_random_sample(nn_core_sample_item item);
            for (int lane = 0; lane < N; lane++)
                item.activation[lane] = random_data();
            item.activation[0] = data_t'(-5);
            if (N > 1)
                item.activation[1] = data_t'(7);
            item.target = target_t'(0);
        endfunction

        function void fill_edge_sample(nn_core_sample_item item);
            for (int lane = 0; lane < N; lane++)
                item.activation[lane] = (lane % 2) ? max_data() : min_data();
            item.target = target_t'(-2);
        endfunction

        function void set_reduction_weights(nn_core_weight_load_item item);
            for (int lane = 0; lane < N; lane++)
                item.reduction_weights[lane] = reduction_t'((lane % 2) ? -1 : 1);
            item.load_reduction_weights = 1'b1;
        endfunction

        task send_weight_load(nn_core_weight_load_item item);
            start_item(item);
            finish_item(item);
        endtask

        task send_sample(nn_core_sample_item item);
            start_item(item);
            finish_item(item);
            sent_samples++;
            if (!item.reset_after_accept)
                expected_results++;
        endtask
    endclass

    class nn_core_smoke_sequence extends nn_core_sequence_base;
        `uvm_object_utils(nn_core_smoke_sequence)

        function new(string name = "nn_core_smoke_sequence");
            super.new(name);
        endfunction

        task body();
            nn_core_weight_load_item load;
            nn_core_sample_item sample;

            load = nn_core_weight_load_item::type_id::create("smoke_weights");
            fill_identity_weights(load);
            set_reduction_weights(load);
            send_weight_load(load);

            sample = nn_core_sample_item::type_id::create("smoke_sample");
            fill_signed_sample(sample);
            sample.result_stall_percent = 0;
            send_sample(sample);
        endtask
    endclass

    class nn_core_regression_sequence extends nn_core_sequence_base;
        `uvm_object_utils(nn_core_regression_sequence)

        function new(string name = "nn_core_regression_sequence");
            super.new(name);
        endfunction

        task body();
            nn_core_weight_load_item load;
            nn_core_sample_item sample;

            random_state = (random_seed == 0) ? DEFAULT_SEED : random_seed;

            // Initial configuration, followed by one sample that must drain
            // before the next configuration command can request reload.
            load = nn_core_weight_load_item::type_id::create("identity_weights");
            fill_identity_weights(load);
            set_reduction_weights(load);
            load.weight_bubble = 1'b1;
            send_weight_load(load);

            sample = nn_core_sample_item::type_id::create("first_sample");
            fill_signed_sample(sample);
            sample.activation_bubble = 1'b1;
            sample.result_stall_percent = 35;
            send_sample(sample);

            // Reload after exactly one fully drained sample.
            load = nn_core_weight_load_item::type_id::create("signed_weights");
            fill_signed_weights(load);
            set_reduction_weights(load);
            load.reload_before = 1'b1;
            load.weight_bubble = 1'b1;
            send_weight_load(load);

            // Reuse the resident weights for a few independent samples.
            for (int sample_index = 0; sample_index < 4; sample_index++) begin
                sample = nn_core_sample_item::type_id::create($sformatf(
                    "resident_sample_%0d", sample_index));
                if (sample_index == 0)
                    fill_zero_sample(sample);
                else
                    fill_random_sample(sample);
                sample.activation_bubble = (sample_index != 2);
                sample.result_stall_percent = 20 + 10*sample_index;
                send_sample(sample);
            end

            // This reload occurs after an arbitrary drained count (four), not
            // because a hidden N-sample frame boundary was reached.
            load = nn_core_weight_load_item::type_id::create("edge_weights");
            fill_edge_weights(load);
            set_reduction_weights(load);
            load.reload_before = 1'b1;
            load.weight_bubble = 1'b1;
            send_weight_load(load);

            // Hold resultReady low until the public activation path reports
            // backpressure, then continue with randomized output stalls.
            for (int stress_index = 0; stress_index < 16; stress_index++) begin
                sample = nn_core_sample_item::type_id::create($sformatf(
                    "backpressure_sample_%0d", stress_index));
                fill_random_sample(sample);
                sample.result_stall_percent = (stress_index == 0) ? 0 : 55;
                sample.hold_result_until_activation_backpressure =
                    (stress_index == 0);
                send_sample(sample);
            end

            // Reset during a partial configuration load.  The interrupted
            // configuration is not expected to produce a sample result.
            load = nn_core_weight_load_item::type_id::create(
                "reset_partial_weight_load");
            fill_random_weights(load);
            load.reload_before = 1'b1;
            load.weight_bubble = 1'b1;
            load.reset_after_rows = 1;
            send_weight_load(load);

            // Recovery after that reset uses a fresh complete configuration.
            load = nn_core_weight_load_item::type_id::create(
                "recovered_after_weight_reset");
            fill_identity_weights(load);
            set_reduction_weights(load);
            load.weight_bubble = 1'b1;
            send_weight_load(load);

            sample = nn_core_sample_item::type_id::create("recovered_sample");
            fill_edge_sample(sample);
            sample.activation_bubble = 1'b1;
            sample.result_stall_percent = 40;
            sample.wait_for_drain_after = 1'b1;
            send_sample(sample);

            // Accept one sample and reset immediately, while its result is
            // still in flight.  The scoreboard discards that outstanding item.
            sample = nn_core_sample_item::type_id::create(
                "reset_with_sample_outstanding");
            fill_random_sample(sample);
            sample.result_stall_percent = 100;
            sample.reset_after_accept = 1'b1;
            send_sample(sample);

            // Recovery after an activation-side reset also reloads weights,
            // because reset clears the resident configuration.
            load = nn_core_weight_load_item::type_id::create(
                "recovered_after_sample_reset");
            fill_identity_weights(load);
            set_reduction_weights(load);
            load.weight_bubble = 1'b1;
            send_weight_load(load);

            sample = nn_core_sample_item::type_id::create("final_inference");
            fill_signed_sample(sample);
            sample.activation_bubble = 1'b1;
            sample.result_stall_percent = 45;
            send_sample(sample);

            // Exercise the real target/training pins for liveness only.  The
            // scoreboard deliberately does not model the learning transition.
            sample = nn_core_sample_item::type_id::create("training_liveness");
            fill_random_sample(sample);
            sample.target = target_t'(3);
            sample.training_enable = 1'b1;
            sample.activation_bubble = 1'b1;
            sample.result_stall_percent = 30;
            send_sample(sample);
        endtask
    endclass

    // ------------------------------------------------------------------
    // Tests
    // ------------------------------------------------------------------

    class nn_uvm_base_test extends uvm_test;
        `uvm_component_utils(nn_uvm_base_test)

        nn_core_env env;
        int unsigned seed;

        function new(string name, uvm_component parent);
            super.new(name, parent);
            seed = DEFAULT_SEED;
        endfunction

        function void build_phase(uvm_phase phase);
            super.build_phase(phase);
            if (!$value$plusargs("NN_SEED=%d", seed))
                seed = DEFAULT_SEED;
            env = nn_core_env::type_id::create("env", this);
            uvm_config_db #(int unsigned)::set(
                this, "env.driver", "seed", seed);
            `uvm_info("SEED", $sformatf(
                "NN_SEED=%0d (0x%08h)", seed, seed), UVM_NONE)
        endfunction

        task wait_for_results(input int unsigned expected_results);
            env.wait_for_completion(expected_results);
        endtask

        task assert_common_counts(nn_core_sequence_base seq);
            if (env.input_monitor.samples_observed != seq.sent_samples)
                `uvm_error("SAMPLE_COUNT", $sformatf(
                    "observed %0d accepted samples, expected %0d",
                    env.input_monitor.samples_observed, seq.sent_samples))
            if (env.result_monitor.results_observed != seq.expected_results)
                `uvm_error("RESULT_COUNT", $sformatf(
                    "observed %0d retired results, expected %0d",
                    env.result_monitor.results_observed,
                    seq.expected_results))
            if (env.scoreboard.numeric_mismatch_count != 0)
                `uvm_error("NUMERIC_CHECK", $sformatf(
                    "scoreboard found %0d inference mismatches",
                    env.scoreboard.numeric_mismatch_count))
        endtask
    endclass

    class nn_uvm_smoke_test extends nn_uvm_base_test;
        `uvm_component_utils(nn_uvm_smoke_test)

        function new(string name, uvm_component parent);
            super.new(name, parent);
        endfunction

        task run_phase(uvm_phase phase);
            nn_core_smoke_sequence smoke_seq;
            phase.raise_objection(this);
            smoke_seq = nn_core_smoke_sequence::type_id::create("smoke_seq");
            smoke_seq.start(env.sequencer);
            wait_for_results(smoke_seq.expected_results);
            assert_common_counts(smoke_seq);
            phase.drop_objection(this);
        endtask
    endclass

    class nn_uvm_regression_test extends nn_uvm_base_test;
        `uvm_component_utils(nn_uvm_regression_test)

        function new(string name, uvm_component parent);
            super.new(name, parent);
        endfunction

        task run_phase(uvm_phase phase);
            nn_core_regression_sequence regression_seq;
            phase.raise_objection(this);
            regression_seq = nn_core_regression_sequence::type_id::create(
                "regression_seq");
            regression_seq.random_seed = seed;
            regression_seq.start(env.sequencer);
            wait_for_results(regression_seq.expected_results);
            assert_common_counts(regression_seq);

            // Scenario assertions are intentionally local and explicit.  A
            // standalone generic coverage component adds no information here.
            if (env.driver.weight_bubbles_injected == 0)
                `uvm_error("SCENARIO", "weight-load bubble was not injected")
            if (env.driver.activation_bubbles_injected == 0)
                `uvm_error("SCENARIO", "activation bubble was not injected")
            if (env.result_monitor.output_stall_cycles == 0)
                `uvm_error("SCENARIO", "output backpressure was not observed")
            if (env.input_monitor.activation_backpressure_cycles == 0)
                `uvm_error("SCENARIO", "activation backpressure was not observed")
            if (env.input_monitor.reload_count < 3)
                `uvm_error("SCENARIO", $sformatf(
                    "expected at least three drained reloads, observed %0d",
                    env.input_monitor.reload_count))
            if (env.input_monitor.reset_count < 2)
                `uvm_error("SCENARIO", $sformatf(
                    "expected partial-load and outstanding-sample resets, observed %0d",
                    env.input_monitor.reset_count))
            if (env.scoreboard.samples_discarded_on_reset == 0)
                `uvm_error("SCENARIO",
                           "reset did not discard an outstanding sample")

            `uvm_info("SCENARIO", $sformatf(
                "weightBubbles=%0d activationBubbles=%0d outputStallCycles=%0d activationBackpressureCycles=%0d reloads=%0d resets=%0d",
                env.driver.weight_bubbles_injected,
                env.driver.activation_bubbles_injected,
                env.result_monitor.output_stall_cycles,
                env.input_monitor.activation_backpressure_cycles,
                env.input_monitor.reload_count,
                env.input_monitor.reset_count), UVM_NONE)

            phase.drop_objection(this);
        endtask
    endclass
