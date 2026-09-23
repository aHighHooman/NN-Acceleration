    // Environment

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
            input_monitor = nn_core_input_monitor::type_id::create("input_monitor", this);
            result_monitor = nn_core_result_monitor::type_id::create("result_monitor", this);
            scoreboard = nn_core_scoreboard::type_id::create("scoreboard", this);
        endfunction

        function void connect_phase(uvm_phase phase);
            super.connect_phase(phase);
            driver.seq_item_port.connect(sequencer.seq_item_export);
            input_monitor.sample_ap.connect(scoreboard.sample_imp);
            result_monitor.result_ap.connect(scoreboard.result_imp);
        endfunction

        task wait_for_completion(input int unsigned expected_results);
            int unsigned drain_cycles, expected_retired;
            expected_retired = expected_results;
            if (scoreboard.samples_discarded_on_reset < expected_retired)
                expected_retired -= scoreboard.samples_discarded_on_reset;
            else
                expected_retired = 0;
            scoreboard.wait_for_completion(expected_retired);
            drain_cycles = 0;
            while (vif.reloadReady !== 1'b1) begin
                @(posedge vif.clk);
                if (++drain_cycles > 100000)
                    `uvm_fatal("DRAIN_TIMEOUT", "core did not drain after the sequence")
            end
        endtask
    endclass

    // Compact stimulus sequences

    class nn_core_sequence_base extends uvm_sequence #(uvm_sequence_item);
        virtual nn_core_if #(WIDTH, N, TARGET_WIDTH,
                             REDUCTION_WEIGHT_WIDTH) vif;
        int unsigned random_seed, random_state;
        int unsigned expected_results, sent_samples, training_samples;

        `uvm_object_utils(nn_core_sequence_base)

        function new(string name = "nn_core_sequence_base");
            super.new(name);
            random_seed = DEFAULT_SEED;
            random_state = DEFAULT_SEED;
        endfunction

        function int unsigned next_random();
            random_state = $urandom(random_state);
            return random_state;
        endfunction

        function void fill_weights(nn_core_config_item item, bit signed_weights);
            for (int row = 0; row < N; row++)
                for (int col = 0; col < N; col++)
                    if (!signed_weights)
                        item.weights[row][col] = (row == col) ? 1 : 0;
                    else
                        item.weights[row][col] = (row == col) ? 1 :
                            (((row + col) % 2) ? -1 : 2);
        endfunction

        function void fill_sample(nn_core_sample_item item, int mode);
            for (int lane = 0; lane < N; lane++) begin
                if (mode == 0)
                    item.input_vector[lane] = (lane == 0) ? 2 :
                        ((lane == 1) ? -3 : lane + 1);
                else if (mode == 1)
                    item.input_vector[lane] = '0;
                else
                    item.input_vector[lane] = data_t'(next_random());
            end
            if (mode == 2) begin
                item.input_vector[0] = -5;
                if (N > 1) item.input_vector[1] = 7;
            end
            item.target = (mode == 0) ? target_t'(1) : '0;
        endfunction

        function void fill_reduction_weights(nn_core_config_item item);
            for (int lane = 0; lane < N; lane++)
                item.reduction_weights[lane] = (lane % 2) ? -1 : 1;
            item.load_reduction_weights = 1'b1;
        endfunction

        task configure(input bit signed_weights, input bit reload,
                       input bit weight_bubble);
            nn_core_config_item item;
            item = nn_core_config_item::type_id::create("configuration");
            fill_weights(item, signed_weights);
            fill_reduction_weights(item);
            item.reload_before = reload;
            item.weight_bubble = weight_bubble;
            start_item(item);
            finish_item(item);
        endtask

        task send_sample(nn_core_sample_item item);
            start_item(item);
            finish_item(item);
            sent_samples++;
            expected_results++;
            if (item.training_enable) training_samples++;
        endtask

        task send_filled_sample(input int mode, input int unsigned stall,
                                input bit bubble, input bit hold);
            nn_core_sample_item item;
            item = nn_core_sample_item::type_id::create("sample");
            fill_sample(item, mode);
            item.input_bubble = bubble;
            item.result_stall_percent = stall;
            item.hold_result_until_input_backpressure = hold;
            send_sample(item);
        endtask

        // The public drain indication includes both pipeline completion and
        // output retirement, so this cannot be satisfied by input acceptance.
        task wait_for_drain();
            int unsigned cycles;
            cycles = 0;
            do begin
                @(negedge vif.clk);
                if (++cycles > 100000)
                    `uvm_fatal("DRAIN_TIMEOUT", "sequence did not drain")
            end while (vif.reloadReady !== 1'b1);
        endtask
    endclass

    class nn_core_smoke_sequence extends nn_core_sequence_base;
        `uvm_object_utils(nn_core_smoke_sequence)

        function new(string name = "nn_core_smoke_sequence");
            super.new(name);
        endfunction

        task body();
            configure(0, 0, 0);
            send_filled_sample(0, 0, 0, 0);
        endtask
    endclass

    class nn_core_regression_sequence extends nn_core_sequence_base;
        `uvm_object_utils(nn_core_regression_sequence)

        function new(string name = "nn_core_regression_sequence");
            super.new(name);
        endfunction

        task body();
            nn_core_sample_item item;
            random_state = (random_seed == 0) ? DEFAULT_SEED : random_seed;

            configure(0, 0, 1);
            send_filled_sample(0, 25, 1, 0);
            send_filled_sample(1, 15, 0, 0);

            configure(1, 1, 1);
            for (int index = 0; index < 12; index++)
                send_filled_sample(2, (index == 0) ? 0 : 50,
                                   (index % 3) != 1, index == 0);

            // A known baseline makes the learned-state boundary observable.
            configure(0, 1, 0);
            item = nn_core_sample_item::type_id::create("training_sample");
            for (int lane = 0; lane < N; lane++)
                item.input_vector[lane] = 16;
            item.target = target_t'(127);
            item.training_enable = 1'b1;
            item.input_bubble = 1'b1;
            item.result_stall_percent = 30;
            send_sample(item);
            wait_for_drain();

            item = nn_core_sample_item::type_id::create("learned_inference");
            for (int lane = 0; lane < N; lane++)
                item.input_vector[lane] = 16;
            item.result_stall_percent = 35;
            send_sample(item);
            wait_for_drain();

            // A complete reload restores an exact-checkable epoch.
            configure(0, 1, 0);
            send_filled_sample(0, 20, 0, 0);
            wait_for_drain();

            item = nn_core_sample_item::type_id::create("reset_sample");
            fill_sample(item, 2);
            item.result_stall_percent = 100;
            item.reset_after_accept = 1'b1;
            send_sample(item);

            configure(0, 0, 1);
            send_filled_sample(0, 40, 1, 0);
        endtask
    endclass

    // Tests

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
            if (!$value$plusargs("NN_SEED=%d", seed)) seed = DEFAULT_SEED;
            env = nn_core_env::type_id::create("env", this);
            uvm_config_db #(int unsigned)::set(this, "env.driver", "seed", seed);
            `uvm_info("SEED", $sformatf("NN_SEED=%0d (0x%08h)", seed, seed), UVM_NONE)
        endfunction

        task assert_common_counts(nn_core_sequence_base seq_obj);
            int unsigned expected_retired;
            expected_retired = seq_obj.expected_results;
            if (env.scoreboard.samples_discarded_on_reset < expected_retired)
                expected_retired -= env.scoreboard.samples_discarded_on_reset;
            else
                expected_retired = 0;
            if (env.input_monitor.samples_observed != seq_obj.sent_samples)
                `uvm_error("SAMPLE_COUNT", "input monitor lost an accepted sample")
            if (env.result_monitor.results_observed != expected_retired)
                `uvm_error("RESULT_COUNT", "result monitor lost or duplicated a result")
            if (env.scoreboard.accepted_sample_count != seq_obj.sent_samples)
                `uvm_error("SCOREBOARD_COUNT", "sample analysis stream lost traffic")
            if (env.scoreboard.accepted_sample_count !=
                env.scoreboard.retired_result_count +
                env.scoreboard.samples_discarded_on_reset)
                `uvm_error("SCOREBOARD_BALANCE", "sample traffic was lost across reset")
            if (env.scoreboard.numeric_mismatch_count != 0)
                `uvm_error("NUMERIC_CHECK", "inference predictor found a mismatch")
            if (env.scoreboard.result_without_sample_count != 0)
                `uvm_error("ORDER_CHECK", "result stream duplicated or reordered traffic")
        endtask
    endclass

    class nn_uvm_smoke_test extends nn_uvm_base_test;
        `uvm_component_utils(nn_uvm_smoke_test)

        function new(string name, uvm_component parent);
            super.new(name, parent);
        endfunction

        task run_phase(uvm_phase phase);
            nn_core_smoke_sequence seq_obj;
            phase.raise_objection(this);
            seq_obj = nn_core_smoke_sequence::type_id::create("smoke_sequence");
            seq_obj.start(env.sequencer);
            env.wait_for_completion(seq_obj.expected_results);
            assert_common_counts(seq_obj);
            phase.drop_objection(this);
        endtask
    endclass

    class nn_uvm_regression_test extends nn_uvm_base_test;
        `uvm_component_utils(nn_uvm_regression_test)

        function new(string name, uvm_component parent);
            super.new(name, parent);
        endfunction

        task run_phase(uvm_phase phase);
            nn_core_regression_sequence seq_obj;
            phase.raise_objection(this);
            seq_obj = nn_core_regression_sequence::type_id::create("regression_sequence");
            seq_obj.random_seed = seed;
            seq_obj.vif = env.vif;
            seq_obj.start(env.sequencer);
            env.wait_for_completion(seq_obj.expected_results);
            assert_common_counts(seq_obj);

            if (env.driver.weight_bubbles_injected == 0)
                `uvm_error("SCENARIO", "weight-load bubble was not injected")
            if (env.driver.input_bubbles_injected == 0)
                `uvm_error("SCENARIO", "input bubble was not injected")
            if (env.result_monitor.output_stall_cycles == 0)
                `uvm_error("SCENARIO", "output backpressure was not observed")
            if (env.input_monitor.input_backpressure_cycles == 0)
                `uvm_error("SCENARIO", "input backpressure was not observed")
            if (env.input_monitor.reload_count < 1)
                `uvm_error("SCENARIO", "reload/recovery was not observed")
            if (env.input_monitor.reset_count < 1)
                `uvm_error("SCENARIO", "reset/recovery was not observed")
            if (env.scoreboard.samples_discarded_on_reset == 0)
                `uvm_error("SCENARIO", "reset did not discard an in-flight sample")
            if (seq_obj.training_samples == 0 ||
                env.scoreboard.retired_training_count != seq_obj.training_samples)
                `uvm_error("SCENARIO", "training sample did not retire before reset")
            if (env.scoreboard.discarded_training_count != 0)
                `uvm_error("SCENARIO", "reset discarded a training sample")
            if (env.scoreboard.scoped_inference_count != 1)
                `uvm_error("SCENARIO", "expected one learned-state inference outside numeric scope")
            if (env.scoreboard.exact_inference_count == 0)
                `uvm_error("SCENARIO", "no known-weight inference was checked")
            phase.drop_objection(this);
        endtask
    endclass
