    // ------------------------------------------------------------------
    // Environment
    // ------------------------------------------------------------------

    class nn_core_env extends uvm_env;
        `uvm_component_utils(nn_core_env)

        virtual nn_core_if #(WIDTH, N) vif;
        nn_core_sequencer sequencer;
        nn_core_driver driver;
        nn_core_input_monitor input_monitor;
        nn_core_result_monitor result_monitor;
        nn_core_scoreboard scoreboard;
        nn_core_coverage coverage;

        function new(string name, uvm_component parent);
            super.new(name, parent);
        endfunction

        function void build_phase(uvm_phase phase);
            super.build_phase(phase);
            if (!uvm_config_db #(virtual nn_core_if #(WIDTH, N))::get(
                    this, "", "vif", vif))
                `uvm_fatal("NO_VIF", "nn_core_env did not receive nn_core_if")

            sequencer = nn_core_sequencer::type_id::create("sequencer", this);
            driver = nn_core_driver::type_id::create("driver", this);
            input_monitor = nn_core_input_monitor::type_id::create("input_monitor", this);
            result_monitor = nn_core_result_monitor::type_id::create("result_monitor", this);
            scoreboard = nn_core_scoreboard::type_id::create("scoreboard", this);
            coverage = nn_core_coverage::type_id::create("coverage", this);
        endfunction

        function void connect_phase(uvm_phase phase);
            super.connect_phase(phase);
            driver.seq_item_port.connect(sequencer.seq_item_export);
            input_monitor.matrix_ap.connect(scoreboard.expected_fifo.analysis_export);
            input_monitor.matrix_ap.connect(coverage.matrix_imp);
            result_monitor.result_ap.connect(scoreboard.actual_fifo.analysis_export);
            result_monitor.result_ap.connect(coverage.result_imp);
        endfunction

        task wait_for_completion(input int unsigned expected_matrices);
            int unsigned drain_cycles;

            scoreboard.wait_for_matrices(expected_matrices);

            // reloadReady is asserted only when the activation FIFOs, skew
            // registers, systolic pipeline, and output FIFOs are all empty.
            drain_cycles = 0;
            while (vif.reloadReady !== 1'b1) begin
                @(posedge vif.clk);
                drain_cycles++;
                if (drain_cycles > 100000)
                    `uvm_fatal("DRAIN_TIMEOUT", $sformatf(
                        "core did not become quiescent after checking %0d matrices",
                        expected_matrices))
            end

            if (result_monitor.rows_observed != expected_matrices * N)
                `uvm_error("RESULT_COUNT", $sformatf(
                    "observed %0d result rows, expected %0d",
                    result_monitor.rows_observed, expected_matrices * N))
            if (result_monitor.framing_errors != 0)
                `uvm_error("FRAMING", $sformatf(
                    "result monitor recorded %0d framing errors",
                    result_monitor.framing_errors))
        endtask
    endclass

    // ------------------------------------------------------------------
    // Matrix sequences
    // ------------------------------------------------------------------

    class nn_core_sequence_base extends uvm_sequence #(nn_core_matrix_item);
        int unsigned random_seed;
        int unsigned random_state;
        int unsigned expected_matrices;

        `uvm_object_utils(nn_core_sequence_base)

        function new(string name = "nn_core_sequence_base");
            super.new(name);
            random_seed = DEFAULT_SEED;
            random_state = DEFAULT_SEED;
            expected_matrices = 0;
        endfunction

        function int unsigned next_random();
            random_state = $urandom(random_state);
            return random_state;
        endfunction

        function data_t random_data();
            logic [WIDTH-1:0] value;

            value = '0;
            for (int bit_index = 0; bit_index < WIDTH; bit_index += 32)
                value = (value << 32) | next_random();
            return data_t'(value);
        endfunction

        function data_t min_data();
            return data_t'({1'b1, {(WIDTH-1){1'b0}}});
        endfunction

        function data_t max_data();
            return data_t'({1'b0, {(WIDTH-1){1'b1}}});
        endfunction

        function void fill_identity_weights(nn_core_matrix_item item);
            for (int row = 0; row < N; row++) begin
                for (int col = 0; col < N; col++) begin
                    item.weights[row][col] = (row == col) ? data_t'(1) : data_t'(0);
                end
            end
        endfunction

        function void fill_signed_activations(nn_core_matrix_item item);
            for (int row = 0; row < N; row++) begin
                for (int col = 0; col < N; col++) begin
                    if (row == col)
                        item.activations[row][col] = data_t'(row + 2);
                    else if ((row + col) % 2)
                        item.activations[row][col] = data_t'(-3);
                    else
                        item.activations[row][col] = data_t'(1);
                end
            end
            item.activations[0][0] = data_t'(2);
            item.activations[0][1] = data_t'(-3);
        endfunction

        function void fill_random_weights(nn_core_matrix_item item);
            for (int row = 0; row < N; row++) begin
                for (int col = 0; col < N; col++) begin
                    item.weights[row][col] = random_data();
                end
            end
            // Keep sign coverage deterministic instead of relying on chance.
            item.weights[0][0] = data_t'(-3);
        endfunction

        function void fill_random_activations(nn_core_matrix_item item);
            for (int row = 0; row < N; row++) begin
                for (int col = 0; col < N; col++) begin
                    item.activations[row][col] = random_data();
                end
            end
            // Keep sign coverage deterministic instead of relying on chance.
            item.activations[0][0] = data_t'(-5);
            item.activations[0][1] = data_t'(7);
        endfunction

        function void fill_edge_case_weights(nn_core_matrix_item item);
            for (int row = 0; row < N; row++) begin
                for (int col = 0; col < N; col++) begin
                    item.weights[row][col] = (row == col) ?
                                             data_t'(1) :
                                             (((row + col) % 2) ? min_data() : max_data());
                end
            end
        endfunction

        function void fill_edge_case_activations(nn_core_matrix_item item);
            for (int row = 0; row < N; row++) begin
                for (int col = 0; col < N; col++) begin
                    item.activations[row][col] = (row == col) ?
                                                 min_data() :
                                                 (((row + col) % 2) ? max_data() : min_data());
                end
            end
        endfunction

        task send_item(nn_core_matrix_item item);
            start_item(item);
            finish_item(item);
            if (!item.is_aborted_by_reset())
                expected_matrices++;
        endtask
    endclass

    class nn_core_smoke_sequence extends nn_core_sequence_base;
        `uvm_object_utils(nn_core_smoke_sequence)

        function new(string name = "nn_core_smoke_sequence");
            super.new(name);
        endfunction

        task body();
            nn_core_matrix_item item;
            item = nn_core_matrix_item::type_id::create("smoke_matrix");
            fill_identity_weights(item);
            fill_signed_activations(item);
            item.pass_through = 1'b1;
            item.load_weights = 1'b1;
            item.stall_percent = 0;
            send_item(item);
        endtask
    endclass

    class nn_core_regression_sequence extends nn_core_sequence_base;
        `uvm_object_utils(nn_core_regression_sequence)

        function new(string name = "nn_core_regression_sequence");
            super.new(name);
        endfunction

        task body();
            nn_core_matrix_item item;

            random_state = (random_seed == 0) ? DEFAULT_SEED : random_seed;

            // Matrix 0: signed pass-through under identity weights.
            item = nn_core_matrix_item::type_id::create("identity_signed");
            fill_identity_weights(item);
            fill_signed_activations(item);
            item.pass_through = 1'b1;
            item.load_weights = 1'b1;
            item.stall_percent = 35;
            send_item(item);

            // Matrix 1: no reload; this is the back-to-back stationary-weight
            // case and includes activation bubbles.
            item = nn_core_matrix_item::type_id::create("identity_random");
            fill_random_activations(item);
            item.pass_through = 1'b1;
            item.load_weights = 1'b0;
            item.activation_bubbles = 1'b1;
            item.stall_percent = 45;
            send_item(item);

            // Matrix 2: mode change after the preceding results have drained.
            item = nn_core_matrix_item::type_id::create("identity_relu");
            fill_signed_activations(item);
            item.pass_through = 1'b0;
            item.activation_bubbles = 1'b1;
            item.stall_percent = 30;
            send_item(item);

            // Matrix 3: new stationary weights, with bubbles in both input
            // streams and an explicit reload boundary.
            item = nn_core_matrix_item::type_id::create("random_weights_pass");
            fill_random_weights(item);
            fill_random_activations(item);
            item.pass_through = 1'b1;
            item.load_weights = 1'b1;
            item.reload_before = 1'b1;
            item.weight_bubbles = 1'b1;
            item.activation_bubbles = 1'b1;
            item.stall_percent = 40;
            send_item(item);

            // Matrix 4: reuse those weights in ReLU mode.
            item = nn_core_matrix_item::type_id::create("random_weights_relu");
            fill_random_activations(item);
            item.pass_through = 1'b0;
            item.activation_bubbles = 1'b1;
            item.stall_percent = 25;
            send_item(item);

            // Stress burst: deterministic output backpressure fills the existing
            // result/activation buffering and forces the producer to hold a row.
            for (int stress_matrix = 0; stress_matrix < 5; stress_matrix++) begin
                item = nn_core_matrix_item::type_id::create($sformatf(
                    "activation_backpressure_%0d", stress_matrix));
                fill_random_activations(item);
                item.pass_through = 1'b0;
                item.activation_bubbles = 1'b0;
                item.stall_percent = 0;
                item.stall_until_activation_backpressure = (stress_matrix == 0);
                send_item(item);
            end

            // Reset while a new weight frame is only partially accepted.  No
            // matrix is expected from this intentionally aborted item.
            item = nn_core_matrix_item::type_id::create("reset_partial_weights");
            fill_random_weights(item);
            item.load_weights = 1'b1;
            item.reload_before = 1'b1;
            item.weight_bubbles = 1'b1;
            item.reset_phase = NN_RESET_DURING_WEIGHT_LOAD;
            item.reset_after_rows = 1;
            item.stall_percent = 20;
            send_item(item);

            item = nn_core_matrix_item::type_id::create("recovered_after_weight_reset");
            fill_edge_case_weights(item);
            fill_edge_case_activations(item);
            item.pass_through = 1'b1;
            item.load_weights = 1'b1;
            item.activation_bubbles = 1'b1;
            item.stall_percent = 35;
            send_item(item);

            // Reset with a stationary matrix loaded and a partial activation
            // frame in flight.  wait_for_drain keeps earlier expected rows
            // from being intentionally discarded by this reset.
            item = nn_core_matrix_item::type_id::create("reset_partial_activation");
            fill_edge_case_activations(item);
            item.pass_through = 1'b1;
            item.wait_for_drain = 1'b1;
            item.reset_phase = NN_RESET_DURING_ACTIVATION;
            item.reset_after_rows = 1;
            item.activation_bubbles = 1'b1;
            item.stall_percent = 30;
            send_item(item);

            // Post-reset recovery must reload weights because reset clears the
            // stationary array, then verify both output activation modes again.
            item = nn_core_matrix_item::type_id::create("recovered_after_activation_reset");
            fill_identity_weights(item);
            fill_signed_activations(item);
            item.pass_through = 1'b1;
            item.load_weights = 1'b1;
            item.weight_bubbles = 1'b1;
            item.stall_percent = 35;
            send_item(item);

            item = nn_core_matrix_item::type_id::create("final_relu_random");
            fill_random_activations(item);
            item.pass_through = 1'b0;
            item.activation_bubbles = 1'b1;
            item.stall_percent = 45;
            send_item(item);
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

        task wait_for_results(input int unsigned expected_matrices);
            env.wait_for_completion(expected_matrices);
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
            wait_for_results(smoke_seq.expected_matrices);
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
            regression_seq = nn_core_regression_sequence::type_id::create("regression_seq");
            regression_seq.random_seed = seed;
            regression_seq.start(env.sequencer);
            wait_for_results(regression_seq.expected_matrices);
            env.coverage.check_broad_coverage();
            phase.drop_objection(this);
        endtask
    endclass
