package nn_uvm_pkg;
    import uvm_pkg::*;
    `include "uvm_macros.svh"

    // The UVM compile is intentionally a small, fast N=3 configuration.  The
    // RTL and directed regression continue to cover the other supported array
    // sizes; changing these two package constants retargets this environment.
    localparam int WIDTH = 8;
    localparam int N = 3;
    localparam int RESULT_WIDTH = 2*WIDTH + $clog2(N);
    localparam int unsigned DEFAULT_SEED = 32'h5eed_2026;

    typedef logic signed [WIDTH-1:0] data_t;
    typedef logic signed [RESULT_WIDTH-1:0] result_t;
    typedef data_t data_matrix_t[N][N];
    typedef result_t result_matrix_t[N][N];

    typedef enum int {
        NN_RESET_NONE = 0,
        NN_RESET_DURING_WEIGHT_LOAD = 1,
        NN_RESET_DURING_ACTIVATION = 2
    } reset_phase_e;

    `uvm_analysis_imp_decl(_matrix)
    `uvm_analysis_imp_decl(_result)

    // ------------------------------------------------------------------
    // Transactions
    // ------------------------------------------------------------------

    // This item is a matrix-level stimulus description.  The driver uses it
    // only to create pin-level ready/valid traffic.  The scoreboard never
    // consumes this item: it receives a separately reconstructed copy from
    // the passive input monitor.
    class nn_core_matrix_item extends uvm_sequence_item;
        data_t weights[N][N];
        data_t activations[N][N];
        bit pass_through;

        bit load_weights;
        bit reload_before;
        bit weight_bubbles;
        bit activation_bubbles;
        int unsigned stall_percent;
        bit wait_for_drain;

        reset_phase_e reset_phase;
        int unsigned reset_after_rows;

        // Filled only by the passive monitor.  It is useful in coverage and
        // is deliberately not used as stimulus configuration.
        int unsigned weight_generation;

        `uvm_object_utils(nn_core_matrix_item)

        function new(string name = "nn_core_matrix_item");
            super.new(name);
            pass_through = 1'b1;
            load_weights = 1'b0;
            reload_before = 1'b0;
            weight_bubbles = 1'b0;
            activation_bubbles = 1'b0;
            stall_percent = 0;
            wait_for_drain = 1'b0;
            reset_phase = NN_RESET_NONE;
            reset_after_rows = 0;
            weight_generation = 0;
        endfunction

        function bit is_aborted_by_reset();
            return reset_phase != NN_RESET_NONE;
        endfunction

        function string convert2string();
            return $sformatf(
                "passThrough=%0b loadWeights=%0b reload=%0b bubbles(w/a)=%0b/%0b stall=%0d reset=%0d",
                pass_through, load_weights, reload_before,
                weight_bubbles, activation_bubbles, stall_percent,
                reset_phase);
        endfunction
    endclass

    class nn_core_result_row extends uvm_sequence_item;
        result_t data[N];
        bit last;
        int unsigned row_index;

        `uvm_object_utils(nn_core_result_row)

        function new(string name = "nn_core_result_row");
            super.new(name);
            last = 1'b0;
            row_index = 0;
        endfunction
    endclass

    // ------------------------------------------------------------------
    // Sequencer and active ready/valid driver
    // ------------------------------------------------------------------

    class nn_core_sequencer extends uvm_sequencer #(nn_core_matrix_item);
        `uvm_component_utils(nn_core_sequencer)

        function new(string name, uvm_component parent);
            super.new(name, parent);
        endfunction
    endclass

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
                    // A periodic forced stall makes the backpressure bin
                    // deterministic, while the random decision varies the
                    // remaining cycles.  The bound prevents deadlock.
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
                vif.weightValid = 1'b0;
                while (vif.weightReady !== 1'b1)
                    @(negedge vif.clk);
                for (int lane = 0; lane < N; lane++)
                    vif.weightData[lane] = vector[lane];
                vif.weightValid = 1'b1;
            end else begin
                vif.activationValid = 1'b0;
                while (vif.activationReady !== 1'b1)
                    @(negedge vif.clk);
                for (int lane = 0; lane < N; lane++)
                    vif.activationData[lane] = vector[lane];
                vif.activationValid = 1'b1;
            end

            // ready was sampled high before this clock edge, so the vector is
            // accepted on this edge even if the FIFO changes ready afterward.
            @(posedge vif.clk);
            @(negedge vif.clk);
            if (is_weight)
                vif.weightValid = 1'b0;
            else
                vif.activationValid = 1'b0;
        endtask
    endclass

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

    // ------------------------------------------------------------------
    // Independent reference model and scoreboard
    // ------------------------------------------------------------------

    class nn_core_scoreboard extends uvm_scoreboard;
        `uvm_component_utils(nn_core_scoreboard)

        uvm_tlm_analysis_fifo #(nn_core_matrix_item) expected_fifo;
        uvm_tlm_analysis_fifo #(nn_core_result_row) actual_fifo;
        virtual nn_core_if #(WIDTH, N) vif;
        int unsigned matrices_checked;
        int unsigned mismatches;
        event matrix_checked_event;

        function new(string name, uvm_component parent);
            super.new(name, parent);
            matrices_checked = 0;
            mismatches = 0;
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
        // truncation and post-accumulation ReLU behavior.
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
                    if (!item.pass_through && expected[row][col][RESULT_WIDTH-1])
                        expected[row][col] = '0;
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
                        mismatches++;
                        `uvm_error("FRAMING", $sformatf(
                            "scoreboard saw row index %0d while expecting %0d",
                            actual.row_index, row))
                    end
                    if (actual.last !== (row == N-1)) begin
                        mismatches++;
                        `uvm_error("FRAMING", $sformatf(
                            "scoreboard saw resultLast=%0b on row %0d",
                            actual.last, row))
                    end
                    for (int col = 0; col < N; col++) begin
                        if (actual.data[col] !== expected[row][col]) begin
                            mismatches++;
                            `uvm_error("MISMATCH", $sformatf(
                                "C[%0d][%0d] got %0d (0x%0h) expected %0d (0x%0h)",
                                row, col, actual.data[col], actual.data[col],
                                expected[row][col], expected[row][col]))
                        end
                    end
                end

                matrices_checked++;
                -> matrix_checked_event;
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

    // Questa FPGA Starter installations commonly lack the SystemVerilog
    // coverage feature license.  These counters provide runnable functional
    // coverage without covergroup syntax; the broad test checks the required
    // bins explicitly.
    class nn_core_coverage extends uvm_component;
        `uvm_component_utils(nn_core_coverage)

        virtual nn_core_if #(WIDTH, N) vif;
        uvm_analysis_imp_matrix #(nn_core_matrix_item, nn_core_coverage) matrix_imp;
        uvm_analysis_imp_result #(nn_core_result_row, nn_core_coverage) result_imp;

        int unsigned matrix_count;
        int unsigned pass_through_count;
        int unsigned relu_count;
        int unsigned negative_operand_count;
        int unsigned repeated_weight_matrix_count;
        int unsigned result_row_count;
        int unsigned result_last_count;
        int unsigned negative_result_row_count;
        int unsigned positive_result_row_count;
        int unsigned weight_bubble_cycles;
        int unsigned activation_bubble_cycles;
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
            pass_through_count = 0;
            relu_count = 0;
            negative_operand_count = 0;
            repeated_weight_matrix_count = 0;
            result_row_count = 0;
            result_last_count = 0;
            negative_result_row_count = 0;
            positive_result_row_count = 0;
            weight_bubble_cycles = 0;
            activation_bubble_cycles = 0;
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
            if (item.pass_through)
                pass_through_count++;
            else
                relu_count++;

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
            bit has_negative;
            bit has_positive;
            result_row_count++;
            if (row.last)
                result_last_count++;

            has_negative = 1'b0;
            has_positive = 1'b0;
            for (int lane = 0; lane < N; lane++) begin
                has_negative |= row.data[lane][RESULT_WIDTH-1];
                has_positive |= !row.data[lane][RESULT_WIDTH-1] &&
                                (row.data[lane] != 0);
            end
            if (has_negative)
                negative_result_row_count++;
            if (has_positive)
                positive_result_row_count++;
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
            if (pass_through_count == 0)
                `uvm_error("COVERAGE", "pass-through mode bin was not observed")
            if (relu_count == 0)
                `uvm_error("COVERAGE", "ReLU mode bin was not observed")
            if (negative_operand_count == 0)
                `uvm_error("COVERAGE", "negative operand bin was not observed")
            if (repeated_weight_matrix_count == 0)
                `uvm_error("COVERAGE", "repeated matrix under stationary weights was not observed")
            if (weight_bubble_cycles == 0)
                `uvm_error("COVERAGE", "weight input bubble bin was not observed")
            if (activation_bubble_cycles == 0)
                `uvm_error("COVERAGE", "activation input bubble bin was not observed")
            if (output_stall_cycles == 0)
                `uvm_error("COVERAGE", "output backpressure bin was not observed")
            if (reload_count == 0)
                `uvm_error("COVERAGE", "weight reload bin was not observed")
            if (reset_count == 0)
                `uvm_error("COVERAGE", "injected reset bin was not observed")
            if (result_last_count == 0)
                `uvm_error("COVERAGE", "resultLast bin was not observed")
        endfunction

        function void report_phase(uvm_phase phase);
            `uvm_info("COVERAGE", $sformatf(
                "matrices=%0d passThrough=%0d ReLU=%0d negativeOperands=%0d repeatedUnderWeights=%0d",
                matrix_count, pass_through_count, relu_count,
                negative_operand_count, repeated_weight_matrix_count), UVM_NONE)
            `uvm_info("COVERAGE", $sformatf(
                "weightBubbles=%0d activationBubbles=%0d outputStallCycles=%0d reloads=%0d injectedResets=%0d",
                weight_bubble_cycles, activation_bubble_cycles,
                output_stall_cycles, reload_count, reset_count), UVM_NONE)
            `uvm_info("COVERAGE", $sformatf(
                "resultRows=%0d resultLastRows=%0d negativeResultRows=%0d positiveResultRows=%0d",
                result_row_count, result_last_count,
                negative_result_row_count, positive_result_row_count), UVM_NONE)
        endfunction
    endclass

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
            scoreboard.wait_for_matrices(expected_matrices);
            // Allow the passive result monitor to observe the final transfer
            // and expose any extra/duplicated rows before the test ends.
            repeat (3) @(posedge vif.clk);

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
            int unsigned word;
            value = '0;
            for (int bit_index = 0; bit_index < WIDTH; bit_index++) begin
                word = next_random();
                value[bit_index] = word[0];
            end
            return data_t'(value);
        endfunction

        function data_t min_data();
            return data_t'({1'b1, {(WIDTH-1){1'b0}}});
        endfunction

        function data_t max_data();
            return data_t'({1'b0, {(WIDTH-1){1'b1}}});
        endfunction

        function void fill_identity(nn_core_matrix_item item);
            for (int row = 0; row < N; row++) begin
                for (int col = 0; col < N; col++) begin
                    item.weights[row][col] = (row == col) ? data_t'(1) : data_t'(0);
                end
            end
        endfunction

        function void fill_signed_activation(nn_core_matrix_item item);
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

        function void fill_random(nn_core_matrix_item item);
            for (int row = 0; row < N; row++) begin
                for (int col = 0; col < N; col++) begin
                    item.weights[row][col] = random_data();
                    item.activations[row][col] = random_data();
                end
            end
            // Guarantee signed cases even if a future random generator is
            // changed to a distribution that happens to miss the sign bit.
            item.weights[0][0] = data_t'(-3);
            item.activations[0][0] = data_t'(-5);
            item.activations[0][1] = data_t'(7);
        endfunction

        function void fill_edge_case(nn_core_matrix_item item);
            for (int row = 0; row < N; row++) begin
                for (int col = 0; col < N; col++) begin
                    item.weights[row][col] = (row == col) ?
                                             data_t'(1) :
                                             (((row + col) % 2) ? min_data() : max_data());
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
            fill_identity(item);
            fill_signed_activation(item);
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
            fill_identity(item);
            fill_signed_activation(item);
            item.pass_through = 1'b1;
            item.load_weights = 1'b1;
            item.stall_percent = 35;
            send_item(item);

            // Matrix 1: no reload; this is the back-to-back stationary-weight
            // case and includes activation bubbles.
            item = nn_core_matrix_item::type_id::create("identity_random");
            fill_identity(item);
            fill_random(item);
            item.pass_through = 1'b1;
            item.load_weights = 1'b0;
            item.activation_bubbles = 1'b1;
            item.stall_percent = 45;
            send_item(item);

            // Matrix 2: mode change after the preceding results have drained.
            item = nn_core_matrix_item::type_id::create("identity_relu");
            fill_identity(item);
            fill_signed_activation(item);
            item.pass_through = 1'b0;
            item.activation_bubbles = 1'b1;
            item.stall_percent = 30;
            send_item(item);

            // Matrix 3: new stationary weights, with bubbles in both input
            // streams and an explicit reload boundary.
            item = nn_core_matrix_item::type_id::create("random_weights_pass");
            fill_random(item);
            item.pass_through = 1'b1;
            item.load_weights = 1'b1;
            item.reload_before = 1'b1;
            item.weight_bubbles = 1'b1;
            item.activation_bubbles = 1'b1;
            item.stall_percent = 40;
            send_item(item);

            // Matrix 4: reuse those weights in ReLU mode.
            item = nn_core_matrix_item::type_id::create("random_weights_relu");
            fill_random(item);
            item.pass_through = 1'b0;
            item.activation_bubbles = 1'b1;
            item.stall_percent = 25;
            send_item(item);

            // Reset while a new weight frame is only partially accepted.  No
            // matrix is expected from this intentionally aborted item.
            item = nn_core_matrix_item::type_id::create("reset_partial_weights");
            fill_random(item);
            item.load_weights = 1'b1;
            item.reload_before = 1'b1;
            item.weight_bubbles = 1'b1;
            item.reset_phase = NN_RESET_DURING_WEIGHT_LOAD;
            item.reset_after_rows = 1;
            item.stall_percent = 20;
            send_item(item);

            // Recovery after the partial weight reset.
            item = nn_core_matrix_item::type_id::create("recovered_after_weight_reset");
            fill_edge_case(item);
            item.pass_through = 1'b1;
            item.load_weights = 1'b1;
            item.activation_bubbles = 1'b1;
            item.stall_percent = 35;
            send_item(item);

            // Reset with a stationary matrix loaded and a partial activation
            // frame in flight.  wait_for_drain keeps earlier expected rows
            // from being intentionally discarded by this reset.
            item = nn_core_matrix_item::type_id::create("reset_partial_activation");
            fill_edge_case(item);
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
            fill_identity(item);
            fill_signed_activation(item);
            item.pass_through = 1'b1;
            item.load_weights = 1'b1;
            item.weight_bubbles = 1'b1;
            item.stall_percent = 35;
            send_item(item);

            item = nn_core_matrix_item::type_id::create("final_relu_random");
            fill_identity(item);
            fill_random(item);
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
            smoke_seq.random_seed = seed;
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

endpackage
