package nn_uvm_pkg;
    import uvm_pkg::*;
    `include "uvm_macros.svh"

    localparam int WIDTH = 8;
    localparam int N = 2;
    localparam int RESULT_WIDTH = 2*WIDTH + $clog2(N);

    typedef logic signed [WIDTH-1:0] data_t;
    typedef logic signed [RESULT_WIDTH-1:0] result_t;

    // ---------------------------------------------------------------------
    // Transactions
    // ---------------------------------------------------------------------

    // One item describes a complete A x B operation.  A production SPI VIP
    // would usually use a smaller word/frame item and layer matrix sequences
    // above it.  Starting at matrix level keeps this teaching example small.
    class nn_matrix_item extends uvm_sequence_item;
        data_t weights[N][N];
        data_t activations[N][N];
        bit pass_through;

        // These fields are intentionally not consumed by the starter driver.
        // They provide obvious extension points for issue #3 exercises.
        int unsigned sclk_half_period_ns;
        int unsigned input_bubbles;

        `uvm_object_utils(nn_matrix_item)

        function new(string name = "nn_matrix_item");
            super.new(name);
            sclk_half_period_ns = 7;
            input_bubbles = 0;
        endfunction

        function string convert2string();
            return $sformatf("passThrough=%0b A=[[0x%0h,0x%0h],[0x%0h,0x%0h]] B=[[0x%0h,0x%0h],[0x%0h,0x%0h]]",
                pass_through,
                activations[0][0], activations[0][1],
                activations[1][0], activations[1][1],
                weights[0][0], weights[0][1],
                weights[1][0], weights[1][1]);
        endfunction
    endclass

    class nn_result_row extends uvm_sequence_item;
        result_t data[N];

        `uvm_object_utils(nn_result_row)

        function new(string name = "nn_result_row");
            super.new(name);
        endfunction
    endclass

    // ---------------------------------------------------------------------
    // Sequences: two complete examples, then room for your own cases
    // ---------------------------------------------------------------------

    class nn_identity_sequence extends uvm_sequence #(nn_matrix_item);
        `uvm_object_utils(nn_identity_sequence)

        function new(string name = "nn_identity_sequence");
            super.new(name);
        endfunction

        task body();
            nn_matrix_item item;
            item = nn_matrix_item::type_id::create("identity_item");
            start_item(item);
            item.weights[0][0] = 1;  item.weights[0][1] = 0;
            item.weights[1][0] = 0;  item.weights[1][1] = 1;
            item.activations[0][0] = 2;  item.activations[0][1] = -3;
            item.activations[1][0] = 4;  item.activations[1][1] = 5;
            item.pass_through = 1;
            item.sclk_half_period_ns = 7;
            item.input_bubbles = 0;
            finish_item(item);
        endtask
    endclass

    class nn_relu_sequence extends uvm_sequence #(nn_matrix_item);
        `uvm_object_utils(nn_relu_sequence)

        function new(string name = "nn_relu_sequence");
            super.new(name);
        endfunction

        task body();
            nn_matrix_item item;
            item = nn_matrix_item::type_id::create("relu_item");
            start_item(item);
            item.weights[0][0] = 1;   item.weights[0][1] = 2;
            item.weights[1][0] = -1;  item.weights[1][1] = 1;
            item.activations[0][0] = 2;   item.activations[0][1] = -3;
            item.activations[1][0] = -4;  item.activations[1][1] = 1;
            item.pass_through = 0;
            item.sclk_half_period_ns = 7;
            item.input_bubbles = 0;
            finish_item(item);
        endtask
    endclass

    class nn_matrix_sequencer extends uvm_sequencer #(nn_matrix_item);
        `uvm_component_utils(nn_matrix_sequencer)

        function new(string name, uvm_component parent);
            super.new(name, parent);
        endfunction
    endclass

    // ---------------------------------------------------------------------
    // Active host driver
    // ---------------------------------------------------------------------

    class nn_spi_driver extends uvm_driver #(nn_matrix_item);
        `uvm_component_utils(nn_spi_driver)

        virtual nn_uvm_if vif;
        uvm_analysis_port #(nn_matrix_item) expected_ap;

        function new(string name, uvm_component parent);
            super.new(name, parent);
            expected_ap = new("expected_ap", this);
        endfunction

        function void build_phase(uvm_phase phase);
            super.build_phase(phase);
            if (!uvm_config_db #(virtual nn_uvm_if)::get(this, "", "vif", vif))
                `uvm_fatal("NO_VIF", "nn_spi_driver did not receive nn_uvm_if")
        endfunction

        task run_phase(uvm_phase phase);
            nn_matrix_item sent;
            reset_dut();

            forever begin
                seq_item_port.get_next_item(req);
                `uvm_info("DRV", {"Driving ", req.convert2string()}, UVM_MEDIUM)

                prepare_for_new_weights();
                vif.passThrough = req.pass_through;
                // TODO(student): use req.sclk_half_period_ns here, then create a
                // sequence that randomizes it between transactions.
                drive_weights(req);
                drive_activations(req);

                // Publish the prediction before the result monitor can finish
                // its first row.  A more reusable environment would predict
                // from a passive input monitor instead of from the driver.
                sent = nn_matrix_item::type_id::create("sent");
                copy_item(req, sent);
                expected_ap.write(sent);

                pull_all_result_rows();
                seq_item_port.item_done();
            end
        endtask

        task reset_dut();
            vif.rst_n = 0;
            vif.passThrough = 1;
            vif.reloadWeights = 0;
            drive_idle();
            repeat (3) @(posedge vif.sclk);
            repeat (3) @(posedge vif.clk);
            @(negedge vif.sclk);
            vif.rst_n = 1;
        endtask

        task drive_idle();
            for (int lane = 0; lane < N; lane++) begin
                vif.cs_n[lane] = 1;
                vif.weightCs_n[lane] = 1;
                vif.activationCs_n[lane] = 1;
                vif.weightMosi[lane] = 0;
                vif.activationMosi[lane] = 0;
            end
        endtask

        task prepare_for_new_weights();
            int cycles;
            if (vif.weightsLoaded !== 1'b1)
                return;

            cycles = 0;
            while (vif.reloadReady !== 1'b1) begin
                @(posedge vif.clk);
                cycles++;
                if (cycles > 10000)
                    `uvm_fatal("RELOAD_TIMEOUT", "reloadReady never asserted")
            end

            @(negedge vif.clk);
            vif.reloadWeights = 1;
            @(posedge vif.clk);
            @(negedge vif.clk);
            vif.reloadWeights = 0;

            cycles = 0;
            while (vif.weightsLoaded !== 1'b0) begin
                @(posedge vif.clk);
                cycles++;
                if (cycles > 10000)
                    `uvm_fatal("RELOAD_TIMEOUT", "weightsLoaded never cleared")
            end
        endtask

        task drive_weights(nn_matrix_item item);
            data_t vector[N];
            // The RTL contract loads B from its final row to its first row.
            for (int row = N-1; row >= 0; row--) begin
                wait_for_input_ready(1);
                for (int lane = 0; lane < N; lane++)
                    vector[lane] = item.weights[row][lane];
                send_input_vector(vector, 1);
            end

            while (vif.weightsLoaded !== 1'b1)
                @(posedge vif.clk);
        endtask

        task drive_activations(nn_matrix_item item);
            data_t vector[N];
            for (int row = 0; row < N; row++) begin
                wait_for_input_ready(0);
                for (int lane = 0; lane < N; lane++)
                    vector[lane] = item.activations[row][lane];
                send_input_vector(vector, 0);
            end
        endtask

        task wait_for_input_ready(bit is_weight);
            int cycles;
            cycles = 0;
            while ((is_weight && vif.weightReady !== 1'b1) ||
                   (!is_weight && vif.activationReady !== 1'b1)) begin
                @(posedge vif.sclk);
                cycles++;
                if (cycles > 10000)
                    `uvm_fatal("READY_TIMEOUT", "SPI input ready never asserted")
            end
        endtask

        task send_input_vector(data_t vector[N], bit is_weight);
            @(negedge vif.sclk);
            for (int lane = 0; lane < N; lane++) begin
                if (is_weight)
                    vif.weightCs_n[lane] = 0;
                else
                    vif.activationCs_n[lane] = 0;
            end

            for (int bit_index = WIDTH-1; bit_index >= 0; bit_index--) begin
                for (int lane = 0; lane < N; lane++) begin
                    if (is_weight)
                        vif.weightMosi[lane] = vector[lane][bit_index];
                    else
                        vif.activationMosi[lane] = vector[lane][bit_index];
                end
                @(posedge vif.sclk);
                @(negedge vif.sclk);
            end

            for (int lane = 0; lane < N; lane++) begin
                if (is_weight)
                    vif.weightCs_n[lane] = 1;
                else
                    vif.activationCs_n[lane] = 1;
            end
        endtask

        task pull_all_result_rows();
            for (int row = 0; row < N; row++)
                pull_one_result_row();
        endtask

        task pull_one_result_row();
            bit all_valid;
            int polls;
            polls = 0;
            all_valid = 0;

            while (!all_valid) begin
                @(negedge vif.sclk);
                for (int lane = 0; lane < N; lane++)
                    vif.cs_n[lane] = 0;
                #1;
                all_valid = 1;
                for (int lane = 0; lane < N; lane++)
                    all_valid &= vif.misoValid[lane];
                if (!all_valid)
                    for (int lane = 0; lane < N; lane++)
                        vif.cs_n[lane] = 1;
                polls++;
                if (polls > 10000)
                    `uvm_fatal("RESULT_TIMEOUT", "No complete result row became available")
            end

            // The passive monitor samples these bits on the same rising edges.
            repeat (RESULT_WIDTH) @(posedge vif.sclk);
            @(negedge vif.sclk);
            for (int lane = 0; lane < N; lane++)
                vif.cs_n[lane] = 1;
        endtask

        function void copy_item(nn_matrix_item source, nn_matrix_item destination);
            for (int row = 0; row < N; row++)
                for (int col = 0; col < N; col++) begin
                    destination.weights[row][col] = source.weights[row][col];
                    destination.activations[row][col] = source.activations[row][col];
                end
            destination.pass_through = source.pass_through;
            destination.sclk_half_period_ns = source.sclk_half_period_ns;
            destination.input_bubbles = source.input_bubbles;
        endfunction
    endclass

    // ---------------------------------------------------------------------
    // Passive result monitor
    // ---------------------------------------------------------------------

    class nn_result_monitor extends uvm_monitor;
        `uvm_component_utils(nn_result_monitor)

        virtual nn_uvm_if vif;
        uvm_analysis_port #(nn_result_row) result_ap;
        result_t partial[N];
        int unsigned bits_seen;

        function new(string name, uvm_component parent);
            super.new(name, parent);
            result_ap = new("result_ap", this);
        endfunction

        function void build_phase(uvm_phase phase);
            super.build_phase(phase);
            if (!uvm_config_db #(virtual nn_uvm_if)::get(this, "", "vif", vif))
                `uvm_fatal("NO_VIF", "nn_result_monitor did not receive nn_uvm_if")
        endfunction

        task run_phase(uvm_phase phase);
            bit transfer_active;
            nn_result_row row;
            bits_seen = 0;
            for (int lane = 0; lane < N; lane++)
                partial[lane] = '0;

            forever begin
                @(posedge vif.sclk);
                if (!vif.rst_n) begin
                    bits_seen = 0;
                    for (int lane = 0; lane < N; lane++)
                        partial[lane] = '0;
                end else begin
                    transfer_active = 1;
                    for (int lane = 0; lane < N; lane++)
                        transfer_active &= (!vif.cs_n[lane] && vif.misoValid[lane]);

                    if (transfer_active) begin
                        for (int lane = 0; lane < N; lane++)
                            partial[lane] = (partial[lane] << 1) | vif.miso[lane];
                        bits_seen++;

                        if (bits_seen == RESULT_WIDTH) begin
                            row = nn_result_row::type_id::create("observed_row");
                            for (int lane = 0; lane < N; lane++) begin
                                row.data[lane] = partial[lane];
                                partial[lane] = '0;
                            end
                            bits_seen = 0;
                            result_ap.write(row);
                        end
                    end
                    // Deliberately preserve partial/bits_seen while CS is high:
                    // the documented result protocol permits pause-and-resume.
                end
            end
        endtask
    endclass

    // ---------------------------------------------------------------------
    // End-to-end reference model and scoreboard
    // ---------------------------------------------------------------------

    class nn_scoreboard extends uvm_scoreboard;
        `uvm_component_utils(nn_scoreboard)

        uvm_tlm_analysis_fifo #(nn_matrix_item) expected_fifo;
        uvm_tlm_analysis_fifo #(nn_result_row) actual_fifo;
        int unsigned matrices_checked;

        function new(string name, uvm_component parent);
            super.new(name, parent);
        endfunction

        function void build_phase(uvm_phase phase);
            super.build_phase(phase);
            expected_fifo = new("expected_fifo", this);
            actual_fifo = new("actual_fifo", this);
        endfunction

        task run_phase(uvm_phase phase);
            nn_matrix_item item;
            nn_result_row actual;
            result_t expected[N][N];
            forever begin
                expected_fifo.get(item);
                predict(item, expected);
                for (int row = 0; row < N; row++) begin
                    actual_fifo.get(actual);
                    for (int col = 0; col < N; col++) begin
                        if (actual.data[col] !== expected[row][col])
                            `uvm_error("MISMATCH", $sformatf(
                                "C[%0d][%0d] got %0d expected %0d",
                                row, col, actual.data[col], expected[row][col]))
                    end
                end
                matrices_checked++;
                `uvm_info("SCOREBOARD", $sformatf(
                    "Checked matrix %0d: %s", matrices_checked,
                    item.convert2string()), UVM_LOW)
            end
        endtask

        function void predict(nn_matrix_item item, output result_t expected[N][N]);
            longint signed sum;
            for (int row = 0; row < N; row++) begin
                for (int col = 0; col < N; col++) begin
                    sum = 0;
                    for (int k = 0; k < N; k++)
                        sum += $signed(item.activations[row][k]) *
                               $signed(item.weights[k][col]);
                    expected[row][col] = sum[RESULT_WIDTH-1:0];
                    if (!item.pass_through && expected[row][col][RESULT_WIDTH-1])
                        expected[row][col] = '0;
                end
            end
        endfunction
    endclass

    // Lightweight counters show the subscriber pattern without requiring the
    // optional Questa "svverification" license feature used by covergroups.
    // Replace these counters with the covergroup in EXERCISES.md when that
    // feature is available.
    class nn_matrix_stats extends uvm_subscriber #(nn_matrix_item);
        `uvm_component_utils(nn_matrix_stats)

        int unsigned pass_through_count;
        int unsigned relu_count;
        int unsigned negative_operand_count;

        function new(string name, uvm_component parent);
            super.new(name, parent);
        endfunction

        function void write(nn_matrix_item t);
            bit has_negative;
            if (t.pass_through)
                pass_through_count++;
            else
                relu_count++;
            has_negative = 0;
            for (int row = 0; row < N; row++)
                for (int col = 0; col < N; col++)
                    has_negative |= t.activations[row][col][WIDTH-1] |
                                    t.weights[row][col][WIDTH-1];
            if (has_negative)
                negative_operand_count++;
        endfunction

        function void report_phase(uvm_phase phase);
            `uvm_info("STATS", $sformatf(
                "observed pass-through=%0d ReLU=%0d negative-operands=%0d",
                pass_through_count, relu_count, negative_operand_count), UVM_LOW)
        endfunction
    endclass

    // ---------------------------------------------------------------------
    // Environment and tests
    // ---------------------------------------------------------------------

    class nn_uvm_env extends uvm_env;
        `uvm_component_utils(nn_uvm_env)

        nn_matrix_sequencer sequencer;
        nn_spi_driver driver;
        nn_result_monitor monitor;
        nn_scoreboard scoreboard;
        nn_matrix_stats stats;

        function new(string name, uvm_component parent);
            super.new(name, parent);
        endfunction

        function void build_phase(uvm_phase phase);
            super.build_phase(phase);
            sequencer = nn_matrix_sequencer::type_id::create("sequencer", this);
            driver = nn_spi_driver::type_id::create("driver", this);
            monitor = nn_result_monitor::type_id::create("monitor", this);
            scoreboard = nn_scoreboard::type_id::create("scoreboard", this);
            stats = nn_matrix_stats::type_id::create("stats", this);
        endfunction

        function void connect_phase(uvm_phase phase);
            super.connect_phase(phase);
            driver.seq_item_port.connect(sequencer.seq_item_export);
            driver.expected_ap.connect(scoreboard.expected_fifo.analysis_export);
            driver.expected_ap.connect(stats.analysis_export);
            monitor.result_ap.connect(scoreboard.actual_fifo.analysis_export);
        endfunction
    endclass

    class nn_uvm_base_test extends uvm_test;
        `uvm_component_utils(nn_uvm_base_test)
        nn_uvm_env env;

        function new(string name, uvm_component parent);
            super.new(name, parent);
        endfunction

        function void build_phase(uvm_phase phase);
            super.build_phase(phase);
            env = nn_uvm_env::type_id::create("env", this);
        endfunction
    endclass

    class nn_uvm_smoke_test extends nn_uvm_base_test;
        `uvm_component_utils(nn_uvm_smoke_test)

        function new(string name, uvm_component parent);
            super.new(name, parent);
        endfunction

        task run_phase(uvm_phase phase);
            nn_identity_sequence identity_seq;
            phase.raise_objection(this);
            identity_seq = nn_identity_sequence::type_id::create("identity_seq");
            identity_seq.start(env.sequencer);
            phase.drop_objection(this);
        endtask
    endclass

    class nn_uvm_relu_test extends nn_uvm_base_test;
        `uvm_component_utils(nn_uvm_relu_test)

        function new(string name, uvm_component parent);
            super.new(name, parent);
        endfunction

        task run_phase(uvm_phase phase);
            nn_relu_sequence relu_seq;
            phase.raise_objection(this);
            relu_seq = nn_relu_sequence::type_id::create("relu_seq");
            relu_seq.start(env.sequencer);
            phase.drop_objection(this);
        endtask
    endclass

    // Runs both completed examples and demonstrates environment reuse plus the
    // driver's weight-reload path.
    class nn_uvm_starter_regression_test extends nn_uvm_base_test;
        `uvm_component_utils(nn_uvm_starter_regression_test)

        function new(string name, uvm_component parent);
            super.new(name, parent);
        endfunction

        task run_phase(uvm_phase phase);
            nn_identity_sequence identity_sequence;
            nn_relu_sequence relu_sequence;
            phase.raise_objection(this);
            identity_sequence = nn_identity_sequence::type_id::create("identity_sequence");
            relu_sequence = nn_relu_sequence::type_id::create("relu_sequence");
            identity_sequence.start(env.sequencer);
            relu_sequence.start(env.sequencer);
            phase.drop_objection(this);
        endtask
    endclass

endpackage
