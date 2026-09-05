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

        bit load_weights;
        bit reload_before;
        bit weight_bubbles;
        bit activation_bubbles;
        int unsigned stall_percent;
        bit stall_until_activation_backpressure;
        bit wait_for_drain;

        reset_phase_e reset_phase;
        int unsigned reset_after_rows;

        // Filled only by the passive monitor.  It is useful in coverage and
        // is deliberately not used as stimulus configuration.
        int unsigned weight_generation;

        `uvm_object_utils(nn_core_matrix_item)

        function new(string name = "nn_core_matrix_item");
            super.new(name);
            load_weights = 1'b0;
            reload_before = 1'b0;
            weight_bubbles = 1'b0;
            activation_bubbles = 1'b0;
            stall_percent = 0;
            stall_until_activation_backpressure = 1'b0;
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
                "loadWeights=%0b reload=%0b bubbles(w/a)=%0b/%0b stall=%0d stallUntilActivationBackpressure=%0b reset=%0d",
                load_weights, reload_before,
                weight_bubbles, activation_bubbles, stall_percent,
                stall_until_activation_backpressure, reset_phase);
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
    // Sequencer
    // ------------------------------------------------------------------

    class nn_core_sequencer extends uvm_sequencer #(nn_core_matrix_item);
        `uvm_component_utils(nn_core_sequencer)

        function new(string name, uvm_component parent);
            super.new(name, parent);
        endfunction
    endclass
