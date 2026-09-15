    // ------------------------------------------------------------------
    // Transactions
    // ------------------------------------------------------------------

    // Architectural sample data observed at the public activation boundary.
    // This class intentionally contains no weight configuration or grouping
    // state: one accepted activation vector is one sample transaction.
    class nn_core_sample_transaction extends uvm_sequence_item;
        data_t activation[N];
        target_t target;
        bit training_enable;

        `uvm_object_utils(nn_core_sample_transaction)

        function new(string name = "nn_core_sample_transaction");
            super.new(name);
            target = '0;
            training_enable = 1'b0;
        endfunction
    endclass

    // Stimulus policy is kept on the active item and is not part of the
    // architectural payload published by the passive input monitor.
    class nn_core_sample_item extends nn_core_sample_transaction;
        bit activation_bubble;
        int unsigned result_stall_percent;
        bit hold_result_until_activation_backpressure;
        bit reset_after_accept;
        bit wait_for_drain_after;

        `uvm_object_utils(nn_core_sample_item)

        function new(string name = "nn_core_sample_item");
            super.new(name);
            activation_bubble = 1'b0;
            result_stall_percent = 0;
            hold_result_until_activation_backpressure = 1'b0;
            reset_after_accept = 1'b0;
            wait_for_drain_after = 1'b0;
        endfunction

        function string convert2string();
            return $sformatf(
                "training=%0b bubble=%0b resultStall=%0d holdForInputPressure=%0b resetAfterAccept=%0b waitForDrain=%0b",
                training_enable, activation_bubble, result_stall_percent,
                hold_result_until_activation_backpressure, reset_after_accept,
                wait_for_drain_after);
        endfunction
    endclass

    // Weight/reduction loading is configuration traffic, separate from every
    // ordinary sample item.  The matrix is carried only on this item.
    class nn_core_weight_load_item extends uvm_sequence_item;
        data_t weights[N][N];
        reduction_t reduction_weights[N];
        bit load_reduction_weights;
        bit reload_before;
        bit weight_bubble;
        int unsigned reset_after_rows;

        `uvm_object_utils(nn_core_weight_load_item)

        function new(string name = "nn_core_weight_load_item");
            super.new(name);
            load_reduction_weights = 1'b0;
            reload_before = 1'b0;
            weight_bubble = 1'b0;
            reset_after_rows = 0;
            for (int lane = 0; lane < N; lane++)
                reduction_weights[lane] = '0;
        endfunction

        function string convert2string();
            return $sformatf(
                "reload=%0b reductionLoad=%0b weightBubble=%0b resetAfterRows=%0d",
                reload_before, load_reduction_weights, weight_bubble,
                reset_after_rows);
        endfunction
    endclass

    class nn_core_result_transaction extends uvm_sequence_item;
        result_t data[N];

        `uvm_object_utils(nn_core_result_transaction)

        function new(string name = "nn_core_result_transaction");
            super.new(name);
        endfunction
    endclass

    class nn_core_weight_row_transaction extends uvm_sequence_item;
        data_t data[N];
        int unsigned row_index;
        bit completes_load;

        `uvm_object_utils(nn_core_weight_row_transaction)

        function new(string name = "nn_core_weight_row_transaction");
            super.new(name);
            row_index = 0;
            completes_load = 1'b0;
        endfunction
    endclass

    class nn_core_reduction_load_transaction extends uvm_sequence_item;
        reduction_t data[N];

        `uvm_object_utils(nn_core_reduction_load_transaction)

        function new(string name = "nn_core_reduction_load_transaction");
            super.new(name);
        endfunction
    endclass

    class nn_core_reload_transaction extends uvm_sequence_item;
        `uvm_object_utils(nn_core_reload_transaction)

        function new(string name = "nn_core_reload_transaction");
            super.new(name);
        endfunction
    endclass

    class nn_core_reset_transaction extends uvm_sequence_item;
        int unsigned generation;

        `uvm_object_utils(nn_core_reset_transaction)

        function new(string name = "nn_core_reset_transaction");
            super.new(name);
            generation = 0;
        endfunction
    endclass

    // ------------------------------------------------------------------
    // Sequencer
    // ------------------------------------------------------------------

    class nn_core_sequencer extends uvm_sequencer #(uvm_sequence_item);
        `uvm_component_utils(nn_core_sequencer)

        function new(string name, uvm_component parent);
            super.new(name, parent);
        endfunction
    endclass
