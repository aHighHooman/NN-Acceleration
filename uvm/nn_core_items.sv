    // ------------------------------------------------------------------
    // The small set of UVM transactions
    // ------------------------------------------------------------------

    // One sample item is used on both sides of the environment.  The active
    // driver consumes the policy fields; the passive monitor fills the
    // configuration snapshot before publishing the accepted sample.
    class nn_core_sample_item extends uvm_sequence_item;
        data_t activation[N];
        target_t target;
        bit training_enable;
        bit activation_bubble;
        int unsigned result_stall_percent;
        bit hold_result_until_activation_backpressure;
        bit reset_after_accept;

        data_t weights[N][N];
        reduction_t reduction_weights[N];
        bit weights_valid;
        bit pass_through;
        bit reduce_output;

        `uvm_object_utils(nn_core_sample_item)

        function new(string name = "nn_core_sample_item");
            super.new(name);
        endfunction

        function string convert2string();
            return $sformatf(
                "training=%0b bubble=%0b resultStall=%0d holdForInputPressure=%0b resetAfterAccept=%0b",
                training_enable, activation_bubble, result_stall_percent,
                hold_result_until_activation_backpressure, reset_after_accept);
        endfunction
    endclass

    // A configuration item streams one complete W matrix and can request a
    // drained reload and/or a reduction-vector load.
    class nn_core_config_item extends uvm_sequence_item;
        data_t weights[N][N];
        reduction_t reduction_weights[N];
        bit load_reduction_weights;
        bit reload_before;
        bit weight_bubble;

        `uvm_object_utils(nn_core_config_item)

        function new(string name = "nn_core_config_item");
            super.new(name);
        endfunction

        function string convert2string();
            return $sformatf("reload=%0b reductionLoad=%0b weightBubble=%0b",
                             reload_before, load_reduction_weights,
                             weight_bubble);
        endfunction
    endclass

    class nn_core_result_transaction extends uvm_sequence_item;
        result_t data[N];

        `uvm_object_utils(nn_core_result_transaction)

        function new(string name = "nn_core_result_transaction");
            super.new(name);
        endfunction
    endclass

    // Keep a named sequencer so the example retains the standard UVM flow.
    class nn_core_sequencer extends uvm_sequencer #(uvm_sequence_item);
        `uvm_component_utils(nn_core_sequencer)

        function new(string name, uvm_component parent);
            super.new(name, parent);
        endfunction
    endclass
