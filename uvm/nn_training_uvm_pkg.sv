package nn_training_uvm_pkg;
    import uvm_pkg::*;
    `include "uvm_macros.svh"

    // Keep the accelerator-level environment independent from the core UVM
    // package.  Questa then resolves only the virtual interface instantiated
    // by the selected top instead of trying to bind both environments.
    localparam int WIDTH = 8;
    localparam int N = 3;

    typedef logic signed [WIDTH-1:0] data_t;

    `include "nn_training_tests.sv"

endpackage
