package nn_uvm_pkg;
    import uvm_pkg::*;
    `include "uvm_macros.svh"

    // The UVM compile is intentionally a small, fast N=3 configuration.  The
    // RTL and directed regression continue to cover the other supported array
    // sizes; changing these two package constants retargets this environment.
    localparam int WIDTH = 8;
    localparam int N = 3;
    localparam int TARGET_WIDTH = WIDTH;
    localparam int REDUCTION_WEIGHT_WIDTH = 8;
    localparam int FRACTION_BITS = 4;
    localparam int MATRIX_RESULT_WIDTH = 2*WIDTH + $clog2(N);
    localparam int RESULT_WIDTH = MATRIX_RESULT_WIDTH + $clog2(N);
    localparam int unsigned DEFAULT_SEED = 32'h5eed_2026;

    typedef logic signed [WIDTH-1:0] data_t;
    typedef logic signed [TARGET_WIDTH-1:0] target_t;
    typedef logic signed [REDUCTION_WEIGHT_WIDTH-1:0] reduction_t;
    typedef logic signed [MATRIX_RESULT_WIDTH-1:0] matrix_result_t;
    typedef logic signed [RESULT_WIDTH-1:0] result_t;

    // The environment has only two semantic streams: accepted samples and
    // retired results. Configuration is reconstructed locally by the input
    // monitor and is carried in the sample snapshot.
    `uvm_analysis_imp_decl(_sample)
    `uvm_analysis_imp_decl(_result)

    // Keep this package as the single compile entry point.  The implementation
    // is grouped by verification role and included in dependency order.
    `include "nn_core_items.sv"
    `include "nn_core_driver.sv"
    `include "nn_core_monitors.sv"
    `include "nn_core_checking.sv"
    `include "nn_core_sequences_tests.sv"

endpackage
