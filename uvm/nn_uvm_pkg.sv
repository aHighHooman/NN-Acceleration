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

    // Keep this package as the single compile entry point.  The implementation
    // is grouped by verification role and included in dependency order.
    `include "nn_core_items.sv"
    `include "nn_core_driver.sv"
    `include "nn_core_monitors.sv"
    `include "nn_core_checking.sv"
    `include "nn_core_sequences_tests.sv"

endpackage
