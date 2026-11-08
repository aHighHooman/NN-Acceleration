// Center-out PE network with combinational joins of registered boundary sums.
// Forward it computes in_data * W. Backward each PE uses its copy of the
// transposed weight, so the same routes compute in_data * W' = W * in_data.
module weightStationarySystolicArray #(
    parameter int N = 5,
    parameter int INPUT_WIDTH = 8,
    parameter int WIDTH = 8,
    parameter int RESULT_WIDTH = INPUT_WIDTH+WIDTH+$clog2(N)
)(
    input logic clk, rst_n, advance, backward,
    input logic cfg_we,
    input logic [((N>1)?$clog2(N):1)-1:0] cfg_row,
    input logic signed [WIDTH-1:0] cfg_data[N],
    input logic update_valid,
    input logic signed [1:0] row_direction[N], column_direction[N],
    input logic signed [INPUT_WIDTH-1:0] in_data[N],
    input logic in_valid[N],
    output logic signed [RESULT_WIDTH-1:0] result[N],
    output logic resultValid[N],
    output logic signed [WIDTH-1:0] weights[N][N],
    output logic loaded, busy
);
    localparam int C = (N-1)/2;
    localparam int INDEX_WIDTH = (N>1)?$clog2(N):1;
    logic [N-1:0] loaded_rows;
    logic accepted_cfg;
    logic signed [INPUT_WIDTH-1:0] activations[N][N], activation_in[N][N];
    logic activation_valids[N][N], activation_valid_in[N][N];
    logic signed [RESULT_WIDTH-1:0] sums[N][N], sum_in[N][N];
    initial begin
        if (N<3 || N%2==0 || RESULT_WIDTH<INPUT_WIDTH+WIDTH+$clog2(N))
            $fatal(1, "Fountain requires odd N>=3 and a full-width accumulator");
    end
    // Ternary signs need comparisons, not a DSP multiplier per PE.
    function automatic logic signed [1:0] combined(input logic signed [1:0] row, column);
        if (row==0 || column==0) return 0;
        return (row==column) ? 2'sd1 : -2'sd1;
    endfunction
    assign loaded = &loaded_rows;
    assign accepted_cfg = cfg_we && !busy && cfg_row<N;
    always_ff @(posedge clk) begin
        if (!rst_n) loaded_rows <= '0;
        else if (accepted_cfg) begin
            if (loaded) loaded_rows <= '0;
            loaded_rows[cfg_row] <= 1;
        end
        // synthesis translate_off
        if (rst_n) begin
            if (cfg_we) assert (!busy && cfg_row<N)
                else $fatal(1, "Array configuration requires an idle valid row");
            if (update_valid) assert (!busy && loaded)
                else $fatal(1, "Array learning update requires an idle loaded array");
            if (advance) for (int lane=0; lane<N; lane++)
                if (in_valid[lane]) assert (loaded && !cfg_we)
                    else $fatal(1, "Array input before weight commit");
        end
        // synthesis translate_on
    end
    // The vector leaves column C along the rows, and sums start at rows C-1 and
    // C and run along the columns. Only forward vectors carry valid bits, so a
    // backward pass never reads as busy.
    always_comb begin
        for (int i=0; i<N; i++) for (int j=0; j<N; j++) begin
            activation_in[i][j] = (j==C) ? in_data[i] : activations[i][(j<C)?j+1:j-1];
            activation_valid_in[i][j] = (j==C) ? in_valid[i] : activation_valids[i][(j<C)?j+1:j-1];
            sum_in[i][j] = (i==C || i==C-1) ? '0 : sums[(i<C)?i+1:i-1][j];
        end
    end
    // Writing row r loads the forward weights of row r and the transposed weights
    // of column r. The transposed copy of W[j][i] takes W[j][i]'s update.
    genvar i,j;
    generate
    for (i=0; i<N; i++) begin : rows
        for (j=0; j<N; j++) begin : columns
            weightStationaryProcessingElement #(.INPUT_WIDTH(INPUT_WIDTH), .WIDTH(WIDTH),
                          .RESULT_WIDTH(RESULT_WIDTH)) pe (
                .clk, .rst_n, .advance, .backward,
                .cfg_we(accepted_cfg && cfg_row==INDEX_WIDTH'(i)), .cfg_data(cfg_data[j]),
                .cfg_transposed_we(accepted_cfg && cfg_row==INDEX_WIDTH'(j)), .cfg_transposed_data(cfg_data[i]),
                .update_valid(update_valid && !busy && loaded),
                .update_direction(combined(row_direction[i], column_direction[j])),
                .transposed_update_direction(combined(row_direction[j], column_direction[i])),
                .activation_in(activation_in[i][j]), .activation_valid(activation_valid_in[i][j]),
                .sum_in(sum_in[i][j]),
                .activation_out(activations[i][j]),
                .activation_out_valid(activation_valids[i][j]),
                .sum_out(sums[i][j]), .weight(weights[i][j])
            );
            // synthesis translate_off
            always_ff @(posedge clk) begin
                if (rst_n && loaded) assert (pe.transposed_weight==weights[j][i])
                    else $fatal(1, "Transposed weight %0d,%0d differs from W[%0d][%0d]", i, j, j, i);
            end
            // synthesis translate_on
        end
    end
    endgenerate
    always_comb begin
        busy = 0;
        for (int i=0; i<N; i++)
            for (int j=0; j<N; j++) busy |= activation_valids[i][j];
    end
    generate
    for (j=0; j<N; j++) begin : columns
        assign result[j] = sums[0][j] + sums[N-1][j];
        assign resultValid[j] = activation_valids[0][j];
        // The join and activation settle between boundary capture and the
        // next layer's PE capture; there is no separate join register.
        // synthesis translate_off
        always_ff @(posedge clk) begin
            if (rst_n && advance)
                assert (activation_valids[0][j]==activation_valids[N-1][j])
                    else $fatal(1, "Fountain join valid mismatch");
        end
        // synthesis translate_on
    end
    endgenerate
endmodule

// A lane delay measured in enabled steps, used by the source skew. DEPTH=0 is a
// real bypass.
module weightStationaryLaneDelay #(
    parameter int WIDTH = 8,
    parameter int DEPTH = 0
)(
    input logic clk, rst_n, advance,
    input logic signed [WIDTH-1:0] in_data,
    input logic in_valid,
    output logic signed [WIDTH-1:0] out_data,
    output logic out_valid,
    output logic busy
);
    initial begin
        if (WIDTH < 1 || DEPTH < 0)
            $fatal(1, "Fountain delay requires a positive width and DEPTH>=0");
    end
    generate
        if (DEPTH <= 0) begin : bypass
            assign out_data = in_data;
            assign out_valid = in_valid;
            assign busy = 1'b0;
        end else begin : delayed
            logic signed [WIDTH-1:0] data_pipe[DEPTH];
            logic [DEPTH-1:0] valid_pipe;
            assign out_data = data_pipe[DEPTH-1];
            assign out_valid = valid_pipe[DEPTH-1];
            assign busy = |valid_pipe;
            always_ff @(posedge clk) begin
                if (!rst_n) begin
                    valid_pipe <= '0;
                    for (int s=0; s<DEPTH; s++) data_pipe[s] <= '0;
                end else if (advance) begin
                    data_pipe[0] <= in_data;
                    valid_pipe[0] <= in_valid;
                    for (int s=1; s<DEPTH; s++) begin
                        data_pipe[s] <= data_pipe[s-1];
                        valid_pipe[s] <= valid_pipe[s-1];
                    end
                end
            end
        end
    endgenerate
endmodule
