// A PE at (i,j) holds W[i][j] for the forward pass and W[j][i] for the
// backward pass, so both directions use the same activation and sum routes.
module weightStationaryProcessingElement #(
    parameter int INPUT_WIDTH = 8,
    parameter int WIDTH = 8,
    parameter int RESULT_WIDTH = INPUT_WIDTH+WIDTH+3
)(
    input logic clk, rst_n, advance, backward,
    input logic cfg_we, cfg_transposed_we,
    input logic signed [WIDTH-1:0] cfg_data, cfg_transposed_data,
    input logic update_valid,
    input logic signed [1:0] update_direction, transposed_update_direction,
    input logic signed [INPUT_WIDTH-1:0] activation_in,
    input logic activation_valid,
    input logic signed [RESULT_WIDTH-1:0] sum_in,
    output logic signed [INPUT_WIDTH-1:0] activation_out,
    output logic activation_out_valid,
    output logic signed [RESULT_WIDTH-1:0] sum_out,
    output logic signed [WIDTH-1:0] weight
);
    localparam int PW = INPUT_WIDTH+WIDTH;
    localparam logic signed [WIDTH-1:0] MIN_WEIGHT = {1'b1, {(WIDTH-1){1'b0}}};
    localparam logic signed [WIDTH-1:0] MAX_WEIGHT = {1'b0, {(WIDTH-1){1'b1}}};
    localparam logic signed [WIDTH-1:0] ONE = {{(WIDTH-1){1'b0}}, 1'b1};
    logic signed [WIDTH-1:0] transposed_weight;
    logic signed [PW-1:0] product;
    logic signed [RESULT_WIDTH-1:0] extended_product;
    initial begin
        if (INPUT_WIDTH<1 || WIDTH<1 || RESULT_WIDTH<PW)
            $fatal(1, "Invalid Fountain PE widths");
    end
    // The select is static during a pass, so the weight mux stays off the
    // activation path.
    assign product = activation_in * (backward ? transposed_weight : weight);
    assign extended_product = {{(RESULT_WIDTH-PW){product[PW-1]}}, product};

    // A saturating one-LSB step. Both copies of a weight take the same step, so
    // they stay equal.
    function automatic logic signed [WIDTH-1:0] stepped(input logic signed [WIDTH-1:0] value,
                                                        input logic signed [1:0] direction);
        if (direction == 2'sd1 && value != MAX_WEIGHT) return value + ONE;
        if (direction == -2'sd1 && value != MIN_WEIGHT) return value - ONE;
        return value;
    endfunction

    // Configuration is independent of compute advance and is legal only while
    // the owning layer is drained. Training updates use an idle barrier too.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            weight <= '0;
            transposed_weight <= '0;
        end else begin
            if (cfg_we) weight <= cfg_data;
            else if (advance && update_valid) weight <= stepped(weight, update_direction);
            if (cfg_transposed_we) transposed_weight <= cfg_transposed_data;
            else if (advance && update_valid) transposed_weight <= stepped(transposed_weight, transposed_update_direction);
        end
    end
    // The fixed lane front delivers an incoming partial sum on the same step
    // as its activation, so one valid bit marks both registered outputs.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            activation_out <= '0;
            activation_out_valid <= 0;
            sum_out <= '0;
        end else if (advance) begin
            activation_out <= activation_in;
            activation_out_valid <= activation_valid;
            sum_out <= sum_in + extended_product;
        end
    end
endmodule
