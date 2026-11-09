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
    logic signed [WIDTH-1:0] transposed_weight;
    // The multiplier's weight. It follows backward and the weights one edge
    // late; the controller never computes on the edge after either changes.
    logic signed [WIDTH-1:0] selected_weight;
    logic signed [PW-1:0] product;
    initial begin
        if (INPUT_WIDTH<1 || WIDTH<1 || RESULT_WIDTH<PW)
            $fatal(1, "Invalid Fountain PE widths");
    end
    // Registering the select keeps the weight mux and its fanout out of the
    // multiply's cycle.
    always_ff @(posedge clk) begin
        if (!rst_n) selected_weight <= '0;
        else selected_weight <= backward ? transposed_weight : weight;
    end
    assign product = activation_in * selected_weight;

    // A saturating one-LSB step. A weight already at the limit it is moving
    // toward holds. Both copies of a weight take the same step, so they stay equal.
    function automatic logic signed [WIDTH-1:0] stepped(input logic signed [WIDTH-1:0] value,
                                                        input logic signed [1:0] direction);
        return (value == ((direction > 0) ? MAX_WEIGHT : MIN_WEIGHT)) ? value : value + direction;
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
            sum_out <= sum_in + product;
        end
    end
    // synthesis translate_off
    always_ff @(posedge clk) begin
        if (rst_n && advance && activation_valid)
            assert (selected_weight == (backward ? transposed_weight : weight))
                else $fatal(1, "PE computed on a stale weight");
    end
    // synthesis translate_on
endmodule
