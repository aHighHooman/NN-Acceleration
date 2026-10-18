module weightStationaryProcessingElement #(
    parameter int INPUT_WIDTH = 8,
    parameter int WIDTH = 8,
    parameter int RESULT_WIDTH = INPUT_WIDTH+WIDTH+3,
    parameter int TAG_WIDTH = 16,
    parameter bit SEED = 0
)(
    input logic clk, rst_n, advance,
    input logic cfg_we,
    input logic signed [WIDTH-1:0] cfg_data,
    input logic update_valid,
    input logic signed [1:0] update_direction,
    input logic signed [INPUT_WIDTH-1:0] activation_in,
    input logic activation_valid,
    input logic [TAG_WIDTH-1:0] activation_tag,
    input logic signed [RESULT_WIDTH-1:0] sum_in,
    input logic sum_valid,
    input logic [TAG_WIDTH-1:0] sum_tag,
    output logic signed [INPUT_WIDTH-1:0] activation_out,
    output logic activation_out_valid,
    output logic [TAG_WIDTH-1:0] activation_out_tag,
    output logic signed [RESULT_WIDTH-1:0] sum_out,
    output logic sum_out_valid,
    output logic [TAG_WIDTH-1:0] sum_out_tag,
    output logic signed [WIDTH-1:0] weight
);
    localparam int PW = INPUT_WIDTH+WIDTH;
    localparam logic signed [WIDTH-1:0] MIN_WEIGHT = {1'b1, {(WIDTH-1){1'b0}}};
    localparam logic signed [WIDTH-1:0] MAX_WEIGHT = {1'b0, {(WIDTH-1){1'b1}}};
    localparam logic signed [WIDTH-1:0] ONE = {{(WIDTH-1){1'b0}}, 1'b1};
    logic signed [PW-1:0] product;
    logic signed [RESULT_WIDTH-1:0] extended_product;
    initial begin
        if (INPUT_WIDTH<1 || WIDTH<1 || RESULT_WIDTH<PW || TAG_WIDTH<1)
            $fatal(1, "Invalid Fountain PE widths");
    end
    assign product = activation_in * weight;
    assign extended_product = {{(RESULT_WIDTH-PW){product[PW-1]}}, product};

    // Configuration is independent of compute advance and is legal only while
    // the owning layer is drained. Training updates use an idle barrier too.
    always_ff @(posedge clk) begin
        if (!rst_n) weight <= '0;
        else if (cfg_we) weight <= cfg_data;
        else if (advance && update_valid) begin
            if (update_direction == 2'sd1 && weight != MAX_WEIGHT)
                weight <= weight + ONE;
            else if (update_direction == -2'sd1 && weight != MIN_WEIGHT)
                weight <= weight - ONE;
        end
    end
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            activation_out <= '0;
            activation_out_valid <= 0;
            activation_out_tag <= '0;
            sum_out <= '0;
            sum_out_valid <= 0;
            sum_out_tag <= '0;
        end else if (advance) begin
            activation_out <= activation_in;
            activation_out_valid <= activation_valid;
            activation_out_tag <= activation_tag;
            sum_out <= (SEED ? RESULT_WIDTH'('0) : sum_in) + extended_product;
            sum_out_valid <= activation_valid &&
                (SEED || (sum_valid && activation_tag == sum_tag));
            sum_out_tag <= activation_tag;
            // synthesis translate_off
            if (!SEED) begin
                assert (activation_valid == sum_valid)
                    else $fatal(1, "Fountain MAC valid mismatch");
                if (activation_valid && sum_valid)
                    assert (activation_tag == sum_tag)
                        else $fatal(1, "Fountain MAC tag mismatch");
            end
            // synthesis translate_on
        end
    end
endmodule
