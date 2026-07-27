module SPI_Slave_Output_Module #(
    parameter int WIDTH = 32
)(
    input  logic rst_n,

    input  logic signed [WIDTH-1:0] data_in,

    input  logic cs_n,
    input  logic sclk,

    output logic miso,
    output logic miso_valid,

    output logic ready
);

    localparam int COUNT_WIDTH = $clog2(WIDTH + 1);
    
    logic [WIDTH-1:0]       miso_buf;
    logic [COUNT_WIDTH-1:0] bit_count;
    logic                   inputNeeded;

    assign inputNeeded = (bit_count >= WIDTH);
    assign miso_valid = (~cs_n && bit_count < WIDTH);
    assign ready = inputNeeded;

    // multiplex my beloved
    always_comb begin 
        if (~cs_n && bit_count < WIDTH)
            miso = miso_buf[WIDTH - 1 - bit_count];
        else
            miso = 0;
    end

    always_ff @(posedge sclk) begin
        if (!rst_n) begin
            miso_buf <= 0;
            bit_count <= WIDTH;
        end
        else if (inputNeeded) begin
            miso_buf <= data_in;
            bit_count <= 0;
        end
        else if (~cs_n && bit_count < WIDTH) begin
            bit_count <= bit_count + 1;
        end
    end

endmodule
