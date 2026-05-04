module memory #(
    parameter int WIDTH = 16,
    parameter int DEPTH = 256
)(
    input  logic                            clk,
    input  logic                            rst_n,
    input  logic        [7:0]               addr,
    output logic signed [WIDTH-1:0]         outputData
);

logic signed [WIDTH-1:0] mem [0:DEPTH-1];

always_ff @(posedge clk) begin
    outputData <= mem[addr];    
end

endmodule