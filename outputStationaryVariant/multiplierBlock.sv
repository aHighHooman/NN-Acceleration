module multiplierBlock #(
    parameter int WIDTH = 16
)(
    input  logic        clk,
    input  logic        rst_n,
    input  logic signed [WIDTH-1:0] leftIn,
    input  logic signed [WIDTH-1:0] topIn,
    output logic signed [WIDTH-1:0] rightOut,
    output logic signed [WIDTH-1:0] bottomOut,
    output logic signed [2*WIDTH-1:0] partialSum
);

always_ff @(posedge clk) begin
    if (!rst_n) begin
        partialSum <= 0;
        rightOut   <= 0;
        bottomOut   <= 0;
    end else begin
        partialSum <= partialSum + leftIn * topIn;
        rightOut   <= leftIn;
        bottomOut   <= topIn; 
    end   
end

endmodule