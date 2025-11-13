module multiplierBlock #(
    parameter int WIDTH = 16
)(
    input  logic        clk,
    input  logic        rst_n,
    input  logic signed [WIDTH-1:0] a,
    input  logic signed [WIDTH-1:0] b,
    output logic signed [WIDTH-1:0] aOut,
    output logic signed [WIDTH-1:0] bOut,
    output logic signed [2*WIDTH-1:0] partialSum
);

always_ff @(posedge clk) begin
    if (!rst_n) begin
        partialSum <= 0;
        aOut   <= 0;
        bOut   <= 0;
    end else begin
        partialSum <= partialSum + a * b;
        aOut   <= a;
        bOut   <= b; 
    end   
end

endmodule