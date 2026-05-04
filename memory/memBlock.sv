module memBlock #(
    parameter int WIDTH = 16,
    parameter int DEPTH = 256,
    parameter int N = 3
)(
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic        [7:0]       addr    [N],
    output logic signed [WIDTH-1:0] dataOut [N]
);

genvar i;
generate
    for (i = 0; i < N; i = i + 1) begin : memArray
        memory #(
            .WIDTH(WIDTH),
            .DEPTH(DEPTH)
        ) memInst (
            .clk        (clk),
            .rst_n      (rst_n),
            .addr       (addr[i]),
            .outputData (dataOut[i])
        );
    end
endgenerate


endmodule