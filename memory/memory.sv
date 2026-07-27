module memory #(
    parameter int WIDTH = 16,
    parameter int DEPTH = 256,
    parameter bit ASYNC_READ = 1'b0
)(
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic [$clog2(DEPTH)-1:0]     readAddr,
    input  logic [$clog2(DEPTH)-1:0]     writeAddr,
    input  logic                         writeEnable,
    input  logic signed [WIDTH-1:0]      inputData,
    output logic signed [WIDTH-1:0]      outputData
);

    logic signed [WIDTH-1:0] mem [0:DEPTH-1];

    always_ff @(posedge clk) begin
        if (writeEnable) mem[writeAddr] <= inputData;
    end

    generate
        if (ASYNC_READ) begin : asynchronous_read
            assign outputData = mem[readAddr];
        end else begin : synchronous_read
            always_ff @(posedge clk) outputData <= mem[readAddr];
        end
    endgenerate

endmodule
