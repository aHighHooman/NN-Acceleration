module multiplierBlockWeightStationary #(
    parameter int WIDTH = 16
)(
    input  logic        clk,
    input  logic        rst_n,
    input  logic        writeEnable,
    input  logic signed [WIDTH-1:0] leftIn,
    input  logic signed [2*WIDTH-1:0] topIn,
    output logic signed [WIDTH-1:0] rightOut,
    output logic signed [2*WIDTH-1:0] bottomOut
);

logic signed [WIDTH-1:0] weightReg;

always_ff @(posedge clk) begin
    if (!rst_n) begin
        weightReg  <= 0;
        rightOut   <= 0;
        bottomOut  <= 0;
    end else begin
        if (writeEnable) begin
            weightReg <= topIn[WIDTH-1:0];
        end else begin
            weightReg <= weightReg;
        end 

        rightOut   <= leftIn;
        bottomOut  <= topIn + leftIn * weightReg; 
    end   
end

endmodule