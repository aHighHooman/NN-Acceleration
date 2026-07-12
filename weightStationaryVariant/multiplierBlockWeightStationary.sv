module multiplierBlockWeightStationary #(
    parameter int WIDTH = 16
)(
    input  logic                        clk,
    input  logic                        rst_n,
    input  logic                        advance,
    input  logic                        loadWeight,
    input  logic signed [WIDTH-1:0]     leftIn,
    input  logic                        leftValid,
    input  logic                        leftLast,
    input  logic signed [2*WIDTH-1:0]   topIn,
    input  logic                        topValid,
    output logic signed [WIDTH-1:0]     rightOut,
    output logic                        rightValid,
    output logic                        rightLast,
    output logic signed [2*WIDTH-1:0]   bottomOut,
    output logic                        bottomValid,
    output logic                        bottomLast
);

    logic signed [WIDTH-1:0] weightReg;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            weightReg   <= '0;
            rightOut    <= '0;
            rightValid  <= 1'b0;
            rightLast   <= 1'b0;
            bottomOut   <= '0;
            bottomValid <= 1'b0;
            bottomLast  <= 1'b0;
        end else if (advance) begin
            rightOut   <= leftIn;
            rightValid <= loadWeight ? 1'b0 : leftValid;
            rightLast  <= loadWeight ? 1'b0 : leftLast;

            if (loadWeight) begin
                weightReg   <= topIn[WIDTH-1:0];
                bottomOut   <= topIn;
                bottomValid <= 1'b0;
                bottomLast  <= 1'b0;
            end else begin
                bottomOut   <= topIn + leftIn * weightReg;
                bottomValid <= topValid && leftValid;
                bottomLast  <= leftLast;
            end
        end
    end

endmodule
