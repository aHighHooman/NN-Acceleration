module multiplierBlockWeightStationary #(
    parameter int WIDTH = 16,
    parameter int RESULT_WIDTH = 2*WIDTH
)(
    input  logic                        clk,
    input  logic                        rst_n,
    input  logic                        advance,
    input  logic                        loadWeight,
    input  logic signed [WIDTH-1:0]     leftIn,
    input  logic                        leftValid,
    input  logic signed [RESULT_WIDTH-1:0] topIn,
    input  logic                        topValid,
    output logic signed [WIDTH-1:0]     rightOut,
    output logic                        rightValid,
    output logic signed [RESULT_WIDTH-1:0] bottomOut,
    output logic                        bottomValid
);

    logic signed [WIDTH-1:0]        weightReg;
    logic signed [2*WIDTH-1:0]      product;
    logic signed [RESULT_WIDTH-1:0] extendedProduct;

    assign product          = leftIn * weightReg;
    assign extendedProduct  = {{ (RESULT_WIDTH-2*WIDTH){product[2*WIDTH-1]} } , product};

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            weightReg   <= 0;
            rightOut    <= 0;
            rightValid  <= 0;
            bottomOut   <= 0;
            bottomValid <= 0;
        end else if (advance) begin
            rightOut    <= leftIn;
            rightValid  <= !loadWeight && leftValid;

            if (loadWeight) begin
                weightReg   <= topIn[WIDTH-1:0];
                bottomOut   <= topIn;
                bottomValid <= 0;
            end else begin
                bottomOut   <= topIn + extendedProduct;
                bottomValid <= topValid && leftValid;
            end
        end
    end

endmodule
