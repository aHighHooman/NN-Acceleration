module multiplierBlockWeightStationary #(
    parameter int WIDTH = 16,
    parameter int RESULT_WIDTH = 2*WIDTH
)(
    input  logic                        clk,
    input  logic                        rst_n,
    input  logic                        advance,
    input  logic                        loadWeight,
    input  logic                        updateWeight,
    input  logic signed [1:0]           updateDirection,
    input  logic signed [WIDTH-1:0]     leftIn,
    input  logic                        leftValid,
    input  logic signed [RESULT_WIDTH-1:0] topIn,
    input  logic                        topValid,
    output logic signed [WIDTH-1:0]     rightOut,
    output logic                        rightValid,
    output logic signed [RESULT_WIDTH-1:0] bottomOut,
    output logic                        bottomValid
);

    localparam logic signed [WIDTH-1:0]
        WEIGHT_MIN = {1'b1, {(WIDTH-1){1'b0}}};
    localparam logic signed [WIDTH-1:0]
        WEIGHT_MAX = {1'b0, {(WIDTH-1){1'b1}}};
    localparam logic signed [WIDTH-1:0]
        WEIGHT_ONE = {{(WIDTH-1){1'b0}}, 1'b1};

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

                case (updateDirection)
                    2'sd1: begin
                        if (updateWeight && (weightReg != WEIGHT_MAX))
                            weightReg <= weightReg + WEIGHT_ONE;
                    end
                    -2'sd1: begin
                        if (updateWeight && (weightReg != WEIGHT_MIN))
                            weightReg <= weightReg - WEIGHT_ONE;
                    end
                    default: weightReg <= weightReg;
                endcase
            end
        end
    end

endmodule
