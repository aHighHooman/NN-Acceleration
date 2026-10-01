module weightStationaryProcessingElement #(
    parameter int WIDTH = 16,
    parameter int RESULT_WIDTH = 2*WIDTH,
    // 0: top-left, 1: anti-diagonal, 2: bottom-right.
    parameter int ROLE = 0
)(
    input  logic                            clk,
    input  logic                            rst_n,
    input  logic                            advance,
    input  logic                            loadWeight,
    input  logic                            captureWeight,
    input  logic                            updateWeight,
    input  logic signed [1:0]               updateDirection,
    input  logic signed [WIDTH-1:0]         activationIn,
    input  logic signed [RESULT_WIDTH-1:0]  psumIn,
    input  logic signed [RESULT_WIDTH-1:0]  lowerPsumIn,
    output logic signed [WIDTH-1:0]         activationOut,
    output logic signed [RESULT_WIDTH-1:0]  psumOut
);

    localparam int ROLE_AD = 1;
    localparam logic signed [WIDTH-1:0]
        WEIGHT_MIN = {1'b1, {(WIDTH-1){1'b0}}};
    localparam logic signed [WIDTH-1:0]
        WEIGHT_MAX = {1'b0, {(WIDTH-1){1'b1}}};
    localparam logic signed [WIDTH-1:0]
        WEIGHT_ONE = {{(WIDTH-1){1'b0}}, 1'b1};

    initial begin
        if (WIDTH < 1 || RESULT_WIDTH < 2*WIDTH || ROLE < 0 || ROLE > 2)
            $fatal(1, "WIDTH>=1, RESULT_WIDTH>=2*WIDTH, and ROLE in {0,1,2}");
    end

    logic signed [WIDTH-1:0]         weightReg;
    logic signed [2*WIDTH-1:0]       product;
    logic signed [RESULT_WIDTH-1:0]  extendedProduct;
    logic signed [RESULT_WIDTH-1:0]  nextPsum;

    // The array injects zero activations during loading, allowing the existing
    // MAC to shift weight values without a load mux in each PE's datapath.
    assign product = activationIn * weightReg;
    assign extendedProduct = {{(RESULT_WIDTH-2*WIDTH){product[2*WIDTH-1]}}, product};

    generate
        if (ROLE == ROLE_AD) begin : antidiagonal_sum
            logic signed [RESULT_WIDTH-1:0] lowerOperand;
            logic [RESULT_WIDTH-1:0] carrySaveSum;
            logic [RESULT_WIDTH-1:0] carrySaveCarry;

            // The AD belongs to the downward load chain. Weight data from the
            // upward chain must not be added to its loading value.
            assign lowerOperand = (loadWeight || captureWeight) ? '0 : lowerPsumIn;
            // Compress all three operands, then use a single carry-propagate
            // addition. The PE result register is the sole output register.
            assign carrySaveSum = psumIn ^ lowerOperand ^ extendedProduct;
            assign carrySaveCarry = ((psumIn & lowerOperand) |
                                     (psumIn & extendedProduct) |
                                     (lowerOperand & extendedProduct)) << 1;
            assign nextPsum = $signed(carrySaveSum) + $signed(carrySaveCarry);
        end else begin : directional_sum
            assign nextPsum = psumIn + extendedProduct;
        end
    endgenerate

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            weightReg <= '0;
            activationOut <= '0;
            psumOut <= '0;
        end else if (advance) begin
            // AD is a sink; its unused activation output is tied to zero.
            activationOut <= (ROLE == ROLE_AD) ? '0 : activationIn;
            psumOut <= nextPsum;

            // Capture the pre-edge psum register after all N shift advances.
            // A same-edge compute/update always multiplies with the old W.
            if (captureWeight) begin
                weightReg <= psumOut[WIDTH-1:0];
            end else if (!loadWeight && updateWeight) begin
                case (updateDirection)
                    2'sd1: begin
                        if (weightReg != WEIGHT_MAX)
                            weightReg <= weightReg + WEIGHT_ONE;
                    end
                    -2'sd1: begin
                        if (weightReg != WEIGHT_MIN)
                            weightReg <= weightReg - WEIGHT_ONE;
                    end
                    default: weightReg <= weightReg;
                endcase
            end
        end
    end

endmodule
