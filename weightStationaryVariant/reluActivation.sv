module reluActivation #(
    parameter int WIDTH = 16
)(
    input  logic signed [WIDTH-1:0] inputData,
    output logic signed [WIDTH-1:0] outputData
);

    localparam logic signed [WIDTH-1:0] ZERO = '0;

    assign outputData = (inputData > ZERO) ? inputData : ZERO;

endmodule
