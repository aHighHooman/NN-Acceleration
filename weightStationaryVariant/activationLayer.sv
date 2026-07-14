module activationLayer #(
    parameter int WIDTH = 16,
    parameter int N = 3
)(
    input  logic signed [WIDTH-1:0] inputData [N],
    input  logic                    passThrough,
    output logic signed [WIDTH-1:0] outputData [N]
);

    logic signed [WIDTH-1:0] reluData[N];

    genvar lane;
    generate
        for (lane = 0; lane < N; lane = lane + 1) begin : activation_lanes
            reluActivation #(.WIDTH(WIDTH)) relu (
                .inputData(inputData[lane]),
                .outputData(reluData[lane])
            );

            assign outputData[lane] = passThrough ? inputData[lane] : reluData[lane];
        end
    endgenerate

endmodule
