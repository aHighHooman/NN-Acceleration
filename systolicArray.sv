module systolicArray #(
    parameter int WIDTH = 16,
    parameter int N = 3
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic signed [WIDTH-1:0] row[N]   ,
    input  logic signed [WIDTH-1:0] col[N]   ,
    output logic signed [2*WIDTH-1:0] resultMatrix [N][N]
);

logic signed [WIDTH-1:0] horizontalData [N][N+1];
logic signed [WIDTH-1:0] verticalData   [N+1][N];

genvar i, j;
generate
    for (i = 0; i < N; i = i + 1) begin : row_loop
        for (j = 0; j < N; j = j + 1) begin : col_loop
            
            // Initialize the first row and first column of the systolic array (this is to feed in the data)
            if (i==0) begin
                assign verticalData[i][j] = col[j];
            end
            if (j==0) begin
                assign horizontalData[i][j] = row[i];
            end
            
            // Instantiate multiplier blocks and their interconnections
            multiplierBlock #(
                .WIDTH(WIDTH)
            ) mb (
                .clk   (clk),
                .rst_n (rst_n),
                .a     (horizontalData[i][j]),
                .b     (verticalData[i][j]),
                .aOut  (horizontalData[i][j+1]),
                .bOut  (verticalData[i+1][j]),
                .partialSum(resultMatrix[i][j])
            );

        end
    end
endgenerate

endmodule