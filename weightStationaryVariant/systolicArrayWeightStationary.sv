module systolicArrayWeightStationary #(
    parameter int WIDTH = 16,
    parameter int N = 3
)(
    input  logic                        clk,
    input  logic                        rst_n,
    input  logic                        advance,
    input  logic                        loadWeight,
    input  logic signed [WIDTH-1:0]     row [N],
    input  logic                        rowValid [N],
    input  logic                        rowLast [N],
    input  logic signed [WIDTH-1:0]     col [N],
    output logic signed [2*WIDTH-1:0]   result [N],
    output logic                        resultValid [N],
    output logic                        resultLast [N],
    output logic                        pipelineBusy
);

    logic signed [WIDTH-1:0]   horizontalData [N][N+1];
    logic                      horizontalValid[N][N+1];
    logic                      horizontalLast [N][N+1];
    logic signed [2*WIDTH-1:0] verticalData   [N+1][N];
    logic                      verticalValid  [N+1][N];
    logic                      verticalLast   [N+1][N];

    genvar i, j;
    generate
        for (i = 0; i < N; i++) begin : boundary_rows
            assign horizontalData[i][0]  = row[i];
            assign horizontalValid[i][0] = rowValid[i];
            assign horizontalLast[i][0]  = rowLast[i];
        end
        for (j = 0; j < N; j++) begin : boundary_cols
            assign verticalData[0][j]  = loadWeight ? {{WIDTH{col[j][WIDTH-1]}}, col[j]} : '0;
            assign verticalValid[0][j] = loadWeight ? 1'b0 : 1'b1;
            assign verticalLast[0][j]  = 1'b0;
        end

        for (i = 0; i < N; i++) begin : row_loop
            for (j = 0; j < N; j++) begin : col_loop
                multiplierBlockWeightStationary #(.WIDTH(WIDTH)) mb (
                    .clk(clk), .rst_n(rst_n), .advance(advance), .loadWeight(loadWeight),
                    .leftIn(horizontalData[i][j]), .leftValid(horizontalValid[i][j]),
                    .leftLast(horizontalLast[i][j]), .topIn(verticalData[i][j]),
                    .topValid(verticalValid[i][j]), .rightOut(horizontalData[i][j+1]),
                    .rightValid(horizontalValid[i][j+1]), .rightLast(horizontalLast[i][j+1]),
                    .bottomOut(verticalData[i+1][j]), .bottomValid(verticalValid[i+1][j]),
                    .bottomLast(verticalLast[i+1][j])
                );
            end
        end
        for (j = 0; j < N; j++) begin : result_loop
            assign result[j]      = verticalData[N][j];
            assign resultValid[j] = verticalValid[N][j];
            assign resultLast[j]  = verticalLast[N][j];
        end
    endgenerate

    always_comb begin
        pipelineBusy = 1'b0;
        for (int r = 0; r <= N; r++)
            for (int c = 0; c < N; c++)
                pipelineBusy |= verticalValid[r][c] && (r != 0);
        for (int r = 0; r < N; r++)
            for (int c = 1; c <= N; c++)
                pipelineBusy |= horizontalValid[r][c];
    end

endmodule
