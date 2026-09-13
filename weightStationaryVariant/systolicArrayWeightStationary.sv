module systolicArrayWeightStationary #(
    parameter int WIDTH = 16,
    parameter int N = 3
)(
    input  logic                        clk,
    input  logic                        rst_n,
    input  logic                        advance,
    input  logic                        loadWeight,
    input  logic signed [1:0]           rowDirection [N],
    input  logic signed [1:0]           columnDirection [N],
    input  logic                        updateValid,
    input  logic signed [WIDTH-1:0]     row [N],
    input  logic                        rowValid [N],
    input  logic signed [WIDTH-1:0]     col [N],
    output logic signed [2*WIDTH+$clog2(N)-1:0] result [N],
    output logic                        resultValid [N],
    output logic                        pipelineBusy,
    output logic                        updateComplete
);

    localparam int FINAL_RESULT_WIDTH = 2*WIDTH + $clog2(N);
    localparam int UPDATE_STAGES = 2*N - 1;
    localparam int UPDATE_PIPE_STAGES = UPDATE_STAGES - 1;
    localparam int UPDATE_PIPE_STORAGE = (UPDATE_PIPE_STAGES > 0) ?
                                         UPDATE_PIPE_STAGES : 1;

    logic signed [WIDTH-1:0]                horizontalData [N][N+1];
    logic                                   horizontalValid[N][N+1];
    logic signed [FINAL_RESULT_WIDTH-1:0]   verticalData   [N+1][N];
    logic                                   verticalValid  [N+1][N];
    logic signed [1:0]                      updateRowPipe   [UPDATE_PIPE_STORAGE][N];
    logic signed [1:0]                      updateColumnPipe[UPDATE_PIPE_STORAGE][N];
    logic                                   updateValidPipe [UPDATE_PIPE_STORAGE];

    // Diagonal zero consumes the live package on its acceptance edge.  The
    // remaining 2N-2 diagonals consume the registered package on successive
    // advancing edges.  Completion therefore coincides with the final PE
    // update rather than an otherwise unused extra pipeline stage.
    generate
        if (N == 1) begin : single_pe_completion
            assign updateComplete = advance && updateValid;
        end else begin : wave_completion
            assign updateComplete = advance &&
                                    updateValidPipe[UPDATE_PIPE_STAGES-1];
        end
    endgenerate

    // Update packages advance with exactly the same enable as the data array.
    // Keeping both complete vectors in every stage permits one new package on
    // every advancing cycle while each stage addresses one anti-diagonal.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int stage = 0; stage < UPDATE_PIPE_STORAGE; stage++) begin
                updateValidPipe[stage] <= 1'b0;
                for (int lane = 0; lane < N; lane++) begin
                    updateRowPipe[stage][lane] <= 2'sd0;
                    updateColumnPipe[stage][lane] <= 2'sd0;
                end
            end
        end else if (advance) begin
            for (int stage = UPDATE_PIPE_STAGES-1; stage > 0; stage--) begin
                updateValidPipe[stage] <= updateValidPipe[stage-1];
                for (int lane = 0; lane < N; lane++) begin
                    updateRowPipe[stage][lane] <= updateRowPipe[stage-1][lane];
                    updateColumnPipe[stage][lane] <= updateColumnPipe[stage-1][lane];
                end
            end

            updateValidPipe[0] <= updateValid;
            for (int lane = 0; lane < N; lane++) begin
                updateRowPipe[0][lane] <= rowDirection[lane];
                updateColumnPipe[0][lane] <= columnDirection[lane];
            end
        end
    end

    genvar i, j;
    generate
        for (i = 0; i < N; i++) begin : boundary_rows
            assign horizontalData[i][0]  = row[i];
            assign horizontalValid[i][0] = rowValid[i];
        end

        for (j = 0; j < N; j++) begin : boundary_cols
            assign verticalData[0][j]  = loadWeight ?
                {{(FINAL_RESULT_WIDTH-WIDTH){col[j][WIDTH-1]}}, col[j]} : '0;
            assign verticalValid[0][j] = !loadWeight;
        end

        for (i = 0; i < N; i++) begin : row_loop
            for (j = 0; j < N; j++) begin : col_loop
                logic              localUpdateValid;
                logic signed [1:0] localRowDirection;
                logic signed [1:0] localColumnDirection;
                logic signed [1:0] localUpdateDirection;

                if ((i+j) == 0) begin : live_update_entry
                    assign localUpdateValid = updateValid;
                    assign localRowDirection = rowDirection[i];
                    assign localColumnDirection = columnDirection[j];
                end else begin : piped_update_wave
                    assign localUpdateValid = updateValidPipe[i+j-1];
                    assign localRowDirection = updateRowPipe[i+j-1][i];
                    assign localColumnDirection = updateColumnPipe[i+j-1][j];
                end

                always_comb begin
                    if ((localRowDirection == 2'sd0) ||
                        (localColumnDirection == 2'sd0))
                        localUpdateDirection = 2'sd0;
                    else if (localRowDirection == localColumnDirection)
                        localUpdateDirection = 2'sd1;
                    else
                        localUpdateDirection = -2'sd1;
                end

                multiplierBlockWeightStationary #(.WIDTH(WIDTH), .RESULT_WIDTH(FINAL_RESULT_WIDTH)) mb (
                    .clk(clk), .rst_n(rst_n), .advance(advance), .loadWeight(loadWeight),
                    .updateWeight(localUpdateValid),
                    .updateDirection(localUpdateDirection),
                    .leftIn(horizontalData[i][j]), .leftValid(horizontalValid[i][j]),
                    .topIn(verticalData[i][j]), .topValid(verticalValid[i][j]),
                    .rightOut(horizontalData[i][j+1]), .rightValid(horizontalValid[i][j+1]),
                    .bottomOut(verticalData[i+1][j]), .bottomValid(verticalValid[i+1][j])
                );
            end
        end
        for (j = 0; j < N; j++) begin : result_loop
            assign result[j]      = verticalData[N][j];
            assign resultValid[j] = verticalValid[N][j];
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
        for (int stage = 0; stage < UPDATE_PIPE_STAGES; stage++)
            pipelineBusy |= updateValidPipe[stage];
    end

endmodule
