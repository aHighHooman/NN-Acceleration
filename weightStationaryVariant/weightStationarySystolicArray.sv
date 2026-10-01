module weightStationarySystolicArray #(
    parameter int WIDTH = 16,
    parameter int N = 3
)(
    input  logic                            clk,
    input  logic                            rst_n,
    input  logic                            advance,
    input  logic                            loadWeight,
    input  logic                            captureWeight,
    input  logic signed [1:0]               rowDirection [N],
    input  logic signed [1:0]               columnDirection [N],
    input  logic                            updateValid,
    input  logic signed [WIDTH-1:0]         rowLeft [N],
    input  logic signed [WIDTH-1:0]         rowRight [N],
    input  logic                            actValid,
    input  logic signed [WIDTH-1:0]         colTop [N],
    input  logic signed [WIDTH-1:0]         colBottom [N],
    output logic signed [2*WIDTH+$clog2(N)-1:0] result [N],
    output logic                            resultValid [N],
    output logic                            pipelineBusy
);

    localparam int FINAL_RESULT_WIDTH = 2*WIDTH + $clog2(N);
    localparam int ROLE_TL = 0;
    localparam int ROLE_AD = 1;
    localparam int ROLE_BR = 2;
    localparam int UPDATE_PIPE_STAGES = N-1;
    // Preserve elaboration of unsupported N=1 long enough to report the
    // parameter contract above, rather than a zero-size storage error.
    localparam int UPDATE_PIPE_STORAGE = (UPDATE_PIPE_STAGES > 0) ? UPDATE_PIPE_STAGES : 1;

    initial begin
        if (WIDTH < 1 || N < 2)
            $fatal(1, "WIDTH>=1 and N>=2");
    end

    // Indexed by producing PE, rather than by grid boundaries. TL links point
    // right/down, BR links left/up, and each AD result goes directly to its lane.
    // Load-mode zeroing is only at the activation edges. Downward weights start
    // after j padded shifts, when column j's left-flowing activations are zero;
    // upward weights start after N-j pads, after its right-flowing path clears.
    logic signed [WIDTH-1:0]               horizontalData [N][N];
    logic signed [FINAL_RESULT_WIDTH-1:0]  verticalData [N][N];
    logic [N-1:0]                         validPipe;
    logic signed [1:0]                    updateRowPipe [UPDATE_PIPE_STORAGE][N];
    logic signed [1:0]                    updateColumnPipe [UPDATE_PIPE_STORAGE][N];
    logic                                 updateValidPipe [UPDATE_PIPE_STORAGE];

    // Phase zero consumes the live update package at both corners. Delayed
    // packages follow the same inward phase as the corresponding sample MAC.
    // Bubbles advance; a stalled output freezes both compute and learning.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            validPipe <= '0;
            for (int stage = 0; stage < UPDATE_PIPE_STORAGE; stage++) begin
                updateValidPipe[stage] <= 1'b0;
                for (int lane = 0; lane < N; lane++) begin
                    updateRowPipe[stage][lane] <= 2'sd0;
                    updateColumnPipe[stage][lane] <= 2'sd0;
                end
            end
        end else if (advance) begin
            if (loadWeight || captureWeight) begin
                validPipe <= '0;
                for (int stage = 0; stage < UPDATE_PIPE_STORAGE; stage++) begin
                    updateValidPipe[stage] <= 1'b0;
                    for (int lane = 0; lane < N; lane++) begin
                        updateRowPipe[stage][lane] <= 2'sd0;
                        updateColumnPipe[stage][lane] <= 2'sd0;
                    end
                end
            end else begin
                validPipe[0] <= actValid;
                for (int stage = 1; stage < N; stage++)
                    validPipe[stage] <= validPipe[stage-1];
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
    end

    genvar i, j;
    generate
        for (i = 0; i < N; i++) begin : row_loop
            for (j = 0; j < N; j++) begin : col_loop
                localparam int ROLE = (i+j < N-1) ? ROLE_TL :
                                      (i+j == N-1) ? ROLE_AD : ROLE_BR;
                localparam int UPDATE_PHASE = (i+j <= N-1) ? i+j : 2*N-2-i-j;
                logic signed [WIDTH-1:0] localActivation;
                logic signed [FINAL_RESULT_WIDTH-1:0] localPsum;
                logic signed [FINAL_RESULT_WIDTH-1:0] localLowerPsum;
                logic localUpdateValid;
                logic signed [1:0] localRowDirection;
                logic signed [1:0] localColumnDirection;
                logic signed [1:0] localUpdateDirection;

                if (ROLE == ROLE_BR) begin : upward_path
                    if (j == N-1) begin : right_boundary
                        assign localActivation = (loadWeight || captureWeight) ? '0 : rowRight[i];
                    end else begin : right_neighbor
                        assign localActivation = horizontalData[i][j+1];
                    end
                    if (i == N-1) begin : bottom_boundary
                        assign localPsum = loadWeight ?
                            {{(FINAL_RESULT_WIDTH-WIDTH){colBottom[j][WIDTH-1]}}, colBottom[j]} : '0;
                    end else begin : bottom_neighbor
                        assign localPsum = verticalData[i+1][j];
                    end
                    assign localLowerPsum = '0;
                end else begin : downward_path
                    if (j == 0) begin : left_boundary
                        assign localActivation = (loadWeight || captureWeight) ? '0 : rowLeft[i];
                    end else begin : left_neighbor
                        assign localActivation = horizontalData[i][j-1];
                    end
                    if (i == 0) begin : top_boundary
                        assign localPsum = loadWeight ?
                            {{(FINAL_RESULT_WIDTH-WIDTH){colTop[j][WIDTH-1]}}, colTop[j]} : '0;
                    end else begin : top_neighbor
                        assign localPsum = verticalData[i-1][j];
                    end
                    if (ROLE == ROLE_AD && i < N-1) begin : lower_merge
                        assign localLowerPsum = verticalData[i+1][j];
                    end else begin : no_lower_merge
                        assign localLowerPsum = '0;
                    end
                end

                if (UPDATE_PHASE == 0) begin : live_update_entry
                    assign localUpdateValid = updateValid;
                    assign localRowDirection = rowDirection[i];
                    assign localColumnDirection = columnDirection[j];
                end else begin : piped_update_wave
                    assign localUpdateValid = updateValidPipe[UPDATE_PHASE-1];
                    assign localRowDirection = updateRowPipe[UPDATE_PHASE-1][i];
                    assign localColumnDirection = updateColumnPipe[UPDATE_PHASE-1][j];
                end

                always_comb begin
                    if ((localRowDirection == 2'sd0) || (localColumnDirection == 2'sd0))
                        localUpdateDirection = 2'sd0;
                    else if (localRowDirection == localColumnDirection)
                        localUpdateDirection = 2'sd1;
                    else
                        localUpdateDirection = -2'sd1;
                end

                weightStationaryProcessingElement #(
                    .WIDTH(WIDTH), .RESULT_WIDTH(FINAL_RESULT_WIDTH), .ROLE(ROLE)
                ) pe (
                    .clk(clk), .rst_n(rst_n), .advance(advance),
                    .loadWeight(loadWeight), .captureWeight(captureWeight),
                    .updateWeight(localUpdateValid), .updateDirection(localUpdateDirection),
                    .activationIn(localActivation), .psumIn(localPsum),
                    .lowerPsumIn(localLowerPsum),
                    .activationOut(horizontalData[i][j]), .psumOut(verticalData[i][j])
                );
            end
        end
        for (j = 0; j < N; j++) begin : result_loop
            assign result[j] = verticalData[N-1-j][j];
            assign resultValid[j] = validPipe[N-1];
        end
    endgenerate

    always_comb begin
        pipelineBusy = |validPipe;
        for (int stage = 0; stage < UPDATE_PIPE_STAGES; stage++)
            pipelineBusy |= updateValidPipe[stage];
    end

endmodule
