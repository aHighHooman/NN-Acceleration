// AXI4-Stream output adapter for the weight-stationary matrix-multiplier core.
// One AXI beat carries one complete result row.  A frame contains exactly N beats.
module matrixMultiplierWeightStationaryAxisOut #(
    parameter int WIDTH = 16,
    parameter int N = 3,
    parameter int INPUT_FIFO_DEPTH = 2*N,
    parameter int OUTPUT_FIFO_DEPTH = 2*N,
    parameter int AXIS_LANE_WIDTH = 8*((2*WIDTH + $clog2(N) + 7) / 8)
)(
    input  logic                                      clk,
    input  logic                                      rst_n,

    input  logic signed [WIDTH-1:0]                   weightData [N],
    input  logic                                      weightValid,
    output logic                                      weightReady,
    input  logic signed [WIDTH-1:0]                   activationData [N],
    input  logic                                      activationValid,
    output logic                                      activationReady,
    input  logic                                      passThrough,
    output logic                                      weightsLoaded,
    input  logic                                      reloadWeights,
    output logic                                      reloadReady,

    // Command channel.  Each valid/ready handshake requests one result frame.
    input  logic                                      output_start_valid,
    output logic                                      output_start_ready,
    output logic                                      result_frame_available,

    // AXI4-Stream-compatible result channel.
    output logic [N*AXIS_LANE_WIDTH-1:0]              m_axis_tdata,
    output logic                                      m_axis_tvalid,
    input  logic                                      m_axis_tready,
    output logic                                      m_axis_tlast
);

    localparam int RESULT_WIDTH = 2*WIDTH + $clog2(N);
    localparam int ROW_COUNT_WIDTH = (N <= 1) ? 1 : $clog2(N);

    logic signed [RESULT_WIDTH-1:0] coreResultData [N];
    logic coreResultValid, coreResultReady, coreResultLast;
    logic coreResultFrameAvailable;
    logic transactionActive;
    logic [ROW_COUNT_WIDTH-1:0] transmittedRow;
    logic axisTransfer, outputStartAccepted;

    assign result_frame_available = coreResultFrameAvailable;
    assign output_start_ready = !transactionActive && result_frame_available;
    assign outputStartAccepted = output_start_valid && output_start_ready;

    assign m_axis_tvalid = transactionActive && coreResultValid;
    assign coreResultReady = transactionActive && m_axis_tready;
    assign axisTransfer = m_axis_tvalid && m_axis_tready;
    assign m_axis_tlast = m_axis_tvalid && (transmittedRow == N-1);

    genvar lane;
    generate
        for (lane = 0; lane < N; lane = lane + 1) begin : pack_axis_row
            assign m_axis_tdata[lane*AXIS_LANE_WIDTH +: AXIS_LANE_WIDTH] = {
                {(AXIS_LANE_WIDTH-RESULT_WIDTH){coreResultData[lane][RESULT_WIDTH-1]}},
                coreResultData[lane]
            };
        end

        if (AXIS_LANE_WIDTH < RESULT_WIDTH || (AXIS_LANE_WIDTH % 8) != 0) begin : invalid_axis_lane_width
            initial $fatal(1, "AXIS_LANE_WIDTH must be byte aligned and at least RESULT_WIDTH");
        end
    endgenerate

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            transactionActive <= 1'b0;
            transmittedRow    <= '0;
        end else if (!transactionActive) begin
            if (outputStartAccepted) begin
                transactionActive <= 1'b1;
                transmittedRow    <= '0;
            end
        end else if (axisTransfer) begin
            if (transmittedRow == N-1) begin
                transactionActive <= 1'b0;
                transmittedRow    <= '0;
            end else begin
                transmittedRow <= transmittedRow + 1'b1;
            end
        end
    end

    matrixMultiplierWeightStationary #(
        .WIDTH(WIDTH),
        .N(N),
        .INPUT_FIFO_DEPTH(INPUT_FIFO_DEPTH),
        .OUTPUT_FIFO_DEPTH(OUTPUT_FIFO_DEPTH)
    ) core (
        .clk(clk),
        .rst_n(rst_n),
        .weightData(weightData),
        .weightValid(weightValid),
        .weightReady(weightReady),
        .activationData(activationData),
        .activationValid(activationValid),
        .activationReady(activationReady),
        .resultData(coreResultData),
        .resultValid(coreResultValid),
        .resultReady(coreResultReady),
        .passThrough(passThrough),
        .resultLast(coreResultLast),
        .resultFrameAvailable(coreResultFrameAvailable),
        .weightsLoaded(weightsLoaded),
        .reloadWeights(reloadWeights),
        .reloadReady(reloadReady)
    );

`ifndef SYNTHESIS
    logic [N*AXIS_LANE_WIDTH-1:0] heldAxisData;
    logic heldAxisLast, axisWasStalled;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            heldAxisData  <= '0;
            heldAxisLast  <= 1'b0;
            axisWasStalled <= 1'b0;
        end else begin
            if (axisWasStalled) begin
                assert(m_axis_tvalid) else $error("AXI TVALID dropped under backpressure");
                assert(m_axis_tdata == heldAxisData) else $error("AXI TDATA changed under backpressure");
                assert(m_axis_tlast == heldAxisLast) else $error("AXI TLAST changed under backpressure");
            end

            assert((coreResultValid && coreResultReady) == axisTransfer)
                else $error("Core output pop did not match AXI transfer");
            if (!transactionActive)
                assert(!m_axis_tvalid) else $error("AXI output valid while no transaction is active");
            if (m_axis_tvalid)
                assert(m_axis_tlast == coreResultLast)
                    else $error("AXI TLAST did not match the core row boundary");
            if (axisTransfer)
                assert(m_axis_tlast == (transmittedRow == N-1))
                    else $error("AXI transaction did not end on beat N");

            if (m_axis_tvalid && !m_axis_tready) begin
                heldAxisData  <= m_axis_tdata;
                heldAxisLast  <= m_axis_tlast;
                axisWasStalled <= 1'b1;
            end else begin
                axisWasStalled <= 1'b0;
            end
        end
    end
`endif

endmodule
