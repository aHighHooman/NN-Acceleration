module signedFifo #(
    parameter int WIDTH = 16,
    parameter int DEPTH = 4
)(
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    push,
    input  logic signed [WIDTH-1:0] pushData,
    input  logic                    pop,
    output logic signed [WIDTH-1:0] popData,
    output logic                    full,
    output logic                    empty,
    output logic [$clog2(DEPTH+1)-1:0] count
);

    localparam int PTR_WIDTH = (DEPTH <= 1) ? 1 : $clog2(DEPTH);
    logic [PTR_WIDTH-1:0] readPtr, writePtr;
    logic pushAccepted, popAccepted;

    assign empty   = (count == 0);
    assign full    = (count == DEPTH);
    assign popAccepted  = pop && !empty;
    assign pushAccepted = push && (!full || popAccepted);

    memory #(.WIDTH(WIDTH), .DEPTH(DEPTH), .ASYNC_READ(1'b1)) storage (
        .clk(clk), .rst_n(rst_n), .readAddr(readPtr), .writeAddr(writePtr),
        .writeEnable(pushAccepted), .inputData(pushData), .outputData(popData)
    );

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            readPtr  <= '0;
            writePtr <= '0;
            count    <= '0;
        end else begin
            if (pushAccepted) begin
                writePtr <= (writePtr == DEPTH-1) ? '0 : writePtr + 1'b1;
            end
            if (popAccepted) begin
                readPtr <= (readPtr == DEPTH-1) ? '0 : readPtr + 1'b1;
            end

            case ({pushAccepted, popAccepted})
                2'b10: count <= count + 1'b1;
                2'b01: count <= count - 1'b1;
            endcase
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (rst_n) begin
            assert (!(push && full && !popAccepted)) else $error("FIFO overflow");
            assert (!(pop && empty)) else $error("FIFO underflow");
        end
    end
`endif

endmodule
