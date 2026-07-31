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
    output logic [$clog2(DEPTH+1)-1:0] values
);

    localparam int PTR_WIDTH = $clog2(DEPTH);

    logic signed [WIDTH-1:0]     data                        [0:DEPTH-1];
    logic        [PTR_WIDTH-1:0] readPtr, writePtr;
    logic                        pushAccepted, popAccepted;

    assign empty        = (values == 0);
    assign full         = (values == DEPTH);
    assign popAccepted  = pop && !empty;
    assign pushAccepted = push && (!full || popAccepted);

    // Lookahead for the Fifo, allows to deal with 1 cycle latency.
    assign popData      = empty ? 0 : data[readPtr];

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            readPtr  <= 0;
            writePtr <= 0;
            values   <= 0;
        end else begin
            if (pushAccepted) begin
                data[writePtr]  <= pushData;
                writePtr        <= (writePtr == DEPTH-1) ? 0 : writePtr + 1;
            end

            if (popAccepted) begin
                readPtr <= (readPtr == DEPTH-1) ? 0 : readPtr + 1;
            end

            case ({pushAccepted, popAccepted})
                2'b10: values   <= values + 1;
                2'b01: values   <= values - 1;
                default: values <= values;
            endcase
        end
    end

endmodule
