module weightLoader #(
    parameter int WIDTH = 16,
    parameter int N = 3
)(
    input  logic        clk,
    input  logic        rst_n,
    input  logic signed [WIDTH-1:0] colB    [N],
    output logic signed [WIDTH-1:0] col     [N],   
    output logic        [7:0]       addrB   [N],
    output logic                    writeEnable
);

logic [N:0] counter;


always_ff @(posedge clk) begin
    
    // Set all outputs to 0 on reset
    if (!rst_n) begin
        counter  <= 0;
        writeEnable <= 1;
        
        for (int i = 0; i < N; i = i + 1) begin
            col[i] <= 0;
            addrB[i] <= 25;
        
        end
    
    
    // Begin Data Orchestration
    end else begin
        if (counter < N + 2) begin
            counter <= counter + 1;
            writeEnable <= 1;
        end else begin
            writeEnable <= 0;
        end

        // Call initial memory addresses
        if (counter == 0) begin
            for (int i = 0; i < N; i = i + 1) begin
                addrB[i] <= N*(N-1) + i;
            end
        end

        // Call the next addresses
        for (int i = 0; i < N; i = i + 1) begin
            if (counter > 0 && counter < N) begin
                addrB[i] <= addrB[i] - N;
            end
        end

        // Send the weights to the systolic array
        for (int i = 0; i < N; i = i + 1) begin
            if (counter > 1 && counter < N+2) begin
                col[i]  <= colB[i];
            end else begin 
                col[i]  <= 0;
            end
        end
    end
end

endmodule