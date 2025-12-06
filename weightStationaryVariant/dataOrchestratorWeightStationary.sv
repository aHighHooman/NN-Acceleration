module dataOrchestratorWeightStationary #(
    parameter int WIDTH = 16,
    parameter int N = 3
)(
    input  logic        clk,
    input  logic        rst_n,
    input  logic        writeEnable,
    input  logic signed [WIDTH-1:0] rowA    [N],
    output logic signed [WIDTH-1:0] row     [N],
    output logic        [7:0]       addrA   [N]
);

logic [N:0] counter;


always_ff @(posedge clk) begin
    
    // Set all outputs to 0 on reset
    if (!rst_n) begin
        counter  <= 0;
        for (int i = 0; i < N; i = i + 1) begin
            // Set outputs to 0
            row[i] <= 0;
            addrA[i] <= 25;
        end
    
    
    // Begin Data Orchestration
    end else begin
        if (~writeEnable) begin
            counter <= counter + 1;
            
            // Call initial addresses
            if (counter == 0) begin
                for (int i = 0; i < N; i = i + 1) begin
                    addrA[i] <= i;
                end
            end

            // Call the next addresses
            for (int i = 0; i < N; i = i + 1) begin
                // The if condition basically gives the window during which each row is active
                if (counter > i && counter <= i + N) begin
                    addrA[i] <= addrA[i] + N;
                end
            end

            // Send the row data to the systolic array
            for (int i = 0; i < N; i = i + 1) begin    
                if (counter > i + 1 && counter <= i + N + 1) begin
                    row[i]  <= rowA[i];
                end else begin 
                    row[i]  <= 0;
                end
            end
        end
    end
end

endmodule