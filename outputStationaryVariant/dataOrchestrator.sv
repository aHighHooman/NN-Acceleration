module dataOrchestrator #(
    parameter int WIDTH = 16,
    parameter int N = 3
)(
    input  logic        clk,
    input  logic        rst_n,
    input  logic signed [WIDTH-1:0] colA    [N],
    input  logic signed [WIDTH-1:0] rowB    [N],
    output logic signed [WIDTH-1:0] row     [N],
    output logic signed [WIDTH-1:0] col     [N],   
    output logic        [7:0]       addrA   [N],
    output logic        [7:0]       addrB   [N]
);

logic [N:0] counter;


always_ff @(posedge clk) begin
    
    // Set all outputs to 0 on reset
    if (!rst_n) begin
        counter  <= 0;
        for (int i = 0; i < N; i = i + 1) begin
            // Set outputs to 0
            row[i] <= 0;
            col[i] <= 0;
            addrA[i] <= 25;
            addrB[i] <= 25;
        end
    
    
    // Begin Data Orchestration
    end else begin
        // Call initial memory addresses
        
        counter <= counter + 1;
        if (counter == 0) begin
            for (int i = 0; i < N; i = i + 1) begin
                addrA[i] <= i * N;
                addrB[i] <= i;
            end
        end


        for (int i = 0; i < N; i = i + 1) begin
            if (counter >= i+1 && counter < i + N + 1) begin
                addrA[i] <= addrA[i] + 1;
                addrB[i] <= addrB[i] + N;
            end
            
            if (counter >= i+2 && counter < i + N+2) begin
                // Send the row and column data to the systolic array
                row[i]  <= colA[i];
                col[i]  <= rowB[i];
                
            end else begin 
                row[i]  <= 0;
                col[i]  <= 0;
            end
        end
    end
end

endmodule