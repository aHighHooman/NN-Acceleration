module dataOrchestrator #(
    parameter int WIDTH = 16,
    parameter int N = 3
)(
    input  logic        clk,
    input  logic        rst_n,
    input  logic signed [WIDTH-1:0] matrixA [N][N],
    input  logic signed [WIDTH-1:0] matrixB [N][N],
    output logic signed [WIDTH-1:0] row     [N]   ,
    output logic signed [WIDTH-1:0] col     [N]   
);

logic [N:0] counter;

// Not fully parameterized yet, for N=3 only
always_ff @(posedge clk) begin
    
    // Set all outputs to 0 on reset
    if (!rst_n) begin
        counter  <= 4'd0;
        for (int i = 0; i < N; i = i + 1) begin
            row[i] <= 0;
            col[i] <= 0;
    end
    
    // Begin Data Orchestration
    end else begin
        counter <= counter + 1;

        if (counter >= 0 && counter < 3) begin
            row[0]  <= matrixA[0][counter];
            col[0]  <= matrixB[counter][0];
        end 
        if (counter >= 1 && counter < 4) begin
            row[1]  <= matrixA[1][counter - 1];
            col[1]  <= matrixB[counter - 1][1];
        end
        if (counter >= 2 && counter < 5) begin
            row[2] <= matrixA[2][counter - 2];
            col[2] <= matrixB[counter - 2][2];
        end
        
    end
end

endmodule