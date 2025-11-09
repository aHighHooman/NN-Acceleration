module dataOrchestrator (
    input  logic        clk,
    input  logic        rst_n,
    input  logic signed [15:0] matrixA [3][3],
    input  logic signed [15:0] matrixB [3][3],
    output logic signed [15:0] rowOne        ,
    output logic signed [15:0] rowTwo        ,
    output logic signed [15:0] rowThree      ,
    output logic signed [15:0] colOne        ,
    output logic signed [15:0] colTwo        ,
    output logic signed [15:0] colThree
);

logic [3:0] counter;

always_ff @(posedge clk) begin
    if (!rst_n) begin
        rowOne   <= 16'sd0;
        rowTwo   <= 16'sd0;
        rowThree <= 16'sd0;
        colOne   <= 16'sd0;
        colTwo   <= 16'sd0;
        colThree <= 16'sd0;
        counter  <= 4'd0;
    end else begin
        counter <= counter + 4'd1;

        if (counter >= 4'd0 && counter < 4'd3) begin
            rowOne  <= matrixA[0][counter];
            colOne  <= matrixB[counter][0];
        end 
        if (counter >= 4'd1 && counter < 4'd4) begin
            rowTwo  <= matrixA[1][counter - 4'd1];
            colTwo  <= matrixB[counter - 4'd1][1];
        end
        if (counter >= 4'd2 && counter < 4'd5) begin
            rowThree <= matrixA[2][counter - 4'd2];
            colThree <= matrixB[counter - 4'd2][2];
        end
        
    end
end

endmodule