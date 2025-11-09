module multiplierBlock (
    input  logic        clk,
    input  logic        rst_n,
    input  logic signed [15:0] a,
    input  logic signed [15:0] b,
    output logic signed [31:0] aOut,
    output logic signed [31:0] bOut,
    output logic signed [31:0] output
);

  always_ff @(posedge clk) begin
    if (!rst_n)
      output <= 32'sd0;
      aOut   <= 0;
      bOut   <= 0;
    else 
        output <= output + a * b;
        aOut   <= a;
        bOut   <= b;    
  end

endmodule