// tb_matrixMultiplierSmall.sv
`timescale 1ns/1ps

module tb_matrixMultiplierSmall;

  // match the DUT defaults; change if you compiled DUT with other params
  localparam int WIDTH = 16;
  localparam int N     = 3;

  // clock / reset
  logic clk;
  logic rst_n;

  // DUT I/O
  logic signed [WIDTH-1:0] matrixA [N][N];
  logic signed [WIDTH-1:0] matrixB [N][N];
  logic signed [2*WIDTH-1:0] resultMatrix [N][N];

  // instantiate DUT
  matrixMultiplierSmall #(
    .WIDTH(WIDTH),
    .N(N)
  ) dut (
    .clk         (clk),
    .rst_n       (rst_n),
    .matrixA     (matrixA),
    .matrixB     (matrixB),
    .resultMatrix(resultMatrix)
  );

  // clock generator: 10ns period
  initial clk = 0;
  always #5 clk = ~clk;

  // test stimulus
  initial begin
    // waveform + tidy
    $dumpfile("tb_matrixMultiplierSmall.vcd");
    $dumpvars(0, tb_matrixMultiplierSmall);

    // initialize
    rst_n = 0;
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++) begin
        matrixA[i][j] = '0;
        matrixB[i][j] = '0;
      end

    rst_n = 0;

    // Set matrices during reset
    matrixA[0] = '{ 1,  2,  3 };
    matrixA[1] = '{ 4,  5,  6 };
    matrixA[2] = '{ 7,  8,  9 };

    matrixB[0] = '{ 1,  0,  0 };
    matrixB[1] = '{ 0,  1,  0 };
    matrixB[2] = '{ 0,  0,  1 };

    #20;
    rst_n = 1;

    // give the DUT several cycles to process (depends on real implementation latency)
    #100;

    // print out resultMatrix
    $display("---- resultMatrix ----");
    for (int i = 0; i < N; i++) begin
      for (int j = 0; j < N; j++) begin
        $write("%0d\t", resultMatrix[i][j]);
      end
      $write("\n");
    end
    $display("----------------------");

    #10;
    $finish;
  end

endmodule