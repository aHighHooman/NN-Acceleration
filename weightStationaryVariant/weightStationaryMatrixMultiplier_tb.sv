`timescale 1ns/1ps
// Independent dense scoreboard and enabled-step timing checks at each lane.
// Identity comes from order: each lane sees vectors in admission order, so the
// bench counts arrivals instead of trusting identity carried by the design.
module weightStationaryMatrixMultiplier_testcase #(
    parameter int N=5, WIDTH=8, LAYERS=3, ACTIVATION_WIDTH=19, FRACTION_BITS=0
)(output logic done);
    localparam int D=N/2+1;
    // Array 0 takes D steps and every later array D+1, one for its lane register.
    localparam int CENTER_AGE=LAYERS*(D+1)-1;
    localparam int LIMIT=512;
    localparam int REDUCTION_WEIGHT_WIDTH=8;
    typedef logic signed [WIDTH-1:0] data_t;
    typedef logic signed [ACTIVATION_WIDTH-1:0] activation_t;
    // Every dense sum here fits 64 bits: at most 19+16+$clog2(9) bits.
    typedef logic signed [63:0] wide_t;
    logic clk=0, rst_n, weightValid, weightReady, inputValid, inputReady;
    logic outputBlocked, weightsLoaded, reloadWeights, reloadReady, advance;
    logic passThrough, backward;
    data_t weightData[N], matrix[LAYERS][N][N];
    activation_t inputData[N];
    wire signed [WIDTH-1:0] weights[LAYERS][N][N];
    logic signed [REDUCTION_WEIGHT_WIDTH-1:0] reductionWeight[N];
    logic activationGates[LAYERS][N];
    logic signed [1:0] rowDirection[LAYERS][N], columnDirection[LAYERS][N];
    logic matrixUpdateValid;
    activation_t resultData[N];
    wire resultValid[N];
    wire tapValid[LAYERS][N], tapGate[LAYERS][N];
    wire signed [1:0] tapSign[LAYERS][N], tapGradientSign[LAYERS][N];
    wire activation_t laneGradient[LAYERS][N];
    activation_t expected[LIMIT][N];
    int acceptedStep[LIMIT], acceptedCount, cycle, enabledStep;
    int laneArrivals[N];
    int unsigned randomState;
    always #5 clk=~clk;
    weightStationaryMatrixMultiplier #(.WIDTH(WIDTH), .ACTIVATION_WIDTH(ACTIVATION_WIDTH),
        .FRACTION_BITS(FRACTION_BITS), .N(N), .LAYERS(LAYERS)) dut (
        .clk, .rst_n, .backward, .weightData, .weightValid, .weightReady,
        .inputData, .inputValid, .inputReady, .passThrough, .reductionWeight, .activationGates,
        .rowDirection, .columnDirection, .matrixUpdateValid,
        .resultData, .resultValid, .outputBlocked,
        .weightsLoaded, .reloadWeights, .reloadReady, .arrayAdvance(advance),
        .weights, .tapValid, .tapGate, .tapSign, .tapGradientSign
    );
    for (genvar l=0; l<LAYERS; l++) for (genvar j=0; j<N; j++)
        assign laneGradient[l][j]=dut.laneGradient[l][j];
    // An independent statement of the format rule: shift out the extra fraction
    // bits, then cap at the activation limits.
    function automatic activation_t requantize(input wide_t value);
        wide_t shifted, limit;
        shifted=value>>>FRACTION_BITS;
        limit=wide_t'(1)<<(ACTIVATION_WIDTH-1);
        if (shifted>=limit) return activation_t'(limit-1);
        if (shifted<-limit) return activation_t'(-limit);
        return activation_t'(shifted);
    endfunction
    function automatic int unsigned random_word();
        randomState=1664525*randomState+1013904223;
        return randomState;
    endfunction
    always @(posedge clk) begin : check
        activation_t activation[N], nextActivation[N];
        wide_t sum;
        activation_t held[N];
        bit heldValid[N];
        bit blocked, stepped;
        cycle++;
        blocked=rst_n && outputBlocked;
        stepped=rst_n && advance;
        for (int j=0; j<N; j++) begin held[j]=resultData[j]; heldValid[j]=resultValid[j]; end
        if (!rst_n) begin
            acceptedCount=0; enabledStep=0;
            for (int j=0; j<N; j++) laneArrivals[j]=0;
        end else begin
            if (advance) enabledStep++;
            if (inputValid && inputReady) begin
                if (acceptedCount>=LIMIT) $fatal(1,"Fountain test sample limit");
                for (int j=0; j<N; j++) activation[j]=$signed(inputData[j]);
                for (int l=0; l<LAYERS; l++) begin
                    for (int j=0; j<N; j++) begin
                        sum=0;
                        for (int i=0; i<N; i++) sum+=activation[i]*$signed(matrix[l][i][j]);
                        nextActivation[j]=requantize((passThrough || sum>=0)?sum:0);
                    end
                    for (int j=0; j<N; j++) activation[j]=nextActivation[j];
                end
                for (int j=0; j<N; j++) expected[acceptedCount][j]=activation[j];
                acceptedStep[acceptedCount]=enabledStep;
                acceptedCount++;
            end
            #1;
            if (blocked) for (int j=0; j<N; j++) begin
                if (heldValid[j]!==resultValid[j]) $fatal(1,"Fountain stalled validity changed");
                if (held[j]!==resultData[j]) $fatal(1,"Fountain stalled data changed");
            end
            // Lane registers change only on an advancing edge, so each valid
            // value seen after one is the next vector in that lane's order.
            if (stepped) for (int j=0; j<N; j++) begin
                if (tapValid[LAYERS-1][j]!==resultValid[j])
                    $fatal(1,"Fountain tap valid N=%0d L=%0d lane=%0d",N,LAYERS,j);
                if (resultValid[j]) begin
                    int sample, q;
                    sample=laneArrivals[j]++;
                    q=(j<N/2)?N/2-j:j-N/2;
                    if (sample>=acceptedCount || enabledStep-acceptedStep[sample]!=CENTER_AGE+q)
                        $fatal(1,"Fountain lane timing N=%0d L=%0d lane=%0d sample=%0d age=%0d expected=%0d",
                            N,LAYERS,j,sample,enabledStep-acceptedStep[sample],CENTER_AGE+q);
                    if (resultData[j]!==expected[sample][j])
                        $fatal(1,"Fountain result N=%0d L=%0d sample=%0d lane=%0d got=%0d expected=%0d",
                            N,LAYERS,sample,j,resultData[j],expected[sample][j]);
                end
            end
        end
    end
    task automatic reset_dut();
        @(negedge clk); rst_n=0; inputValid=0; weightValid=0; reloadWeights=0;
        repeat(2) @(posedge clk);
        @(negedge clk); rst_n=1;
    endtask
    task automatic load_weights(input bit extremes);
        int lastHost, guard;
        for (int l=0; l<LAYERS; l++) for (int i=N-1; i>=0; i--) begin
            @(negedge clk);
            if (!weightReady || inputReady) $fatal(1,"Fountain loader readiness");
            weightValid=1;
            for (int j=0; j<N; j++) begin
                matrix[l][i][j]=extremes ? data_t'((i+j)%2 ? (1<<(WIDTH-1))-1 : -(1<<(WIDTH-1))) :
                    data_t'((l+1)*9+i*N+j+1);
                weightData[j]=matrix[l][i][j];
            end
            @(posedge clk); #1; lastHost=cycle;
            if ((i+l)%2) begin @(negedge clk); weightValid=0; repeat(2) @(posedge clk); end
        end
        @(negedge clk); weightValid=0;
        guard=0;
        while (!weightsLoaded) begin @(posedge clk); #1; if (++guard>LAYERS*N+4) $fatal(1,"Fountain load timeout"); end
        // Gaps after the final row can occur inside this task; check resident data
        // rather than counting those optional host-idle cycles as transfer stages.
        for (int l=0; l<LAYERS; l++) for (int i=0; i<N; i++) for (int j=0; j<N; j++)
            if (weights[l][i][j]!==matrix[l][i][j]) $fatal(1,"Fountain row mapping failed");
    endtask
    task automatic stream(input int count, mode);
        int sent, guard;
        sent=0; guard=0;
        while (sent<count) begin
            @(negedge clk);
            outputBlocked=(mode==2 && (random_word()>>16)%4!=0);
            inputValid=(mode!=1 || random_word()%3!=0);
            for (int i=0; i<N; i++) begin
                if (mode==3) inputData[i]=activation_t'((sent+i)%2 ? -(1<<(ACTIVATION_WIDTH-1)) : (1<<(ACTIVATION_WIDTH-1))-1);
                else if (mode==4) inputData[i]=activation_t'(i==sent%N);
                else inputData[i]=activation_t'(random_word());
            end
            @(posedge clk);
            if (inputValid && inputReady) sent++;
            #1;
            if (++guard>3000) $fatal(1,"Fountain input timeout");
        end
        @(negedge clk); inputValid=0; outputBlocked=0;
        guard=0;
        // The outermost lane finishes each vector last.
        while (laneArrivals[0]<acceptedCount || !reloadReady) begin
            @(posedge clk); #1;
            if (++guard>400) $fatal(1,"Fountain drain timeout");
        end
        // Nothing may go missing: a drained array has delivered every vector on every lane.
        for (int j=0; j<N; j++) if (laneArrivals[j]!=acceptedCount)
            $fatal(1,"Fountain lane completeness N=%0d L=%0d lane=%0d arrivals=%0d accepted=%0d",
                N,LAYERS,j,laneArrivals[j],acceptedCount);
    endtask
    // Hold the readout weights for one edge less than the accelerator's backward
    // count, because this bench reads the signs before they are captured, and
    // compare every layer's lanes with a dense calculation.
    task automatic check_backward();
        activation_t gradient[N], next[N];
        wide_t sum;
        @(negedge clk);
        backward=1;
        for (int j=0; j<N; j++) reductionWeight[j]=REDUCTION_WEIGHT_WIDTH'(random_word());
        for (int l=0; l<LAYERS; l++) for (int j=0; j<N; j++) activationGates[l][j]=random_word()%4!=0;
        repeat(LAYERS*(D+1)-1) @(posedge clk);
        #1;
        for (int j=0; j<N; j++) gradient[j]=activationGates[LAYERS-1][j]?reductionWeight[j]:0;
        for (int l=LAYERS-1; l>=0; l--) begin
            for (int j=0; j<N; j++) begin
                if (laneGradient[l][j]!==gradient[j])
                    $fatal(1,"Fountain gradient N=%0d L=%0d layer=%0d lane=%0d got=%0d expected=%0d",
                        N,LAYERS,l,j,laneGradient[l][j],gradient[j]);
                if (tapGradientSign[l][j]!==((gradient[j]==0)?2'sd0:((gradient[j]<0)?-2'sd1:2'sd1)))
                    $fatal(1,"Fountain gradient sign N=%0d L=%0d layer=%0d lane=%0d",N,LAYERS,l,j);
            end
            if (l>0) begin
                for (int i=0; i<N; i++) begin
                    sum=0;
                    for (int j=0; j<N; j++) sum+=$signed(matrix[l][i][j])*gradient[j];
                    next[i]=activationGates[l-1][i]?requantize(sum):0;
                end
                gradient=next;
            end
        end
        @(negedge clk); backward=0;
    endtask
    task automatic reload();
        int previous;
        @(negedge clk);
        previous=acceptedCount;
        reloadWeights=1; inputValid=1;
        #1;
        if (inputReady) $fatal(1,"Fountain reload failed to exclude admission");
        @(posedge clk); #1;
        if (weightsLoaded || acceptedCount!=previous) $fatal(1,"Fountain reload epoch failure");
        @(negedge clk); reloadWeights=0; inputValid=0;
    endtask
    initial begin
        done=0; rst_n=0; weightValid=0; inputValid=0;
        outputBlocked=0; reloadWeights=0; passThrough=1; matrixUpdateValid=0; backward=0;
        cycle=0; enabledStep=0; acceptedCount=0;
        randomState=32'(N*913+LAYERS*43);
        for (int l=0; l<LAYERS; l++) for (int i=0; i<N; i++) begin
            rowDirection[l][i]=0; columnDirection[l][i]=0; activationGates[l][i]=1;
        end
        for (int i=0; i<N; i++) begin inputData[i]=0; weightData[i]=0; reductionWeight[i]=0; end
        reset_dut(); load_weights(0);
        stream(N,4); stream(50,0); stream(40,1); stream(45,2);
        // A backward pass leaves the PE registers dirty; forward results must not care.
        check_backward(); stream(16,0);
        reload(); passThrough=0; load_weights(0); stream(32,0);
        reload(); passThrough=1; load_weights(1); stream(8,3);
        check_backward(); stream(8,3);
        @(negedge clk); inputValid=1;
        @(posedge clk); #1;
        reset_dut(); load_weights(0); stream(8,0);
        $display("PASS: Fountain matrix N=%0d WIDTH=%0d ACTIVATION_WIDTH=%0d F=%0d LAYERS=%0d arithmetic, lane timing, II=1, stalls, reset/reload",
            N,WIDTH,ACTIVATION_WIDTH,FRACTION_BITS,LAYERS);
        done=1;
    end
endmodule

module weightStationaryMatrixMultiplier_tb;
    wire [9:0] done;
    weightStationaryMatrixMultiplier_testcase #(.N(3),.LAYERS(1)) a(done[0]);
    weightStationaryMatrixMultiplier_testcase #(.N(5),.LAYERS(1)) b(done[1]);
    weightStationaryMatrixMultiplier_testcase #(.N(5),.LAYERS(3)) c(done[2]);
    weightStationaryMatrixMultiplier_testcase #(.N(9),.LAYERS(3)) d(done[3]);
    weightStationaryMatrixMultiplier_testcase #(.N(5),.WIDTH(16),.LAYERS(1)) e(done[4]);
    weightStationaryMatrixMultiplier_testcase #(.N(9),.LAYERS(1)) f(done[5]);
    weightStationaryMatrixMultiplier_testcase #(.N(3),.WIDTH(16),.LAYERS(3)) g(done[6]);
    weightStationaryMatrixMultiplier_testcase #(.N(3),.LAYERS(4)) h(done[7]);
    // The Quartus format, and a narrow format that saturates often.
    weightStationaryMatrixMultiplier_testcase #(.N(5),.WIDTH(16),.LAYERS(3),.FRACTION_BITS(8)) k(done[8]);
    weightStationaryMatrixMultiplier_testcase #(.N(5),.LAYERS(3),.ACTIVATION_WIDTH(12),.FRACTION_BITS(2)) m(done[9]);
    initial begin wait(&done); $display("PASS: Fountain matrix regression"); $finish; end
    initial begin #1000000; $fatal(1,"Fountain matrix regression timeout"); end
endmodule
