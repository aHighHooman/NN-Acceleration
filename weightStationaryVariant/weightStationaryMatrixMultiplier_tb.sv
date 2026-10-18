`timescale 1ns/1ps
// Independent dense scoreboard and enabled-step timing checks at each lane.
module weightStationaryMatrixMultiplier_testcase #(
    parameter int N=5, WIDTH=8, LAYERS=3, G=0, J=1
)(output logic done);
    localparam int AW=WIDTH+LAYERS*(WIDTH+$clog2(N));
    localparam int D=N/2+1+G+J;
    localparam int LIMIT=512;
    typedef logic signed [WIDTH-1:0] data_t;
    typedef logic signed [AW-1:0] result_t;
    logic clk=0, rst_n, weightValid, weightReady, inputValid, inputReady;
    logic resultValid, resultReady, weightsLoaded, reloadWeights, reloadReady, advance;
    logic passThrough;
    data_t weightData[N], inputData[N], matrix[LAYERS][N][N];
    wire signed [WIDTH-1:0] weights[LAYERS][N][N];
    logic signed [1:0] rowDirection[LAYERS][N], columnDirection[LAYERS][N];
    logic matrixUpdateValid;
    logic [15:0] inputTag;
    wire [15:0] resultTag;
    result_t resultData[N];
    wire tapValid[LAYERS][N], tapGate[LAYERS][N];
    wire [15:0] tapTag[LAYERS][N];
    wire signed [1:0] tapSign[LAYERS][N];
    result_t expected[LIMIT][N];
    int acceptedStep[LIMIT], acceptedCount, retiredCount, cycle, enabledStep;
    int unsigned randomState;
    always #5 clk=~clk;
    weightStationaryMatrixMultiplier #(.WIDTH(WIDTH), .N(N), .LAYERS(LAYERS), .G(G), .J(J)) dut (
        .clk, .rst_n, .weightData, .weightValid, .weightReady,
        .inputData, .inputValid, .inputTag, .inputReady, .passThrough,
        .rowDirection, .columnDirection, .matrixUpdateValid,
        .resultData, .resultValid, .resultTag, .resultReady,
        .weightsLoaded, .reloadWeights, .reloadReady, .arrayAdvance(advance),
        .weights, .tapValid, .tapGate, .tapTag, .tapSign
    );
    function automatic int unsigned random_word();
        randomState=1664525*randomState+1013904223;
        return randomState;
    endfunction
    always @(posedge clk) begin : check
        // Match the conservative final mathematical bound, including profiles
        // whose exact dense results exceed a native 64-bit scoreboard value.
        result_t activation[N], nextActivation[N], sum;
        result_t held[N];
        logic [15:0] heldTag;
        bit blocked;
        cycle++;
        blocked=rst_n && resultValid && !resultReady;
        heldTag=resultTag;
        for (int j=0; j<N; j++) held[j]=resultData[j];
        if (!rst_n) begin acceptedCount=0; retiredCount=0; enabledStep=0; end
        else begin
            if (advance) enabledStep++;
            if (resultValid && resultReady) begin
                if (retiredCount>=acceptedCount || resultTag!=retiredCount)
                    $fatal(1,"Fountain result ordering N=%0d L=%0d",N,LAYERS);
                for (int j=0; j<N; j++) if (resultData[j]!==expected[retiredCount][j])
                    $fatal(1,"Fountain result N=%0d L=%0d sample=%0d lane=%0d got=%0d expected=%0d",
                        N,LAYERS,retiredCount,j,resultData[j],expected[retiredCount][j]);
                retiredCount++;
            end
            if (inputValid && inputReady) begin
                if (acceptedCount>=LIMIT || inputTag!=acceptedCount) $fatal(1,"Invalid test tag");
                for (int j=0; j<N; j++) activation[j]=$signed(inputData[j]);
                for (int l=0; l<LAYERS; l++) begin
                    for (int j=0; j<N; j++) begin
                        sum=0;
                        for (int i=0; i<N; i++) sum+=activation[i]*$signed(matrix[l][i][j]);
                        nextActivation[j]=(passThrough || sum>=0)?sum:0;
                    end
                    for (int j=0; j<N; j++) activation[j]=nextActivation[j];
                end
                for (int j=0; j<N; j++) expected[acceptedCount][j]=result_t'(activation[j]);
                acceptedStep[acceptedCount]=enabledStep;
                acceptedCount++;
            end
            #1;
            if (blocked) begin
                if (!resultValid || resultTag!==heldTag) $fatal(1,"Fountain stalled validity changed");
                for (int j=0; j<N; j++) if (held[j]!==resultData[j]) $fatal(1,"Fountain stalled data changed");
            end
            for (int j=0; j<N; j++) if (tapValid[LAYERS-1][j]) begin
                int tag, q;
                tag=tapTag[LAYERS-1][j];
                q=(j<N/2)?N/2-j:j-N/2;
                if (tag>=acceptedCount || enabledStep-acceptedStep[tag]!=LAYERS*D+q)
                    $fatal(1,"Fountain lane timing N=%0d L=%0d G=%0d J=%0d lane=%0d tag=%0d age=%0d expected=%0d",
                        N,LAYERS,G,J,j,tag,enabledStep-acceptedStep[tag],LAYERS*D+q);
            end
            if (resultValid && enabledStep-acceptedStep[resultTag]!=LAYERS*D+N/2)
                $fatal(1,"Fountain collector timing N=%0d L=%0d",N,LAYERS);
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
            resultReady=(mode!=2 || (random_word()>>16)%4==0);
            inputValid=(mode!=1 || random_word()%3!=0);
            inputTag=16'(acceptedCount);
            for (int i=0; i<N; i++) begin
                if (mode==3) inputData[i]=data_t'((sent+i)%2 ? -(1<<(WIDTH-1)) : (1<<(WIDTH-1))-1);
                else if (mode==4) inputData[i]=data_t'(i==sent%N);
                else inputData[i]=data_t'(random_word());
            end
            @(posedge clk);
            if (inputValid && inputReady) sent++;
            #1;
            if (++guard>3000) $fatal(1,"Fountain input timeout");
        end
        @(negedge clk); inputValid=0; resultReady=1;
        guard=0;
        while (retiredCount<acceptedCount || !reloadReady) begin
            @(posedge clk); #1;
            if (++guard>400) $fatal(1,"Fountain drain timeout");
        end
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
        done=0; rst_n=0; weightValid=0; inputValid=0; inputTag=0;
        resultReady=1; reloadWeights=0; passThrough=1; matrixUpdateValid=0;
        cycle=0; enabledStep=0; acceptedCount=0; retiredCount=0;
        randomState=32'(N*913+LAYERS*43+G*7+J);
        for (int l=0; l<LAYERS; l++) for (int i=0; i<N; i++) begin rowDirection[l][i]=0; columnDirection[l][i]=0; end
        for (int i=0; i<N; i++) begin inputData[i]=0; weightData[i]=0; end
        reset_dut(); load_weights(0);
        stream(N,4); stream(50,0); stream(40,1); stream(45,2);
        reload(); passThrough=0; load_weights(0); stream(32,0);
        reload(); passThrough=1; load_weights(1); stream(8,3);
        @(negedge clk); inputValid=1; inputTag=16'(acceptedCount);
        @(posedge clk); #1;
        reset_dut(); load_weights(0); stream(8,0);
        $display("PASS: Fountain matrix N=%0d WIDTH=%0d LAYERS=%0d G=%0d J=%0d arithmetic, lane timing, II=1, stalls, reset/reload",N,WIDTH,LAYERS,G,J);
        done=1;
    end
endmodule

module weightStationaryMatrixMultiplier_tb;
    wire [6:0] done;
    weightStationaryMatrixMultiplier_testcase #(.N(3),.LAYERS(1)) a(done[0]);
    weightStationaryMatrixMultiplier_testcase #(.N(5),.LAYERS(1)) b(done[1]);
    weightStationaryMatrixMultiplier_testcase #(.N(5),.LAYERS(3)) c(done[2]);
    weightStationaryMatrixMultiplier_testcase #(.N(5),.LAYERS(3),.G(1)) d(done[3]);
    weightStationaryMatrixMultiplier_testcase #(.N(3),.WIDTH(16),.LAYERS(1),.J(2)) e(done[4]);
    weightStationaryMatrixMultiplier_testcase #(.N(9),.LAYERS(1)) f(done[5]);
    weightStationaryMatrixMultiplier_testcase #(.N(3),.WIDTH(16),.LAYERS(3)) g(done[6]);
    initial begin wait(&done); $display("PASS: Fountain matrix regression"); $finish; end
    initial begin #1000000; $fatal(1,"Fountain matrix regression timeout"); end
endmodule
