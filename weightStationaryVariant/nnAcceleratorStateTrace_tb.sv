`timescale 1ns/1ps
// One input bundle per edge; Python owns the independent expected state.
module nnAcceleratorStateTrace_tb #(
    parameter int N=3, LAYERS=3, FRACTION_BITS=0, G=0, J=1
);
    localparam int WIDTH=8, TAG_WIDTH=5;
    localparam int RAW_WIDTH=WIDTH+LAYERS*(WIDTH+$clog2(N));
    localparam int RESULT_WIDTH=RAW_WIDTH+$clog2(N);
    logic clk=0, rst_n=0, weightValid=0, inputValid=0, trainingEnable=0, resultReady=1;
    logic loadReductionWeights=0, reloadWeights=0, passThrough=1, reduceOutput=0;
    logic signed [7:0] weightData[N], inputData[N], targetData, reductionWeight[N];
    wire weightReady, inputReady, resultValid, weightsLoaded, reloadReady;
    wire signed [RESULT_WIDTH-1:0] resultData[N];
    integer stimulus_fd, trace_fd, status, cycle_count, accepted, retired;
    integer fields[11+3*N];
    string stimulus_path,trace_path;
    always #5 clk=~clk;
    nnAccelerator #(.WIDTH(WIDTH),.N(N),.LAYERS(LAYERS),.FRACTION_BITS(FRACTION_BITS),
        .TAG_WIDTH(TAG_WIDTH),.G(G),.J(J),.IN_FLIGHT_DEPTH(16),.OUTPUT_FIFO_DEPTH(2)) dut(.*);

    for (genvar l=0; l<LAYERS; l++) begin : freeze_layer
        localparam int XW=WIDTH+l*(WIDTH+$clog2(N));
        localparam int AW=XW+WIDTH+$clog2(N);
        for (genvar i=0; i<N; i++) begin : freeze_row
            for (genvar j=0; j<N; j++) begin : freeze_column
                always @(posedge clk) begin
                    logic signed [XW-1:0] activation;
                    logic signed [AW-1:0] partial;
                    logic av,pv;
                    logic [TAG_WIDTH-1:0] at,pt;
                    if (rst_n && weightsLoaded && !dut.arrayAdvance) begin
                        activation=dut.matrixEngine.layers[l].systolicArray.rows[i].columns[j].pe.activation_out;
                        partial=dut.matrixEngine.layers[l].systolicArray.rows[i].columns[j].pe.sum_out;
                        av=dut.matrixEngine.layers[l].systolicArray.rows[i].columns[j].pe.activation_out_valid;
                        pv=dut.matrixEngine.layers[l].systolicArray.rows[i].columns[j].pe.sum_out_valid;
                        at=dut.matrixEngine.layers[l].systolicArray.rows[i].columns[j].pe.activation_out_tag;
                        pt=dut.matrixEngine.layers[l].systolicArray.rows[i].columns[j].pe.sum_out_tag;
                        #1;
                        if (activation!==dut.matrixEngine.layers[l].systolicArray.rows[i].columns[j].pe.activation_out ||
                            partial!==dut.matrixEngine.layers[l].systolicArray.rows[i].columns[j].pe.sum_out ||
                            av!==dut.matrixEngine.layers[l].systolicArray.rows[i].columns[j].pe.activation_out_valid ||
                            pv!==dut.matrixEngine.layers[l].systolicArray.rows[i].columns[j].pe.sum_out_valid ||
                            at!==dut.matrixEngine.layers[l].systolicArray.rows[i].columns[j].pe.activation_out_tag ||
                            pt!==dut.matrixEngine.layers[l].systolicArray.rows[i].columns[j].pe.sum_out_tag)
                            $fatal(1,"PE changed during a global stall");
                    end
                end
            end
        end
    end
    always @(posedge clk) begin
        logic signed [RESULT_WIDTH-1:0] held[N];
        bit blocked;
        blocked=rst_n && resultValid && !resultReady;
        for (int j=0; j<N; j++) held[j]=resultData[j];
        #1;
        if (blocked) begin
            if (!resultValid) $fatal(1,"Public valid dropped during backpressure");
            for (int j=0; j<N; j++) if (resultData[j]!==held[j]) $fatal(1,"Public result changed while blocked");
        end
    end
    task automatic dump();
        $fwrite(trace_fd,"%0d %0d %0d",cycle_count,accepted,retired);
        $fwrite(trace_fd," %0d %0d %0d %0d %0d %0d %0d %0d",weightReady,inputReady,resultValid,
            weightsLoaded,reloadReady,dut.trainingBusy,dut.configurationError,dut.arrayAdvance);
        $fwrite(trace_fd," %0d %0d %0d %0d %0d %0d %0d",dut.outstanding,
            dut.matrixEngine.hostLayer,dut.matrixEngine.loadedWeightRows,dut.matrixEngine.loadingWeights,
            dut.matrixEngine.captureWeights,dut.matrixEngine.loadLayer,dut.matrixEngine.loadStep);
        $fwrite(trace_fd," %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
            dut.matrixEngine.inputVectorValid,dut.matrixEngine.inputVectorTag,dut.nextTag,dut.trainingTag,
            dut.trainingState,dut.backwardLayer,dut.backwardRow,dut.backwardColumn,
            $signed(dut.backwardAccumulator),$signed(dut.learningDirection),dut.resultFifo.values,dut.resultTag);
        for (int j=0; j<N; j++) $fwrite(trace_fd," %0d",$signed(resultData[j]));
        for (int l=0; l<LAYERS; l++) for (int i=0; i<N; i++) for (int j=0; j<N; j++)
            $fwrite(trace_fd," %0d",$signed(dut.residentWeights[l][i][j]));
        for (int j=0; j<N; j++) $fwrite(trace_fd," %0d",$signed(dut.residentReductionWeight[j]));
        for (int l=0; l<LAYERS; l++) for (int j=0; j<N; j++) $fwrite(trace_fd," %0d",$signed(dut.sensitivity[l][j]));
        for (int l=0; l<LAYERS; l++) for (int j=0; j<N; j++) $fwrite(trace_fd," %0d",dut.activationGates[l][j]);
        for (int l=0; l<LAYERS; l++) for (int j=0; j<N; j++) $fwrite(trace_fd," %0d",$signed(dut.inputSigns[l][j]));
        for (int j=0; j<N; j++) $fwrite(trace_fd," %0d",$signed(dut.finalSigns[j]));
        for (int l=0; l<LAYERS; l++) for (int j=0; j<N; j++) begin
            if (dut.matrixEngine.chainValid[l+1][j])
                $fwrite(trace_fd," 1 %0d %0d",dut.matrixEngine.chainTag[l+1][j],$signed(dut.matrixEngine.chainData[l+1][j]));
            else $fwrite(trace_fd," 0 0 0");
        end
        $fwrite(trace_fd,"\n");
    endtask
    initial begin
        for (int j=0; j<N; j++) begin weightData[j]=0; inputData[j]=0; reductionWeight[j]=0; end
        targetData=0;
        if (!$value$plusargs("STIMULUS=%s",stimulus_path) || !$value$plusargs("TRACE=%s",trace_path))
            $fatal(1,"STIMULUS and TRACE paths are required");
        stimulus_fd=$fopen(stimulus_path,"r");
        trace_fd=$fopen(trace_path,"w");
        if (!stimulus_fd || !trace_fd) $fatal(1,"Cannot open state-trace files");
        while (!$feof(stimulus_fd)) begin
            @(negedge clk);
            status=$fscanf(stimulus_fd,"%d",fields[0]);
            if (status==1) begin
                for (int field=1; field<11+3*N; field++) begin
                    status=$fscanf(stimulus_fd," %d",fields[field]);
                    if (status!=1) $fatal(1,"Malformed stimulus field %0d",field);
                end
                cycle_count=fields[0]; rst_n=fields[1]; weightValid=fields[2]; inputValid=fields[3+N];
                for (int j=0; j<N; j++) begin
                    weightData[j]=8'(fields[3+j]); inputData[j]=8'(fields[4+N+j]);
                    reductionWeight[j]=8'(fields[7+2*N+j]);
                end
                targetData=8'(fields[4+2*N]); trainingEnable=fields[5+2*N]; resultReady=fields[6+2*N];
                loadReductionWeights=fields[7+3*N]; reloadWeights=fields[8+3*N]; passThrough=fields[9+3*N]; reduceOutput=fields[10+3*N];
                @(posedge clk);
                accepted=rst_n && inputValid && inputReady;
                retired=rst_n && resultValid && resultReady;
                #1; dump();
            end else if (status!=-1) $fatal(1,"Malformed stimulus row %0d fields",status);
        end
        $fclose(stimulus_fd); $fclose(trace_fd);
        $display("PASS: Fountain public stimulus/state trace written");
        $finish;
    end
    initial begin #2000000; $fatal(1,"State trace timeout"); end
endmodule
