`timescale 1ns/1ps

// Accelerator-level interface used by the Phase 5F UVM test.  Debug signals
// are observation-only connections to architectural queues and update stages.
interface nn_training_if #(
    parameter int WIDTH = 8,
    parameter int N = 3,
    parameter int TARGET_WIDTH = WIDTH,
    parameter int REDUCTION_WEIGHT_WIDTH = 8
);
    localparam int PREDICTION_WIDTH = 2*WIDTH + 2*$clog2(N);
    localparam int UPDATE_STAGES = 2*N - 1;

    logic clk;
    logic rst_n;
    logic signed [WIDTH-1:0] weightData[N];
    logic weightValid, weightReady;
    logic signed [WIDTH-1:0] activationData[N];
    logic signed [TARGET_WIDTH-1:0] targetData, resultTargetData;
    logic trainingEnable, activationValid, activationReady;
    logic signed [REDUCTION_WEIGHT_WIDTH-1:0] reductionWeight[N];
    logic loadReductionWeights, reduceOutput;
    logic signed [PREDICTION_WIDTH-1:0] resultData[N];
    logic signed [1:0] learningDirection;
    logic signed [1:0] rowDirection[N], columnDirection[N];
    logic matrixUpdateValid, resultValid, resultReady, resultLast;
    logic weightsLoaded, reloadWeights, reloadReady, passThrough;

    logic samplePush, samplePop, bufferedTrainingEnable;
    logic matrixUpdateAccepted, matrixUpdateComplete, arrayAdvance;
    logic reductionUpdateEmpty;
    logic [UPDATE_STAGES-1:0] updateStageValid;
    logic signed [WIDTH-1:0] residentMatrixWeight[N][N];
    logic signed [REDUCTION_WEIGHT_WIDTH-1:0] residentReductionWeight[N];

    property p_training_enable_is_buffered;
        @(posedge clk) disable iff (!rst_n)
            matrixUpdateValid == (samplePop && bufferedTrainingEnable);
    endproperty
    assert property (p_training_enable_is_buffered)
        else $error("NN_UVM_TRAIN_ALIGN: update valid did not use buffered trainingEnable");

    property p_no_commit_during_array_stall;
        @(posedge clk) disable iff (!rst_n)
            !arrayAdvance |-> !matrixUpdateComplete;
    endproperty
    assert property (p_no_commit_during_array_stall)
        else $error("NN_UVM_UPDATE_STALL: matrix update completed while arrayAdvance was low");

    property p_update_stages_stable_during_array_stall;
        @(posedge clk) disable iff (!rst_n)
            !arrayAdvance && (updateStageValid != '0) |=>
            $stable(updateStageValid);
    endproperty
    assert property (p_update_stages_stable_during_array_stall)
        else $error("NN_UVM_UPDATE_STALL: update stage occupancy moved while arrayAdvance was low");

    generate
        for (genvar lane = 0; lane < N; lane++) begin : stable_output_assertions
            property p_result_tuple_stable;
                @(posedge clk) disable iff (!rst_n)
                    resultValid && !resultReady |=>
                    resultValid && $stable(resultData[lane]) &&
                    $stable(resultTargetData) &&
                    $stable(bufferedTrainingEnable) &&
                    $stable(rowDirection[lane]);
            endproperty
            assert property (p_result_tuple_stable)
                else $error("NN_UVM_TRAIN_STABLE: stalled sample tuple changed at lane %0d", lane);

            property p_reduction_weight_stable_during_array_stall;
                @(posedge clk) disable iff (!rst_n)
                    !arrayAdvance && (updateStageValid != '0) |=>
                    $stable(residentReductionWeight[lane]);
            endproperty
            assert property (p_reduction_weight_stable_during_array_stall)
                else $error("NN_UVM_REDUCTION_STALL: resident reduction weight %0d changed", lane);
        end
    endgenerate

endinterface
