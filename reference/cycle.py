"""Clocked Fountain PE network, enabled-step adapters, loader and learning barrier.

Lane arithmetic advances from previous registered neighbors. Whole-vector
functional calculations are used only as an independent result oracle. Each
lane carries a model-side sample index for bookkeeping and self-checks; the
hardware carries no identity beyond its fixed lane front.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from collections.abc import Sequence
from .arithmetic import to_signed, ternary_sign, saturating_lsb_update
from .functional import ReferenceConfig, Sample, analyze, normalize_weights


@dataclass(frozen=True)
class CycleConfig(ReferenceConfig):
    in_flight_depth: int | None = None
    output_fifo_depth: int | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.output_fifo_depth is None:
            object.__setattr__(self,'output_fifo_depth',2*self.n)
        if self.in_flight_depth is None:
            object.__setattr__(self,'in_flight_depth',self.layers*(self.n+2)+self.output_fifo_depth+2)
        for name in ('in_flight_depth','output_fifo_depth'):
            value=getattr(self,name)
            if not isinstance(value,int) or isinstance(value,bool) or value<1:
                raise ValueError(f'invalid {name}')

    @property
    def center_latency(self):
        return self.n//2+1

    @property
    def backward_cycles(self):
        # One edge to switch the arrays to their transposed weights, then the
        # longest path through LAYERS-1 lane registers and the PE registers, then
        # one capture edge.
        return self.layers*(self.center_latency+1)


@dataclass(frozen=True)
class CycleInputs:
    input_valid: bool = False
    input_data: tuple[int,...] = ()
    target_data: int = 0
    training_enable: bool = False
    result_ready: bool = True
    weight_valid: bool = False
    weight_data: tuple[int,...] = ()
    reduction_weight: tuple[int,...] = ()
    load_reduction_weights: bool = False
    reload_weights: bool = False
    reset_n: bool = True
    pass_through: bool = True


@dataclass(frozen=True)
class Lane:
    value: int
    sample: int


class _Delay:
    def __init__(self, depth):
        self.words=[None]*depth

    def read(self, incoming):
        return self.words[-1] if self.words else incoming

    def tick(self, incoming):
        if self.words:
            self.words=[incoming]+self.words[:-1]

    @property
    def busy(self):
        return any(word is not None for word in self.words)


def _join(a,b):
    if (a is None)!=(b is None):
        raise AssertionError('Fountain valid mismatch')
    if a is None:
        return None
    if a.sample!=b.sample:
        raise AssertionError('Fountain sample mismatch')
    return Lane(a.value+b.value,a.sample)


class _Layer:
    def __init__(self, config, index, weights):
        self.config=config
        self.weights=weights
        self.sum_width=config.sum_width
        n=config.n
        self.act=[[None]*n for _ in range(n)]
        self.sums=[[None]*n for _ in range(n)]

    def outputs(self):
        return [_join(self.sums[0][j],self.sums[-1][j]) for j in range(self.config.n)]

    @property
    def busy(self):
        return any(value is not None for row in self.act+self.sums for value in row)

    def tick(self, incoming):
        n=self.config.n
        c=n//2
        next_act=[[None]*n for _ in range(n)]
        next_sums=[[None]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                activation=incoming[i] if j==c else self.act[i][j+1 if j<c else j-1]
                seed=i in (c,c-1)
                partial=None if seed else self.sums[i+1 if i<c else i-1][j]
                if not seed:
                    if (activation is None)!=(partial is None):
                        raise AssertionError(f'MAC valid mismatch at {i},{j}')
                    if activation is not None and activation.sample!=partial.sample:
                        raise AssertionError(f'MAC sample mismatch at {i},{j}')
                next_act[i][j]=activation
                if activation is not None:
                    value=activation.value*self.weights[i][j]+(partial.value if partial else 0)
                    assert to_signed(value,self.sum_width)==value, 'MAC overflow'
                    next_sums[i][j]=Lane(value,activation.sample)
        self.act,self.sums=next_act,next_sums

    def begin_backward(self):
        # The RTL's PE registers still hold forward leftovers here. Poison stands
        # in for them, so stopping before everything settles fails the oracle.
        n=self.config.n
        self.back_act=[[1<<self.sum_width]*n for _ in range(n)]
        self.back_sums=[[1<<self.sum_width]*n for _ in range(n)]

    def gradient_outputs(self):
        return [self.back_sums[0][j]+self.back_sums[-1][j] for j in range(self.config.n)]

    def tick_backward(self, incoming):
        # Forward's routes, with each PE (i,j) multiplying by its transposed copy W[j][i].
        n=self.config.n
        c=n//2
        next_act=[[0]*n for _ in range(n)]
        next_sums=[[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                gradient=incoming[i] if j==c else self.back_act[i][j+1 if j<c else j-1]
                partial=0 if i in (c,c-1) else self.back_sums[i+1 if i<c else i-1][j]
                next_act[i][j]=gradient
                next_sums[i][j]=gradient*self.weights[j][i]+partial
        self.back_act,self.back_sums=next_act,next_sums


class _Reduction:
    """Nx1 column: each row takes one lane of the last layer on its own front."""
    def __init__(self, config):
        self.config=config
        self.sums=[None]*config.n

    def output(self):
        return _join(self.sums[0],self.sums[-1])

    @property
    def busy(self):
        return any(value is not None for value in self.sums)

    def tick(self, incoming, weights):
        n=self.config.n
        c=n//2
        next_sums=[None]*n
        for i,activation in enumerate(incoming):
            seed=i in (c,c-1)
            partial=None if seed else self.sums[i+1 if i<c else i-1]
            if not seed:
                if (activation is None)!=(partial is None):
                    raise AssertionError(f'readout valid mismatch at {i}')
                if activation is not None and activation.sample!=partial.sample:
                    raise AssertionError(f'readout sample mismatch at {i}')
            if activation is not None:
                next_sums[i]=Lane(activation.value*weights[i]+(partial.value if partial else 0),activation.sample)
        self.sums=next_sums


@dataclass(frozen=True)
class ResultEntry:
    prediction: int
    sample: int


@dataclass
class SampleTiming:
    accepted_at: int
    enqueued_at: int | None = None
    retired_at: int | None = None


@dataclass(frozen=True)
class CycleSnapshot:
    cycle: int
    accepted: bool
    retired: bool
    state: tuple[int,...]


class CycleReference:
    def __init__(self,config: CycleConfig,W=None,R=None):
        self.config=config
        self.cycle=0
        self.snapshots=[]
        self._hardware_reset()
        if W is not None:
            initial=normalize_weights(config,W)
            for l in range(config.layers):
                for i in range(config.n): self._W[l][i][:]=initial[l][i]
            self.weights_loaded=True
        if R is not None:
            if len(R)!=config.n: raise ValueError('readout length must equal n')
            self._R=[to_signed(value,config.reduction_weight_width) for value in R]
        self.mode_pass=config.pass_through

    def _hardware_reset(self):
        cfg=self.config
        n,k=cfg.n,cfg.layers
        self._W=[[[0]*n for _ in range(n)] for _ in range(k)]
        self._R=[0]*n
        self.layers=[_Layer(cfg,l,self._W[l]) for l in range(k)]
        self.source=[_Delay(abs(i-n//2)) for i in range(n)]
        self.reduction=_Reduction(cfg)
        # The register after each array's lane mux. Index 0 is unused: array 0
        # reads the source directly. The readout has one too.
        self.held=[[None]*n for _ in range(k)]
        self.back_held=[[0]*n for _ in range(k)]
        self.readout_input=[None]*n
        self.array_backward=False
        self.weight_buffer=[[[0]*n for _ in range(n)] for _ in range(k)]
        self.weights_loaded=False
        self.host_layer=self.loaded_rows=self.load_layer=self.load_step=0
        self.loading=self.capture=False
        self.input_stage=None
        self.next_sample=0
        self.outstanding=0
        self.fifo=[]
        self.training_pending=False
        self.training_sample=0
        self.saved_target=0
        self.input_signs=[[0]*n for _ in range(k)]
        self.gates=[[False]*n for _ in range(k)]
        self.final_signs=[0]*n
        self.gradient=[[0]*n for _ in range(k)]
        self.training_state=0
        self.back_count=0
        self.direction=0
        self.mode_pass=False
        self.configuration_error=False
        self.expected={}
        self.timings={}
        self.records=[]

    @property
    def W(self):
        return [[row[:] for row in matrix] for matrix in self._W]

    @property
    def R(self):
        return self._R[:]

    def lane_outputs(self):
        """Each layer's lanes after ReLU and requantization, as the next layer sees them."""
        return [[None if lane is None else Lane(self.config.requantize(lane.value if self.mode_pass or lane.value>=0 else 0),lane.sample)
                 for lane in layer.outputs()] for layer in self.layers]

    def gradient_lanes(self,readout):
        """The gradient vector at each layer's output, as the wires show it now."""
        lanes=[None]*self.config.layers
        for l in range(self.config.layers):
            source=readout if l==self.config.layers-1 else [self.config.requantize(value) for value in self.layers[l+1].gradient_outputs()]
            lanes[l]=[value if gate else 0 for value,gate in zip(source,self.gates[l])]
        return lanes

    @property
    def pipeline_busy(self):
        held=any(lane is not None for lanes in self.held+[self.readout_input] for lane in lanes)
        return self.input_stage is not None or held or any(delay.busy for delay in self.source) or any(layer.busy for layer in self.layers) or self.reduction.busy

    def signals(self,inputs: CycleInputs | None=None):
        inputs=inputs or CycleInputs(pass_through=self.mode_pass)
        output=self.reduction.output()
        result_valid=bool(self.fifo)
        retire=result_valid and inputs.result_ready
        matrix_reload=self.weights_loaded and not self.pipeline_busy and self.training_state!=2
        quiescent=self.outstanding==0 and not self.training_pending and (not self.weights_loaded or matrix_reload)
        reload_ready=self.weights_loaded and matrix_reload and quiescent and not self.fifo
        reload=inputs.reload_weights and reload_ready
        can_accept=len(self.fifo)<self.config.output_fifo_depth or retire
        advance=self.weights_loaded and not(output is not None and not can_accept)
        matrix_input=self.weights_loaded and not reload and self.training_state!=2 and (self.input_stage is None or advance)
        capacity=(self.outstanding<self.config.in_flight_depth or retire) and not self.training_pending and self.training_state==0 and not inputs.load_reduction_weights and not reload
        return dict(weight_ready=inputs.reset_n and not self.weights_loaded and not self.loading and not self.capture,
            input_ready=matrix_input and capacity,result_valid=result_valid,
            reload_ready=reload_ready,advance=advance,quiescent=quiescent,
            retire=retire,reload=reload,push=output is not None and can_accept,
            output=output)

    def step(self,inputs: CycleInputs | None=None):
        inputs=inputs or CycleInputs(pass_through=self.mode_pass)
        cfg=self.config
        sig=self.signals(inputs)
        accepted=bool(inputs.reset_n and inputs.input_valid and sig['input_ready'])
        retired=bool(inputs.reset_n and sig['retire'])
        if not inputs.reset_n:
            self._hardware_reset()
        else:
            old_pending=self.training_pending
            old_state=self.training_state
            old_pass=self.mode_pass
            old_r=self._R[:]
            old_input=self.input_stage
            old_outstanding=self.outstanding
            raw_outputs=[layer.outputs() for layer in self.layers]
            activated=self.lane_outputs()
            head=self.fifo[0] if self.fifo else None
            if sig['quiescent']:
                self.mode_pass=inputs.pass_through
            elif inputs.pass_through!=self.mode_pass:
                self.configuration_error=True
            if inputs.load_reduction_weights:
                if sig['quiescent']:
                    if len(inputs.reduction_weight)!=cfg.n: raise ValueError('readout length')
                    self._R=[to_signed(v,cfg.reduction_weight_width) for v in inputs.reduction_weight]
                else: self.configuration_error=True
            # The pending training sample is the last one admitted, so the last
            # valid value seen at each tap is its own.
            if old_pending and sig['advance']:
                for l in range(cfg.layers):
                    for j,lane in enumerate(raw_outputs[l]):
                        if lane is not None:
                            self.gates[l][j]=old_pass or lane.value>0
                            sign=ternary_sign(activated[l][j].value)
                            if l<cfg.layers-1: self.input_signs[l+1][j]=sign
                            else: self.final_signs[j]=sign
            if sig['advance']:
                for j,lane in enumerate(activated[-1]):
                    if lane is not None:
                        assert lane.value==self.expected[lane.sample].activated_result[j], 'clocked/functional mismatch'
                self.reduction.tick(self.readout_input,old_r)
                self.readout_input=activated[-1]
                source_words=[None if old_input is None else Lane(old_input[0][i],old_input[1]) for i in range(cfg.n)]
                layer_inputs=[[delay.read(word) for delay,word in zip(self.source,source_words)]]+self.held[1:]
                for layer,lane_inputs in zip(self.layers,layer_inputs): layer.tick(lane_inputs)
                for delay,word in zip(self.source,source_words): delay.tick(word)
                self.held=[[None]*cfg.n]+activated[:-1]
            if retired:
                self.fifo.pop(0)
                self.timings[head.sample].retired_at=self.cycle
                self.records.append(self.expected[head.sample])
            if sig['push']:
                sample=sig['output'].sample
                prediction=sig['output'].value >> cfg.rescale_shift
                assert prediction==self.expected[sample].prediction, 'clocked/functional mismatch'
                self.fifo.append(ResultEntry(prediction,sample))
                self.timings[sample].enqueued_at=self.cycle
            if accepted:
                values=tuple(to_signed(value,cfg.activation_width) for value in inputs.input_data)
                if len(values)!=cfg.n: raise ValueError('input length')
                sample=self.next_sample
                self.expected[sample]=analyze(replace(cfg,pass_through=self.mode_pass),self._W,old_r,
                    Sample(values,inputs.target_data,inputs.training_enable),sample)
                self.timings[sample]=SampleTiming(self.cycle)
                self.next_sample=sample+1
                self.input_stage=(values,sample)
                if inputs.training_enable:
                    self.training_pending=True
                    self.training_sample=sample
                    self.saved_target=to_signed(inputs.target_data,cfg.target_width)
                    self.input_signs[0]=[ternary_sign(v) for v in values]
            elif old_input is not None and sig['advance']:
                self.input_stage=None
            self.outstanding+=int(accepted)-int(retired)
            if old_state==0 and retired and old_pending and old_outstanding==1:
                assert head.sample==self.training_sample, 'training sample did not retire last'
                self.direction=ternary_sign(self.saved_target-head.prediction)
                for layer in self.layers: layer.begin_backward()
                self.back_held=[[1<<cfg.sum_width]*cfg.n for _ in range(cfg.layers)]
                self.gradient=self.gradient_lanes(old_r)
                self.back_count=0
                self.training_state=2 if cfg.layers==1 or self.direction==0 else 1
            elif old_state==1 and sig['advance']:
                self.gradient=self.gradient_lanes(old_r)
                for l in range(1,cfg.layers):
                    # Until the registered select switches, the arrays still run
                    # forward on leftovers, which poison stands in for.
                    if self.array_backward: self.layers[l].tick_backward(self.back_held[l])
                    else: self.layers[l].begin_backward()
                self.back_held=[lanes[:] if self.array_backward else [1<<cfg.sum_width]*cfg.n for lanes in self.gradient]
                self.back_count+=1
                if self.back_count==cfg.backward_cycles: self.training_state=2
            elif old_state==2 and sig['advance']:
                record=self.expected[self.training_sample]
                for l in range(cfg.layers):
                    for i in range(cfg.n):
                        for j in range(cfg.n):
                            direction=self.input_signs[l][i]*self.direction*ternary_sign(self.gradient[l][j])
                            assert direction==record.matrix_update_directions[l][i][j]
                            self._W[l][i][j]=saturating_lsb_update(self._W[l][i][j],direction,cfg.width)
                self._R=[saturating_lsb_update(r,self.direction*sign,cfg.reduction_weight_width) for r,sign in zip(old_r,self.final_signs)]
                self.training_pending=False
                self.training_state=0
            # Loader writes after all pre-edge PE reads, exactly like a clocked bank.
            if self.capture:
                self.capture=False
                self.weights_loaded=True
                self.host_layer=0
            elif self.loading:
                self._W[self.load_layer][self.load_step][:]=self.weight_buffer[self.load_layer][self.load_step]
                if self.load_step==cfg.n-1:
                    self.load_step=0
                    if self.load_layer==cfg.layers-1:
                        self.loading=False
                        self.capture=True
                    else: self.load_layer+=1
                else: self.load_step+=1
            elif sig['weight_ready'] and inputs.weight_valid:
                if len(inputs.weight_data)!=cfg.n: raise ValueError('weight row length')
                self.weight_buffer[self.host_layer][cfg.n-1-self.loaded_rows]=[to_signed(v,cfg.width) for v in inputs.weight_data]
                if self.loaded_rows==cfg.n-1:
                    self.loaded_rows=0
                    if self.host_layer==cfg.layers-1:
                        self.loading=True
                        self.load_layer=self.load_step=0
                    else: self.host_layer+=1
                else: self.loaded_rows+=1
            if sig['reload']:
                self.weights_loaded=False
                self.host_layer=self.loaded_rows=0
            self.array_backward=old_state==1
        snapshot=CycleSnapshot(self.cycle,accepted,retired,self.state_vector(inputs))
        self.snapshots.append(snapshot)
        self.cycle+=1
        return snapshot

    def state_vector(self,inputs=None):
        """Architectural state compared with the RTL on every edge: public
        handshakes, the training barrier, result data, resident parameters, the
        per-lane raw fronts and the readout output. Controller and loader
        internals stay private."""
        sig=self.signals(inputs)
        head=self.fifo[0].prediction if self.fifo else 0
        state=[int(sig[name]) for name in ('weight_ready','input_ready','result_valid')]
        state += [int(self.weights_loaded),int(sig['reload_ready']),int(self.training_pending),head]
        state += [v for matrix in self._W for row in matrix for v in row]
        state += self._R
        for layer in self.lane_outputs():
            for value in layer:
                state.extend((0,0) if value is None else (1,value.value))
        output=sig['output']
        state.extend((0,0) if output is None else (1,output.value >> self.config.rescale_shift))
        return tuple(state)

    def run(self,cycles: Sequence[CycleInputs]):
        return [self.step(value) for value in cycles]
