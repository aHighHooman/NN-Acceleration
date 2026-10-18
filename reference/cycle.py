"""Clocked Fountain PE network, enabled-step adapters, loader and learning barrier.

Lane arithmetic advances from previous registered neighbors. Whole-vector
functional calculations are used only as an independent result oracle.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from collections.abc import Sequence
from .arithmetic import to_signed, ternary_sign, saturating_lsb_update
from .functional import ReferenceConfig, Sample, analyze, normalize_weights


@dataclass(frozen=True)
class CycleConfig(ReferenceConfig):
    route_delay: int = 0
    join_latency: int = 1
    tag_width: int = 16
    in_flight_depth: int | None = None
    output_fifo_depth: int | None = None

    def __post_init__(self):
        super().__post_init__()
        for name, minimum in (('route_delay',0), ('join_latency',1), ('tag_width',1)):
            value=getattr(self,name)
            if not isinstance(value,int) or isinstance(value,bool) or value<minimum:
                raise ValueError(f'invalid {name}')
        if self.tag_width>30:
            raise ValueError('tag_width must be <=30')
        if self.output_fifo_depth is None:
            object.__setattr__(self,'output_fifo_depth',2*self.n)
        if self.in_flight_depth is None:
            object.__setattr__(self,'in_flight_depth',self.layers*(self.n+self.route_delay+self.join_latency+2)+self.output_fifo_depth+2)
        for name in ('in_flight_depth','output_fifo_depth'):
            value=getattr(self,name)
            if not isinstance(value,int) or isinstance(value,bool) or value<1:
                raise ValueError(f'invalid {name}')
        if self.in_flight_depth>1<<self.tag_width:
            raise ValueError('live tags would alias')

    @property
    def center_latency(self):
        return self.n//2+1+self.route_delay+self.join_latency

    @property
    def sample_context_depth(self):
        return self.in_flight_depth


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
    reduce_output: bool = False


@dataclass(frozen=True)
class Lane:
    value: int
    tag: int


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
    if a.tag!=b.tag:
        raise AssertionError('Fountain tag mismatch')
    return Lane(a.value+b.value,a.tag)


class _Layer:
    def __init__(self, config, index, weights):
        self.config=config
        self.weights=weights
        self.aw=config.width+(index+1)*(config.width+(config.n-1).bit_length())
        n=config.n
        self.act=[[None]*n for _ in range(n)]
        self.sums=[[None]*n for _ in range(n)]
        self.joined=[None]*n
        self.upper=[_Delay(config.route_delay) for _ in range(n)]
        self.lower=[_Delay(config.route_delay) for _ in range(n)]
        self.tail=[_Delay(config.join_latency-1) for _ in range(n)]

    def outputs(self):
        return [delay.read(value) for delay,value in zip(self.tail,self.joined)]

    @property
    def busy(self):
        return any(value is not None for row in self.act+self.sums for value in row) or any(
            value is not None for value in self.joined) or any(
            delay.busy for delay in self.upper+self.lower+self.tail)

    def tick(self, incoming):
        n=self.config.n
        c=n//2
        next_act=[[None]*n for _ in range(n)]
        next_sums=[[None]*n for _ in range(n)]
        next_joined=[]
        for j in range(n):
            upper=self.sums[0][j]
            lower=self.sums[-1][j]
            next_joined.append(_join(self.upper[j].read(upper),self.lower[j].read(lower)))
            self.upper[j].tick(upper)
            self.lower[j].tick(lower)
            self.tail[j].tick(self.joined[j])
        for i in range(n):
            for j in range(n):
                activation=incoming[i] if j==c else self.act[i][j+1 if j<c else j-1]
                seed=i in (c,c-1)
                partial=None if seed else self.sums[i+1 if i<c else i-1][j]
                if not seed:
                    if (activation is None)!=(partial is None):
                        raise AssertionError(f'MAC valid mismatch at {i},{j}')
                    if activation is not None and activation.tag!=partial.tag:
                        raise AssertionError(f'MAC tag mismatch at {i},{j}')
                next_act[i][j]=activation
                if activation is not None:
                    value=activation.value*self.weights[i][j]+(partial.value if partial else 0)
                    assert to_signed(value,self.aw)==value, 'MAC overflow'
                    next_sums[i][j]=Lane(value,activation.tag)
        self.act,self.sums,self.joined=next_act,next_sums,next_joined


@dataclass(frozen=True)
class ResultEntry:
    activated_result: tuple[int,...]
    prediction: int
    tag: int


@dataclass(frozen=True)
class SampleContext:
    target: int
    input_signs: tuple
    gates: tuple


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
        self.mode_reduce=config.reduce_output

    def _hardware_reset(self):
        cfg=self.config
        n,k=cfg.n,cfg.layers
        self._W=[[[0]*n for _ in range(n)] for _ in range(k)]
        self._R=[0]*n
        self.layers=[_Layer(cfg,l,self._W[l]) for l in range(k)]
        self.source=[_Delay(abs(i-n//2)) for i in range(n)]
        self.collector=[_Delay(n//2-abs(i-n//2)) for i in range(n)]
        self.weight_buffer=[[[0]*n for _ in range(n)] for _ in range(k)]
        self.weights_loaded=False
        self.host_layer=self.loaded_rows=self.load_layer=self.load_step=0
        self.loading=self.capture=False
        self.input_stage=None
        self.input_tag=0
        self.next_tag=0
        self.outstanding=0
        self.fifo=[]
        self.training_pending=False
        self.training_tag=0
        self.saved_target=0
        self.input_signs=[[0]*n for _ in range(k)]
        self.gates=[[False]*n for _ in range(k)]
        self.final_signs=[0]*n
        self.sensitivity=[[0]*n for _ in range(k)]
        self.training_state=0
        self.back_layer=self.back_row=self.back_col=self.back_acc=0
        self.direction=0
        self.mode_pass=self.mode_reduce=False
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
        return [[None if lane is None else Lane(lane.value if self.mode_pass or lane.value>=0 else 0,lane.tag)
                 for lane in layer.outputs()] for layer in self.layers]

    def collected(self):
        lanes=[delay.read(value) for delay,value in zip(self.collector,self.lane_outputs()[-1])]
        if any(value is not None for value in lanes):
            assert all(value is not None for value in lanes), 'collector valid mismatch'
            assert len({value.tag for value in lanes})==1, 'collector tag mismatch'
            return lanes
        return None

    @property
    def pipeline_busy(self):
        return self.input_stage is not None or any(delay.busy for delay in self.source+self.collector) or any(layer.busy for layer in self.layers) or self.collected() is not None

    def signals(self,inputs: CycleInputs | None=None):
        inputs=inputs or CycleInputs(pass_through=self.mode_pass,reduce_output=self.mode_reduce)
        collected=self.collected()
        result_valid=bool(self.fifo)
        retire=result_valid and inputs.result_ready
        matrix_reload=self.weights_loaded and not self.pipeline_busy and self.training_state!=2
        quiescent=self.outstanding==0 and not self.training_pending and (not self.weights_loaded or matrix_reload)
        reload_ready=self.weights_loaded and matrix_reload and quiescent and not self.fifo
        reload=inputs.reload_weights and reload_ready
        can_accept=len(self.fifo)<self.config.output_fifo_depth or retire
        advance=self.weights_loaded and not(collected is not None and not can_accept)
        matrix_input=self.weights_loaded and not reload and self.training_state!=2 and (self.input_stage is None or advance)
        capacity=(self.outstanding<self.config.in_flight_depth or retire) and not self.training_pending and self.training_state==0 and not inputs.load_reduction_weights and not reload
        return dict(weight_ready=inputs.reset_n and not self.weights_loaded and not self.loading and not self.capture,
            input_ready=matrix_input and capacity,result_valid=result_valid,
            reload_ready=reload_ready,advance=advance,quiescent=quiescent,
            retire=retire,reload=reload,push=collected is not None and can_accept,
            collected=collected)

    def step(self,inputs: CycleInputs | None=None):
        inputs=inputs or CycleInputs(pass_through=self.mode_pass,reduce_output=self.mode_reduce)
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
            old_weights_loaded=self.weights_loaded
            raw_outputs=[layer.outputs() for layer in self.layers]
            activated=self.lane_outputs()
            head=self.fifo[0] if self.fifo else None
            collected=sig['collected']
            if sig['quiescent']:
                self.mode_pass=inputs.pass_through
                self.mode_reduce=inputs.reduce_output
            elif inputs.pass_through!=self.mode_pass or inputs.reduce_output!=self.mode_reduce:
                self.configuration_error=True
            if inputs.load_reduction_weights:
                if sig['quiescent']:
                    if len(inputs.reduction_weight)!=cfg.n: raise ValueError('readout length')
                    self._R=[to_signed(v,cfg.reduction_weight_width) for v in inputs.reduction_weight]
                else: self.configuration_error=True
            if old_pending and sig['advance']:
                for l in range(cfg.layers):
                    for j,lane in enumerate(raw_outputs[l]):
                        if lane is not None and lane.tag==self.training_tag:
                            self.gates[l][j]=old_pass or lane.value>0
                            sign=ternary_sign(activated[l][j].value)
                            if l<cfg.layers-1: self.input_signs[l+1][j]=sign
                            else: self.final_signs[j]=sign
            if sig['advance']:
                for j,delay in enumerate(self.collector): delay.tick(activated[-1][j])
                source_words=[None if old_input is None else Lane(old_input[0][i],old_input[1]) for i in range(cfg.n)]
                layer_inputs=[[delay.read(word) for delay,word in zip(self.source,source_words)]]+activated[:-1]
                for layer,lane_inputs in zip(self.layers,layer_inputs): layer.tick(lane_inputs)
                for delay,word in zip(self.source,source_words): delay.tick(word)
            if retired:
                self.fifo.pop(0)
                self.timings[head.tag].retired_at=self.cycle
                self.records.append(self.expected[head.tag])
            if sig['push']:
                values=tuple(lane.value for lane in collected)
                tag=collected[0].tag
                prediction=sum(v*r for v,r in zip(values,old_r)) >> cfg.rescale_shift
                record=self.expected[tag]
                assert values==record.activated_result and prediction==record.prediction, 'clocked/functional mismatch'
                self.fifo.append(ResultEntry(values,prediction,tag))
                self.timings[tag].enqueued_at=self.cycle
            if accepted:
                values=tuple(to_signed(value,cfg.width) for value in inputs.input_data)
                if len(values)!=cfg.n: raise ValueError('input length')
                tag=self.next_tag
                self.expected[tag]=analyze(replace(cfg,pass_through=self.mode_pass,reduce_output=self.mode_reduce),self._W,old_r,
                    Sample(values,inputs.target_data,inputs.training_enable),tag)
                self.timings[tag]=SampleTiming(self.cycle)
                self.next_tag=(tag+1) % (1<<cfg.tag_width)
                self.input_stage=(values,tag)
                self.input_tag=tag
                if inputs.training_enable:
                    self.training_pending=True
                    self.training_tag=tag
                    self.saved_target=to_signed(inputs.target_data,cfg.target_width)
                    self.input_signs[0]=[ternary_sign(v) for v in values]
            elif old_input is not None and sig['advance']:
                self.input_stage=None
            self.outstanding+=int(accepted)-int(retired)
            if old_state==0 and retired and old_pending and head.tag==self.training_tag:
                self.direction=ternary_sign(self.saved_target-head.prediction)
                self.sensitivity[-1]=[r if gate else 0 for r,gate in zip(old_r,self.gates[-1])]
                self.back_layer=max(0,cfg.layers-2)
                self.back_row=self.back_col=self.back_acc=0
                self.training_state=2 if cfg.layers==1 or self.direction==0 else 1
            elif old_state==1 and sig['advance']:
                value=(self.back_acc if self.back_col else 0)+self._W[self.back_layer+1][self.back_row][self.back_col]*self.sensitivity[self.back_layer+1][self.back_col]
                if self.back_col==cfg.n-1:
                    self.sensitivity[self.back_layer][self.back_row]=value if self.gates[self.back_layer][self.back_row] else 0
                    self.back_col=self.back_acc=0
                    if self.back_row==cfg.n-1:
                        self.back_row=0
                        if self.back_layer==0: self.training_state=2
                        else: self.back_layer-=1
                    else: self.back_row+=1
                else:
                    self.back_acc=value
                    self.back_col+=1
            elif old_state==2 and sig['advance']:
                record=self.expected[self.training_tag]
                for l in range(cfg.layers):
                    for i in range(cfg.n):
                        for j in range(cfg.n):
                            direction=self.input_signs[l][i]*self.direction*ternary_sign(self.sensitivity[l][j])
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
        snapshot=CycleSnapshot(self.cycle,accepted,retired,self.state_vector(inputs))
        self.snapshots.append(snapshot)
        self.cycle+=1
        return snapshot

    def state_vector(self,inputs=None):
        sig=self.signals(inputs)
        head=self.fifo[0] if self.fifo else ResultEntry((0,)*self.config.n,0,0)
        result=(head.prediction,)+(0,)*(self.config.n-1) if self.mode_reduce else head.activated_result
        state=[int(sig[name]) for name in ('weight_ready','input_ready','result_valid')]
        state += [int(self.weights_loaded),int(sig['reload_ready']),int(self.training_pending),int(self.configuration_error),int(sig['advance']),
            self.outstanding,self.host_layer,self.loaded_rows,int(self.loading),int(self.capture),self.load_layer,self.load_step,
            int(self.input_stage is not None),self.input_tag,self.next_tag,self.training_tag,
            self.training_state,self.back_layer,self.back_row,self.back_col,self.back_acc,self.direction,len(self.fifo),head.tag,*result]
        state += [v for matrix in self._W for row in matrix for v in row]
        state += self._R
        state += [v for row in self.sensitivity for v in row]
        state += [int(v) for row in self.gates for v in row]
        state += [v for row in self.input_signs for v in row]
        state += self.final_signs
        for layer in self.lane_outputs():
            for value in layer:
                state.extend((0,0,0) if value is None else (1,value.tag,value.value))
        return tuple(state)

    def run(self,cycles: Sequence[CycleInputs]):
        return [self.step(value) for value in cycles]
