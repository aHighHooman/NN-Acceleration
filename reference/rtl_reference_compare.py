"""Generate public stimuli and compare every semantic RTL edge with Fountain."""
from __future__ import annotations
import argparse
from dataclasses import replace
from pathlib import Path
from random import Random
from .cycle import CycleConfig,CycleInputs,CycleReference
from .functional import Sample,analyze

N=3
WIDTH=8
TAG_WIDTH=5
IN_FLIGHT_DEPTH=16
OUTPUT_FIFO_DEPTH=2


def configuration(layers=3,fraction_bits=0,g=0,j=1,n=N):
    return CycleConfig(n=n,width=WIDTH,layers=layers,fraction_bits=fraction_bits,
        target_width=8,reduction_weight_width=8,tag_width=TAG_WIDTH,
        in_flight_depth=IN_FLIGHT_DEPTH,output_fifo_depth=OUTPUT_FIFO_DEPTH,
        route_delay=g,join_latency=j)


def encode(cycle,inputs,n=N):
    return [cycle,int(inputs.reset_n),int(inputs.weight_valid),*(inputs.weight_data or (0,)*n),
        int(inputs.input_valid),*(inputs.input_data or (0,)*n),inputs.target_data,
        int(inputs.training_enable),int(inputs.result_ready),*(inputs.reduction_weight or (0,)*n),
        int(inputs.load_reduction_weights),int(inputs.reload_weights),int(inputs.pass_through),int(inputs.reduce_output)]


def decode(row,n=N):
    if len(row)!=11+3*n: raise ValueError('wrong stimulus column count')
    return row[0],CycleInputs(reset_n=bool(row[1]),weight_valid=bool(row[2]),weight_data=tuple(row[3:3+n]),
        input_valid=bool(row[3+n]),input_data=tuple(row[4+n:4+2*n]),target_data=row[4+2*n],training_enable=bool(row[5+2*n]),
        result_ready=bool(row[6+2*n]),reduction_weight=tuple(row[7+2*n:7+3*n]),load_reduction_weights=bool(row[7+3*n]),
        reload_weights=bool(row[8+3*n]),pass_through=bool(row[9+3*n]),reduce_output=bool(row[10+3*n]))


class Driver:
    def __init__(self,config):
        self.model=CycleReference(config)
        self.lines=[]
        self.pass_through=True
        self.reduce_output=False
        self.scenarios=[]

    def tick(self,**kwargs):
        if 'pass_through' in kwargs: self.pass_through=kwargs['pass_through']
        if 'reduce_output' in kwargs: self.reduce_output=kwargs['reduce_output']
        values=dict(pass_through=self.pass_through,reduce_output=self.reduce_output)
        values.update(kwargs)
        n=self.model.config.n
        for field in ('input_data','reduction_weight'):
            if field in values and len(values[field])!=n:
                pattern=values[field]
                values[field]=tuple(pattern[i%len(pattern)] for i in range(n))
        inputs=CycleInputs(**values)
        self.lines.append(encode(self.model.cycle,inputs,self.model.config.n))
        return self.model.step(inputs)

    def idle(self,count=1,**kwargs):
        for _ in range(count): self.tick(**kwargs)

    def mark(self,name):
        self.scenarios.append((name,self.model.cycle))

    def drain(self):
        for _ in range(2000):
            if self.model.signals()['reload_ready']: return
            self.tick()
        raise AssertionError('driver drain timeout')

    def load(self,weights,readout,reset=True):
        if reset: self.idle(2,reset_n=False)
        for layer,matrix in enumerate(weights):
            for row in range(self.model.config.n-1,-1,-1):
                assert self.model.signals()['weight_ready']
                self.tick(weight_valid=True,weight_data=tuple(matrix[row]),reduction_weight=tuple(readout),
                    load_reduction_weights=layer==0 and row==self.model.config.n-1)
                if (row+layer)%2: self.idle(2)
        for _ in range(len(weights)*self.model.config.n+4):
            if self.model.weights_loaded: break
            self.tick()
        assert self.model.weights_loaded and self.model.W==weights

    def offer(self,x,target=0,training=False,ready=True):
        for _ in range(2000):
            result_ready=ready(self.model.cycle) if callable(ready) else ready
            snapshot=self.tick(input_valid=True,input_data=tuple(x),target_data=target,
                training_enable=training,result_ready=result_ready)
            if snapshot.accepted: return
        raise AssertionError('driver admission timeout')


def build(config):
    rng=Random(19110)
    n=config.n
    driver=Driver(config)
    def weights():
        return [[[rng.randint(-12,12) for _ in range(n)] for _ in range(n)] for _ in range(config.layers)]
    driver.mark('reverse row loading with gaps')
    driver.load(weights(),[64,-48,32])
    driver.mark('continuous inference and tag wrap')
    for _ in range(85): driver.offer([rng.randint(-128,127) for _ in range(n)])
    driver.drain()
    driver.mark('bubbles')
    for index in range(28):
        driver.offer([rng.randint(-128,127) for _ in range(n)])
        driver.idle(index%3)
    driver.drain()
    driver.mark('long downstream stall and simultaneous FIFO pop/push')
    # Four admissions fill the two-entry sink even for the shortest profile,
    # without asking a blocked producer to exceed physical pipeline capacity.
    for _ in range(4): driver.offer((2,-3,7),ready=False)
    driver.idle(40,result_ready=False)
    assert not driver.model.signals(CycleInputs(result_ready=False))['advance']
    for _ in range(20): driver.offer((3,-1,4),ready=lambda t:t%4!=0)
    driver.drain()
    driver.mark('busy mode/readout changes rejected')
    before=driver.model.R
    driver.offer((1,2,3))
    driver.tick(pass_through=False,load_reduction_weights=True,reduction_weight=(1,1,1))
    driver.pass_through=True
    driver.drain()
    assert driver.model.configuration_error and driver.model.R==before
    driver.mark('quiescent scalar mode and ReLU')
    driver.tick(pass_through=False,reduce_output=True)
    for _ in range(15): driver.offer([rng.randint(-8,8) for _ in range(n)])
    driver.drain()
    driver.mark('reload wins over simultaneous input')
    snapshot=driver.tick(reload_weights=True,input_valid=True,input_data=(1,2,3))
    assert not snapshot.accepted and not driver.model.weights_loaded
    driver.load(weights(),[-128,127,-1],reset=False)
    for _ in range(12): driver.offer((-128,127,-128))
    driver.drain()
    driver.mark('reset while output and pipeline are frozen')
    for _ in range(4): driver.offer((1,2,3),ready=False)
    driver.idle(30,result_ready=False)
    assert not driver.model.signals(CycleInputs(result_ready=False))['advance']
    driver.tick(reset_n=False,result_ready=False)
    driver.idle(20)
    assert not driver.model.pipeline_busy and not driver.model.fifo
    driver.mark('full sensitivity learning and hidden sign cancellation')
    identity=[[int(i==j) for j in range(n)] for i in range(n)]
    matrices=[[row[:] for row in identity] for _ in range(config.layers)]
    matrices[-1][0][0]=2
    matrices[-1][0][1]=-1
    matrices[-1][1][1]=2
    driver.pass_through=False
    driver.reduce_output=False
    driver.load(matrices,[1,3]+[0]*(n-2))
    for target in (1,-1,127):
        driver.offer((1,1,1),target=target,training=True)
        # Hold the next valid input through the complete learning barrier.
        driver.offer((2,1,3))
        driver.drain()
    driver.mark('zero error, zero activation and gated learning')
    driver.offer((0,0,0),target=0,training=True)
    driver.drain()
    driver.mark('reset during backward sensitivity calculation')
    driver.offer((2,1,3),target=-128,training=True)
    if config.layers>1:
        for _ in range(100):
            if driver.model.training_state==1: break
            driver.tick()
        assert driver.model.training_state==1
    else:
        while not driver.model.fifo: driver.tick()
    driver.tick(reset_n=False)
    driver.pass_through=True
    driver.reduce_output=True
    driver.mark('negative extrema and saturation')
    matrices=[[[(-128 if (i+j+l)%2 else 127) for j in range(n)] for i in range(n)] for l in range(config.layers)]
    driver.load(matrices,[-128,127,-128])
    for x in ((-128,127,-128),(127,-128,127),(1,-1,1)):
        driver.offer(x,target=127,training=True)
        driver.drain()
    driver.offer((1,2,3))
    driver.drain()
    return driver


def labels(config):
    n=config.n
    result=['weightReady','inputReady','resultValid','weightsLoaded','reloadReady','trainingBusy',
        'configurationError','arrayAdvance','outstanding','hostLayer','loadedWeightRows','loadingWeights',
        'captureWeights','loadLayer','loadStep','inputVectorValid','inputVectorTag','nextTag','trainingTag',
        'trainingState','backwardLayer','backwardRow','backwardColumn','backwardAccumulator','learningDirection',
        'resultFifo.values','resultTag']
    result += [f'resultData[{i}]' for i in range(n)]
    result += [f'W[{l}][{i}][{j}]' for l in range(config.layers) for i in range(n) for j in range(n)]
    result += [f'r[{i}]' for i in range(n)]
    for name in ('sensitivity','activationGates','inputSigns'):
        result += [f'{name}[{l}][{i}]' for l in range(config.layers) for i in range(n)]
    result += [f'finalSigns[{i}]' for i in range(n)]
    result += [f'lane[{l}][{i}].{field}' for l in range(config.layers) for i in range(n) for field in ('valid','tag','value')]
    return ['cycle','accepted','retired']+result


def compare(stimulus,trace,config):
    model=CycleReference(config)
    stimuli=[decode([int(v) for v in line.split()],config.n) for line in Path(stimulus).read_text().splitlines() if line.strip()]
    observed=[[int(v) for v in line.split()] for line in Path(trace).read_text().splitlines() if line.strip()]
    if len(stimuli)!=len(observed): raise AssertionError(f'trace length {len(observed)} != {len(stimuli)}')
    names=labels(config)
    accepted=retired=0
    for (cycle,inputs),actual in zip(stimuli,observed):
        snapshot=model.step(inputs)
        expected=[cycle,int(snapshot.accepted),int(snapshot.retired),*snapshot.state]
        if len(expected)!=len(names): raise AssertionError('label count mismatch')
        if len(actual)!=len(expected): raise AssertionError(f'cycle {cycle}: column count {len(actual)} != {len(expected)}')
        for field,got,want in zip(names,actual,expected):
            if got!=want: raise AssertionError(f'cycle {cycle} {field}: RTL={got} model={want}')
        accepted+=snapshot.accepted
        retired+=snapshot.retired
    print(f'PASS: {len(observed)} RTL edges, {len(names)} state fields, {accepted} admissions, {retired} retirements; '
          f'N={config.n} LAYERS={config.layers} F={config.fraction_bits} G={config.route_delay} J={config.join_latency}.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('generate','compare'))
    parser.add_argument('stimulus')
    parser.add_argument('trace',nargs='?')
    parser.add_argument('--layers',type=int,default=3)
    parser.add_argument('--n',type=int,default=3)
    parser.add_argument('--fraction-bits',type=int,default=0)
    parser.add_argument('--g',type=int,default=0)
    parser.add_argument('--j',type=int,default=1)
    args=parser.parse_args()
    config=configuration(args.layers,args.fraction_bits,args.g,args.j,args.n)
    if args.command=='generate':
        driver=build(config)
        Path(args.stimulus).parent.mkdir(parents=True,exist_ok=True)
        Path(args.stimulus).write_text(''.join(' '.join(map(str,row))+'\n' for row in driver.lines))
        print(f'Generated {len(driver.lines)} edges in {len(driver.scenarios)} directed Fountain scenarios.')
    elif args.trace is None: parser.error('compare requires a trace file')
    else: compare(args.stimulus,args.trace,config)


if __name__=='__main__': main()
