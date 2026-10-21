"""Clocked neighbor dependencies, timing, control, and learning barrier tests."""
from dataclasses import replace
import random
import unittest
from reference.cycle import CycleConfig,CycleInputs,CycleReference,Lane,_Layer
from reference.functional import FunctionalReference,Sample


def identity(n): return [[int(i==j) for j in range(n)] for i in range(n)]


class CycleTests(unittest.TestCase):
    def model(self,**kwargs):
        cfg=CycleConfig(n=5,width=8,layers=3,**kwargs)
        return CycleReference(cfg,[identity(cfg.n)]*cfg.layers,[64]*cfg.n)

    def drain(self,model):
        for _ in range(1000):
            if model.signals()['reload_ready']: return
            model.step()
        self.fail('drain timed out')

    def test_raw_front_and_public_boundaries(self):
        for n in (3,5,9):
            for layers in (1,3,10):
                for g,j in ((0,1),(1,1),(0,2)):
                    cfg=CycleConfig(n=n,width=8,layers=layers,route_delay=g,join_latency=j)
                    model=CycleReference(cfg,[identity(n)]*layers,[64]*n)
                    model.step(CycleInputs(input_valid=True,input_data=tuple(range(1,n+1))))
                    seen={}
                    for _ in range(layers*cfg.center_latency+n+5):
                        model.step()
                        for index,lane in enumerate(model.lane_outputs()[-1]):
                            if lane is not None: seen.setdefault(index,model.cycle-1)
                    self.assertEqual([seen[i] for i in range(n)],
                        [layers*cfg.center_latency+abs(i-n//2) for i in range(n)])
                    self.assertEqual(model.timings[0].enqueued_at,layers*cfg.center_latency+n//2+1)
                    self.assertEqual(model.timings[0].retired_at,layers*cfg.center_latency+n//2+2)

    def test_streaming_random_signed_vectors_and_bubbles(self):
        rng=random.Random(124)
        for n in (3,5,9):
            cfg=CycleConfig(n=n,layers=3)
            weights=[[[rng.randint(-128,127) for _ in range(n)] for _ in range(n)] for _ in range(3)]
            model=CycleReference(cfg,weights,[rng.randint(-128,127) for _ in range(n)])
            for _ in range(70):
                model.step(CycleInputs(input_valid=rng.randrange(3)!=0,
                    input_data=tuple(rng.randint(-128,127) for _ in range(n))))
            self.drain(model)
            times=[v.enqueued_at for v in model.timings.values()]
            self.assertTrue(any(b-a==1 for a,b in zip(times,times[1:])))
            self.assertEqual(len(model.records),len(model.timings))

    def test_output_backpressure_and_small_capacity(self):
        model=self.model(output_fifo_depth=1,in_flight_depth=5)
        held=None
        froze=False
        for cycle in range(100):
            inputs=CycleInputs(input_valid=True,input_data=(1,2,3,4,5),result_ready=False)
            sig=model.signals(inputs)
            if not sig['advance']:
                froze=True
                current=[layer.outputs() for layer in model.layers]
                if held is not None: self.assertEqual(current,held)
                held=current
            model.step(inputs)
            self.assertLessEqual(model.outstanding,5)
            self.assertLessEqual(len(model.fifo),1)
        self.assertTrue(froze)
        self.drain(model)
        self.assertEqual(len(model.records),5)

    def test_loader_reverse_rows_gaps_commit_and_reload(self):
        cfg=CycleConfig(n=3,layers=3)
        model=CycleReference(cfg)
        model.step(CycleInputs(reset_n=False))
        w=[[[l*30+i*3+j+1 for j in range(3)] for i in range(3)] for l in range(3)]
        for l in range(3):
            for i in range(2,-1,-1):
                self.assertTrue(model.signals()['weight_ready'])
                last=model.cycle
                model.step(CycleInputs(weight_valid=True,weight_data=tuple(w[l][i])))
                if i!=0: model.step()
        self.assertFalse(model.weights_loaded)
        while not model.weights_loaded: model.step()
        self.assertEqual(model.cycle-1-last,10)
        self.assertEqual(model.W,w)
        model.step(CycleInputs(reload_weights=True,input_valid=True,input_data=(9,9,9)))
        self.assertFalse(model.weights_loaded)
        self.assertEqual(model.outstanding,0)
        self.assertEqual(model.host_layer,0)

    def test_training_uses_serial_exact_oracle_and_blocks_following_samples(self):
        cfg=CycleConfig(n=3,layers=3,pass_through=False)
        w=[identity(3),identity(3),[[2,-1,0],[0,2,0],[0,0,1]]]
        model=CycleReference(cfg,w,[1,3,0])
        oracle=FunctionalReference(cfg,w,[1,3,0])
        sample=Sample((1,1,1),1,True)
        expected=oracle.step(sample)
        model.step(CycleInputs(input_valid=True,input_data=sample.x,target_data=sample.target,training_enable=True,pass_through=False))
        old=model.W
        cycles=0
        while model.training_pending:
            inputs=CycleInputs(input_valid=True,input_data=(2,2,2),pass_through=False)
            self.assertFalse(model.signals(inputs)['input_ready'])
            self.assertEqual(model.W,old)
            model.step(inputs)
            cycles+=1
            self.assertLess(cycles,100)
        self.assertEqual(model.W,oracle.W)
        self.assertEqual(model.R,oracle.R)
        self.assertEqual(model.sensitivity,[list(row) for row in expected.sensitivities])
        self.assertEqual(len(model.records),1)
        model.step(CycleInputs(input_valid=True,input_data=(2,2,2),pass_through=False))
        self.drain(model)
        self.assertEqual(len(model.records),2)

    def test_training_behind_inflight_inference_uses_its_own_context(self):
        cfg=CycleConfig(n=3,layers=3,pass_through=False)
        w=[identity(3),identity(3),[[2,-1,0],[0,2,0],[0,0,1]]]
        model=CycleReference(cfg,w,[1,3,0])
        oracle=FunctionalReference(cfg,w,[1,3,0])
        samples=[Sample((-1,2,-3))]*3+[Sample((1,1,1),1,True)]
        for sample in samples:
            oracle.step(sample)
            inputs=CycleInputs(input_valid=True,input_data=sample.x,target_data=sample.target,
                training_enable=sample.training_enable,pass_through=False)
            while not model.step(inputs).accepted: pass
        self.assertEqual(model.outstanding,4)
        self.drain(model)
        self.assertEqual(model.W,oracle.W)
        self.assertEqual(model.R,oracle.R)
        self.assertNotEqual(model.W,w)

    def test_reset_discards_every_pipeline_and_pending_training(self):
        for age in (1,5,13,20):
            model=self.model()
            model.step(CycleInputs(input_valid=True,input_data=(1,2,3,4,5),target_data=127,training_enable=True))
            for _ in range(age): model.step(CycleInputs(result_ready=False))
            model.step(CycleInputs(reset_n=False,result_ready=False))
            self.assertFalse(model.weights_loaded)
            self.assertFalse(model.training_pending)
            self.assertFalse(model.pipeline_busy)
            self.assertEqual(model.outstanding,0)
            self.assertFalse(model.fifo)
            for _ in range(20): model.step()
            self.assertFalse(model.fifo)

    def test_reconfiguration_is_rejected_while_busy(self):
        model=self.model()
        old=model.R
        model.step(CycleInputs(input_valid=True,input_data=(1,2,3,4,5)))
        model.step(CycleInputs(load_reduction_weights=True,reduction_weight=(1,)*5,pass_through=False))
        self.assertEqual(model.R,old)
        self.assertTrue(model.configuration_error)
        self.assertTrue(model.mode_pass)
        self.drain(model)
        model.step(CycleInputs(load_reduction_weights=True,reduction_weight=(1,)*5,pass_through=False))
        self.assertEqual(model.R,[1]*5)
        self.assertFalse(model.mode_pass)

    def test_model_rejects_broken_fronts_and_parameters(self):
        cfg=CycleConfig(n=3,layers=1)
        layer=_Layer(cfg,0,identity(3))
        layer.tick([None,Lane(1,1),None])
        with self.assertRaisesRegex(AssertionError,'valid mismatch'): layer.tick([None,None,None])
        layer=_Layer(cfg,0,identity(3))
        layer.tick([None,Lane(1,1),None])
        with self.assertRaisesRegex(AssertionError,'sample mismatch'): layer.tick([Lane(1,1),None,Lane(1,2)])
        with self.assertRaises(ValueError): CycleConfig(route_delay=-1)
        with self.assertRaises(ValueError): CycleConfig(join_latency=0)


if __name__=='__main__': unittest.main()
