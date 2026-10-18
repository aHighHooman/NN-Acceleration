"""Independent numbers and gradient signs for the implemented Fountain network."""
import random
import unittest
from reference.arithmetic import matrix_multiply, weighted_vector_reduction
from reference.functional import ReferenceConfig, FunctionalReference, Sample, analyze


def identity(n):
    return [[int(i==j) for j in range(n)] for i in range(n)]


class FunctionalTests(unittest.TestCase):
    def test_worked_example_and_exact_widths(self):
        w=[[5*i+j+1 for j in range(5)] for i in range(5)]
        cfg=ReferenceConfig(n=5,width=8,layers=3)
        model=FunctionalReference(cfg,[w,identity(5),identity(5)],[64]*5)
        record=model.step(Sample((1,2,3,4,5)))
        self.assertEqual(record.activated_result,(215,230,245,260,275))
        self.assertEqual(record.prediction,612)
        self.assertEqual(cfg.matrix_result_width,41)
        self.assertEqual(cfg.prediction_width,44)
        self.assertEqual(model.W,[w,identity(5),identity(5)])

    def test_single_layer_matches_existing_fixed_point_arithmetic(self):
        rng=random.Random(123)
        for n in (3,5,9):
            for fraction in (0,4):
                cfg=ReferenceConfig(n=n,width=8,layers=1,fraction_bits=fraction)
                for _ in range(10):
                    w=[[rng.randint(-128,127) for _ in range(n)] for _ in range(n)]
                    x=[rng.randint(-128,127) for _ in range(n)]
                    r=[rng.randint(-128,127) for _ in range(n)]
                    record=analyze(cfg,w,r,Sample(tuple(x)))
                    self.assertEqual(list(record.activated_result),matrix_multiply(x,w,8))
                    self.assertEqual(record.prediction,weighted_vector_reduction(record.activated_result,r,8,8,fraction))

    def test_every_layer_activates_once_and_keeps_fractional_bits(self):
        w=[identity(3),identity(3),identity(3)]
        record=analyze(ReferenceConfig(n=3,layers=3,pass_through=False),w,[64]*3,Sample((-3,0,7)))
        self.assertEqual(record.activations,((-3,0,7),(0,0,7),(0,0,7),(0,0,7)))
        self.assertEqual(record.gates,((False,False,True),)*3)
        cfg=ReferenceConfig(n=3,layers=3,fraction_bits=2)
        scaled=[[[4*int(i==j) for j in range(3)] for i in range(3)] for _ in range(3)]
        record=analyze(cfg,scaled,[64]*3,Sample((4,8,12)))
        self.assertEqual(record.activated_result,(256,512,768))
        self.assertEqual(record.prediction,12)

    def test_hidden_sign_cancellation_and_ternary_counterexample(self):
        cfg=ReferenceConfig(n=3,layers=3,pass_through=False)
        w=[identity(3),identity(3),[[2,-1,0],[0,2,0],[0,0,1]]]
        record=analyze(cfg,w,[1,3,0],Sample((1,1,1),1,True))
        self.assertEqual(record.sensitivities,((-1,6,0),(-1,6,0),(1,3,0)))
        self.assertEqual(record.matrix_update_directions[1][0][0],-1)
        self.assertEqual(sum(w[2][0][j]*(1 if [1,3,0][j]>0 else 0) for j in range(3)),1)

    def test_atomic_learning_is_visible_to_next_admitted_sample(self):
        cfg=ReferenceConfig(n=3,layers=3,target_width=32)
        model=FunctionalReference(cfg,[identity(3)]*3,[64]*3)
        first=model.step(Sample((1,2,3),127,True))
        self.assertTrue(first.update_generated)
        self.assertEqual(model.R,[65]*3)
        self.assertEqual(model.W,[[[1+int(i==j) for j in range(3)] for i in range(3)]]*3)
        second=model.step(Sample((1,2,3)))
        self.assertEqual(second.W_used,tuple(tuple(tuple(row) for row in matrix) for matrix in model.W))
        self.assertGreater(second.prediction,first.prediction)
        self.assertFalse(second.update_generated)

    def test_step_saturates_weights_and_zero_error_holds(self):
        cfg=ReferenceConfig(n=3,width=4,layers=3,target_width=64,reduction_weight_width=4)
        model=FunctionalReference(cfg,[[[7]*3 for _ in range(3)]]*3,[7]*3)
        base=model.step(Sample((1,1,1)))
        model.step(Sample((1,1,1),base.prediction+1,True))
        self.assertEqual(model.R,[7]*3)
        self.assertTrue(all(v==7 for matrix in model.W for row in matrix for v in row))
        before=model.W
        model.step(Sample((1,1,1),base.prediction,True))
        self.assertEqual(model.W,before)

    def test_full_sensitivity_signs_match_finite_differences(self):
        rng=random.Random(1963)
        cfg=ReferenceConfig(n=3,layers=3,target_width=64,pass_through=False)
        checked=0
        for attempt in range(3000):
            w=[[[rng.randint(-3,3) for _ in range(3)] for _ in range(3)] for _ in range(3)]
            x=[rng.randint(-3,3) for _ in range(3)]
            r=[rng.randint(-3,3) for _ in range(3)]
            def output():
                a=x[:]
                pre=[]
                for matrix in w:
                    z=[sum(a[i]*matrix[i][j] for i in range(3)) for j in range(3)]
                    pre.extend(z)
                    a=[max(0,value) for value in z]
                return sum(v*c for v,c in zip(a,r)),pre
            raw,pre=output()
            if any(value==0 for value in pre): continue
            record=analyze(cfg,w,r,Sample(tuple(x), (raw>>7)+1,True))
            target=raw+1
            epsilon=1e-5
            for layer in range(3):
                for i in range(3):
                    for j in range(3):
                        old=w[layer][i][j]
                        w[layer][i][j]=old+epsilon
                        plus=0.5*(target-output()[0])**2
                        w[layer][i][j]=old-epsilon
                        minus=0.5*(target-output()[0])**2
                        w[layer][i][j]=old
                        descent=-(plus-minus)/(2*epsilon)
                        direction=0 if abs(descent)<1e-5 else (1 if descent>0 else -1)
                        self.assertEqual(direction,record.matrix_update_directions[layer][i][j])
            checked+=1
            if checked==12: break
        self.assertEqual(checked,12)

    def test_parameter_and_shape_rejections(self):
        for n in (1,2,4,8):
            with self.assertRaises(ValueError): ReferenceConfig(n=n)
        for name in ('width','layers','target_width','reduction_weight_width'):
            with self.assertRaises(ValueError): ReferenceConfig(**{name:0})
        with self.assertRaises(ValueError): ReferenceConfig(fraction_bits=-1)
        with self.assertRaises(ValueError): FunctionalReference(ReferenceConfig(),[identity(5)],[1]*5)
        model=FunctionalReference(ReferenceConfig(n=3,layers=1),identity(3),[1]*3)
        with self.assertRaises(TypeError): model.step(((1,2,3),0,False))
        with self.assertRaises(ValueError): model.step(Sample((1,2)))


if __name__=='__main__': unittest.main()
