# Fountain Three-Layer Learning Analysis

Version 0.1 | 16 October 2026 | Analysis only

Companion to `Fountain_Architecture_Specification.md`, version 0.3. No RTL, training model, or learning algorithm is implemented by this note.

## 1 Proposed forward network

Interpret three layers as three Fountain matrix stages, with the scalar weighted readout attached only to the third:

```text
x -> W1 -> activation -> W2 -> activation -> W3 -> activation -> scalar readout r -> prediction
```

The scalar readout computes prediction = sum(a3[j]*r[j]). The intermediate stages emit vectors. Each Fountain stage still needs its per-column upper/lower sum joins; those joins are different from the final scalar readout.

For three equal N by N stages, the forward hardware contains 3*N squared matrix product paths, 3*N column joining adders, and N scalar-readout products, before control, adapters, pipeline choices, or arithmetic-width growth. At N = 5, this is 75 matrix product paths, 15 column joining adders, and 5 scalar-readout products. These are logical operation paths, not predicted FPGA DSP counts.

The ideal N = 5 raw output times remain [14,13,12,13,14], with G = 0, J = 1, and no additional inter-layer processing or transport. The scalar readout and its output capture must then be counted. Width conversion and activation timing remain explicit boundary decisions.

Use nonlinear activations between stages when the aim is a more expressive model. If all stages pass their values through linearly, with no quantization or other nonlinear operation, the complete map collapses algebraically to x*(W1*W2*W3*r). Three matrices still cost hardware but represent a linear predictor. A ReLU at every matrix output is the analysis case below; individual stages may instead be configured as pass-through if that is part of the intended network.

## 2 What SSLMS provides and what depth changes

Classic sign-sign LMS updates a linear predictor using the signs of its input and output error:

```text
e = target - prediction
weight[i] += mu * sign(input[i]) * sign(e)
```

This is the linear-filter rule documented by [MathWorks](https://www.mathworks.com/help/dsp/ref/dsp.lmsfilter-system-object.html). The repository extends a sign-based update to its matrix plus scalar readout by multiplying the output error sign by the readout-coefficient sign and the activation gate.

That shallow structure makes the matrix's error direction simple: every hidden unit has one direct scalar-readout connection. With more layers, a hidden unit reaches the output through multiple weighted paths. Their contributions can cancel. The final scalar error sign alone does not identify which earlier weights should increase or decrease.

Sign-based training of a deep network is possible. The unresolved choice is how to obtain each hidden unit's error direction, and which approximations to accept. [signSGD](https://proceedings.mlr.press/v80/bernstein18a.html) studies updates based on gradient signs. It does not establish that discarding magnitudes at every intermediate backward step is equivalent, or that the repository's constant one-LSB, delayed, saturating implementation has the same convergence properties.

## 3 A mathematically defined reference for all three layers

Use the repository's row-vector weight orientation. For one sample:

```text
a0 = x
z1 = a0 * W1; a1 = ReLU(z1)
z2 = a1 * W2; a2 = ReLU(z2)
z3 = a2 * W3; a3 = ReLU(z3)
prediction = sum(a3[j] * r[j])
e = target - prediction
loss = 0.5 * e squared
```

For ReLU, g_l[j] = 1 when z_l[j] > 0 and 0 otherwise, choosing derivative zero at zero. For a pass-through stage, g_l[j] = 1, including at zero.

Let s_l[j] be the sensitivity of the prediction to pre-activation z_l[j]. Treat s_l and r as column vectors in the matrix expressions below. Elementwise multiplication is denoted `.*`:

```text
s3 = g3 .* r
s2 = g2 .* (W3 * s3)
s1 = g1 .* (W2 * s2)
delta_l = e * s_l
```

The weighted sums are explicit in the repository's orientation:

```text
s2[i] = g2[i] * sum(W3[i,j] * s3[j], over j)
s1[i] = g1[i] * sum(W2[i,j] * s2[j], over j)
```

These are the chain-rule backward sensitivities, with error sign chosen so positive delta gives an increasing-weight descent direction. The standard layer-by-layer principle is described by [Rumelhart, Hinton, and Williams](https://doi.org/10.1038/323533a0). The expressions here are a derivation for this particular scalar-output network.

A sign-gradient update is then:

```text
W_l[i,j] += mu_l * sign(a_(l-1)[i]) * sign(e) * sign(s_l[j])
r[j]     += mu_r * sign(e) * sign(a3[j])
```

Choose mu_l as the layer's weight step, initially one stored LSB if matching the existing policy, and saturate the updated stored weights. The sign of zero is zero. Each PE can still apply a ternary increment, decrement, or hold. Obtaining the correct column directions is the additional work.

For W3, sign(s3[j]) = g3[j]*sign(r[j]), reproducing the current matrix-update form with a2 as its input. W1 and W2 require the extra backward weighted sums. Thus the final matrix and readout can retain the existing form, while extending that form to earlier layers requires credit assignment.

Because the output error is scalar, its magnitude can be factored out: the backward sensitivity calculation needs W2, W3, r, and gates, but only sign(e) is needed for the final update directions. The magnitudes within s3 and s2 generally remain necessary to determine earlier signs correctly.

These equations are exact local gradient directions for the unquantized real-valued network away from activation boundaries. They do not guarantee that a finite one-LSB step reduces loss. Integer rounding, saturation, and inter-layer requantization introduce additional behavior. If using a surrogate derivative through those operations, specify it; do not call that the exact derivative of the quantized forward map.

## 4 A cheaper ternary backward candidate

An implementation candidate carries only ternary error directions between layers:

```text
q3 = g3 .* (sign(e) * sign(r))
q2 = g2 .* sign(W3 * q3)
q1 = g1 .* sign(W2 * q2)
W_l[i,j] += mu_l * sign(a_(l-1)[i]) * q_l[j]
```

Here `sign` applies elementwise. A product of a stored weight and q in {-1,0,+1} becomes a signed add, subtract, or zero. A separate backward implementation therefore need not contain general weight-by-error multipliers. It still needs weighted reduction hardware or scheduled reuse of existing arithmetic, data movement across the opposite matrix dimension, and accumulator guard bits. Sign-extend before negating the most negative stored weight.

This rule quantizes the backward signal after each layer. It is an approximation to the reference in section 3. Further replacing W2 and W3 by their signs would introduce another approximation, because the magnitudes of the downstream weights would also be discarded.

### Counterexample: a hidden update can reverse

Take a2 = [1,1], W3 = [[2,-1],[0,2]], and r = [1,3]. Both last-stage ReLUs are active: z3 = [2,1]. Let the target exceed the prediction, so sign(e) = +1.

For the first unit in layer 2:

```text
exact sensitivity:       2*1 + (-1)*3 = -1
ternary-error estimate:  2*1 + (-1)*1 = +1
```

The reference decreases its incoming weights for positive input activations; the ternary candidate increases them. This demonstrates why the candidate must be evaluated as its own learning rule rather than assumed to be exact backpropagation with smaller signals.

## 5 Practical choices for the architecture

| Choice | What learns | Additional learning work | Role in evaluation |
| --- | --- | --- | --- |
| Fixed prefix | W3 and r; W1/W2 remain fixed or externally pretrained | Save a2 input signs and the current suffix learning context; preserve coherent W3/r updates | First integration baseline; does not train the complete network |
| Full backward sensitivities, sign weight steps | W1, W2, W3, and r | Two weighted sensitivity matrix-vector operations, with sufficient precision and matched weight versions | Reference for the intended per-sample gradient signs |
| Ternary backward candidate | All three matrices and r | Two conditional-add/subtract matrix-vector operations, sign extraction, and gates | Hardware-oriented approximation to compare with the reference |
| Direct feedback alignment candidate | All matrices and r | Fixed feedback from output error to each hidden layer, plus local signs and gates | Alternative if a layer-by-layer backward datapath is too costly |

[Direct feedback alignment](https://papers.nips.cc/paper_files/paper/2016/file/d490d7b4576290fa60eb31b5fc917ad1-Paper.pdf) uses fixed random connections from output errors to hidden layers. For this scalar-output architecture, a possible sign-based adaptation assigns a fixed feedback vector b_l to each hidden layer and uses q_l = g_l .* sign(b_l*e). Using ternary/one-LSB weight updates on top is a proposed adaptation, not the published algorithm unchanged. Its effectiveness on this network and workload needs testing.

For ReLU hidden outputs, input signs to later layers are usually zero or positive. Sign updates discard the relative strengths of positive activations. Measure the effect of that information loss, step size, dead units, and saturated weights rather than assuming the shallow rule's behavior is preserved with depth.

Check initialization as part of that study. A bias-free three-stage ReLU network with all matrix weights zero has zero hidden activations and closed derivative gates, so these update rules cannot bootstrap it. Nonzero initial weights and measured hidden-unit activity are needed for a useful experiment.

## 6 Learning state, weight versions, and throughput

The three-stage forward stream is simpler than the complete training stream. The target error becomes available only after the final prediction, while earlier activations may already have left their stages.

Preserve at least the following per in-flight training sample, or provide a reproducible replay mechanism:

- Vector tag and target, retained through the selected retirement/error event.
- Input signs for each layer: sign(x), sign(a1), and sign(a2).
- Activation derivative gates g1, g2, and g3, and final activation signs for the readout update.
- The readout coefficient information and downstream weight versions needed by the selected backward rule.
- Backward/update state until every scheduled update has completed.

With inter-layer rounding, a positive pre-quantization activation can become a stored zero. Keep the derivative-gate convention distinct from the stored activation sign when these differ.

To reproduce the section 3 reference, backward sensitivities must use the downstream parameter values associated with that sample's forward computation. Saving the signs of r, as the current shallow design does, is insufficient for full sensitivities: r magnitudes and W2/W3 values matter. Re-reading live weights after intervening updates generally computes a different direction.

Start with one training sample at a time, freezing all parameters from forward computation through sensitivity formation, then applying the sample's updates. This establishes an unambiguous software reference. Freezing a configuration for a batch is another possible baseline, but the batch update rule must distinguish sign of a summed gradient from sums or votes of per-sample signs.

An eventual streaming implementation may use delayed updates, parameter versions, or another explicitly approximate consistency policy. Per-layer forward/backward matching and agreement on one network generation across layers are separate questions. [PipeDream](https://cs.stanford.edu/~deepakn/assets/papers/pipedream-sosp19.pdf) illustrates this distinction with weight stashing and optional cross-stage synchronization; it is background, not a prescribed Fountain implementation.

The earlier-stage backward operations also use the transposed direction relative to forward xW: the required sum is over output columns to produce a row-indexed error. The current Fountain forward links do not provide that operation automatically. Options include a separate backward network, new PE routing modes, or time-multiplexed arithmetic.

Forward II = 1 is not a training-throughput guarantee. At that rate, every fully provisioned forward product path is already occupied. Reusing those paths or their adders for backward work requires a new schedule; separate ternary conditional-add hardware has its own area and bandwidth cost. Keeping only one forward scalar readout does not remove the backward weighted sums.

Preserve the chosen public learning trigger, currently result retirement, or specify a deliberate change. Include backward context, version storage, and pending updates in stall, reset, reload, and busy accounting. The existing N+2 sample update-visibility delay does not carry over to this deeper network.

## 7 Proposed study before hardware changes

1. Fix the three-stage activation and inter-layer numeric formats. Begin with a real-valued reference so rounding does not hide the credit-assignment question.
2. Compare the fixed-prefix baseline, full-sensitivity sign updates, and ternary backward candidate on the same tasks, initializations, and data order. Treat direct feedback alignment as an additional experiment if needed.
3. Measure error versus processed samples, variation across seeds, exact/candidate update-direction agreement, inactive units, weight saturation, and sensitivity to layer step sizes. Include a task that benefits from nonlinear hidden features.
4. Add the intended fixed-point formats and requantization policy. Specify any surrogate derivative and compare against the real-valued reference.
5. Introduce delayed updates and explicit weight-version policies. Compare learning against the serial reference before claiming continuous online-training throughput.
6. Only after choosing a learning rule, derive the backward datapath, storage, update phases, resource cost, and timing for RTL.

For this analysis, 1,240 parameter derivatives across 30 three-stage ReLU networks of widths 2, 3, and 5 were checked against central finite differences. They agreed with the section 3 formulas. The section 4 reversal example was also checked. These are algebra checks from in-session calculations, not learning curves, convergence evidence, a checked-in training implementation, or hardware results.

The practical starting point is three forward Fountain stages with one final readout, an exact sign-gradient software reference for all trainable weights, and a measured comparison with the ternary backward candidate. The forward architecture can be specified now; the learning approximation and streaming consistency policy remain experimental choices.
