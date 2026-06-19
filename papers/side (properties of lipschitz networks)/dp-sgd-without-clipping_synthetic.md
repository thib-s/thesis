# DP-SGD Without Clipping: The Lipschitz Neural Network Way

**Synthetic conversion of the paper**

**Authors:** Louis Bethune, Thomas Massena, Thibaut Boissin, Aurelien Bellet, Franck Mamalet, Yannick Prudent, Corentin Friedrich, Mathieu Serrurier, David Vigouroux  
**Affiliations:** IRIT / Universite Paul Sabatier; IRT Saint Exupery; Inria / Universite de Montpellier  
**Code:** https://github.com/Algue-Rythme/lip-dp

---

## Abstract

Training deep neural networks with differential privacy usually relies on DP-SGD: per-sample gradients are clipped to bound sensitivity, then Gaussian noise is added. This clipping is expensive in memory and computation, biases the average gradient, and requires tuning a broad clipping hyperparameter. The paper proposes **Clipless DP-SGD**, which replaces per-sample gradient clipping by analytically bounded gradient sensitivities obtained from **Lipschitz constrained neural networks**. The key theoretical link is between a network's Lipschitz constant with respect to inputs and its Lipschitz constant with respect to parameters. By constraining each layer and propagating sensitivity bounds through the network, the method can train neural networks with formal `(epsilon, delta)`-DP guarantees without computing or clipping per-sample parameter gradients.

---

## 1. Motivation and problem setting

Differential privacy bounds how much changing one individual record can alter the output distribution of a randomized algorithm. For neighboring datasets `D` and `D'` differing by one sample, an algorithm `A` is `(epsilon, delta)`-DP if for all measurable sets `S`:

```text
P[A(D) in S] <= exp(epsilon) P[A(D') in S] + delta.
```

The paper uses the add/remove-one neighboring relation and focuses on supervised learning datasets:

```text
D = {(x_1, y_1), ..., (x_N, y_N)}.
```

A standard way to privatize a query is the **Gaussian mechanism**: compute the query sensitivity

```text
Delta(M) = sup_{neighboring D,D'} || M(D) - M(D') ||_2,
```

then add Gaussian noise with variance proportional to `Delta^2 sigma^2`.

### DP-SGD bottleneck

In ordinary DP-SGD, each minibatch gradient query is made private by bounding per-sample gradients. Since the true bound is usually unknown, gradients are clipped to norm `C`, giving sensitivity roughly `C / batch_size` in the add/remove setting. This creates three main issues:

1. **Hyperparameter burden:** the clipping threshold `C` strongly affects the privacy/utility trade-off and must be tuned.
2. **Runtime and memory cost:** per-sample gradients are expensive, especially for large batch sizes.
3. **Bias:** clipping changes the direction of the averaged gradient, especially when misclassified samples dominate the useful learning signal.

### Proposed alternative

The paper proposes to use neural networks whose parameter-wise gradients are **provably bounded by construction**. If the sensitivity of each layer's gradient can be computed analytically, the algorithm can add calibrated Gaussian noise directly, without clipping the per-sample parameter gradient. This is the central idea of **Clipless DP-SGD**.

---

## 2. Lipschitz neural networks and sensitivity bounds

A feedforward network of depth `D` is written as

```text
f(theta, x) = (f_D(theta_D) o ... o f_1(theta_1))(x),
```

where `theta = (theta_1, ..., theta_D)`. A layer is `ell_d`-Lipschitz with respect to its input if

```text
|| f_d(theta_d, x) - f_d(theta_d, y) ||_2 <= ell_d ||x - y||_2.
```

A Lipschitz feedforward network constrains every layer, so the full network is Lipschitz with constant

```text
ell = product_d ell_d.
```

Prior Lipschitz network work mostly studies `||nabla_x f||`, useful for robustness and generalization. This paper studies the less-explored quantity `||nabla_theta f||`, which is what matters for differential privacy of training.

### Requirements

The framework relies on three requirements:

1. **Lipschitz loss:** the loss must be `L`-Lipschitz with respect to logits for every label. Common supervised losses are analyzed in the appendix.
2. **Bounded input:** there is `X_0 > 0` such that `||x|| <= X_0` for all inputs.
3. **Lipschitz projection:** constraints are enforced with a projection operator `Pi : R^p -> Theta`, not by an arbitrary differentiable reparametrization. Projection is post-processing and does not create privacy leakage.

The paper focuses on projected training because reparametrizations may have unknown or unbounded Lipschitz constants with respect to parameters.

---

## 3. Backpropagation for bounds

The paper introduces a scalar analogue of backpropagation. Ordinary backpropagation propagates gradient vectors via vector-Jacobian products. Clipless DP-SGD instead propagates **upper bounds** on norms by replacing those products with scalar products of Lipschitz constants.

For a layer `y_d = f_d(theta_d, x_d)`, ordinary backpropagation has

```text
nabla_{x_d} L = (nabla_{y_d} L) * partial f_d / partial x_d.
```

The corresponding bound is

```text
||nabla_{x_d} L||_2 <= ||nabla_{y_d} L||_2 * || partial f_d / partial x_d ||_2.
```

The parameter-gradient bound is

```text
||nabla_{theta_d} L||_2 <= ||nabla_{y_d} L||_2 * || partial f_d / partial theta_d ||_2.
```

For standard parameterized layers, such as dense and convolutional layers, the parameter map is bilinear, giving bounds of the form

```text
|| partial f_d(theta_d, x) / partial theta_d ||_2 <= K(f_d, theta_d) ||x||_2 <= K(f_d, theta_d) X_{d-1}.
```

### Algorithm: Backpropagation for Bounds

Input: network `f`, weights `theta`, input norm bound `X_0`.

1. **Forward input-bound propagation:** for each layer,
   ```text
   X_d = max_{||x|| <= X_{d-1}} || f_d(theta_d, x) ||_2.
   ```
2. Initialize gradient-bound scalar:
   ```text
   G = L / b
   ```
   where `b` is batch size.
3. Traverse layers backward. For each layer `d`:
   ```text
   Delta_d = G * max_{||x|| <= X_{d-1}} || partial f_d(theta_d, x) / partial theta_d ||_2
   G = G * max_{||x|| <= X_{d-1}} || partial f_d(theta_d, x) / partial x ||_2
   ```
4. Return per-layer sensitivities `Delta_1, ..., Delta_D`.

The diagram on page 4 of the PDF visualizes this as three flows through each DP layer: red for input-bound propagation, green for backpropagated gradient bounds, and yellow for layer sensitivity.

---

## 4. Clipless DP-SGD

Clipless DP-SGD replaces per-sample clipping by sensitivity computation.

For every training step:

1. Compute per-layer sensitivities with Backpropagation for Bounds.
2. Sample a minibatch.
3. Compute the ordinary averaged gradient for each layer:
   ```text
   g_d = (1 / b) sum_i nabla_{theta_d} L(f(theta, x_i), y_i).
   ```
4. Add per-layer Gaussian noise:
   ```text
   zeta_d ~ N(0, sigma Delta_d).
   ```
5. Apply the perturbed update:
   ```text
   theta_d <- theta_d - eta (g_d + zeta_d).
   ```
6. Project back onto the Lipschitz feasible set:
   ```text
   theta_d <- Pi(theta_d).
   ```
7. Update the privacy accountant.

### Accounting strategies

The paper discusses two accounting strategies:

- **Global sensitivity:** aggregate layer sensitivities as
  ```text
  Delta = sqrt(sum_d Delta_d^2)
  ```
  and treat the full update as one isotropic Gaussian mechanism. This is simple and compatible with existing DP-SGD accountants but may overestimate sensitivity.

- **Per-layer sensitivity:** privatize each layer gradient with its own Gaussian mechanism and compose across layers. This can add less effective noise to each layer, because different layers may have different sensitivities, but it may produce a larger `epsilon` per epoch at the same noise multiplier.

The implementation uses Rényi Differential Privacy accounting via `autodp` and reports guarantees under Poisson-sampling assumptions for amplification, while experiments use sampling without replacement by shuffling each epoch.

---

## 5. Signal-to-noise ratio analysis

The analysis studies when the sensitivity bounds remain useful for deep networks.

### Informal theorem: gradient norm of Lipschitz networks

Assume every layer is `K`-Lipschitz, every bias is bounded by `B`, activations satisfy `f_d(0)=0`, the input norm is bounded by `X_0`, and the loss is `L`-Lipschitz.

- If `K < 1`, gradients vanish approximately like `K^D`. The network output becomes nearly independent of the input, and training is difficult.
- If `K > 1`, bounds grow approximately like `K^D`. The sensitivity bound becomes too large and the required noise becomes vacuous.
- If `K = 1`, the most favorable scaling occurs. For linear layers without biases, the gradient norm bound simplifies asymptotically to
  ```text
  O(L sqrt(D) (1 + X_0)).
  ```

The main message is that **1-Lipschitz layers are the best regime for privacy/utility**, because they avoid both vanishing and exploding sensitivity bounds.

### Gradient Norm Preserving networks

Gradient Norm Preserving (GNP) networks are 1-Lipschitz networks whose layer Jacobians are orthogonal. They satisfy an Eikonal-like property:

```text
|| partial f_d(theta_d, x_d) / partial x_d ||_2 = 1.
```

Then the bound for a layer simplifies because the product of downstream Jacobian norms remains one:

```text
||nabla_{theta_d} L|| <= ||nabla_{y_D} L|| * || partial f_d(theta_d, x_d) / partial theta_d ||.
```

For weight matrices this becomes proportional to the norm of the previous activation:

```text
||nabla_{W_d} L|| <= ||nabla_{y_D} L|| * || f_{d-1}(theta_{d-1}, x_{d-1}) ||.
```

This motivates two interventions:

1. **Control input norms** through preprocessing, clipping, normalization, or color-space choices.
2. **Control the loss-gradient norm** to maintain a good gradient-to-noise ratio during training.

### Loss-logits gradient clipping

The paper proposes an optional hybrid technique: clip the gradient of the loss with respect to logits or intermediate activations, not the per-sample parameter gradients. This is cheaper because the clipped vector is small, e.g. size `batch_size x number_of_classes` rather than `batch_size x number_of_parameters`.

For binary classification with BCE loss, sufficiently small loss-logits clipping yields a descent direction identical to the Kantorovich-Rubinstein objective:

```text
L_KR(y_hat, y) = - y y_hat.
```

This may improve signal-to-noise ratio but can bias the model toward robust classifiers with lower clean accuracy. In experiments, the authors use adaptive clipping thresholds no lower than the 90th percentile to limit utility loss.

---

## 6. The `lip-dp` library

The paper provides an open-source TensorFlow/Keras library, `lip-dp`, built on `deel-lip`, with:

- DP-aware model wrappers: `DP_Model`, `DP_Sequential`.
- DP layers with known sensitivity bounds: bounded input, spectral dense and convolutional layers, GroupSort, flattening, pooling, residual helpers.
- DP losses with known Lipschitz constants.
- DP accounting callbacks.
- Utilities for common datasets such as MNIST, Fashion-MNIST, and CIFAR-10.

Example workflow:

```python
model = DP_Sequential([
    DP_BoundedInput(input_shape=(28, 28, 1), upper_bound=20.),
    DP_SpectralConv2D(filters=16, kernel_size=3, use_bias=False),
    DP_GroupSort(2),
    DP_Flatten(),
    DP_SpectralDense(1),
], dp_parameters=dp_parameters, dataset_metadata=dataset_metadata)

model.compile(loss=DP_TauBCE(tau=20.), optimizer=Adam(1e-3))
model.fit(train_dataset, validation_data=val_dataset,
          epochs=num_epochs, callbacks=[DP_Accountant()])
```

The library enforces Lipschitz constraints with projections such as spectral normalization and power iteration. For GNP-like behavior it uses GroupSort activations, scaled L2 pooling, and orthogonal or near-orthogonal linear operators.

---

## 7. Experimental results

### Tabular data

The method is evaluated on binary classification tasks from the Adbench suite, using MLPs and `(epsilon, delta)`-DP with `epsilon = 1`. Reported validation AUROC values are competitive with DP-SGD:

| Dataset | Samples | Features | delta | DP-SGD AUROC | Clipless DP-SGD AUROC |
|---|---:|---:|---:|---:|---:|
| ALOI | 39,627 | 27 | 1e-5 | 56.5 | 56.2 |
| campaign | 32,950 | 62 | 1e-5 | 90.0 | 82.2 |
| celeba | 162,079 | 39 | 1e-6 | 96.6 | 96.5 |
| census | 239,428 | 500 | 1e-6 | 93.3 | 92.5 |
| donors | 495,460 | 10 | 1e-6 | 100.0 | 100.0 |
| magic | 15,216 | 10 | 1e-5 | 90.7 | 89.7 |
| shuttle | 39,277 | 9 | 1e-5 | 98.3 | 99.4 |
| skin | 196,045 | 3 | 1e-6 | 100.0 | 99.8 |
| yeast | 1,187 | 8 | 1e-4 | 66.8 | 75.1 |

An extended appendix table also reports wall-clock runtimes. Clipless DP-SGD is not always faster for these small tabular MLPs, but the main speed advantage appears at large batch sizes and larger models.

### MNIST, Fashion-MNIST, CIFAR-10

The paper presents Pareto fronts of validation accuracy versus privacy budget. Each green dot corresponds to an `(accuracy, epsilon)` pair at an epoch, and the blue line is the Pareto front. The results align with baselines from the DP learning literature, though the authors explicitly avoid heavy task-specific engineering such as pretraining, augmentation, or handcrafted features.

### Robustness and privacy

Because the models are Lipschitz constrained, they can provide certified robustness radii. For an `ell`-Lipschitz classifier, the predicted class is invariant under perturbations whose norm is below a margin-dependent radius. On CIFAR-10, the paper reports privacy/accuracy/robustness Pareto fronts for multiple perturbation radii. Unconstrained DP-SGD baselines cannot provide these deterministic Lipschitz certificates.

A table in the appendix shows that the softmax temperature strongly affects certified accuracy under `(20, 1e-5)`-DP training on CIFAR-10. Lower temperatures improve clean accuracy but can destroy larger-radius certificates; higher temperatures improve certificates at large radii but reduce clean accuracy.

### Speed and memory

The speed benchmark compares Clipless DP-SGD to TensorFlow Privacy, Opacus, and Optax on CNNs ranging from about 130K to 2M parameters on CIFAR-10. The main observation is:

- DP-SGD methods based on per-sample gradients become slower or run out of memory as batch size grows.
- Clipless DP-SGD's projection overhead depends on weights, not batch size.
- This makes it compatible with large batches, which reduce sensitivity roughly like `1 / b` and reduce stochastic gradient variance at the expected parametric rate.

The paper emphasizes that this batch-size scaling advantage is one of the main practical benefits of avoiding per-sample parameter-gradient clipping.

---

## 8. Comparison with other DP-SGD implementations

The appendix compares the method to standard and optimized clipping approaches.

| Method | Per-sample gradient? | Stores every layer's gradient? | Non-DP gradient? | Backprops | Overhead independent of batch size? |
|---|---:|---:|---:|---:|---:|
| non-DP | No | No | Yes | 1 | Yes |
| TF-Privacy / Abadi et al. | Yes | No | Yes | B | No |
| Opacus | Yes | Yes | Yes | 1 | No |
| FastGradClip | Yes | No | No | 2 | No |
| GhostClip | No | No | Yes | 1 | No |
| Book-Keeping | No | No | No | 1 | No |
| Clipless with Lipschitz | No | No | Yes | 1 | Yes |
| Clipless with GNP | No | No | Yes | 1 | Yes |

For feedforward layers with weight matrices of shape `p x d`, physical batch size `B`, and image spatial size `T`, the paper summarizes overheads as:

| Method | Time overhead | Memory overhead |
|---|---:|---:|
| non-DP | 0 | 0 |
| TF-Privacy / Abadi et al. | `O(B T p d)` | 0 |
| Opacus | `O(B T p d)` | `O(B p d)` |
| FastGradClip | `O(B T p d)` | `O(B p d)` |
| GhostClip | `O(B T p d + B T^2)` | `O(B T^2)` |
| Book-Keeping | `O(B T p d)` | `O(B min(p d, T^2))` |
| Clipless with Lipschitz | `O(U p d)` | 0 |
| Clipless with GNP | `O(U p d + V p d min(p,d))` | 0 |

Here `U` is the number of power-iteration steps and `V` the number of Bjorck projection steps, typically less than 15.

---

## 9. Appendix: important technical details

### Lipschitz and GNP background

For dense networks, the paper defines recurrent hidden states:

```text
h_0(x) = x
z_t(x) = W_t h_{t-1}(x) + b_t
h_t(x) = sigma(z_t(x))
f(theta, x) = z_{T+1}(x)
```

A Lipschitz network uses `S`-Lipschitz activations and affine maps with `||W||_2 <= U`, so the full network is `U(US)^T`-Lipschitz. GNP networks further require orthogonal layer Jacobians. Without biases, GNP networks are also norm preserving:

```text
||f(theta, x)|| = ||x||.
```

Because orthogonal sets are non-convex, projected approaches are preferred. Arbitrary reparametrizations can produce uncontrolled sensitivity through their own Jacobians.

### Loss Lipschitz constants

The paper gives constants for common losses:

| Loss | Hyperparameters | Lipschitz bound |
|---|---|---:|
| Softmax cross-entropy | temperature `tau > 0` | `sqrt(2) / tau` |
| Cosine similarity | lower norm bound `X_min > 0` | `1 / X_min` |
| Multiclass hinge | margin `m > 0` | `1` |
| Kantorovich-Rubinstein | none | `1` |
| Hinge + Kantorovich-Rubinstein | margin `m`, regularization `alpha` | `1 + alpha` |

### Layer sensitivity constants

For common layers, the parameter-Jacobian bound is proportional to input norm:

```text
|| partial f_t(theta_t, x) / partial theta_t ||_2 <= constant * ||x||_2.
```

Examples:

- **1-Lipschitz dense layer:** factor `1`.
- **Convolution with window size s:** factor `sqrt(s)`.
- **RKO convolution:** tighter factor depending on window size and image dimensions, approximately `1` for large images with small kernels.

Input-Jacobian constants are typically:

- Dense, RKO convolution, layer centering, ReLU, GroupSort: `1`.
- Residual block: `2` unless scaled to preserve 1-Lipschitzness.
- Softplus, sigmoid, tanh: `1`.

### Dense-layer proof sketch

For a dense layer `x -> W^T x`, the map from `W` to `W^T x` is linear, so its Lipschitz constant with respect to the parameter matrix is bounded by `||x||_2`:

```text
|| partial(W^T x) / partial W ||_2 <= ||x||_2.
```

### Convolution proof sketch

For a convolution `Psi * x` with window size `s`, the parameter Jacobian is bounded by

```text
|| partial(Psi * x) / partial Psi ||_2 <= sqrt(s) ||x||_2.
```

The proof decomposes the convolution output into feature maps and spatial positions, applies Pythagoras and Cauchy-Schwarz, and uses the fact that each pixel belongs to at most `s` patches.

### MLP-Mixer adaptation

The paper adapts MLP-Mixer to a 1-Lipschitz setting by:

- replacing ReLU by GroupSort,
- replacing dense layers by GNP equivalents,
- optionally using scaled skip connections,
- choosing patch sizes and hidden dimensions to maintain expressiveness while keeping sensitivity favorable,
- preferring square matrices where possible for exact gradient norm preservation.

The best CIFAR-10 architecture in the experiments was a single-block Lipschitz MLP-Mixer. Empirically, the last layer had relatively tight theoretical bounds, while deeper MLP blocks had looser bounds.

---

## 10. Practical guidance from the paper

The paper's practical recommendations can be condensed as follows:

1. Prefer **1-Lipschitz layers**; `K < 1` risks vanishing gradients, while `K > 1` creates exploding sensitivity bounds.
2. Prefer **GNP architectures** when possible, because they keep gradient norms closer to the computable upper bounds.
3. Control input norms with preprocessing, input clipping, or normalization.
4. Use bounded or Lipschitz-aware losses with known constants.
5. Consider loss-logits gradient clipping when the loss gradient becomes much smaller than its upper bound.
6. Use large batches when possible: Clipless DP-SGD benefits because sensitivity decreases with batch size and overhead is independent of batch size.
7. Be careful with numerical correctness: privacy guarantees rely on correct enforcement of spectral constraints.

---

## 11. Limitations and future work

The method replaces the **optimizer bias** of clipped DP-SGD with a **function-space bias**: models must live in the class of Lipschitz constrained networks. This is often acceptable for classification and useful for robustness certificates, but it changes the architecture search space and may reduce clean accuracy.

Key limitations and open directions:

- High-performing GNP architectures are still less mature than conventional CNNs.
- Orthogonality for convolutions is difficult and remains an active research area.
- Current analysis focuses on projection-based constraints; differentiable reparametrizations need additional sensitivity analysis.
- Sensitivity-bound propagation can be tightened, especially for deeper layers where Cauchy-Schwarz may be pessimistic.
- Floating-point and GPU nondeterminism may affect the validity of spectral projections and therefore sensitivity certificates.
- The experiments intentionally avoid heavy pretraining and feature engineering; integrating these could improve utility.

The appendix reports that no empirical violations of the computed gradient-norm certificates were observed during experiments, suggesting numerical errors can be controlled in practice.

---

## 12. Core takeaway

Clipless DP-SGD shows that differentially private deep learning does not necessarily require per-sample parameter-gradient clipping. By using Lipschitz constrained neural networks and propagating analytical sensitivity bounds layer by layer, one can train with formal privacy guarantees, avoid clipping-induced gradient bias, reduce batch-size-dependent memory and runtime costs, and obtain robustness certificates as a byproduct. The approach is most effective when networks are 1-Lipschitz and gradient-norm preserving, and its future progress is tied to better Lipschitz/GNP architectures and tighter sensitivity analysis.
