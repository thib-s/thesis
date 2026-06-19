# Efficient Robust Conformal Prediction via Lipschitz-Bounded Networks

**Synthetic conversion of the PDF**  
Thomas Massena, Leo Andeol, Thibaut Boissin, Franck Mamalet, Corentin Friedrich, Mathieu Serrurier, Sebastien Gerchinovitz  
Proceedings of ICML 2025 / PMLR 267 - arXiv:2506.05434v2

---

## Executive summary

This paper proposes **lip-rcp**, an efficient method for robust conformal prediction (CP) using **Lipschitz-bounded neural networks**. Standard, or *vanilla*, CP provides finite-sample, distribution-free coverage guarantees under exchangeability, but these guarantees can fail under adversarial input perturbations. Existing robust CP methods either scale poorly or return very large prediction sets.

The paper makes two main contributions:

1. **Auditing vanilla CP under attack**: it derives high-probability lower and upper bounds on the coverage of vanilla CP under adversarial perturbations. These bounds hold **simultaneously for all attack budgets** `epsilon > 0` and for any attack function.
2. **Fast robust CP via Lipschitz bounds**: it introduces **lip-rcp**, which computes conservative and restrictive bounds on conformal scores using global Lipschitz constants. With 1-Lipschitz robust networks, lip-rcp achieves strong robust CP performance with negligible overhead compared with vanilla CP.

Empirically, lip-rcp produces smaller robust prediction sets than prior methods on CIFAR-10, CIFAR-100, TinyImageNet, and ImageNet, while remaining computationally efficient. On ImageNet at `epsilon = 0.02`, `alpha = 0.1`, lip-rcp obtains a robust CP set size of `111.0` with `97.4%` coverage and `0.012 s/sample`, compared with CAS at set size `1000.0`, `100%` coverage, and `7.920 s/sample`.

---

## 1. Motivation

Conformal Prediction (CP) is a post-hoc uncertainty quantification framework that transforms model outputs into prediction sets with finite-sample coverage guarantees. It is attractive because it is:

- distribution-free;
- model-agnostic;
- applicable after model training;
- valid under exchangeability of calibration and test data.

However, neural networks can be misled by small adversarial perturbations. Since CP guarantees rely on the distributional relationship between calibration and test data, adversarially perturbed inputs may invalidate vanilla CP guarantees. This motivates **Robust Conformal Prediction**, where prediction sets should maintain coverage even when the test input is perturbed within an `epsilon`-ball.

Existing robust CP methods include randomized smoothing, CDF-aware smoothing, binary certificates, and formal verification. Their main issues are either poor scalability or overly conservative prediction sets. This paper argues that **Lipschitz-bounded networks** provide a practical route to certifiable, scalable robust CP.

---

## 2. Split Conformal Prediction

The paper focuses on classification. Let:

- `f : X -> R^c` be a classifier returning logits for `c` classes;
- `Y` be the label set;
- `s : X x Y -> R` be a non-conformity score;
- `D_cal = {(X_i, Y_i)}_{i=1}^{n_cal}` be a calibration set.

For each calibration example, compute:

```math
R_i = s(X_i, Y_i).
```

Let

```math
q_alpha = R_{ceil((n_cal + 1)(1 - alpha))}
```

where the calibration scores are sorted increasingly. For a test input `x_test`, vanilla CP returns:

```math
C_alpha(x_test) = { y in Y : s(x_test, y) <= q_alpha }.
```

**Theorem 2.1.** If `D_cal` and `(X_test, Y_test)` are exchangeable, then for any score `s` and risk level

```math
alpha in [1/(n_cal + 1), 1),
```

```math
P(Y_test in C_alpha(X_test)) >= 1 - alpha.
```

This is a marginal guarantee: the average error over all possible test inputs is at most `alpha`.

---

## 3. Robust Conformal Prediction

For any input `x`, define the adversarial ball:

```math
B_epsilon(x) = {x' in X : ||x' - x|| <= epsilon}.
```

A prediction set `C_{alpha,epsilon}` is robust to `epsilon`-bounded perturbations if, for every random perturbed input `X_tilde_test` satisfying `X_tilde_test in B_epsilon(X_test)` almost surely,

```math
P(Y_test in C_{alpha,epsilon}(X_tilde_test)) >= 1 - alpha.
```

Robust CP methods usually build sets by replacing the original non-conformity score with a **conservative score**. A conservative score `s_lower` must satisfy:

```math
s_lower(x, y) <= inf_{x_tilde in B_epsilon(x)} s(x_tilde, y).
```

Then the robust prediction set is:

```math
C_{alpha,epsilon}(x) = { y in Y : s_lower(x, y) <= q_alpha }.
```

The quality of a robust CP method depends on how tightly and efficiently it estimates worst-case conformal scores.

---

## 4. Related work and limitations

### Randomized smoothing methods

RSCP uses a smoothed score:

```math
s_tilde(x, y) = Phi^{-1}( E_{Delta ~ N(0, sigma^2 I)}[s(x + Delta, y)] ),
```

then defines a conservative score by subtracting `epsilon / sigma`. Later methods such as RSCP+, PTT, RCT, and CAS improve smoothing-based robust CP using corrective factors, CDF-aware sets, and holdout splits.

Limitations:

- require many Monte Carlo samples;
- increase memory and inference cost;
- become less effective in high-dimensional inputs;
- can produce large prediction sets.

### Formal verification methods

VRCP uses neural-network verification tools to compute lower bounds on conformal scores around a point. Two variants are discussed:

- **VRCP-I**: standard calibration, then robust inference using conservative scores around test points;
- **VRCP-C**: conservative scores are computed around calibration points, so inference has no extra cost.

Limitations:

- formal verification scales poorly with model size;
- bounds become loose on deeper/larger models;
- practical use is restricted to small or medium networks.

### Positioning of lip-rcp

The paper's method avoids Monte Carlo sampling and formal solvers by using Lipschitz bounds. It therefore offers constant-time score bounding at both calibration and inference.

| Method | Certifiable | Fast calibration | Fast test | Supports l1/l_inf | Poisoning robustness | Hyperparameters |
|---|---:|---:|---:|---:|---:|---:|
| aPRCP | no | no | yes | no | no | 2 |
| RSCP | no | no | no | yes | no | 3 |
| RSCP+ / PTT / RCT | yes | no | no | yes | no | 4 to 6 |
| CAS | yes | no | no | yes | yes | 2 |
| VRCP-I | yes | yes | no | yes | no | 2 |
| VRCP-C | yes | no | yes | yes | no | 2 |
| **lip-rcp** | **yes** | **yes** | **yes** | no | yes | 2 |

---

## 5. Auditing vanilla CP under attack

The paper also studies vanilla CP under adversarial perturbations without enlarging the sets. This is useful when small prediction sets are desired, but certified coverage bounds under attack are still needed.

Let `h(x, epsilon)` be any attack function such that:

```math
||h(x, epsilon) - x|| <= epsilon.
```

The coverage of vanilla CP under attack is:

```math
gamma_h(epsilon) = P_{D_test}(Y_test in C_alpha(h(X_test, epsilon))).
```

Because `h` is unknown, the paper defines two bounding quantities using conservative and restrictive scores:

```math
s_lower_epsilon(x, y) = inf_{x_tilde in B_epsilon(x)} s(x_tilde, y),
```

```math
s_upper_epsilon(x, y) = sup_{x_tilde in B_epsilon(x)} s(x_tilde, y).
```

These define:

```math
C_lower_{alpha,epsilon}(x) = {y : s_lower_epsilon(x, y) <= q_alpha},
```

```math
C_upper_{alpha,epsilon}(x) = {y : s_upper_epsilon(x, y) <= q_alpha}.
```

The resulting coverages satisfy:

```math
gamma_lower(epsilon) <= gamma_h(epsilon) <= gamma_upper(epsilon).
```

### Corrected empirical bounds

The paper identifies an overfitting-type issue in previous lower-bound guarantees from Gendler et al. (2022) and Zargarbashi et al. (2024). To fix this, it introduces an independent evaluation set:

```math
D_eval = {(X_i, Y_i)}_{i=1}^m.
```

Empirical estimates are:

```math
gamma_upper_m(epsilon) = (1/m) sum_i 1{Y_i in C_lower_{alpha,epsilon}(X_i)},
```

```math
gamma_lower_m(epsilon) = (1/m) sum_i 1{Y_i in C_upper_{alpha,epsilon}(X_i)}.
```

The paper then applies binomial-tail corrections, tighter than Hoeffding-style bounds, to produce corrected high-probability bounds `gamma_m^-` and `gamma_m^+`.

### Theorem 3.3

Under mild assumptions:

- `X` is convex and closed;
- `s(x, y)` is continuous in `x` for all labels `y`;
- calibration, evaluation, and test sets are i.i.d.;
- `m >= 2`;

then, with probability at least `1 - delta` over the draw of `D_eval`, for any attack function `h` and almost every calibration set:

```math
forall epsilon > 0,
 gamma_m^-(epsilon, delta') <= gamma_h(epsilon) <= gamma_m^+(epsilon, delta'),
```

where:

```math
delta' = delta / (2m - 2).
```

Key point: the guarantee holds **simultaneously for all attack budgets** `epsilon > 0`.

---

## 6. Lipschitz-bounded networks and lip-rcp

A classifier `f : X -> R^c` is `L`-Lipschitz in `l_p` norm if:

```math
||f(x) - f(y)||_p <= L ||x - y||_p.
```

Computing exact Lipschitz constants of arbitrary deep networks is NP-hard. The paper therefore focuses on **Lipschitz-by-design** architectures, especially 1-Lipschitz networks with orthogonality constraints. These networks offer:

- explicit robustness-accuracy control;
- tighter certified Lipschitz bounds;
- better gradient propagation;
- efficient training with specialized parametrizations.

### Lipschitz score bounds

Assume the non-conformity score is:

```math
s(x, y) = psi(f(x), y),
```

where:

- `f` is `L_n`-Lipschitz;
- `psi(., y)` is `L_s`-Lipschitz.

Then:

```math
|s(x, y) - s(x + delta, y)| <= L_n L_s ||delta||_p.
```

For any adversarial point `x_tilde in B_epsilon(x)`:

```math
s(x, y) - L_n L_s epsilon <= s(x_tilde, y) <= s(x, y) + L_n L_s epsilon.
```

This is the basis of **lip-rcp**.

### LAC sigmoid score

Instead of the classic softmax LAC score, the paper proposes a sigmoid-based score:

```math
s(x, y) = 1 - sigmoid((f(x)_y - b) / T),
```

where:

- `T` is a temperature;
- `b` is a bias.

The Lipschitz constant of this score with respect to `f(x)_y` is:

```math
L_s = 1 / (4T).
```

In robust CP, the sigmoid score yields smaller robust sets than the softmax score in the reported experiments.

### Unified calibration and inference

Unlike VRCP, which distinguishes robust calibration and robust inference, lip-rcp uses global Lipschitz bounds. Because the robust shift is additive and the quantile computation is translation equivariant, robust calibration and robust inference yield the same result.

### Complexity

| Method | Calibration complexity | Test complexity |
|---|---:|---:|
| RSCP | O(n_mc) | O(n_mc) |
| CAS | O(n_mc) | O(n_mc * t_b) |
| VRCP-I | O(1) | O(t_v) |
| VRCP-C | O(t_v) | O(1) |
| **lip-rcp** | **O(1)** | **O(1)** |

Here `n_mc` is the number of Monte Carlo samples, `t_b` the CAS-bound computation cost, and `t_v` the verification solver cost.

---

## 7. Empirical validation

The paper validates lip-rcp on CIFAR-10, CIFAR-100, TinyImageNet, and ImageNet.

### Experimental setup

The authors use 1-Lipschitz feature extractors followed by classification layers enforcing a 1-Lipschitz condition on every output. The training scheme uses Lipschitz and orthogonality constraints from Boissin et al. (2025). Models are trained with Optimal Transport-inspired losses such as the Hinge Kantorovich-Rubinstein loss.

All experiments use held-out calibration and test data unseen during training. Several experiments repeat over multiple random calibration/test splits.

### Robust CP comparison

Datasets:

- CIFAR-10;
- CIFAR-100;
- TinyImageNet;
- ImageNet.

The method is compared with VRCP, RSCP+, CAS, PTT, and related robust CP methods. In Figure 3, robust CP set sizes and empirical robust coverage are shown for `alpha = 0.1`. The ideal point is low set size and coverage close to the target `1 - alpha`.

Interpretation:

- lip-rcp obtains smaller robust CP sets across the tested datasets;
- coverage remains close to the desired target;
- Lipschitz-constrained training promotes robustness;
- orthogonality constraints help produce tight score bounds;
- smoothing methods suffer from Monte Carlo correction and finite-sample penalties.

### ImageNet scalability

On ImageNet, VRCP is not practical because formal verification does not scale. The paper compares lip-rcp with CAS using a ResNet50 baseline.

At `epsilon = 0.02`, `alpha = 0.1`:

| Method | Set size | Coverage (%) | n | Time per sample (s) |
|---|---:|---:|---:|---:|
| CAS | 1000.0 | 100.0 | 5e2 | 7.920 |
| **lip-rcp** | **111.0** | **97.4** | **5e4** | **0.012** |

An additional lip-rcp run with `n = 5e2` gives set size `118.5` and `97.5%` coverage.

---

## 8. Certifiable vanilla CP coverage bounds

The paper also evaluates the auditing bounds from Section 3.

Procedure:

1. Perform vanilla split CP on `D_cal` with `n_cal = 3000`.
2. Estimate empirical lower/upper coverage quantities on `D_eval` with `n_eval = 5000`.
3. Compute corrected estimates `gamma_m^-` and `gamma_m^+` by binary search using the binomial correction.
4. Repeat over 20 randomly sampled non-overlapping calibration/evaluation pairs.

For ImageNet, the paper uses:

```math
n_cal = 15000,
 n_eval = 35000.
```

Figure 4 compares Lipschitz and CROWN verification bounds on CIFAR-10 and then reports corrected bounds across datasets. The results show that Lipschitz-constrained models provide less pessimistic coverage bounds than formal verification, largely because formal verification becomes loose and expensive for larger networks.

The main practical message is that robust networks plus vanilla CP can retain small prediction sets while still allowing certified error bounds under bounded attacks.

---

## 9. Discussion, limitations, and future work

### Main conclusions

The paper presents a fast method for certifiably robust CP:

- lip-rcp is essentially as efficient as vanilla CP;
- it produces smaller robust prediction sets than competing methods;
- it scales to ImageNet;
- it can be used both for robust CP set construction and for auditing vanilla CP under attack;
- the auditing guarantee holds for all attack budgets simultaneously.

### Limitations

The method currently focuses on `l_2` perturbations. It does not directly generalize to `l_1` or `l_inf` norms. Extending the method to other norms is identified as future work.

The approach is not fully model-agnostic in practice, because tight results depend on having a computable and tight Lipschitz constant. The experiments are therefore restricted to Lipschitz-by-design architectures. If tight Lipschitz estimation became available for arbitrary models, the method could become more model-agnostic.

The authors also note that it would be useful to study higher perturbation budgets while retaining tightness and efficiency.

---

## 10. Appendix highlights

### A. Issue in previous lower-bound theorem

The paper identifies a subtle mistake in Gendler et al. (2022, Theorem 2), later reproduced in Zargarbashi et al. (2024). The issue is that a deterministic probability is bounded by a random quantity depending on calibration data. A lemma from Romano et al. (2019) is used with a random parameter, although it only applies to deterministic values.

### B. Proof of Theorem 3.3

The proof proceeds through:

1. continuity properties of the exact lower and upper coverage functions;
2. binomial concentration for fixed `epsilon`;
3. a union-bound and quantile-grid argument to obtain uniform-in-`epsilon` bounds;
4. separate handling of right-continuity and left-continuity for the upper and lower functions.

### C. Measurability

The paper discusses measurability of functions involving infima and suprema over adversarial balls. Under the assumptions that `X` is closed and convex and that scores are continuous, the relevant functions are continuous and therefore measurable.

### D. Extending VRCP

The appendix describes how formal verification could also estimate restrictive prediction sets, not only conservative ones, because verification tools often return lower and upper logit bounds.

### E. Sigmoid vs softmax score

For vanilla CP on CIFAR-10 with a ResNet50 and `alpha = 0.1`:

| Score | Coverage (%) | Set size |
|---|---:|---:|
| LAC softmax | 90.04 | 1.088 |
| LAC sigmoid | 90.28 | 1.144 |

For robust CP on CIFAR-10 with a 1-Lipschitz VGG, `epsilon = 0.03`, `alpha = 0.1`:

| Score | Coverage (%) | Set size |
|---|---:|---:|
| LAC softmax | 95.6 | 2.38 |
| LAC sigmoid | 94.9 | 1.72 |

The sigmoid score is slightly worse in vanilla CP but considerably better for robust CP in this setting.

### F. Model implementation and training overhead

Replacing vanilla layers with Lipschitz-constrained layers introduces overhead, but the cost is batch-size independent and becomes relatively smaller with large batches.

| Batch size | Training time overhead (%) |
|---:|---:|
| 1024 | 20.31 |
| 2048 | 10.26 |

The paper reports a `9.6%` runtime overhead for the ResNeXt model on TinyImageNet compared with an unconstrained counterpart.

### G. Experimental settings

The Lipschitz models for CIFAR-10, CIFAR-100, and TinyImageNet use a ResNeXt-like design with grouped-convolution residual blocks and increasing channels `64 -> 128 -> 256 -> 512`.

The ImageNet model uses an efficient Lipschitz-parametrized architecture from the `orthogonium` library, with stages expanding channels from `256` to `2048`, depthwise `5x5` convolutions, GroupSort2 activations, and residual mixing respecting 1-Lipschitz constraints.

Training hyperparameters for Lipschitz networks:

| Dataset | Margin | Temperature | Epochs | Alpha |
|---|---:|---:|---:|---:|
| CIFAR-10 | 0.6 | 5.0 | 130 | 0.975 |
| CIFAR-100 | 0.6 | 5.0 | 220 | 0.975 |
| TinyImageNet | 0.3 | 5.0 | 80 | 0.975 |

### H. Calibration poisoning

Label-flipping attacks are not directly handled by a Lipschitz certificate because common conformal scores are not Lipschitz in the label. However, the quantile-shift defense of Zargarbashi et al. (2024) remains valid.

For calibration-time feature poisoning, Lipschitz continuity with respect to `x` helps. The paper formulates a linear program to compute maximum and minimum quantile shifts under a poisoning budget `(k, epsilon)`.

On CIFAR-10 with `n_cal = 4750`:

| epsilon | k | Metric | Mean | Std dev |
|---:|---:|---|---:|---:|
| 0 | - | Validation accuracy | 0.7260 | - |
| 0 | - | Set size | 2.0690 | 0.0387 |
| 0.25 | 6 | Robust CP gamma | 0.8982 | 0.0085 |
| 0.25 | 6 | Robust CP set size | 2.0699 | 0.0467 |
| 0.25 | 50 | Robust CP gamma | 0.9090 | 0.0068 |
| 0.25 | 50 | Robust CP set size | 2.1872 | 0.0466 |

### I. Tightness of score bounds

A VGG-like 1-LipNet with 10.7M parameters is used on CIFAR-10 to compare score-bound tightness. For `epsilon = 0.03`, across 10 runs:

| Method | Coverage (%) | Set size | Runtime (s/run) |
|---|---:|---:|---:|
| CAS, n_mc = 1024 | 94.60 | 2.302 | 2615 |
| **lip-rcp** | **92.83** | **1.889** | **10** |
| VRCP-I/C, CROWN | OOM | OOM | N/A |

CROWN does not scale to this deeper network. CAS may improve if its smoothing parameter is tuned, but the reported cost is high.

---

## Takeaway

The paper reframes robust conformal prediction around Lipschitz-certified score variation. With a well-trained 1-Lipschitz network, robust CP becomes nearly as cheap as vanilla CP while retaining formal guarantees. The same Lipschitz bounds also enable efficient, high-probability auditing of vanilla CP under arbitrary bounded adversarial attacks.
