# Robust One-Class Classification with Signed Distance Function using 1-Lipschitz Neural Networks

**Authors:** Louis Bethune, Paul Novello, Guillaume Coiffier, Thibaut Boissin, Mathieu Serrurier, Quentin Vincenot, Andres Troya-Galvis  
**Venue:** ICML 2023, PMLR 202  
**arXiv:** 2303.01978v2, 1 Apr 2024  
**Core contribution:** One Class Signed Distance Function (OCSDF), a one-class classification method that learns a signed distance function to the support boundary of the known class using 1-Lipschitz neural networks and the Hinge Kantorovich-Rubinstein loss.

---

## 1. Executive summary

One-class classification (OCC) learns a decision boundary from positive/in-distribution data only. The paper proposes **OCSDF**, which learns the **Signed Distance Function (SDF)** to the boundary of the data support. The SDF value is used as a normality score: points inside the support have positive distance, points outside have negative distance, and the boundary is the zero level set.

The method combines three ingredients:

1. **A complementary negative distribution** generated without external outliers.
2. **A 1-Lipschitz neural network** used as the classifier/score function.
3. **The HKR loss** (Hinge Kantorovich-Rubinstein), which makes the 1-Lipschitz classifier approximate the SDF under suitable sampling assumptions.

The central claim is that learning the SDF with a 1-Lipschitz network gives **provable robustness certificates** against bounded `l2` adversarial attacks. This leads to a **certified AUROC** computable from scores, at roughly the same cost as ordinary AUROC, without explicitly running adversarial attacks.

OCSDF is empirically competitive on tabular anomaly detection and simple image OCC benchmarks, and it is substantially more robust under adversarial attacks than non-certified baselines. The paper also explores two side applications: visualizing learned support boundaries and reconstructing implicit 3D surfaces from point clouds.

---

## 2. Problem setting

In OCC, the training data contains only examples from the positive class. The goal is to learn a domain containing the support of the unknown data distribution and reject samples outside it. Typical uses include anomaly detection, novelty detection, out-of-distribution detection, open-set recognition, fraud detection, cyber-intrusion detection, and industrial defect detection.

Let positive samples be drawn from a distribution `P_X` with compact support `X subset R^d`. The boundary is denoted `partial X`. The signed distance function is:

```math
S(x) = \begin{cases}
 d(x, \partial X), & x \in X \\
-d(x, \partial X), & x \notin X
\end{cases}
```

where

```math
d(x, \partial X) = \inf_{z \in \partial X} ||x-z||_2.
```

Important SDF properties:

- `S(x) > 0` inside the support, `S(x) < 0` outside, and `S(x)=0` on the boundary.
- `S` satisfies the Eikonal equation: `||grad_x S(x)|| = 1`.
- Therefore `S` is 1-Lipschitz under the `l2` norm.
- Distance from the boundary is directly interpretable as normality or abnormality strength.

---

## 3. Related work, condensed

### One-class classification

Classical OCC includes One-Class SVM, Local Outlier Factor, Isolation Forest, PCA-based methods, kNN methods, and many variants. Deep approaches include Deep SVDD and later neural anomaly detection methods. These methods often perform well but generally lack formal robustness guarantees and must cope with the absence of negative data.

### SDF and neural implicit surfaces

Signed distance functions are widely used in computer graphics to represent surfaces as level sets. Neural implicit surface methods often learn an SDF from supervised distance labels or nearest-neighbor-derived distances. OCSDF differs by learning an SDF from positive samples only, without known point-to-surface distances.

### 1-Lipschitz neural networks

1-Lipschitz networks constrain the function class so that output changes are bounded by input perturbations. The paper uses GroupSort activations and orthogonal transformations. Such networks are connected to optimal transport and can provide `l2` robustness certificates.

### Robustness and certification

Many defenses against adversarial attacks are empirical or attack-specific. OCSDF provides a certificate for all `l2` attacks of bounded radius because the score function is globally 1-Lipschitz.

### Negative sampling

Some OCC methods generate artificial negatives or use outlier exposure. OCSDF uses **adaptive negative sampling**: it starts from uniform samples in a feasible box and moves them toward the learned boundary using gradients of the current 1-Lipschitz model.

---

## 4. Method

OCSDF reframes OCC as binary classification between:

- positive samples from `P_X`, and
- artificial negative samples from a complementary distribution `Q` whose support lies outside `X` but inside a bounded feasible domain `B`.

The challenge is to find a useful `Q`: negatives should not be arbitrary far-away points, but should concentrate near the boundary of `X`, where they are informative.

### 4.1 Complementary distribution

Informally, `Q` is a complementary distribution to `P_X` if:

- `supp(Q) subset B`,
- `supp(Q)` is disjoint from `X`,
- points sampled from `Q` are at least `2 epsilon` away from `X`,
- `Q` fills the remaining feasible space outside that gap.

For images, `B` is the pixel-value hypercube, for example `[0,1]^(W x H x C)` or a normalized equivalent. For tabular data, `B` is a hypercube scaled from data statistics.

### 4.2 HKR loss and SDF learning

The classifier `f` is trained with the HKR loss:

```math
L^{hkr}_{m,\lambda}(y f(x)) = \lambda \max(0, m - y f(x)) - y f(x),
```

where:

- `m` is the margin,
- `lambda > 0` is a regularization weight,
- `y in {-1, 1}` is the label,
- positive samples use `y=+1`, negatives use `y=-1`.

The population risk is:

```math
E^{hkr}(f, P_X, Q) = E_{x \sim P_X}[L^{hkr}_{m,\lambda}(f(x))]
+ E_{z \sim Q}[L^{hkr}_{m,\lambda}(-f(z))].
```

The optimization is restricted to the space of 1-Lipschitz functions:

```math
f^* \in \arg\inf_{f \in Lip_1(R^d, R)} E^{hkr}(f, P_X, Q).
```

**Main theorem.** If `Q` is a valid complementary distribution and `m = epsilon`, then the HKR minimizer approximates the SDF over `B`:

```math
S(x) = f^*(x) - m, \quad x \in X,
```

```math
S(z) = f^*(z) - m, \quad z \in supp(Q).
```

Moreover, `sign(f(x)) = sign(S(x))` on `X union supp(Q)`. Thus the classifier score has the same sign structure as the SDF and approximates signed distance up to the margin shift.

### 4.3 Why use a 1-Lipschitz network?

The network is constrained to be 1-Lipschitz by construction. The paper uses GroupSort activations and orthogonal affine transformations. This ensures:

- bounded output variation: `|f(x)-f(z)| <= ||x-z||_2`,
- direct robustness certificates,
- gradients that are smoother and more interpretable than those of unconstrained networks,
- a natural relation to the SDF, whose gradient norm is 1 almost everywhere.

### 4.4 Finding the complementary distribution

The method begins with uniform negative sampling from `B`, then iteratively improves the negative samples by moving them toward the estimated boundary.

At iteration `t`, train a classifier `f_t` using current negatives `Q_t`. Then define the level set:

```math
L_t = f_t^{-1}({-\epsilon}).
```

A negative point `z_0` is moved toward this level set using a Newton-Raphson-like update. A local linearization gives:

```math
f_t(z_0 + \delta) \approx f_t(z_0) + <grad_x f_t(z_0), \delta>.
```

Solving for `f_t(z_0 + delta) = -epsilon` yields:

```math
\delta = - \frac{f_t(z_0) + \epsilon}{||grad_x f_t(z_0)||_2^2} grad_x f_t(z_0).
```

When the classifier is well fitted and gradient norm is close to 1, this simplifies conceptually to moving along `-f_t(z_0) grad f_t(z_0)`.

### 4.5 Adapted Newton-Raphson sampler

The algorithm generates negative samples as follows:

```text
Input: 1-Lipschitz neural net f_t
Parameter: number of steps T
Output: negative sample z' ~ Q_t(f)

1. Sample learning rate eta ~ Uniform([0,1])
2. Sample z_0 ~ Uniform(B)
3. For t = 1 to T:
     z_{t+1} <- z_t - (eta/T) * grad f(z_t) / ||grad f(z_t)||_2^2 * (f(z_t) + epsilon)
     z_{t+1} <- projection_B(z_{t+1})
4. Return z_T
```

The random `eta` spreads samples along paths toward the boundary rather than placing all negatives at the same level set. In high dimensions, the density of generated negatives increases near the boundary, helping mitigate the curse of dimensionality.

### 4.6 Full training loop

OCSDF alternates between:

1. generating negative samples with the current classifier,
2. sampling positive data,
3. updating the 1-Lipschitz network with HKR loss.

The practical version is a lazy alternating optimization: instead of fully solving the inner minimization at each iteration, it performs a fixed number of gradient updates, similarly to GAN/WGAN training.

---

## 5. Robustness properties

### 5.1 Certified AUROC

For a 1-Lipschitz score function, any `l2` perturbation `delta` of radius `epsilon` changes the score by at most `epsilon`:

```math
f(x+\delta) \in [f(x)-||\delta||_2, f(x)+||\delta||_2].
```

This allows a lower bound on the AUROC under any adversarial attack of radius `epsilon`.

Let:

- `F_0` be the CDF of negative scores,
- `p_1` be the PDF of positive scores.

The certified AUROC under attack radius `epsilon` is:

```math
AUROC_\epsilon(f) = \int_{-\infty}^{\infty} F_0(t) p_1(t - 2\epsilon) dt.
```

This can be computed from score distributions without running attacks. The certificate holds for any `l2` adversarial attack bounded by `epsilon`, independently of the attack algorithm.

### 5.2 Ranking normal and anomalous points

Because the score approximates distance to the boundary, samples far from the boundary require larger adversarial perturbations to flip their status. This makes the normality score meaningful: highly abnormal points have strongly negative scores and require larger attack budgets to hide.

---

## 6. Experiments

All experiments use TensorFlow and 1-Lipschitz networks implemented with the Deel-Lip library. The code is reported as available at `https://github.com/Algue-Rythme/OneClassMetricLearning`.

### 6.1 Toy 2D Scikit-Learn examples

Datasets include one cloud, two circles, two blobs, blob-and-cloud, and two moons. OCSDF is compared against:

- One-Class SVM,
- Isolation Forest,
- a conventional neural network trained with binary cross-entropy.

Findings:

- OCSDF produces smooth, meaningful contours that match the expected data geometry.
- The conventional network can fit the binary task but does not learn a meaningful distance and has uncontrolled local Lipschitz constants.
- The conventional network score magnitude grows above `10^3`, whereas OCSDF scores remain interpretable as approximate signed distances.

Reported lower bounds on the local Lipschitz constant for the conventional network after 10,000 steps:

| Dataset | Local Lipschitz lower bound |
|---|---:|
| One cloud | 26.66 |
| Two clouds | 122.84 |
| Two blobs | 1421.41 |
| Blob and cloud | 53.90 |
| Two moons | 258.73 |

### 6.2 Tabular anomaly detection

The method is evaluated on ODDS benchmark datasets under an unsupervised anomaly detection protocol: both normal and anomalous examples are present during training, but labels are not used. AUROC is computed following ADBench-style practice.

Hyperparameters:

- margin `m in {0.01, 0.05, 0.2, 1}`,
- `lambda = 100`,
- batch size 128,
- Newton-Raphson steps `T=4`,
- 40 epochs with 5 warm-start epochs using `T=0`.

Summary result:

- OCSDF average rank: `7.1 +/- 3.6` among 15 methods.
- Best average rank reported: Isolation Forest, `4.5 +/- 3.2`.
- OCSDF is competitive, not dominant, but adds robustness certificates and a parametric SDF interpretation.

Selected AUROC results from the anomaly detection protocol:

| Dataset | Dimension | Normal + anomalous | OCSDF AUROC |
|---|---:|---:|---:|
| breastw | 9 | 444 + 239 | 82.6 +/- 5.9 |
| cardio | 21 | 1655 + 176 | 95.0 +/- 0.1 |
| glass | 9 | 205 + 9 | 73.9 +/- 4.1 |
| http | 3 | 565287 + 2211 | 67.5 +/- 37 |
| ionosphere | 33 | 225 + 126 | 80.2 +/- 0.1 |
| lymphography | 18 | 142 + 6 | 96.1 +/- 4.9 |
| mammography | 6 | 10923 + 260 | 86.0 +/- 2.5 |
| musk | 166 | 2965 + 97 | 92.6 +/- 20 |
| optdigits | 64 | 5066 + 150 | 51.0 +/- 0.9 |
| pima | 8 | 500 + 268 | 60.7 +/- 1.0 |
| satimage-2 | 36 | 5732 + 71 | 97.9 +/- 0.4 |
| shuttle | 9 | 45586 + 3511 | 99.1 +/- 0.3 |
| smtp | 3 | 95126 + 30 | 87.1 +/- 3.5 |
| speech | 400 | 3625 + 61 | 46.0 +/- 0.2 |
| thyroid | 6 | 3679 + 93 | 95.9 +/- 0.0 |
| vertebral | 6 | 210 + 30 | 48.6 +/- 2.6 |
| vowels | 12 | 1406 + 50 | 94.7 +/- 0.7 |
| WBC | 30 | 357 + 21 | 93.6 +/- 0.1 |
| Wine | 13 | 119 + 10 | 81.5 +/- 0.9 |

A separate one-class tabular protocol trains on only normal data and tests on unseen normal + anomaly data. Reported OCSDF AUROCs include: Arrhythmia 80.0, Thyroid 98.3, Mammography 88.0, Vowels 96.1, Satimage-2 97.8, smtp 83.8.

### 6.3 Image one-class classification

The method is evaluated one-class-vs-all on MNIST and CIFAR-10. For each class, the model trains only on that class and tests against that class plus all other classes.

Baselines:

- Deep SVDD,
- One-Class SVM,
- Isolation Forest.

#### MNIST

Mean AUROC:

| Method / certificate radius | mAUROC |
|---|---:|
| OCSDF, epsilon = 0 | 95.5 +/- 0.4 |
| OCSDF, epsilon = 8/255 | 93.2 +/- 2.1 |
| OCSDF, epsilon = 16/255 | 89.9 +/- 3.5 |
| OCSDF, epsilon = 36/255 | 78.4 +/- 6.4 |
| OCSDF, epsilon = 72/255 | 57.5 +/- 7.5 |
| One-Class SVM | 91.3 +/- 0.0 |
| Deep SVDD | 94.8 +/- 0.9 |
| Isolation Forest | 92.3 +/- 0.5 |

OCSDF is competitive at epsilon 0 and uniquely provides certified AUROC at nonzero radii.

#### CIFAR-10

Mean AUROC:

| Method / certificate radius | mAUROC |
|---|---:|
| OCSDF, epsilon = 0 | 57.4 +/- 2.1 |
| OCSDF, epsilon = 8/255 | 53.1 +/- 2.1 |
| OCSDF, epsilon = 16/255 | 48.8 +/- 2.1 |
| OCSDF, epsilon = 36/255 | 38.4 +/- 1.9 |
| OCSDF, epsilon = 72/255 | 22.5 +/- 1.4 |
| One-Class SVM | 64.8 +/- 8.0 |
| Deep SVDD | 64.8 +/- 6.8 |
| Isolation Forest | 55.4 +/- 8.0 |

OCSDF underperforms the best CIFAR baselines in nominal AUROC but remains the only method among these comparisons with certificates at positive attack radii.

### 6.4 Empirical adversarial robustness

Empirical robustness is tested with `l2` PGD attacks using three random restarts and a step size proportional to the attack radius. Results show:

- OCSDF certificates are empirically respected.
- OCSDF is substantially more robust than Deep SVDD, especially on CIFAR-10.
- For 1-Lipschitz networks trained with HKR loss, PGD tends to find similar adversaries to other attacks, making the empirical results representative.

### 6.5 Visualization of the support

Because the backward pass through the classifier can generate boundary-seeking samples, OCSDF can visualize the learned support. The paper describes the forward computation graph as a classifier based on optimal transport and the backward graph as an image generator. Generated examples from MNIST show recognizable digit-like boundary samples.

### 6.6 Limitations on complex images

On the Cats vs Dogs dataset, training on cats and treating dogs as OOD gives AUROC barely above 55%. The authors attribute this to the limits of Euclidean distance in high-dimensional pixel space. Future work should use more perceptually meaningful distances or priors.

---

## 7. OCSDF for implicit shape parametrization

The paper connects OCC to implicit surface reconstruction. Classical SDF methods in computer graphics often require supervised point-to-surface distances computed via nearest-neighbor search or geometric queries. OCSDF instead learns from points sampled on or inside the object support.

Procedure:

1. Use ModelNet10 meshes.
2. Sample 2048 points per mesh with Trimesh.
3. Fit OCSDF on the point cloud.
4. Extract an isosurface with Lewiner marching cubes on a `200 x 200 x 200` voxel grid.
5. Compare against a Graphite implementation of surface reconstruction by restricted Voronoi cells.

Reported qualitative finding:

- With only 2048 points, many fine details are lost.
- OCSDF often recovers the global shape more smoothly and consistently than the baseline.
- The learned SDF has better extrapolation behavior from sparse point clouds.

Comparison with DeepSDF:

| Aspect | OCSDF | DeepSDF |
|---|---|---|
| Target | Generate complementary `Q` | Compute `S(x)` from point set |
| Cost bottleneck | Backward pass | Nearest-neighbor search |
| Loss | HKR loss | Squared SDF regression |
| Guarantee | `f` is 1-Lipschitz | No comparable guarantee |

---

## 8. Implementation details from appendices

### 8.1 Complementary distribution and proof idea

The formal definition requires compact support, separation from `X` by at least `2 epsilon`, and positive measure over all valid measurable regions outside that separation. Under this condition, the HKR hinge term can be made null, while the Wasserstein-like term encourages maximal signed amplitude under the 1-Lipschitz constraint. This makes the optimal function match signed distance up to the margin.

### 8.2 Preservation of complementary distributions

If `Q_t` is complementary and the next distribution is generated by the adapted Newton-Raphson algorithm, the paper proves that `Q_{t+1}` remains complementary under the exact minimizer assumptions. The 1-Lipschitz constraint prevents generated samples from overshooting the boundary.

### 8.3 Lazy alternating optimization

The exact max-min procedure is expensive. The practical algorithm performs only a fixed number of updates per iteration. This is faster but loses some theoretical guarantees and may introduce oscillations.

### 8.4 1-Lipschitz parametrization

Network kernels are reparametrized as orthogonal projections:

```math
\Theta_i = \Pi(W_i),
```

where `W_i` are unconstrained weights and `Pi` projects onto the Stiefel manifold. The projection uses the differentiable Bjorck algorithm. Optimization is performed over unconstrained weights.

### 8.5 Toy/tabular/shape architecture

For toy 2D, tabular, and implicit shape experiments:

- dense network with 4 hidden layers of width 512,
- GroupSort/fullsort activations,
- orthogonal square matrices,
- final dense linear output,
- about 792K parameters.

### 8.6 Image architecture

For MNIST and CIFAR-10:

- VGG-like convolutional architecture,
- orthogonal convolutional layers from Deel-Lip,
- GroupSort activations,
- `l2` norm-pooling layers,
- global `l2` norm pooling,
- dense linear output,
- about 2.1M parameters.

### 8.7 MNIST settings

- Pixel range: `[-1, 1]`.
- Feasible set: `B = [-1,1]^(28x28)`.
- `m approximately 0.79`, corresponding to about 1% of maximum image norm.
- `lambda = 200`.
- Batch size: 128.
- Newton-Raphson steps: `T = 16`.
- 70 epochs, with 10 warm-start epochs at `T=0`.
- Learning rate decays linearly from `1e-3` to `1e-6`.

### 8.8 CIFAR-10 settings

- Images are channel-centered and reduced.
- Approximate pixel interval: `[-2.1, 2.14]`.
- Feasible set: `B = [-2.1,2.14]^(32x32x3)`.
- `m approximately 0.11`, corresponding to about 0.2% of maximum image norm.
- `lambda = 1000`.
- Batch size: 128.
- Newton-Raphson steps: `T = 32`.
- 90 epochs, with 10 warm-start epochs at `T=0`.
- Learning rate decays linearly from `2.5e-4` to `2.5e-7`.

### 8.9 Deep SVDD comparison details

The paper evaluates multiple Deep SVDD configurations because results are sensitive to architecture and implementation:

1. original paper configuration with LeNet,
2. official PyTorch implementation with LeNet + BatchNorm,
3. PyOD TensorFlow dense implementation,
4. PyOD TensorFlow using the same unconstrained architecture as the OCSDF image network.

The best average among configurations is reported in the main comparison. The authors observe that Deep SVDD can show train/test score discrepancies, suggesting memorization or weak generalization outside the training class.

### 8.10 Per-class tuning on CIFAR-10

The main paper uses shared hyperparameters across all CIFAR-10 classes to avoid unfair per-class tuning. The appendix reports that per-class margin tuning improves CIFAR-10 OCSDF substantially, with a best average AUROC around `62.1 +/- 8.0`, compared with `57.4 +/- 2.1` in the shared-hyperparameter main table.

---

## 9. Main conclusions

OCSDF proposes a principled OCC framework based on learning the signed distance function to the positive data support. Its main strengths are:

- turns OCC into SDF learning,
- uses adaptive negative sampling without external OOD data,
- learns with a 1-Lipschitz neural network and HKR loss,
- provides certified AUROC against any bounded `l2` adversarial attack,
- remains competitive on tabular and simple image benchmarks,
- supports boundary visualization and implicit surface reconstruction.

The main limitations are:

- Euclidean distance in pixel space is inadequate for complex natural images,
- CIFAR-10 nominal performance trails some baselines under shared hyperparameters,
- theoretical guarantees weaken under lazy practical training,
- robust 1-Lipschitz convolutions remain an active research area.

Promising future directions include better priors for complementary sampling, Fourier or Perlin-noise priors, negative-data augmentation, more meaningful distances for images, and stronger 1-Lipschitz convolutional architectures.

---

## 10. Key takeaways

- **OCSDF = OCC via SDF approximation.** The classifier is meant to approximate distance to the support boundary, not just separate positives from arbitrary negatives.
- **HKR + 1-Lipschitz networks are the theoretical core.** Under a valid complementary distribution, the HKR minimizer recovers the SDF up to a margin shift.
- **Adaptive negative sampling is essential.** Uniform negatives are easy but inefficient in high dimension; gradient-based refinement concentrates them near the boundary.
- **Robustness is certified, not just empirical.** The 1-Lipschitz property gives an analytical AUROC lower bound under `l2` attacks.
- **The method is broader than anomaly detection.** It also provides tools for visualizing learned supports and reconstructing implicit surfaces from point clouds.
