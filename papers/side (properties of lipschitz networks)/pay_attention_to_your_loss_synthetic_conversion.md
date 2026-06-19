# Pay attention to your loss: understanding misconceptions about 1-Lipschitz neural networks

**Authors:** Louis Béthune, Thibaut Boissin, Mathieu Serrurier, Franck Mamalet, Corentin Friedrich, Alberto González-Sanz  
**Venue/version:** NeurIPS 2022; arXiv:2104.05097v6, 17 Oct 2022  
**Topic:** 1-Lipschitz neural networks for classification, robustness, expressiveness, loss design, and generalization.

## Synthetic abstract

The paper argues against a common misconception: constraining a neural network to be 1-Lipschitz does **not** inherently reduce its classification power. For classification, a 1-Lipschitz network can represent arbitrarily complex decision boundaries and can match the decision boundary of an unconstrained network after appropriate rescaling or construction. What often causes poor performance in practice is not the hypothesis class itself, but the **loss and its hidden hyperparameters**, especially the temperature in cross-entropy and the margin/regularization parameters in hinge-like losses.

The main message is that 1-Lipschitz networks offer a strong package for classification: they can be expressive, provide certifiable robustness, and enjoy statistical generalization guarantees. The loss controls the accuracy-robustness-generalization trade-off.

## Core terminology

- **AllNet:** conventional unconstrained feed-forward neural networks. They have no enforced bound on their Lipschitz constant.
- **LipNet1:** feed-forward neural networks constrained to be 1-Lipschitz. In the experiments, the networks use orthogonal weight matrices and GroupSort activations through the Deel.Lip library.
- **Lipschitz constant:** the smallest `L` such that `||f(x)-f(z)|| <= L ||x-z||`. Large values allow outputs to change sharply under small input perturbations.
- **Adversarial robustness radius:** the smallest perturbation size required to change a classifier prediction.
- **Certificate for LipNet1:** for binary classifier `sign(f)`, if `f` is 1-Lipschitz, the robustness radius at `x` is at least `|f(x)|`.
- **BCE / cross-entropy temperature `tau`:** the loss is `L_bce_tau(f(x), y) = -log sigma(y tau f(x))`. Changing `tau` is equivalent to changing the effective Lipschitz scale.
- **Hinge margin `m`:** `L_H_m(f(x), y)=max(0, m-yf(x))`.
- **HKR loss:** a hinge-regularized Kantorovich-Rubinstein loss, `L_hkr_m,alpha = -yf(x) + alpha max(0, m-yf(x))`.
- **MCR:** Mean Certifiable Robustness, a signed average certificate that rewards confident correct classification and penalizes confident wrong classification.

## Main claims

1. **Expressiveness:** 1-Lipschitz networks are as expressive as unconstrained networks for classification, even though not necessarily for regression.
2. **Robustness:** LipNet1 classifiers provide free local robustness certificates. The signed distance function is the most robust classifier among maximally accurate classifiers, while the WGAN discriminator maximizes mean certifiable robustness but may be a weak classifier.
3. **Generalization:** LipNet1 networks form a better-behaved statistical class. Their empirical loss converges to test loss for Lipschitz losses, unlike unconstrained networks in general.
4. **Loss parameters matter:** temperature, margin, and HKR regularization parameters move the model along an accuracy-robustness Pareto front.
5. **Architecture is not the only issue:** the paper argues that the loss is often overlooked compared with architectural design.

## 1. Introduction

Lipschitz-constrained networks are used for adversarial robustness, Wasserstein distance estimation, interpretability, and generalization. The usual objection is that they are less expressive. The paper contests this for classification.

For an unconstrained neural network `g : R^n -> R^K` with finite Lipschitz constant `L`, the rescaled network `f = g/L` is 1-Lipschitz and has the same argmax decision boundary. Classification decisions are invariant under positive logit rescaling. Thus, while Lipschitz constraints restrict regression values, they do not necessarily restrict classification boundaries.

The authors support this theoretically and empirically, including an experiment where a LipNet1 network reaches 99.96% accuracy on CIFAR-100 with random labels, illustrating high classification expressiveness.

## 2. Setting and losses

The main theory focuses on binary classification with labels `{-1, +1}` on compact input space `X subset R^n`. The positive and negative input distributions are denoted `P` and `Q`.

The paper uses:

- **BCE / logloss:** with temperature `tau`, usually hidden as `tau=1` in frameworks.
- **Hinge loss:** with margin `m`.
- **HKR loss:** combining Wasserstein/Kantorovich-Rubinstein structure with a hinge term.

The authors emphasize that `tau` is not a harmless implementation detail. For LipNet1, it controls fitting, robustness, and generalization. A low `tau` gives smoother, more robust, more generalizing classifiers but may underfit. A high `tau` improves fitting and accuracy but may reduce robustness and enlarge the generalization gap.

## 3. 1-Lipschitz classifiers are expressive

### Boundary fitting

**Proposition 1:** For any binary classifier with closed pre-images, there exists a 1-Lipschitz function `f` whose sign reproduces the classifier on `X`. Its gradient norm is 1 almost everywhere.

The construction uses the **signed distance function (SDF)** to the decision boundary. The SDF assigns positive distance on one side of the boundary and negative distance on the other. It is 1-Lipschitz and preserves the decision region.

**Corollary:** If the class supports are separated by positive distance, then a LipNet1 network can reach zero classification error.

**Interpretation:** Lipschitzness constrains the slope of the score landscape, not the geometric complexity of the decision boundary. The boundary can still be arbitrarily complex.

### Figure 1a: complex boundary

The authors train a LipNet1 network to fit the SDF of a fourth-iteration Von Koch snowflake-like boundary over 160,000 pixels. The network handles a sharp, nearly fractal boundary, supporting the claim that LipNet1 classification boundaries can be very complex.

### Why LipNet1 is often perceived as not expressive

LipNet1 networks cannot drive BCE to zero in the same way unconstrained networks can, because logits are bounded by the Lipschitz constraint and compact input domain. This may look like underfitting if one monitors loss rather than accuracy.

**Proposition 2:** BCE minimization over 1-Lipschitz functions on compact `X` attains a true minimum. The proof uses Arzela-Ascoli: bounded, equicontinuous sequences of Lipschitz functions have uniformly convergent subsequences.

The minimizer of BCE is not necessarily the minimizer of classification error. As `tau -> infinity`, BCE approaches empirical accuracy maximization. As `tau` decreases, the model becomes more robust/smooth but may ignore low-density components.

### Figure 1b: temperature matters

On a Gaussian-mixture toy example, low `tau` treats a small mixture component as noise, while high `tau` fits it. The result illustrates that bad LipNet1 accuracy in some literature can be caused by the default `tau=1`, not by lack of expressiveness.

## 4. 1-Lipschitz classifiers are certifiably robust

The paper studies the accuracy-robustness trade-off. It does not claim that robustness and accuracy are always incompatible; rather, for a fixed accuracy, robustness can be optimized up to a point, and accepting lower clean accuracy can yield higher certifiable robustness.

### Most robust classifier among maximally accurate classifiers

The SDF of the Bayes classifier gives the largest possible robustness certificates among classifiers with maximal accuracy.

**Corollary 2:** For the SDF of the Bayes classifier, the robustness certificate is tight: the robustness radius equals `|f(x)|`. The adversarial direction is given by `-f(x) grad_x f(x)`.

This SDF cannot be constructed directly in practice because it depends on the unknown Bayes classifier, but it characterizes the ideal maximally accurate and maximally certified classifier.

### Mean Certifiable Robustness and WGAN discriminators

The paper defines **MCR** for class distribution `P` with label `y` as:

`R_(P,y)(f) = E_{x~P} [ y f(x) ]`.

Correct high-confidence predictions increase this quantity; wrong high-confidence predictions decrease it.

**Property 2:** Maximizing MCR over both classes under a 1-Lipschitz constraint is exactly the Wasserstein-1 dual problem. The optimal classifier is a WGAN discriminator / Kantorovich-Rubinstein potential.

However, this classifier can have poor clean accuracy.

**Proposition 3:** There exist separated distributions for which every WGAN discriminator has classification error close to 1/2. Thus WGAN discriminators are optimally robust according to MCR but may be weak classifiers.

### Accuracy-robustness trade-off via losses

The paper connects common losses to this trade-off:

- As `tau -> 0`, BCE behaves like the Wasserstein/MCR objective.
- As `tau -> infinity`, BCE behaves more like empirical accuracy maximization.
- HKR interpolates between Wasserstein behavior (`alpha=0`) and hinge behavior (`alpha -> infinity`).

### Figure 2 and Figure 3

- Figure 2 shows three classifiers on a toy problem: one maximally accurate at small robustness radius, one maximizing MCR with lower clean accuracy, and one compromise.
- Figure 3 shows a Pareto front on CIFAR-10: tuning BCE/CCE temperature, hinge margin, or HKR parameters moves a LipNet1 CNN along the clean accuracy vs robust accuracy trade-off.

## 5. 1-Lipschitz classifiers have generalization guarantees

### Consistency

**Proposition 4:** For a bounded input space and a Lipschitz loss, the empirical loss minimizer over `Lip_L(X, R)` converges almost surely to the test loss minimizer as the sample size grows.

This is because bounded Lipschitz functions form a Glivenko-Cantelli class. Therefore, for LipNet1 networks and Lipschitz losses, train loss becomes a reliable proxy for test loss in the large-sample limit.

The paper stresses that this property is not shared by unconstrained networks in general.

### Figure 4: dataset size and temperature

On CIFAR-10, the authors train LipNet1 CNNs on fractions of the training set with different `tau`. Low `tau` generalizes better but plateaus at lower accuracy. High `tau` achieves higher training accuracy but requires more samples to close the generalization gap. An unconstrained AllNet severely overfits.

### Why BCE over AllNet is ill-behaved

**Proposition 5:** If unconstrained networks minimize BCE to zero on a non-trivial training set, their Lipschitz constants diverge to infinity, at least one weight matrix norm diverges, and predicted probabilities saturate to 0 or 1.

A simple linear example shows the issue: separable data can reach perfect classification, but BCE has no finite minimizer; parameters grow without bound.

The paper argues that this explains why AllNet networks are prone to adversarial vulnerability and overconfidence when trained with cross-entropy.

### PAC learnability with margin

The authors introduce classifiers that abstain when the certificate is below a margin `m`:

- predict `+1` if `f(x) >= m`
- predict `-1` if `f(x) <= -m`
- output `unknown` otherwise

**Proposition 6:** 1-Lipschitz classifiers with margin are PAC learnable. Their VC dimension is bounded by packing-number-like quantities:

`(1/m)^n vol(X)/vol(B) <= VCdim <= (3/m)^n vol(X)/vol(B)`.

This bound is **architecture independent**. The paper interprets this as allowing large LipNet1 architectures without the same kind of overfitting risk, provided the margin is properly chosen.

**Proposition 7:** For a fixed LipNet1 GroupSort architecture with `W` neurons, the VC dimension is `O((n+1) 2^W)`.

## Table 1: summary of differences

The paper contrasts AllNet and LipNet1:

| Property | AllNet | LipNet1 |
|---|---|---|
| Fit arbitrary classification boundary | Yes | Yes |
| Robustness certificates | No practical certificates | Yes, directly from `|f(x)|` |
| Consistent estimator | No in general | Yes for Lipschitz losses |
| BCE minimization | Ill-defined; Lipschitz constant diverges | Minimum is attained |
| Wasserstein minimization | Diverges during training | Attained; maximizes MCR |
| Hinge loss | Attained, but no margin guarantees | Attained; margin must be tuned |
| HKR loss | Ill-defined/diverging in unconstrained setting | Controls accuracy-robustness trade-off |
| VC bounds | Architecture dependent | Margin-based bound can be architecture independent |

## 6. Related work

The paper situates LipNet1 networks within work on:

- spectral normalization and Lipschitz regularization;
- Parseval and orthogonal neural networks;
- GroupSort and expressive Lipschitz activations;
- orthogonal convolution kernels;
- normalizing flows, recurrent networks, graph neural networks, and reinforcement learning;
- large-margin learning and adversarial robustness;
- generalization bounds for Lipschitz classifiers;
- Wasserstein GANs and optimal transport.

A key technical point is that ReLU-based Lipschitz networks may have expressiveness issues, motivating GroupSort-like activations.

## 7. Conclusion

The paper challenges the idea that Lipschitz constraints intrinsically degrade classification performance. For classification, LipNet1 networks can fit arbitrary decision boundaries, provide robustness certificates, and enjoy meaningful generalization guarantees. The practical challenge is not only architectural; it is also about choosing the right loss and tuning its parameters.

The authors argue that cross-entropy is not always the best default choice for LipNet1 networks. Margin-based losses, especially hinge and HKR, offer better control over robustness and accuracy.

## 8. Perspectives

The authors view Lipschitz-constrained networks as a useful bridge between theoretical ML and empirical deep learning. They suggest future work on:

- scaling empirical verification to large datasets such as ImageNet;
- understanding how AllNet practices such as weight decay and mixup implicitly affect Lipschitz constants;
- improving efficient training of LipNet1 networks;
- adapting architectural tools such as residual connections and normalization to the LipNet1 setting;
- separating the roles of architecture, optimization, and generalization in deep learning.

## Appendix synthesis

### A. Proofs of expressiveness

The appendix gives a constructive proof using signed distance functions. The multiclass case is handled by defining a multiclass SDF with one score per class. The paper also states that LipNet1 can imitate AllNet decision boundaries for classification.

### B. Proofs of robustness claims

The appendix proves that the SDF gives tight robustness certificates and derives the Wasserstein/MCR equivalence from Kantorovich-Rubinstein duality. It also constructs pathological distributions where WGAN discriminators are weak classifiers despite optimizing MCR.

### C. Generalization proofs

The appendix proves consistency through Glivenko-Cantelli arguments and shows AllNet can overfit arbitrary training sets while having arbitrarily bad test loss. It also develops the VC dimension arguments for margin-based Lipschitz classifiers and GroupSort networks.

### D. Deel.Lip networks

The experiments use the Deel.Lip library, orthogonal layers, spectral normalization, Bjorck orthogonalization, and GroupSort activations. The authors discuss related approaches for enforcing Lipschitz constraints in affine, convolutional, attention, and recurrent layers.

### E. Multiclass HKR

The binary HKR loss is extended to multiclass classification using a multiclass mean certifiable robustness objective. The construction avoids a simple one-vs-all scheme because that would not yield meaningful certificates.

### F. Gradient Norm Preserving networks

The appendix discusses vanishing/exploding gradients. GNP networks avoid exploding gradients under appropriate assumptions, but residual connections complicate the GNP property. BCE gradients do not vanish elementwise at the LipNet1 minimizer, although batch averages can still cancel at optimum.

### G. Wasserstein discriminator and Lipschitz constant

The Wasserstein discriminator objective is invariant to the chosen finite Lipschitz upper bound after appropriate scaling. This supports the view that the Wasserstein direction is structurally different from accuracy maximization.

### H. BCE through optimal transport

BCE minimization is analyzed as a non-stationary optimal-transport-like process. At small temperature, BCE resembles a Wasserstein objective; at high effective temperature or increasing Lipschitz constant, it approaches hard classification behavior and can increase adversarial vulnerability.

### I. CIFAR-100 with random labels

The authors train a LipNet1 network on CIFAR-100 with random labels and achieve 99.96% train accuracy. This strongly supports the claim that LipNet1 classification capacity is not intrinsically weak.

### J. Consistency experiment protocol

CIFAR-10 experiments vary training-set fraction and temperature. Results show that higher temperature needs more data to reduce the generalization gap; lower temperature generalizes more easily but underfits.

### K. Accuracy/robustness trade-off protocol

A small 0.4M-parameter LipNet1 CNN is trained with different CCE, hinge, and HKR parameters. The resulting CIFAR-10 experiments reveal a Pareto front between validation accuracy and certified robustness. The authors warn that robust metrics and architecture choices affect the apparent comparison.

### L. Hardware

Toy experiments run on a personal workstation with an NVIDIA GeForce GTX 1050 Ti. Larger CIFAR-10 experiments run on Google Cloud TPU v2-8. TensorFlow is used except for one JAX toy experiment.

### M. Divergence of AllNet weights

A simple ConvNet on MNIST demonstrates the BCE divergence phenomenon: validation accuracy remains high, but spectral norms continue to grow. Adam worsens the growth relative to SGD in this experiment.

### N. Stability of LipNet1 training

Fashion-MNIST experiments show that changing temperature during training moves models along the Pareto front. The final position depends mainly on the final temperature, not initialization. Larger architectures shift the Pareto front toward both higher accuracy and higher MCR.

## Practical takeaways

- Do not judge LipNet1 expressiveness by BCE loss alone; classification accuracy and decision boundaries tell a different story.
- Tune the loss temperature or margin explicitly. The framework default `tau=1` is often arbitrary.
- Use low temperature or Wasserstein-like objectives when robustness and generalization are prioritized.
- Use higher temperature or stronger hinge/accuracy terms when clean accuracy is prioritized.
- HKR provides a direct knob for moving along the accuracy-robustness Pareto frontier.
- 1-Lipschitz constraints give certificates essentially for free, unlike many post-hoc certification methods.
- Unconstrained cross-entropy training can drive Lipschitz constants and weight norms upward even when accuracy has already saturated.

## One-sentence summary

For classification, 1-Lipschitz neural networks are not inherently less expressive than unconstrained networks; their main advantage is that, with the right loss parameters, they combine expressive decision boundaries with robustness certificates and stronger generalization behavior.
