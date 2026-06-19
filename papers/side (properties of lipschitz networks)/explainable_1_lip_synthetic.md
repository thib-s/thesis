# On the Explainable Properties of 1-Lipschitz Neural Networks: An Optimal Transport Perspective

**Authors:** Mathieu Serrurier, Franck Mamalet, Thomas Fel, Louis Bethune, Thibaut Boissin  
**Venue/version:** NeurIPS 2023 / arXiv v3, 2 Feb 2024  
**Core topic:** Why 1-Lipschitz neural networks trained with an optimal-transport dual loss produce unusually meaningful input gradients and saliency maps.

---

## Executive summary

The paper studies **Optimal Transport Neural Networks (OTNNs)**: neural networks constrained to be **1-Lipschitz** and trained with the **hinge Kantorovich-Rubinstein (hKR) loss**, which is the dual of an optimal transport problem with hinge regularization.

The central claim is that, unlike standard neural networks, OTNNs have input gradients that are not merely local adversarial directions. Instead, their gradients align with the **optimal transport plan between classes**, point toward the **nearest decision boundary**, and can therefore be interpreted as **counterfactual explanations**.

Empirically, this makes the simple **Saliency Map** method, defined as the absolute input gradient, highly competitive or superior to more expensive explanation methods such as SmoothGrad, Integrated Gradients, and Grad-CAM. OTNN saliency maps are less noisy, more concentrated on relevant image regions, more stable, more faithful, and more aligned with human attention on ImageNet than a large collection of standard models.

---

## Main contributions

1. **Theory of OTNN gradients**
   - The gradient of the optimal hKR solution points in the direction of the optimal transport plan.
   - For separable classes, the point reached by moving from an input in the negative gradient direction by the model score lies on the decision boundary.
   - The nearest adversarial example is therefore explicitly identified by the model gradient.

2. **Reinterpretation of adversarial examples**
   - For OTNNs, adversarial directions are not arbitrary noise-exploiting perturbations.
   - They lie on a transport path from one class distribution to another.
   - This turns gradient-based perturbations into counterfactual explanations: they indicate how an input should change to become another class.

3. **Empirical explainability results**
   - Saliency Maps on OTNNs outperform saliency and many advanced attribution methods on unconstrained networks.
   - OTNN explanations show strong fidelity and stability across FashionMNIST, CelebA, Cat vs Dog, and ImageNet.
   - OTNN Saliency Maps are remarkably aligned with human feature importance on ImageNet.

4. **Scalability**
   - The paper reports large 1-Lipschitz models, including ResNet50-style OTNNs on ImageNet.
   - These models are slower to train than unconstrained networks but have comparable inference cost.

---

## Background

### 1-Lipschitz neural networks

A function \(f : X \to \mathbb{R}\) is **1-Lipschitz** when:

\[
\forall x,z \in X, \quad |f(x)-f(z)| \leq \|x-z\|.
\]

A 1-Lipschitz classifier provides certified robustness because its score cannot change too rapidly with respect to the input. For a classifier based on \(\mathrm{sign}(f)\), the absolute score \(|f(x)|\) is a lower bound on the distance to a decision-changing perturbation.

Several layer-level constraints can enforce or approximate this property, including:

- Frobenius normalization,
- spectral normalization,
- orthogonalization,
- Bjork orthogonalization,
- GroupSort activations.

In this work, OTNNs are built using the **DEEL.LIP** library, with spectral layers and Bjork orthogonalization.

### Optimal transport and hKR loss

The paper builds on earlier work showing that binary classification with 1-Lipschitz networks can be formulated through an optimal transport dual objective.

For two conditional distributions:

- \(\mu = P(x \mid y=1)\),
- \(\nu = P(x \mid y=-1)\),

and a 1-Lipschitz function \(f\), the hKR loss is:

\[
L^{hKR}_{\lambda,m}(f)
= \mathbb{E}_{x \sim \nu}[f(x)]
- \mathbb{E}_{x \sim \mu}[f(x)]
+ \lambda \mathbb{E}_{(x,y)\sim P}(m-yf(x))_+.
\]

where:

- \(m>0\) is the margin,
- \((z)_+ = \max(0,z)\),
- \(\lambda\) weights the hinge regularization.

The resulting classifier is \(\mathrm{sign}(f^*)\), where \(f^*\) minimizes the hKR loss over 1-Lipschitz functions.

The paper calls such models **Optimal Transport Neural Networks (OTNNs)**.

### Standard saliency maps and their problem

A Saliency Map is:

\[
g(x) = |\nabla_x f(x)|.
\]

For standard unconstrained networks, saliency maps are often noisy because gradients may point toward brittle adversarial features rather than meaningful semantic changes. This motivates smoother or more complex alternatives such as:

- SmoothGrad,
- Integrated Gradients,
- Gradient x Input,
- Grad-CAM,
- perturbation-based methods such as RISE.

The paper argues that OTNNs change the situation: the raw gradient already has a meaningful transport and counterfactual interpretation.

---

## Theoretical results

Let \(\pi\) be the optimal transport plan associated with the hKR minimizer \(f^*\). For an input \(x\), let \(\gamma_\pi(x)\) denote its image under the transport plan. When the plan is not deterministic, the paper uses the point with maximal mass under \(\pi\).

### Proposition 1: Gradient gives the transport-plan direction

For \(x \sim \mu\) and \(z = \gamma_\pi(x) \in \nu\), there exists \(t \ge 0\) such that:

\[
\gamma_\pi(x) = x - t \nabla_x f^*(x).
\]

For \(x \sim \nu\), the same holds with \(t \le 0\).

**Interpretation:** the OTNN gradient points along the optimal transport direction from one class distribution to the other. This is the key theoretical reason why OTNN saliency maps are meaningful.

### Proposition 2: Gradient reaches the decision boundary

Assume the class distributions have disjoint supports with minimum distance \(\epsilon\), and \(m < 2\epsilon\). For \(x \sim P\):

\[
x_\delta = x - f^*(x) \nabla_x f^*(x)
\]

lies on the decision boundary:

\[
\partial X = \{x' \in X \mid f^*(x') = 0\}.
\]

**Interpretation:** moving from \(x\) in the gradient direction by the signed model score reaches the model boundary exactly.

### Corollary: Exact adversarial example

Under the same separability assumptions:

\[
\mathrm{adv}(f^*,x) = x_\delta = x - f^*(x)\nabla_x f^*(x).
\]

Thus, for OTNNs, the nearest adversarial example is explicitly obtained by the input gradient.

This explains why attacks such as PGD or Carlini-Wagner can reduce to FGSM-like behavior on OTNNs: the optimal adversarial direction is already given by the gradient.

### Geometric illustration

The paper illustrates these propositions with two concentric Koch snowflake distributions. The learned 0-level set forms the decision boundary, and the segments:

\[
[x, x - f(x)\nabla_x f(x)]
\]

land on that boundary while following the transport direction.

---

## Link to counterfactual explanations

Counterfactual explanations answer questions of the form:

> Why was the decision A rather than B?

A good counterfactual should usually be valid, sparse, close to the data manifold, actionable, and causally meaningful when possible.

The paper connects OTNN gradients to counterfactual explanations through optimal transport:

- Optimal transport gives a global matching between class distributions.
- For a sample \(x\), \(\gamma_\pi(x)\) is a sample in the other class that is close on average under the transport plan.
- Prior work argues that transport-based counterfactuals can act as surrogates for causal counterfactuals, and may coincide with causal counterfactuals in some settings.

Because Proposition 1 shows:

\[
\gamma_\pi(x) = x - t\nabla_x f^*(x),
\]

the OTNN gradient gives the direction toward the transport-based counterfactual.

Using \(t=f^*(x)\) reaches the decision boundary; using larger \(t\) moves further toward the other class. This means OTNN gradients simultaneously encode:

- adversarial direction,
- boundary direction,
- transport direction,
- counterfactual direction.

This is the conceptual reason why the paper expects OTNN Saliency Maps to be less noisy and more semantically meaningful than standard-network gradients.

---

## Experimental setup

### Datasets

The experiments cover small, medium, and large vision tasks:

| Dataset | Task | Size / notes |
|---|---:|---|
| FashionMNIST | 10-class classification | 28 x 28 x 1 images; 50k train, 10k test |
| CelebA | 22 binary facial attributes | 218 x 178 x 3; imbalanced labels |
| Cat vs Dog | binary classification | 224 x 224 x 3 after preprocessing; 17,400 train, 5,800 test |
| ImageNet | 1000-class classification | large-scale natural images |

CelebA labels include attributes such as Bald, Blond_Hair, Blurry, Eyeglasses, Gray_Hair, Mouth_Slightly_Open, Mustache, Rosy_Cheeks, Smiling, Wearing_Hat, Wearing_Lipstick, Young, etc. Several labels are highly imbalanced; for example Bald, Mustache, Wearing_Hat, and Gray_Hair have very low positive rates.

### Architectures

The paper compares OTNNs and unconstrained networks with similar macro-architectures:

- FashionMNIST: VGG-style convolutional network.
- CelebA: VGG-style convolutional network.
- Cat vs Dog and ImageNet: ResNet50-style architecture.
- ImageNet also includes a larger OTNN variant with about 50M parameters.

Main differences:

| Component | OTNN | Unconstrained network |
|---|---|---|
| Convolution / dense layers | SpectralConv2D, SpectralDense | Conv2D, Dense |
| Activation | GroupSort2 | ReLU |
| Normalization | Lipschitz-compatible components, batch centering where needed | BatchNorm |
| Pooling | L2-based Lipschitz pooling | Average/global average pooling |
| Loss | hKR loss or multiclass hKR variant | cross-entropy |

### Multiclass hKR extension

For multiclass problems, the paper uses one 1-Lipschitz output function per class. It also proposes a softmax-weighted sample-wise hKR variant to address convergence and class-weighting issues in large multiclass settings.

The proposed loss contains:

- a weighted one-vs-all KR term,
- a softmax-weighted hinge term,
- hyperparameters \(\lambda\), margin \(m\), and temperature \(\alpha\).

Hyperparameters reported:

| Dataset | Main hKR parameters |
|---|---|
| CelebA | \(\lambda=20, m=1\) |
| FashionMNIST | \(\lambda=5, \alpha=10, m=0.5\) |
| Cat vs Dog | \(\lambda=10, m=18\) |
| ImageNet | \(\lambda=500, \alpha=200, m=0.05\) |

### Training

All networks use Adam. Reported training schedules include:

- CelebA: batch size 128, 200 epochs, learning rate \(10^{-2}\).
- FashionMNIST: batch size 128, 200 epochs, staged learning rates from \(5\cdot10^{-4}\) down to \(10^{-6}\).
- Cat vs Dog: batch size 256, 200 epochs, staged learning rates from \(10^{-2}\) down to \(10^{-9}\).
- ImageNet: batch size 512, 40 epochs, staged learning rates from \(5\cdot10^{-4}\) down to \(10^{-9}\).

---

## Classification performance

OTNNs remain competitive with unconstrained models, though not always equal on the largest tasks.

| Dataset | OTNN performance | Unconstrained performance |
|---|---:|---:|
| FashionMNIST | about 88.5%-88.6% accuracy | about 88.5% accuracy |
| CelebA | about 81% average sensitivity, 82% average specificity | strong but less balanced on rare labels |
| Cat vs Dog | 96% accuracy | 98% accuracy |
| ImageNet | 67% accuracy; 70% for larger OTNN | 75% for standard ResNet50 |

The appendix reports that the proposed multiclass hKR variant dramatically improves FashionMNIST accuracy over the earlier multiclass hKR formulation:

| Model | FashionMNIST accuracy |
|---|---:|
| Unconstrained | 88.5 |
| Earlier multiclass hKR | 72.2 |
| Proposed multiclass hKR variant | 88.6 |

---

## Explanation methods evaluated

The paper evaluates:

- **Saliency:** \(g(x)=|\nabla_x f(x)|\)
- **SmoothGrad:** average gradient over noisy samples
- **Integrated Gradients:** accumulated gradients from baseline to input
- **Gradient x Input**
- **Grad-CAM**
- **RISE**, in supplementary experiments

The library used for attribution maps is **Xplique**.

---

## Explanation metrics

The paper uses several XAI metrics.

### Fidelity metrics

1. **Deletion**  
   Removes important pixels and measures score drop. Lower AUC is better.

2. **Insertion**  
   Starts from a baseline and restores important pixels. Higher AUC is better.

3. **micro-Fidelity**  
   Measures correlation between summed attribution importance and score drop when corresponding variables are removed. Higher is better.

4. **Robustness-Sr**  
   Measures adversarial distance when perturbations are restricted to important features. Lower is better, but not directly comparable across models with different intrinsic robustness.

### Stability and complexity

- **Stability:** explanation variation around nearby samples, measured with L2 or Spearman-rank distance. Lower is better.
- **Distance between explanations:** L2 or \(1-\rho\), where \(\rho\) is Spearman correlation.
- **Explanation complexity:** JPEG compression size as a proxy for Kolmogorov complexity. Lower is better.
- **BlockMNIST proxy accuracy:** fraction of top-k saliency pixels falling inside a null block. Lower is better.

### Human alignment

The paper also evaluates alignment with human feature importance using the **ClickMe** dataset and methodology from prior work on harmonizing deep-network strategies with humans.

---

## Main quantitative results

### micro-Fidelity and stability

Across FashionMNIST, CelebA, Cat vs Dog, and ImageNet, OTNN explanations generally outperform those from unconstrained networks.

Key reported findings from the main table:

- OTNN Saliency has much higher micro-Fidelity than unconstrained Saliency.
- For most datasets and baselines, Saliency on OTNN beats every tested attribution method on unconstrained models.
- OTNN explanations are more stable under perturbations.
- The distance between Saliency and SmoothGrad is much lower for OTNNs, often nearly zero.

Representative values:

| Metric / dataset | OTNN Saliency | Unconstrained Saliency |
|---|---:|---:|
| micro-Fidelity Uniform, FashionMNIST | 0.156 | -0.001 |
| micro-Fidelity Uniform, CelebA | 0.244 | 0.052 |
| micro-Fidelity Uniform, ImageNet | 0.240 | 0.004 |
| micro-Fidelity Zero, FashionMNIST | 0.246 | 0.034 |
| micro-Fidelity Zero, CelebA | 0.325 | 0.082 |
| micro-Fidelity Zero, ImageNet | 0.147 | 0.049 |
| Stability Spearman, FashionMNIST | 0.59 | 0.91 |
| Stability Spearman, CelebA | 0.51 | 0.77 |
| Stability Spearman, ImageNet | 0.60 | 0.74 |

Since lower stability score is better, OTNN explanations are more stable.

### Saliency vs SmoothGrad

For OTNNs, Saliency and SmoothGrad are extremely close. This suggests that the raw OTNN gradient is already smooth and meaningful, so the expensive smoothing process of SmoothGrad is often unnecessary.

Reported Saliency-SmoothGrad distances:

| Dataset | OTNN | Unconstrained |
|---|---:|---:|
| FashionMNIST | \(7.5\cdot10^{-3}\) | \(7.0\cdot10^{-2}\) |
| CelebA | \(3.1\cdot10^{-4}\) | \(1.4\cdot10^{-1}\) |
| Cat vs Dog | \(3.7\cdot10^{-8}\) | \(4.1\cdot10^{-8}\) |
| ImageNet | \(3.7\cdot10^{-8}\) | \(4.3\cdot10^{-8}\) |

The paper visually illustrates this on CelebA, Cat vs Dog, and ImageNet: OTNN Saliency and SmoothGrad are nearly indistinguishable, whereas unconstrained explanations remain noisier.

### Human feature alignment on ImageNet

The paper reports that OTNN Saliency Maps align better with human attention than many modern models, including convolutional networks, transformers, self-supervised models, robust models, and harmonized models.

Reported ImageNet comparison:

| Model | Accuracy | Human alignment |
|---|---:|---:|
| CLIP | 56.0 | 0.03 |
| Swin | 85.2 | 0.03 |
| ViT/ConvNeXt | 85.8 | 0.15 |
| Inception | 81.1 | 0.25 |
| Adversarial ResNet50 | 74.8 | 0.33 |
| ResNet50 | 76.0 | 0.33 |
| VGG16 | 71.3 | 0.35 |
| Harmonized ResNet50 | 77.0 | 0.44 |
| OTNN | 67.0 | 0.54 |
| OTNN large | 70.0 | 0.57 |

The paper emphasizes that OTNNs surpass the previously observed Pareto front of accuracy vs human alignment.

---

## Qualitative results

### Figure 1: overview of OTNN gradient behavior

The first figure shows four kinds of evidence:

1. **MNIST counterfactuals:** following \(x' = x - t\nabla_x f(x)\) transforms digits into counterfactual digits, such as a 0 becoming a 5.
2. **CelebA counterfactuals:** the same procedure changes semantic facial attributes, such as mouth open/closed or blond hair/not blond hair.
3. **Saliency comparison:** OTNN Saliency Maps focus on relevant parts of cats and birds, whereas unconstrained ResNet50 maps are noisy and often highlight irrelevant pixels.
4. **Feature visualization:** gradient ascent with OTNN produces clearer class prototypes than unconstrained ResNet50.

### Figure 2: Koch snowflake toy problem

The figure shows two concentric distributions, the learned decision boundary, and transport-gradient segments. Moving each point by:

\[
x' = x - f(x)\nabla_x f(x)
\]

lands on the decision boundary, matching Proposition 2.

### Figure 3: Saliency vs SmoothGrad

On CelebA, Cat vs Dog, and ImageNet, OTNN Saliency is visually similar to SmoothGrad. For unconstrained networks, the maps are noisier and less consistent.

### Figure 4: human alignment

OTNN and OTNN-large appear above prior models in the accuracy/alignment plot, showing unusually high human-feature alignment for their accuracy levels.

### Figures 5-6: counterfactuals

The paper shows counterfactual transformations on:

- FashionMNIST: trousers, dresses, pullovers, shirts, sneakers, sandals, bags, coats, etc.
- CelebA: mouth open/closed, blurry/not blurry, rosy cheeks/no rosy cheeks.
- Cat vs Dog: cat-to-dog and dog-to-cat changes.
- ImageNet: examples such as cabbage butterfly to monarch butterfly and great grey owl to bald eagle.

The authors note that ImageNet counterfactuals are less compelling, likely because defining a class-pair transport plan is difficult with 1000 classes.

---

## Supplementary results

### CelebA performance

The appendix reports sensitivity/specificity for all 22 CelebA attributes. OTNNs often improve sensitivity on rare labels compared with unconstrained models, though sometimes with lower specificity.

Examples:

| Attribute | Unconstrained sens/spec | OTNN sens/spec |
|---|---:|---:|
| Bald | 0.64 / 1.00 | 0.87 / 0.83 |
| Mustache | 0.47 / 0.99 | 0.86 / 0.76 |
| Receding_Hairline | 0.47 / 0.98 | 0.81 / 0.79 |
| Rosy_Cheeks | 0.46 / 0.99 | 0.82 / 0.80 |
| Wearing_Necktie | 0.75 / 0.98 | 0.87 / 0.86 |

This reflects a more balanced behavior on imbalanced labels.

### Robustness-Sr

OTNN Saliency obtains top or near-top rankings under Robustness-Sr in supplementary experiments.

| Dataset | OTNN Saliency rank | Unconstrained Saliency rank |
|---|---:|---:|
| CelebA | 1 | 2 |
| FashionMNIST | 3 | 4 |

The authors caution that raw Robustness-Sr scores are not directly comparable between OTNNs and unconstrained networks because OTNNs are intrinsically more robust.

### Explanation complexity

JPEG-compression complexity of Saliency Maps is lower for OTNNs, especially on CelebA:

| Dataset | OTNN | Unconstrained |
|---|---:|---:|
| CelebA | 9.48 kB | 16.84 kB |
| FashionMNIST | 0.92 kB | 0.94 kB |

Lower complexity reflects cleaner, less noisy maps.

### BlockMNIST discriminative-feature proxy

The paper evaluates whether saliency top-k pixels fall into a random null block. OTNNs behave similarly to adversarially robust models and much better than standard models.

| k | Unconstrained | Adversarial | OTNN |
|---:|---:|---:|---:|
| 2.5 | 43.8 | <1 | <1 |
| 5 | 42.5 | <1 | <1 |
| 10 | 44.8 | 2.3 | 1.8 |
| 15 | 46.5 | 7.9 | 6.6 |
| 20 | 47.5 | 16.7 | 16.8 |
| 25 | 48.1 | 24.2 | 26.0 |
| 30 | 48.4 | 29.9 | 32.9 |

---

## Appendix: architectures and implementation details

### FashionMNIST architecture

OTNN uses repeated SpectralConv2D + GroupSort2 blocks, strided spectral convolutions, scaled global average pooling, and SpectralDense output. The unconstrained counterpart uses Conv2D + BatchNorm + ReLU, global average pooling, and Dense output.

### CelebA architecture

OTNN uses SpectralConv2D + GroupSort2 blocks, L2NormPooling, SpectralDense layers, and final 22-dimensional output. The unconstrained network mirrors the structure with Conv2D, BatchNorm, ReLU, average pooling, and Dense layers.

### ResNet50-style 1-Lipschitz architecture

The OTNN ResNet uses:

- SpectralConv2D layers,
- BatchCentering,
- GroupSort2,
- invertible downsampling,
- Lipschitz residual additions,
- ScaledL2NormPooling,
- ScaledGlobalL2NormPooling,
- SpectralDense output.

The architecture has roughly 25M parameters, like standard ResNet50. The large version increases hidden channel counts by 1.5x and reaches about 50M parameters.

---

## Limitations and caveats

1. **Training cost**
   - OTNNs train roughly 3-6 times slower than unconstrained networks.
   - Inference cost is similar.

2. **Accuracy trade-off**
   - OTNNs are competitive, but ImageNet accuracy remains below standard unconstrained ResNet50.

3. **ImageNet counterfactuals**
   - The counterfactual visual transformations are less convincing on ImageNet than on simpler or binary tasks.
   - The authors attribute this to the complexity of transport across 1000 classes.

4. **Theoretical assumptions**
   - Some formal results rely on separability and disjoint support assumptions.
   - Experiments suggest some properties hold more broadly, but the proof is under stronger conditions.

5. **Counterfactual realism**
   - The method is not intended to compete with GAN-based or causality-based counterfactual generation in image realism.
   - Its strength is that the gradient has a formal transport interpretation.

---

## Conclusion

The paper argues that OTNNs make raw input gradients meaningfully interpretable. Because the hKR-trained 1-Lipschitz model solves an optimal-transport dual problem, its gradient aligns with the transport plan between classes. Under separability assumptions, this same gradient also identifies the nearest decision boundary and the closest adversarial example.

This gives Saliency Maps for OTNNs a dual interpretation:

- as local feature attributions,
- and as directions toward transport-based counterfactuals.

Experiments show that this produces cleaner, more faithful, more stable, and more human-aligned explanations than standard unconstrained networks. The main practical implication is that, for OTNNs, a fast and simple Saliency Map can outperform or match more computationally expensive attribution methods.

---

## One-sentence takeaway

**Training a 1-Lipschitz neural network with an optimal-transport dual loss makes the input gradient align with class-to-class transport, turning ordinary saliency maps into robust, stable, and counterfactual explanations.**
