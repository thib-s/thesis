# Achieving Robustness in Classification Using Optimal Transport with Hinge Regularization

**Authors:** Mathieu Serrurier, Franck Mamalet, Alberto Gonzalez-Sanz, Thibaut Boissin, Jean-Michel Loubes, Eustasio del Barrio  
**Type/date:** Preprint, April 27, 2021  
**Topic:** Certified and empirical adversarial robustness for classifiers through 1-Lipschitz neural networks and a hinge-regularized optimal-transport loss.

## Abstract - synthetic version

Deep neural networks can be vulnerable to adversarial examples: small, often imperceptible perturbations that change predictions. Bounding a model's Lipschitz constant gives certifiable robustness, but standard losses do not naturally exploit this constraint. This paper proposes a binary classification framework based on optimal transport. The classifier is a 1-Lipschitz neural network trained with a **hinge-regularized Kantorovich-Rubinstein loss**. The loss is interpretable as an optimal-transport problem, provides robustness certificates, admits an optimal solution, and extends to multiclass classification. Experiments on MNIST, CIFAR-10, and CelebA show improved robustness without a major accuracy drop. For the proposed models, adversarial attacks tend to make meaningful, structured changes to the input, which can also serve as explanations.

## 1. Motivation and context

Adversarial attacks exploit the high sensitivity of unconstrained neural networks. White-box attacks such as **FGSM**, **PGD**, and **Carlini-Wagner** use gradients or logits; black-box attacks such as boundary or pointwise attacks iteratively reduce a perturbation until it fools the model.

The paper groups defenses into three families:

1. **Agnostic defenses**, independent of the classifier, such as randomized smoothing or Defense-GAN.
2. **Adversarial training**, framed as saddle-point optimization with adversarial examples in the training loop.
3. **Lipschitz-constrained models**, which bound sensitivity and can provide robustness certificates.

The proposed method belongs to the third family, but its novelty is to connect the 1-Lipschitz constraint directly to an optimal-transport classification objective.

## 2. Wasserstein distance and the vanilla KR classifier

The paper focuses on the 1-Wasserstein distance, also called the Earth Mover distance. For two distributions \(\mu\) and \(\nu\) on \(\Omega\):

\[
W(\mu,\nu)=\inf_{\pi\in\Pi(\mu,\nu)}\mathbb{E}_{x,z\sim\pi}\|x-z\|
=\sup_{f\in Lip_1(\Omega)} \mathbb{E}_{x\sim\mu}[f(x)]-\mathbb{E}_{x\sim\nu}[f(x)].
\]

For binary classification with labels \(Y\in\{-1,1\}\), the class-conditional distributions are:

\[
P_+ = P(X\mid Y=1),\qquad P_- = P(X\mid Y=-1).
\]

A natural idea is to learn a 1-Lipschitz function using the Kantorovich-Rubinstein dual objective and classify by the sign of a centered version of the learned function:

\[
f_c^*(x)=f^*(x)-\frac{1}{2}\left(\mathbb{E}_{z\sim P_+}[f^*(z)]+\mathbb{E}_{z\sim P_-}[f^*(z)]\right).
\]

This vanilla classifier has attractive robustness properties because the optimal potential is linked to transport paths:

\[
P_{x,z\sim\pi^*}\left(f^*(x)-f^*(z)=\|x-z\|\right)=1,
\]

and satisfies a gradient norm property on the transport support:

\[
\|\nabla f^*\|=1 \quad \text{almost surely on the support of } \pi^*.
\]

However, the paper shows on the two-moons dataset that this vanilla Wasserstein classifier is weak: even when classes are visually separable, the centered function values overlap across classes, making the zero-threshold separator suboptimal.

## 3. Hinge-regularized Kantorovich-Rubinstein classifier

To make the optimal-transport formulation useful for classification, the authors add a hinge regularization term:

\[
\sup_{f\in Lip_1(\Omega)} -L^{hKR}_{\lambda}(f)
=
\inf_{f\in Lip_1(\Omega)}
\mathbb{E}_{x\sim P_-}[f(x)]-
\mathbb{E}_{x\sim P_+}[f(x)]
+\lambda\mathbb{E}_{x}(1-Yf(x))_+.
\]

Here \((1-yf(x))_+=\max(0,1-yf(x))\), and \(\lambda\ge 0\). The goal is to learn a 1-Lipschitz neural network minimizing this **hinge-KR loss**.

### Intuition

The objective simultaneously:

- separates the classes with a margin, through the hinge term;
- spreads the two class distributions as much as possible, through the KR/Wasserstein term;
- preserves the transport interpretation and 1-Lipschitz robustness certificate.

Using only hinge loss ignores examples outside the margin. Using only KR loss spreads distributions but may not produce a good separator. The hinge-KR loss combines both effects.

### Main theoretical results

**Theorem 1 - existence.** For every \(\lambda>0\), an optimal solution exists:

\[
f^*_{\lambda}\in \arg\min_{f\in Lip_1(\Omega)} L^{hKR}_{\lambda}(f).
\]

The solution is also uniformly bounded. If \(\psi\) is an optimal transport potential from \(P_+\) to \(P_-\), then:

\[
\|f^*\|_{\infty}\le
1+\operatorname{diam}(\Omega)+\frac{L_1(\psi)}{\inf(p,1-p)}.
\]

**Theorem 2 - duality.** The hinge-regularized KR problem is still the dual of an optimal-transport problem, but with relaxed constraints on the transport measure. The primal is:

\[
\inf_{\pi\in\Pi^p_\lambda(P_+,P_-)}
\int_{\Omega\times\Omega}\|x-z\|d\pi + \pi_x(\Omega)+\pi_z(\Omega)-1.
\]

The feasible set \(\Pi^p_\lambda(P_+,P_-)\) contains positive measures absolutely continuous with respect to \(dP_+\times dP_-\), with relaxed marginal density constraints:

\[
\frac{d\pi_x}{dP_+}\in[p,p(1+\lambda)],
\qquad
\frac{d\pi_z}{dP_-}\in[1-p,(1-p)(1+\lambda)].
\]

**Proposition 1 - gradient norm.** Under regularity assumptions on the optimal measure, there exists an optimal solution satisfying:

\[
\|\nabla f^*\|=1 \quad \text{almost surely}.
\]

**Proposition 2 - separable classes.** If the two classes are sufficiently separated, optimizing the hinge-KR objective produces a perfect classifier. In the balanced case, if there exists \(\varepsilon>0\) such that

\[
\|x-z\|>2\varepsilon \quad dP_+\times dP_-\text{-almost surely},
\]

then the optimal classifier has zero hinge classification loss. If \(\varepsilon\ge 1\), it is also an optimal transport potential.

### Robustness certificate

For any learned 1-Lipschitz function \(\hat f\), define the closest adversarial point as:

\[
adv(f,x)=\arg\min_{z\in\Omega,\ sign(f(z))=-sign(f(x))}\|x-z\|.
\]

Because \(\hat f\) is 1-Lipschitz:

\[
|\hat f(x)|\le \|x-adv(\hat f,x)\|.
\]

Thus \(|\hat f(x)|\) is a lower bound on the distance from \(x\) to the decision boundary and therefore a certified lower bound on \(\ell_2\) adversarial robustness.

Empirically, the paper observes that adversarial examples follow the transport direction:

\[
adv(x)\approx x-c_x f^*(x)\nabla_x f^*(x),
\qquad
tr_{f^*}(x)\approx x-c'_x f^*(x)\nabla_x f^*(x),
\]

with \(1\le c_x\le c'_x\). This is closely related to FGSM-style gradient attacks and supports the view that adversarial perturbations travel along the optimal transport path.

## 4. Architecture: enforcing 1-Lipschitz and gradient norm preservation

The method requires neural networks constrained to be 1-Lipschitz. The paper uses layerwise constraints.

### Dense layers

For a linear layer represented by a matrix \(W\), the Lipschitz constant is the spectral norm \(\|W\|\). The paper uses **spectral normalization**:

\[
W_s=\frac{W}{\|W\|}.
\]

The largest singular value is estimated using power iteration during the forward pass and is included in backpropagation.

### Convolutional layers

Normalizing convolution kernels alone is insufficient. The paper proposes multiplicative constants that account for input duplication in the convolution operator.

For zero-padded convolutions with kernel size \(k=2\bar{k}+1\), input width \(w\), and height \(h\), a tighter empirical normalization factor is:

\[
\Lambda=
\sqrt{\frac{(k w-\bar{k}(\bar{k}+1))(k h-\bar{k}(\bar{k}+1))}{hw}}.
\]

The appendix extends this to strided convolutions and pooling. Max pooling is 1-Lipschitz by definition. Average pooling is treated as a convolution-like operator.

### Gradient norm preserving networks

Since optimal solutions satisfy \(\|\nabla f^*\|=1\), the architecture is also designed to preserve gradient norm. The paper follows the approach of Anil et al. using:

- **GroupSort2** for convolutional activations;
- **FullSort** for dense activations;
- **2-norm pooling**;
- **Bjorck orthonormalization** for linear layers.

Bjorck orthonormalization iteratively pushes a weight matrix toward having all singular values equal to 1.

The authors implemented these layers in the open-source TensorFlow library **DEEL.LIP**.

## 5. Multiclass extension

The multiclass version uses one-vs-all classifiers. For \(q\) classes \(Y=\{C_1,\dots,C_q\}\), define:

\[
P_k=P(X\mid Y=C_k),
\qquad
\neg P_k=P(X\mid Y\ne C_k).
\]

The multiclass hinge-KR objective is:

\[
L^{hKR}_{\lambda}(f_1,\dots,f_q)=
\sum_{k=1}^q
\left[
\mathbb{E}_{x\sim \neg P_k}[f_k(x)]-
\mathbb{E}_{x\sim P_k}[f_k(x)]
+\lambda\mathbb{E}_x\left(m-(2\mathbf{1}_{y=C_k}-1)f_k(x)\right)_+
\right].
\]

The final layer has \(q\) outputs, each independently normalized so that each output function remains 1-Lipschitz. Prediction is by:

\[
\arg\max_k f_k(x).
\]

The certified robustness lower bound is half the gap between the largest and second-largest output:

\[
\frac{1}{2}\left(\max_k f_k(x)-\max_{j\ne k^*}f_j(x)\right).
\]

## 6. Experiments

The paper compares five methods:

| Method | Description |
|---|---|
| Adv | Adversarial training as in Madry et al. |
| 1LIPlog | Log-entropy classifier with Bjorck orthonormalization and ReLU, similar to Parseval networks |
| GNP_hin | Gradient norm preserving classifier with hinge loss |
| GNP_log | Gradient norm preserving classifier with log-entropy loss |
| hKR | Proposed hinge-KR classifier |

Datasets:

- **MNIST**, dense and CNN architectures, 10 classes;
- **CIFAR-10**, 10 classes;
- **CelebA**, binary eyeglasses detection, 128x128x3 images, 38% positive class.

Attacks:

- DeepFool;
- FGSM;
- \(\ell_2\)-PGD;
- \(\ell_2\) Carlini-Wagner;
- combined attacks where an attack succeeds if any attack fools the model.

### Key empirical findings

1. **Two-moons toy problem.** The vanilla KR classifier creates overlapping class score distributions. The hinge-KR classifier creates well-separated score distributions and a near-optimal decision boundary.

2. **DeepFool robustness.** Across MNIST, CIFAR-10, and CelebA, the hKR approach is among the most robust and is systematically better than GNP_log. On multiclass tasks, it is also better than GNP_hin at the same margin.

3. **Combined attacks.** hKR achieves the highest robustness in nearly all tested settings, especially under Carlini-Wagner attacks and for larger perturbations.

4. **CelebA detailed results.** On eyeglasses detection, hKR is especially strong under \(\ell_2\)-PGD and \(\ell_2\)-CW attacks. At \(\varepsilon=5\), the reported accuracies include:

| Attack | Adv | GNP_hin | GNP_log | hKR_50 |
|---|---:|---:|---:|---:|
| Base, \(\varepsilon=0\) | 98.04 | 97.20 | 94.51 | 96.07 |
| FGSM, \(\varepsilon=5\) | 82.96 | 37.70 | 43.90 | 64.01 |
| \(\ell_2\)-PGD, \(\varepsilon=5\) | 61.35 | 34.90 | 48.02 | 69.79 |
| \(\ell_2\)-CW, \(\varepsilon=5\) | 13.17 | 29.46 | 32.81 | 56.44 |

5. **Interpretability of attacks.** On CelebA, adversarial examples for hKR are structured and semantically meaningful. For images without glasses, attacks add perturbations around the eyes and nose bridge. For images with glasses, attacks tend to erase the glasses. This contrasts with other methods where perturbations are less interpretable.

## 7. Appendix material retained synthetically

### Discrete optimal transport

For balanced finite samples \(x_1,\dots,x_n\sim P_+\) and \(x_{n+1},\dots,x_{2n}\sim P_-\), the primal OT problem finds a transport plan \(\Pi\):

\[
\min_\Pi \sum_{i,j}\Pi_{ij}C_{ij},
\quad C_{ij}=\|x_i-x_{n+j}\|,
\]

subject to nonnegativity and uniform marginal constraints:

\[
\Pi_{ij}\ge0,
\qquad
\sum_i\Pi_{ij}=\frac1n,
\qquad
\sum_j\Pi_{ij}=\frac1n.
\]

The discrete dual optimizes a vector \(F\) subject to Lipschitz-like constraints:

\[
\max_F F U^T,
\qquad
F_i-F_{n+j}\le C_{ij}.
\]

The hinge-regularized discrete transport problem relaxes marginal constraints, allowing more mass to be placed on close cross-class pairs, which are also the hardest examples for classification.

### Layer normalization summary

| Layer type | Main Lipschitz control |
|---|---|
| Dense | spectral norm \(\|W\|\) |
| Convolution without stride | kernel norm times convolution duplication factor |
| Convolution with stride | stride-dependent duplication factor |
| Max pooling | 1-Lipschitz |
| Average pooling | bounded as a convolution-like operator |

### Network architecture summaries

**MNIST dense:** input 784, dense 256, dense 256, output 10.  
**MNIST CNN:** conv 16, pooling, conv 32, pooling, dense 100, output 10.  
**CIFAR CNN:** conv blocks with 128, 256, and 512 channels, pooling after each block, dense 512, dense 512, output 10.  
**CelebA CNN:** repeated convolution blocks with 16, 32, 64, 128, 256 channels, pooling after each block, dense 512, dense 512, output 10.

Algorithm-specific layer choices:

| Network | Conv activation | Dense activation | Output | Pooling | Orthonormalization |
|---|---|---|---|---|---|
| Adv | ReLU | ReLU | softmax | max pooling | none |
| 1LIP | ReLU | ReLU | softmax | max pooling | Bjorck |
| GNP_log | GroupSort2 | FullSort | softmax | 2-norm | Bjorck |
| GNP_hin | GroupSort2 | FullSort | linear | 2-norm | Bjorck |
| hKR | GroupSort2 | FullSort | linear | 2-norm | Bjorck |

## 8. Conclusion

The paper introduces a classification framework in which robustness is not just an empirical property but part of the mathematical formulation. The hinge-KR loss combines optimal transport, margin-based classification, and 1-Lipschitz neural-network constraints. The resulting models provide certifiable lower bounds on adversarial robustness and empirically resist several strong attacks. A notable qualitative result is that adversarial examples against hKR models often correspond to meaningful semantic changes, suggesting a connection between robustness and counterfactual explanation.

## 9. Core references retained

The paper builds primarily on Wasserstein GANs, Kantorovich-Rubinstein duality, Lipschitz-constrained networks, gradient norm preserving architectures, adversarial training, and standard adversarial attacks. Key cited works include Arjovsky et al. on WGAN, Villani on optimal transport, Madry et al. on adversarial training, Goodfellow et al. on FGSM, Carlini and Wagner on robustness evaluation, Anil et al. on Lipschitz approximation, and Li et al. on gradient attenuation in Lipschitz-constrained convolutional networks.
