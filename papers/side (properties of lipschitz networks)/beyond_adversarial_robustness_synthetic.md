# Beyond Adversarial Robustness: Breaking the Robustness-Alignment Trade-off in Object Recognition

**Venue:** ICLR 2025 Workshop on Representational Alignment (Re-Align)  
**Authors:** Pinyuan Feng, Drew Linsley, Thibaut Boissin, Alekh Karkada Ashok, Thomas Fel, Stephanie Olaiya, Thomas Serre  
**Code/Data:** `https://github.com/TonyFPY/Adversarial_Alignment`

## Core question

Deep neural networks (DNNs) are well known to be vulnerable to small adversarial perturbations. This paper asks whether the large improvements in ImageNet object recognition over the past decade have also improved adversarial robustness, and whether the resulting perturbations affect image features that humans use for recognition.

The answer is mixed: modern DNNs need larger perturbations to change their decisions, but those perturbations are increasingly misaligned with human-recognizable object features. This creates a robustness-alignment trade-off.

## Main claim

A robust vision model should satisfy two conditions:

1. **High perturbation tolerance:** attacks should require large input changes.
2. **High adversarial human alignment:** successful perturbations should target object regions that humans find diagnostic for recognition.

The paper argues that tolerance alone is insufficient: a large perturbation can still be hard for humans to notice if it affects pixels that are perceptually irrelevant.

## Metrics

### Perturbation Tolerance (PT)

PT is the average `l2` distance between the clean image and the minimally perturbed adversarial image. Higher PT means a larger perturbation is needed to flip the model decision.

### Adversarial Human Alignment (AA)

AA is the average Spearman rank correlation between the adversarial perturbation pattern and a human feature-importance map from ClickMe. Higher AA means the attack targets image regions that humans also consider important for object recognition.

### Accuracy

Model performance is measured with **multi-label ImageNet accuracy**, which accounts for all semantically valid labels and enables comparison with human performance.

## Experimental setup

### Stimuli

- The evaluation uses a 960-image subset of ImageNet.
- Images come from 240 categories.
- The task is simplified to **animal vs. non-animal** classification.
- Each image is paired with a ClickMe human feature-importance map.
- The ImageNet subset is balanced through 12 WordNet-derived superclasses to reduce fine-grained category ambiguity.

### Model zoo

The study evaluates **309 DNNs** implemented with the TIMM toolbox:

| Model family | Count | Notes |
|---|---:|---|
| CNNs | 125 | Supervised ImageNet-trained convolutional models |
| Vision Transformers | 78 | ViT-style architectures |
| Hybrid models | 52 | CNN-ViT combinations |
| Self-supervised models | 54 | Includes CLIP, DINO, DINOv2-style training paradigms |

Additional models are evaluated to test whether the trade-off can be broken:

- **9 harmonized models**, trained with the neural harmonizer.
- **33 adversarially trained / robustified models**, trained with `l2`-bounded or `linf`-bounded PGD perturbations.

### Attack method

The paper uses the **Fast Minimum-Norm (FMN)** adversarial attack, specifically `l2` FMN.

Reasons for choosing FMN:

- It approximates the smallest perturbation needed to change a model prediction.
- It is efficient enough for a large model zoo.
- It produces continuous-valued perturbation maps that can be compared with ClickMe maps.

Attack details:

- `l2` FMN attack
- 1000 iterations
- annealed step size from `1.0` to `1e-5`
- epsilon update initial step size `gamma = 5.0`
- normalized gradient descent with adaptive epsilon projection
- all attacks succeeded for all model-image pairs
- runtime: approximately 30-240 minutes per model on one NVIDIA A6000 GPU

## Results

### 1. DNNs have become more tolerant to adversarial perturbations

As ImageNet accuracy increases, Perturbation Tolerance also increases strongly:

- Spearman correlation: `rho_s = 0.81`
- significance: `p < 0.001`

Important patterns:

- More accurate models require larger adversarial perturbations.
- The most accurate evaluated model, `eva02_large_patch14_448.mim_in22k_ft_in22k_in1k`, rivals several adversarially trained models in PT while being roughly **11.24%-46.62% more accurate** on ImageNet.
- ViTs are significantly more tolerant than CNNs:
  - `T(163) = 9.11`, `p < 0.001`
- Self-supervised models are significantly more tolerant than other models:
  - `T(65) = 7.85`, `p < 0.001`
- Models pretrained on larger datasets and fine-tuned on ImageNet are more tolerant than models trained only on ImageNet:
  - `T(32) = 5.65`, `p < 0.001`

**Interpretation:** scaling, larger datasets, and transformer-style architectures provide real protection against small adversarial attacks.

### 2. DNN perturbations have become less aligned with human perception

As ImageNet accuracy increases, Adversarial Human Alignment decreases:

- Spearman correlation: `rho_s = -0.74`
- significance: `p < 0.001`

Important patterns:

- High-performing models tend to be attacked through image regions that humans do not rely on for recognition.
- The most accurate model has AA around `rho_s = -0.03`.
- The least accurate model, `lcnet_050.ra2_in1k`, has higher AA around `rho_s = 0.22`.
- CNNs are significantly more aligned with human visual features than ViTs:
  - `T(153) = -12.5`, `p < 0.001`
- Self-supervised models have significantly lower AA than supervised models:
  - `T(98) = -7.49`, `p < 0.001`
- Larger pretraining datasets do not significantly improve AA at the strict `p < 0.001` threshold:
  - `T(40) = -2.54`, `p = 0.015`

**Interpretation:** modern scale improves attack tolerance but worsens the human perceptual relevance of successful attacks.

### 3. Perturbation Tolerance and Adversarial Human Alignment form a trade-off

When PT is plotted against AA, standard DNNs form a Pareto frontier:

- Models with high PT tend to have low AA.
- Models with higher AA tend to have lower PT.
- Improving one dimension usually hurts the other.

This suggests that standard DNN scaling does not naturally produce human-like adversarial robustness.

### 4. Harmonization and adversarial training can break the trade-off

Two training strategies improve both PT and AA:

#### Neural harmonization

Harmonization trains models to solve object recognition using human-diagnostic features. It adds a loss encouraging DNN gradients to match ClickMe feature-importance maps across multiple spatial scales.

Key effects:

- Harmonized models improve PT relative to their baselines.
- Harmonized models improve AA relative to their baselines.
- Unlike adversarial training, harmonization maintains or slightly improves ImageNet accuracy.

#### Adversarial training

Robustified models are trained on adversarial examples using a min-max objective:

`min_theta E[(x,y)~D max_{||delta||_p <= epsilon} L(f_theta(x + delta), y)]`

Key effects:

- Robustified models have significantly higher PT than standard models:
  - `T(41) = 6.41`, `p < 0.001`
- Robustified models have significantly higher AA than standard models:
  - `T(106) = 20.85`, `p < 0.001`
- However, robustified models lose ImageNet accuracy:
  - `T(45) = -8`, `p < 0.001`

**Interpretation:** adversarial training gives strong robustness and alignment benefits, but at a substantial accuracy cost. Harmonization gives a more favorable trade-off for accuracy.

## Figure-level synthesis

### Figure 1

Introduces the desired robustness target: attacks should be both strong and human-aligned. The figure contrasts low-tolerance/low-alignment, high-tolerance/low-alignment, and high-tolerance/high-alignment cases using ImageNet images, ClickMe maps, and FMN perturbations.

### Figure 2

Shows that PT increases with multi-label ImageNet accuracy. Human performance is shown as a reference band around **95.14% +/- 2.02%**.

### Figure 3

Shows that AA decreases as ImageNet accuracy improves. Stronger models are less attacked through human-diagnostic object regions.

### Figure 4

Shows the PT-AA Pareto trade-off and highlights that harmonized and robustified models move beyond the standard frontier. Example visualizations show ClickMe maps and perturbations for baseline, robust, and harmonized ResNet18 models.

### Figure 5

Provides example `l2` FMN perturbations for several DNNs across ImageNet images such as a snow leopard, bagel, American chameleon, and cradle.

### Figures 6-7

Compare harmonized models with baselines:

- Figure 6: harmonized models generally increase PT.
- Figure 7: all harmonized models significantly improve AA.

### Figures 8-9

Visualize stimuli, human feature-importance maps, and adversarial perturbations across model families: CNN, ViT, Hybrid, SSL, Robust, and Harmonized.

## Harmonized model results

| Harmonized model | PT | AA |
|---|---:|---:|
| resnetv2_50 | 2.9369 +/- 0.0785 | 36.62% +/- 1.44% |
| vit_tiny_patch16_224 | 2.4101 +/- 0.0563 | 30.51% +/- 1.40% |
| convnext_tiny | 2.5164 +/- 0.0605 | 25.97% +/- 1.38% |
| mobilenetv3_small_050 | 1.5694 +/- 0.0328 | 26.14% +/- 1.13% |
| resnet18 | 1.6884 +/- 0.0156 | 40.11% +/- 1.56% |
| resnet34 | 2.1633 +/- 0.0430 | 40.26% +/- 1.56% |
| resnet50 | 1.6827 +/- 0.0313 | 39.22% +/- 1.48% |
| resnet101 | 1.7925 +/- 0.0413 | 33.35% +/- 1.49% |
| resnet152 | 1.9059 +/- 0.0365 | 37.29% +/- 1.40% |

## Robustified model result ranges

The paper reports 33 robustified models across ResNet18, ResNet50, and Wide ResNet-50-2, trained with `l2` or `linf` PGD.

Notable ranges:

- `l2`-trained models:
  - PT ranges roughly from `1.4511` to `4.8839`.
  - AA ranges roughly from `23.27%` to `30.48%`.
- `linf`-trained models:
  - PT ranges roughly from `3.5693` to `4.7939`.
  - AA ranges roughly from `15.29%` to `35.69%`.

Highest values in the table:

- Highest PT: `wide_resnet50_2_l2_eps0.5`, PT `4.8839 +/- 0.1330`
- Highest AA: `wide_resnet50_2_linf_eps1.0/255`, AA `35.69% +/- 1.43%`

## Discussion

### Scaling helps robustness, but only partially

The paper argues that DNN scale provides valuable protection against adversarial attacks because modern models require larger perturbations to flip predictions. This trend is especially visible for ViTs and models trained with more data.

### Scaling worsens human alignment

The same scaling trend makes adversarial perturbations less aligned with human-diagnostic features. This means modern models may be harder to attack in norm space, but successful attacks may remain difficult for humans to notice.

### Harmonized and robustified models are partial solutions

- Robustified models obtain strong PT and AA but lose accuracy.
- Harmonized models improve PT and AA while preserving accuracy.
- The authors suggest scaling harmonization to larger DNNs and larger ClickMe-like datasets, possibly using pseudo-labels on internet-scale data.
- They also propose combining harmonization and adversarial training through auxiliary losses.

## Limitations

- The study uses only `l2` FMN attacks for evaluation.
- It remains unknown whether the conclusions transfer to other `lp` attacks.
- The paper does not include psychophysical experiments testing whether the adversarial features transfer to human perception.
- The analysis focuses on spatial overlap between perturbations and human feature-importance maps, not direct human behavioral responses to the generated attacks.

## Broader impact

Adversarial attacks expose limitations of DNNs as models of biological vision. The paper argues that today’s scaling-driven progress in computer vision improves perturbation tolerance but deepens adversarial misalignment with humans. Better representational alignment may improve interpretability, reliability, and the use of DNNs as computational models of human vision.

## One-sentence takeaway

Modern DNNs are becoming harder to attack in `l2` norm, but their successful adversarial perturbations are becoming less human-like; harmonization and adversarial training can improve both robustness and alignment, with harmonization preserving accuracy more effectively.
