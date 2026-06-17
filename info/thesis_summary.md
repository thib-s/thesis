## Summary of PhD work

Your PhD, **“Architectures de réseaux de neurones dans l’espace des fonctions 1-Lipschitz,”** studies how to design neural network architectures and optimization methods around **orthogonality and Lipschitz constraints**. The central motivation is that 1-Lipschitz neural networks provide mathematically controlled behavior, especially for **certifiable robustness**, but practical architectures must remain expressive, scalable, and efficient. 

### Year 1 — Building efficient 1-Lipschitz networks

The first part of the PhD focused on **constraining network weights** so that each layer has a controlled Lipschitz constant. Since the Lipschitz constant of a full network can be bounded by the product of the constants of its layers, orthogonal layers are a natural building block for 1-Lipschitz models. 

A major contribution was the development of **Adaptive Orthogonal Convolutions (AOC)**. Orthogonal convolutions are difficult because convolutions are structured linear operators: directly enforcing orthogonality can destroy the convolutional structure, while keeping the structure makes exact orthogonality hard. AOC combines existing ideas to produce convolutional layers that are flexible, efficient, and compatible with modern CNN designs such as U-Net, ResNeXt, MobileNetV2-style dilation, groups, stride, and transposed convolutions. 

This work led to the ICML 2025 paper **“Adaptive Orthogonal Convolutions for Efficient and Flexible CNN Architectures,”** where you are first author. It is one of the core contributions of the thesis because convolution is a central component of Lipschitz networks and orthogonality is hard to enforce on it. 

A second major output was **Orthogonium**, an open-source library that unifies efficient implementations of orthogonal and 1-Lipschitz building blocks. The library includes AOC/adaptive BCOP, adaptive SOC, SLL, AOL, activation functions, normalization layers, skip-connection variants, and infrastructure for model factories and model zoos. This contribution supports reproducibility and gives the community a common implementation basis for comparing 1-Lipschitz architectures. 

### Year 2 — From constrained weights to constrained gradients

The second year shifted from **forward orthogonality in Lipschitz networks** to **backward/optimization-side orthogonality**, especially through the **Muon optimizer**. The conceptual transition is: in the first year, weights were constrained to enforce Lipschitz properties; in the second year, gradients or updates are constrained to guide optimization. 

Muon treats hidden-layer weights as matrices, builds momentum, and orthogonalizes the update before applying it. It can require fewer optimization steps, but each step is more expensive because of the Newton–Schulz approximate orthogonalization procedure. This creates a trade-off between **training efficiency in number of steps** and **wall-clock training speed**. 

Your main contribution here is **Turbo-Muon**, which accelerates orthogonality-based optimization with a preconditioning step. The goal is to reduce the polar-factor approximation error without slowing down the algorithm. The key idea is to reuse computation so that preconditioning has negligible cost, reduce the Newton–Schulz polar error, and remove one iteration while preserving performance. Experiments on FineWeb-edu, CIFAR-10, and ImageNet show comparable performance at lower runtime cost. 

A second line of work investigates why **Muon is theoretically wrong for convolutions but empirically effective**. Standard Muon reshapes convolution kernels and orthogonalizes them as matrices, which does not correspond to true orthogonalization of the convolutional operator. A more theory-compliant convolution-aware Newton–Schulz variant was explored, connected to the block-convolution ideas from the AOC work. However, the results suggest that better theoretical compliance does not automatically translate into better optimization, and this remains an active work-in-progress. 

### Publications and outputs

During the thesis, the main first-author/core contributions are:

1. **Adaptive Orthogonal Convolutions for Efficient and Flexible CNN Architectures** — ICML 2025, first author.
2. **Orthogonium: A Unified, Efficient Library of Orthogonal and 1-Lipschitz Building Blocks** — CODEML Workshop @ ICML 2025, first author.
3. **Turbo-Muon: Accelerating Orthogonality-Based Optimization with Pre-Conditioning** — first author, submitted/preprint according to the slides.
4. **Muon is theoretically wrong for convolutions, but empirically effective** — ongoing work.  

You also contributed to related work on **robust conformal prediction with Lipschitz-bounded networks**, **explainability**, **differential privacy**, and robustness-alignment trade-offs. These side contributions strengthen the broader thesis narrative: Lipschitz and orthogonal neural networks are not only tools for adversarial robustness, but also connect to calibration, conformal guarantees, privacy, explainability, and optimization. 

### Overall thesis narrative

Your PhD can be summarized as follows:

> This thesis develops efficient neural architectures and optimization methods based on orthogonality. It starts from 1-Lipschitz neural networks, where orthogonal layers provide certifiable control of robustness, and addresses practical bottlenecks such as convolutional flexibility, scalability, and implementation quality. It then extends the same orthogonality perspective to optimization, studying Muon-like methods where orthogonalized updates improve training dynamics. The resulting contributions include new orthogonal convolutional layers, an open-source library of 1-Lipschitz components, an accelerated Muon variant, and an ongoing theoretical and empirical study of orthogonal optimization for convolutions.

### Future direction

The natural next step is to connect the two sides of the thesis: **forward orthogonality** in Lipschitz networks and **backward orthogonality** in Muon-style optimization. The slides propose studying the fact that orthogonal layers are invertible, their inverse is their transpose, and backpropagation through linear layers uses the transposed weight matrix. This suggests a possible bridge between orthogonal networks, target propagation, and second-order optimization, with the special case of orthogonal networks still largely unexplored. 
