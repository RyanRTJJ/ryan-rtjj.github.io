---
published: true
title: Feature Splitting & Feature Absorption
date: 2025-03-15 00:00:00 -500
categories: [mechanistic interpretability]
tags: [mechanistic interpretability]
math: true
---

# Coming Soon

My [previous blogpost](https://amagibaba.com/posts/opt-failure/) gives a very clear visualization of how the latents of simple ReLU networks look like, how to interpret them, and a good description of optimization pressures that force them into or away from local optima. I feel well equipped to investigate the problems of [feature splitting](https://transformer-circuits.pub/2023/monosemantic-features/index.html#phenomenology-feature-splitting) and feature absorption. Please reach out to me at ryan.rtjj@gmail.com if you'd like to collaborate.

## A Glimpse

This is still rather early work and the plot here is not exactly 100% correct, but it almost is. This is a 3D plot of the latent zones for SAE features 1085 (the `_short` feature) and 6510 (the `_s` feature), together with the SAE-space embeddings of tokens for several words that start with 's'. This is based on [A is for Absorption](https://openreview.net/pdf?id=LC2KxRwC3n). I'm interested in how the features here almost completely orthogonal to the actual embeddings.

<div style="display: flex; justify-content: center;">
    <video width="500px" controls>
        <source src="../../images/feature_splitting/3d_rotation.mov" type="video/mp4">
    </video>
</div>