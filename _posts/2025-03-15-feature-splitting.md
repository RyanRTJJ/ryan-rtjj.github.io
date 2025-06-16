---
published: true
title: Feature Splitting & Feature Absorption
date: 2025-03-15 00:00:00 -500
categories: [mechanistic interpretability]
tags: [mechanistic interpretability]
math: true
---

My [previous blogpost](https://amagibaba.com/posts/opt-failure/) gives a very clear visualization of how the latents of simple ReLU networks look like, how to interpret them, and a good description of optimization pressures that force them into or away from local optima. I feel well equipped to investigate the problems of [feature splitting](https://transformer-circuits.pub/2023/monosemantic-features/index.html#phenomenology-feature-splitting) and feature absorption.

## TLDR

**I've decided not to address feature splitting and feature absorption for now**, for the reasons that:
- I think feature splitting can be a good thing
- Feature absorption is a result of optimization pressures towards sparsity
- The definition of "feature," especially in papers describing Feature Absorption, is not good. For example, [A is for Absorption](https://openreview.net/pdf?id=LC2KxRwC3n) mentions a `_s` feature. What kind of feature is "tokens that start with s"?! They use this "feature" to illustrate a good point, but the feature itself is stupid and entirely useless. This definition problem is a big obstacle, and I think it's more worthwhile to look at more statistically principled methods / definitions.
- There are sub-problems that I think are worth looking into first, for example, I think SAEs are implementing clustering. I told Neel Nanda about this when he opened Office Hours after a Stanford seminar, and he didn't seem convinced, but provided no good reason other than "a lot of research has shown that SAEs tend to extract interpretable and composable features," which is yet another definitional mess. He raised a good question - what evidence could demonstrate one result versus the other? And to that end, my answer would be the phenomenon of feature absorption, but that seems unsatisfying. So it looks like more math could be done here so that this could be further experimentally investigated.

Also, I think I've got a pretty good sense of how the features extracted by SAEs are screwed up. In particular, the solutions provided by SAEs are not sufficiently constrained (which leads to problems like instability / lack of universality of SAE features). Without loss of generality, suppose you had to extract 2 features using an SAE. So long as the activation zones of the 2 features segment the latent space into a bunch of zones, 2 of which correspond to only either feature activating, you're good. The SAE feature vectors themselves do not have to be aligned with the data points that hypothetically encode only that feature; they could be rotated any which way. To illustrate what I mean, this is a 3D plot of the latent zones for SAE features 1085 (the `_short` feature) and 6510 (the `_s` feature), together with the SAE-space embeddings of tokens for several words that start with 's'. This is based on [A is for Absorption](https://openreview.net/pdf?id=LC2KxRwC3n). Notice how the features here almost completely orthogonal to the actual embeddings.

<div style="display: flex; justify-content: center;">
    <video width="500px" controls>
        <source src="../../images/feature_splitting/3d_rotation.mov" type="video/mp4">
    </video>
</div>
<br/>

As a result, **I'll be moving on to other methods to extract features, and building on foundational understanding of features are arranged in the latent space**. I write about some of my intuition of how features are arranged in latent space in ["Thoughts on Hidden Structure in MLP Space
"](/posts/tegum-factors/), which contain some ideas that I thought were wild and unprovable. But a collaborator, Simon Reich, recently brought to me attention research that corroborates these ideas, including [From Flat to Hierarchical: Extracting Sparse Representations with Matching Pursuit](https://arxiv.org/html/2506.03093v1) by Thomas Fel (amazing amazing researcher) and friends, and [The Geometry of Categorical and Hierarchical Concepts in Large Language Models](https://arxiv.org/abs/2406.01506) by Kiho Park and friends. I remain super excited about these.