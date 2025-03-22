---
published: true
title: Feature Splitting (tmp)
date: 2025-03-21 00:00:00 -500
categories: [mechanistic interpretability]
tags: [mechanistic interpretability]
math: true
---

# Definitions

First thing's first, what do we mean by "features" / "neurons"?

### Features vs Neurons

Unfortunately, there's no truly explicit standardized definition for this, so this is my definition, which I've tried to align with the majority of the literature ([this](https://openai.com/index/multimodal-neurons/), [this](https://distill.pub/2020/circuits/zoom-in/), [this](https://arxiv.org/pdf/2009.05041), [this](https://transformer-circuits.pub/2022/solu/index.html), and many more).

In my framework of thinking, features are the ["high-level human-interpretable features" (Gurnee et. al)](https://arxiv.org/pdf/2305.01610), and are the things that can exist in superposition with each other (multiple non-orthogonal features co-existing in the same space). For there to be superposition, there must be a non-linearity, so everytime we talk about features, we are talking about the part(s) of the model that looks like this (usually the MLP):

$$
\begin{align*}
y = \text{ReLU}(Wx + b), \text{ where } W \in \mathbb{R}^{n \times d}, n > d
\end{align*}
$$

- **Features.** They are:
    - $W_i$, or $i$-th row of $W$, or 'feature vector'
    - $y_i$, or $i$-th entry of $y$, or 'feature activation'
- **Neurons.** Being inspired by biological neurons, this refers to the units of computation that accept a bunch of inputs, and are passed through an activation function. They are:
    - $W_i$, or $i$-th row of $W$. Same as features.
    - $y_i$, or $i$-th entry of $y$. Same as features.
- **Latents.** They simply refer to the latent representation of a data point, and latent representation refers to their coordinates in a latent space, where the latent space refers generally to any sort of vector space that is not the input / output space. Generally, this means $\mathbb{R}^{\text{anything but }n}$, and depends on the context. This is generally vague and should be avoided, unless you're explicit about which latent space you're referring to, e.g. "latents in the model-space" would refer to the representation in $\mathbb{R}^{\text{model}}$, which most often denotes the vector space of the residual stream of a Transformer.

So features are neurons, which in this context, are:
- Each entry of $y$
- Each of the feature vectors (rows) in $W$, which live in $\mathbb{R}^d$.

Confusion arises when we assume that the true number of semantic features (human-interpretable, disentangle-able neabubgs) is more than than the number of feature neurons, or when we are working in multi-layer MLPs / transformers, where features could come from any of the earlier layers, which would necessarily mean that the total number of features across all layers is necessarily larger than the number of feature neurons available in any one particular layer. In these contexts, we might read / think something like "how do we represent $m > n$ features in $n$ features?", which is really confusing, but we will stick to the use of "features" to describe them both, as both these types of "features" were at some point a feature vector of an MLP layer (**big assumption I'm making that I will investigate more**); we simply disambiguate them by describing them more explicitly.

### Monosemanticity vs Polysemanticity (Semanticity)

So then, what is monosemanticity? Generally (and by that, I mean according to Anthropic's papers [Softmax Linear Units](https://transformer-circuits.pub/2022/solu/index.html), [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html), and [Toy Models of Superpositions](https://transformer-circuits.pub/2022/toy_model/index.html)) people use monosemantic neurons / features to refer to entries of $y$ that fire only in the presence of some concept, and virtually nothing else. Polysemantic features hence refer to those that fire reliably in the presence of more than one concept.

### Semanticity vs Superposition

**Superposition** refers to the phenomenon where feature vectors are represented non-orthogonally in the model space. This is entirely separate from semanticity. Note:
- Monosemantic neurons and Polysemantic neurons alike can be in superposition
- Superposition is more a function of model capacity than semanticity. If the model space is $d$-dimensional, but you have to encode more than $d$ features (monosemantic or polysemantic), then you will necessarily have superposition, by pigeon-hole principle.

Here is where I take issue with statements like this one, from the paper [Finding Neurons in a Haystack: Case Studies with Sparse Probing](https://arxiv.org/pdf/2305.01610):

> Superposition implies polysemanticity by the pigeonhole principle — if a layer of $n$ neurons
reacts to a set of $m ≫ n$ features, then neurons must (on average) respond to more than one feature.

Because it's not quite right. They define superposition as follows: "an obvious problem arises when a network has to represent more features than it has neurons. To accomplish this, a model must employ some form of compression to embed $n$ features in $d < n$ dimensions. While this *superposition* of features...." The important thing for them here is representing $n$ features in $d < n$ dimensions, implying non-orthogonality of feature vectors. Anthropic's toy model is a simple counter-example to this:

$$
\begin{align*}
y &= \text{ReLU}(W^\top Wx + b), \text{ where } W \in \mathbb{R}^{2 \times 7} \\
X &= I_7
\end{align*}
$$

The $2$ and $7$ are just arbitrary dimensional choices for a small-dimensional and large-dimensional space. Obviously, since the bottleneck dimension is $2$, but we have $7$ features to learn, there will be superposition, but the model can also be trained to learn all $7$ features as monosemantic features. Each of $y_i$ will only fire in the presence of one $x_i$. **Superposition does not imply polysemanticity.** 

What implies polysemanticity is having a universe of "true semantic features" that need to be represented in the number of features your model allows. Examples of this are:
- You are training a single-layer MLP to encode 10 features, but you only have 7 features. Something like:

$$
\begin{align*}
(y \in \mathbb{R}^{10}) &= M\left(\text{ReLU}(Wx + b) \in \mathbb{R}^7\right), \text{ for some } W \in \mathbb{R}^{7 \times 10}, M \in \mathbb{R}^{10 \times 7}
\end{align*}
$$

- You are looking at MLP layer 10 in a multi-layer MLP ($n$ neurons per layer) model, and assuming that each of your earlier 9 layers have extracted full sets of distinct semantic features, this current MLP layer 10 could encode up to $10n$ semantic features whilst having only sufficient weights to express $n$ feature vectors. I suppose this is what the above paper meant by trying to "embed $n$ features" (analogous to the $10n$ semantic features here) "in $d < n$ dimensions" (analogous to the $n$ feature vectors / neurons here).

**Semanticity and Superposition are just different concepts, period.** Any attempt to confound them is going to get in the way of what this post wishes to investigate: Feature Splitting and Feature Absorption.

# Questions

The core concepts of interest are Feature Splitting & Absorption. These are phenomena when training Sparse Auto-Encoders (SAEs), which are ReLU autoencoders that try to [project a model's features onto an overcomplete basis in hopes that the basis is entirely (or mostly) monosemantic](https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder-dataset):

$$
\begin{align*}
\text{hopefully monosemantic features, } f &= \text{ReLU} \left( W_e \left( \text{model features} - b_d \right) + b_e \right) \\
\text{model feature reconstruction } & = W_d f + b_d \\
\end{align*}
$$
> I simply think of this as a function that goes from $\mathbb{R}^\text{feature dim} \rightarrow \mathbb{R}^{\text{HUGE}} \rightarrow \mathbb{R}^\text{feature dim}$

[**Feature Splitting**](https://transformer-circuits.pub/2023/monosemantic-features/index.html#phenomenology-feature-splitting) is when you learn successively wider SAE features and discover that perhaps a concept corresponded to only 1 SAE feature in the smaller SAE splits into multiple finer SAE features in bigger

- What incentivizes feature splitting?

> Whatever else definition and Questions

In [A is for Absorption](https://arxiv.org/html/2409.14507v3), the authors train an SAE on the residual latents.

# Subproblem

`_short` vs `_s` activations:
- `_short` activation is ~5 times as high as the `_s` activation
- `_short` SAE feature has ~1/5 the dot product with LR probe as that of `_s`

I think one can explain this by using the latent zone plots. It is much easier to overactivate for a feature with a small SAE feature magnitude. This would concur with the observation that the LR probe direction is the main causal direction.

## Hypothesis

Absorbing features are "failed" features, in the same 