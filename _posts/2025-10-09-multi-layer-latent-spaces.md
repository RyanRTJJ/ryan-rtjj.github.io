---
published: true
title: Multi-Layer Latent Space Visualization
date: 2025-10-02 00:00:00 -500
categories: [mechanistic interpretability]
tags: [mechanistic interpretability]
math: true
---

# Simplified Conceptualization of Transformers

A transformer block roughly looks like this (sans the layer norms):

<img src = "../../images/multi_layer_space/transformer_block.png" alt="Transformer Block" width="100%"> 
*Transformer Block*

If we were to anthropomorphize slightly what the transformer block components are doing, we have:
- The `attn` block mixes the token vectors
- The `MLP` block uses `attn`'s outputs, in addition to the original token vectors (for this layer), and **creates features** (thanks to the up-projection followed by ReLU).
- The new features are then added back to the residual stream

There are lots of nuances I'm glazing over (lack of layer norm, the biases in the MLP equations), but we will address them later. For now, just fly with me at an intuitive level. Now let's introduce the notation for the MLP operations:

$$
\begin{align*}
h_\text{MLP} (x) & = \text{ReLU}(W_\text{up} x + b_\text{up}) \\
f_\text{MLP} (x) & = W_\text{down} h_\text{MLP} (x) + b_\text{down}
\end{align*}
$$

We are going to do 3 things conceptually here:
1. We are going to **think in the context of multi-layer attention** (i.e. chaining of several of these blocks together). This means that we are reasoning in terms of how the previous layer's `MLP` block contributes to the features created in the current layer's `MLP` block.
2. We are going to **decouple `attn` and previous-layer `MLP` contributions**. This means that we effectively distribute the current layer's `MLP` operations over the `attn` output and the residual stream representation (which we think of as previous layer `MLP` contributions). This is because we want to ignore `attn`'s contribution to the `MLP` inputs. This allows us to intuit how the decoder weights ($W_\text{L-1, down}$, $b_\text{L-1, down}$) of layer $L-1$ contribute to the features created in layer $L$ (via $W_\text{L, up}$, $b_\text{L, up}$). 
> Because $\text{ReLU}$ is non-linear and non-distributive, this is not correct, and here's an example of why: if the values of $h$ due to `attn` were very negative and the contributions from previous `MLP` layers were only weakly positive, then the `MLP` contributions are effectively 0, because whether or not you factor in the `MLP` contributions, the values of $h$ pre-$\text{ReLU}$ would still be negative, leading to values of 0 post-$\text{ReLU}$. However, when thinking in terms of general model behavior over a large number of inputs, positive previous `MLP` contributions would still equate to an expected positive contribution to $h$ values; likewise for negative `MLP` contributions. Therefore, ignoring `attn`'s contributions is a fine approximation.
3. We are going to **think of `MLP` as feature sets**. As in, every `MLP` block creates and adds features to the residual stream representation.

So, we end up with a simplified conceptualization of attention blocks:

<img src = "../../images/multi_layer_space/feature_sets.png" alt="Simplified Transformer Blocks" width="100%"> 
*Simplified Transformer Blocks*

# Visualizing a Latent Space

Deep models are all performing some sort of information compression. A useful way to think about the latent space of models (usually refers to the residual stream of Large Language Models, or the grey 2-vectors in the graphics above) is as a "superposition of features," for which there is much research material. In ["Superposition - An Actual Image of Latent Spaces"](/posts/viewing-latent-spaces/), I introduce a method to visualize such features. The method relies on the fact that $\text{ReLU}$ effectively segments a lower-dimension (2D in this case) subspace into a superposition of 5 half-space pairs (5 because the up-projection projects the 2D subspace to a 5D subspace in this case).

Here's what would happen if we chose **one** latent space (2D) to visualize, with respect to features determined by an immediate up-projection + $\text{ReLU}$:

<img src = "../../images/multi_layer_space/one_latent_space.png" alt="One Layer Latent Space" width="100%"> 
*Latent Space Visualization for 1-layer MLP*

# Latent Spaces for Chained MLPs

The above plot gives you an idea of where you have to be in the 2D space in order to activate any subset of the 5 features. Now, let's attempt to ask the same question for chained MLPs - given 2 consecutive MLP layers, where would you have to be in the **earlier** 2D space to activate any subset of the 5 features of the **later** MLP block?

<img src = "../../images/multi_layer_space/chained_latent_spaces.png" alt="Two Layer MLP Latent Space" width="100%"> 
*Latent Space Visualization for 2 chained MLPs?*

## What Would It Take To Activate h1?

For our final output to activate the <span style="color: royalblue">1st feature (i.e. $h_1 > 0$)</span>, it must be in the <span style="color: royalblue">blue zone (pointed to by $U_1$)</span> of the 2D input space of Feature Set 2, which I'll call "$\beta$-space."

In turn, what would it take to be embedded onto the <span style="color: royalblue">blue zone</span> of $\beta$-space? Notice that any point in $\beta$-space is given by $Da$, where $a$ is the output feature vector from Feature Set 1 (the previous MLP). This is the region corresponding to the solutions to:

$$
\begin{align*}
(Da) \cdot U_1 + b_{U,1} & > 0
\end{align*}
$$

where $b_{U,1}$ is the first term of the bias corresponding to Feature Set 2, i.e. the $b_{U}$ in $h = \text{ReLU}(Ux + b_{U})$, where $x \in \beta\text{-space}$. Let's express this as a linear inequality of the values of $a$:

$$
\begin{align*}
(Da) \cdot U_1 + b_{U,1} & > 0 \\
(U_1^\top D)a + b_{U,1} & > 0 \\
(U_1^\top D)_1 a_1 + (U_1^\top D)_2 a_2 + \cdots + b_{U,1} & > 0
\end{align*}
$$

This represents an open half-space where the boundary is a hyperplane in $\mathbb{R}^5$. **Let's call this open half-space $\Pi$.**