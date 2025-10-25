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
(Da + b_\text{D}) \cdot U_1 + b_{U,1} & > 0 \\
(U_1^\top D)a + U_1^\top b_D + b_{U,1} & > 0 \\
(U_1^\top D)_1 a_1 + (U_1^\top D)_2 a_2 + \cdots + (U_1^\top D)_5 a_5 + U_1^\top b_{D} + b_{U,1} & > 0
\end{align*}
$$

> For the sake of simplicity, we see that the terms $U_1^\top b_{D}$ and $b_{U,1}$ are independent of the activations $a$. So, I will simply forget about $b_{D}$; if it so turns out that $b_{D}$ needs to be non-zero, the model can compensate for this using $b_{U,1}$.

This represents an open rank-5 half-space where the boundary is a 4-dimensional hyperplane in $\mathbb{R}^5$. **Let's call this open half-space $\Pi$.** In addition to this constraint, which I will call **"Constraint Due To $h_1$,"** there are also the constraints that $a \geq 0$ (due to being immediately post-$\text{ReLU}$). I will call these the **"Constraints Due To $\text{ReLU}$."** Together, the region of $a$ that will end up activating $h_1$ become the intersection of 6 half-spaces (1 from Constraint Due To $h_1$, and 5 from Constraints Due To $\text{ReLU}$). Note that this is convex.

## Back Propagating to $\alpha$-space

The key question we are trying to figure out now is: if we wanted to activate feature $i$ in $h$, where in $\alpha$-space must we be? To derive algorithms that'll allow us to compute this, let's walk through a simple example. Instead of modulating between spaces of 2 and 5 dimensions respectively, we'll scale it down to 2 and 3 dimensions, so that we can visualize all the spaces. Suppose we learnt a matrix $U$ containing columns (features $U_1$, $U_2$, $U_3$) that are regularly spaced apart in a wheel, and a $b_U$ vector that is essentially a small negative constant vector ($\mathbf{-0.5}$). Our $\beta$-space would like this (with the columns of $U$ plotted):

<img src = "../../images/multi_layer_space/beta_space_3_features.png" alt="beta-space with U1, U2, U3" width="100%"> 
*Beta-space with U1, U2, U3*

Let's add in the decoder features of the previous MLP block (the rows of $D$):

<img src = "../../images/multi_layer_space/beta_space_with_D_3_features.png" alt="beta-space with U, D" width="100%"> 
*Beta-space with U, D*

> The $D$-vectors can be any arbitrary thing learnt by the model, but for the sake of simplicity and even representation (all regions of the 2D space are in the half-space pointed to by at least one of the $D$ vectors; we'll see why this is important later) I chose the above grey features as my $D$-vectors.

Let's focus on activating <span style="color: blue">$h_1$</span> (i.e. we want to be in the <span style="color: blue">blue zone</span>). This means that the embedding in $\beta$-space must lie in the <span style="color: blue">blue zone</span>, which means that $Da$ must lie in the <span style="color: blue">blue zone</span>. This is merely a recapitulation of **Constraint Due To $h_1$**:

$$
\begin{align*}
(Da) \cdot U_1 + b_{U, 1} > 0
\end{align*}
$$

Here, we chose $D$ (arbitrarily) to be:

$$
\begin{align*}
D = 0.5 \times
\begin{bmatrix}
\frac{1}{2} & \frac{1}{2} & -\frac{\sqrt{3}}{2} \\
\frac{\sqrt{3}}{2} & - \frac{\sqrt{3}}{2} & 0
\end{bmatrix}
\end{align*}
$$

And the decoder parameters $U$ and $b_U$ as shown above is:

$$
\begin{align*}
U = 
\begin{bmatrix}
1 & 0 \\
- \frac{1}{2}  & -\frac{\sqrt{3}}{2} \\
- \frac{1}{2} & \frac{\sqrt{3}}{2} \\
\end{bmatrix} \text{, }
b_U =
\begin{bmatrix}
-0.5 \\
-0.5 \\
-0.5
\end{bmatrix}
\end{align*} \\
$$

And forgetting about $b_\text{D}$, expanding **Constraint Due To $h_1$**, we get:

$$
\begin{align*}
(U_1^\top D) a + b_\text{U, 1} & > 0 \\ 
\Longrightarrow \left(
\begin{bmatrix}
\frac{1}{4} & \frac{1}{4} & -\frac{\sqrt{3}}{4} \\
\end{bmatrix} \right) a - 0.5 & > 0
\end{align*}
$$

This corresponds to a half-space in 3D. The below image illustrates what imposing this half-space constraint onto a cubic volume of space looks like:

<img src = "../../images/multi_layer_space/constraint_h1.png" alt="constraint_h1" width="100%"> 
*Origin-centered cube gets cut by Constraint Due To h1*

Remember that we also have the Constraints Due To $\text{ReLU}$, which essentially limits us to the positive orthant (hypercube):

<img src = "../../images/multi_layer_space/constraint_relu.png" alt="constraint_relu" width="400px"> 
*Origin-centered cube is limited to positive orthant after imposing Constraint Due To ReLU*

Combining (taking the intersection of) all the above half-spaces induced by their constraints, we get:

<img src = "../../images/multi_layer_space/constraint_all.png" alt="constraint_all" width="400px"> 
*a has to live somewhere in this space*

There is one more complication: $a$ doesn't have access to this entire volume of space, particularly because of the following.

Firstly, the decoder that projects the features (call $x$) from $\alpha$-space to $a$ (i.e. the $W$ and $b_W$ in the equation: $a = \text{ReLU}(Wx + b_W)$) implies a hyperplane that does NOT live in the dense region of space (positive-orthant). This is because, for decoders to be able to learn how to extract all features without also activating other features, the decoder hyperplane has to end up looking something like that (explained in ["Superposition - An Actual Image of Latent Spaces"](/posts/viewing-latent-spaces/)):

<img src = "../../images/opt_failure/latent_zone_intro_2.png" alt="Latent zone intro 2" width="400px">

Secondly, the final step of computing $a$ is applying $\text{ReLU}$. Combined with the above assumption, we see that really, only the axial **surfaces** of the above polygon are accessible.

<img src = "../../images/multi_layer_space/accessible_region.png" alt="accessible_region" width="400px"> 
*Only the darker surfaces (axial planes) are accessible*

If we were to propagate it back to before the $\text{ReLU}$, sort of undoing the projection onto the surfaces of the positive orthant that $\text{ReLU}$ does, we end up with the following, which I've broken down into 2 phases for clarity. The first phase involves un-$\text{ReLU}$-ing 1 dimension at a time so that the relationship between the pre and post-$\text{ReLU}$ regions is geometrically clear, and the second phase involves doing the rest.

<div style="display: flex; justify-content: center;">
    <video width="500px" autoplay loop muted playsinline>
        <source src="../../images/multi_layer_space/poly_to_plane.mov" type="video/mp4">
    </video>
</div>
<br/>
<!-- <img src="../../images/poly_to_plane.gif" width="500px" /> -->
