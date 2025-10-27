---
published: true
title: Multi-Layer Latent Space Visualization
date: 2025-10-02 00:00:00 -500
categories: [mechanistic interpretability]
tags: [mechanistic interpretability]
math: true
---

# In MLPs, What Input Gives What Output?

If you had a series of MLP (`nn.Linear`) layers chained together, you may like to answer the question: if I wanted the final layer outputs' first element (call $h_1$) to be activated (i.e. $> 0$), what would my input have to be? Is that even calculable? How would we do that?

Reverse engineering the inputs required to produce a desired output is definitely theoretically possible; after all, deep networks are not hash functions; they are reversible. However, to do so is very computationally expensive. In this post, we will walk through what that reverse engineering calculation looks like, and what building a tool to do so entails.

# Motivation: Simplified Conceptualization of Transformers

To set the stage, one may ask: who cares? Well, wouldn't it be great if you could answer: I want a model's response to my one-word prompt to be a certain word (e.g. "extermination"), what should my one-word prompt be? Or perhaps, I want my text-to-image model to produce an image that is a little more `{insert description of feature 16243, or whatever}`, what can I say to make it do that? To do so requires us to know how to pass constraints on the output all the way back to the input space. Modern models are not just MLP blocks - there are blocks like Attention, Convolution, Diffusion, that all complicate matters, but they do all have MLP blocks. This makes this reverse engineering process in pure-MLP toy models worth studying.

Under some conditions, we may also be able to approximate what the MLP components of modern models are doing by (lossily) ignoring the other components. For example, a transformer block roughly looks like this (sans the layer norms):

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

The key question we are trying to figure out now is: if we wanted to activate feature $i$ in $h$, where in $\alpha$-space must we be? To derive algorithms that'll allow us to compute this, let's walk through a simple example. Instead of modulating between spaces of 2 and 5 dimensions respectively, we'll scale it down to 2 and 3 dimensions, so that we can visualize all the spaces. Suppose we learnt a matrix $U$ containing columns (features $U_1$, $U_2$, $U_3$) that are regularly spaced apart in a wheel, and a $b_U$ vector that is essentially a small negative constant vector ($\mathbf{-0.3}$). Our $\beta$-space would like this (with the columns of $U$ plotted):

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
-0.3 \\
-0.3 \\
-0.3
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
\end{bmatrix} \right) a - 0.3 & > 0
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

Above shows "phase 1," where we Un-$\text{ReLU}$ the faces (only 1 dimension had been zero-ed by $\text{ReLU}$). This is more intuitive.

<div style="display: flex; justify-content: center;">
    <video width="500px" autoplay loop muted playsinline>
        <source src="../../images/multi_layer_space/poly_to_plane_phase2.mov" type="video/mp4">
    </video>
</div>
<br/>
Above shows "phase 2," where we Un-$\text{ReLU}$ everything else (multiple dimensions had been zero-ed by $\text{ReLU}$). This is less obvious because there are multiple projections going on.

Let's fact check. If we had a model as such:

$$
\begin{align*}
f(x) & = \text{ReLU} \left( U \cdot D \cdot \text{ReLU} \left( W x  + b_W \right) + b_U \right) \text{, with:} \\

W & = \begin{bmatrix}
1 & 0 \\
- \frac{1}{2} & \frac{\sqrt{3}}{2} \\
- \frac{1}{2} & - \frac{\sqrt{3}}{3}
\end{bmatrix}, b_W = \begin{bmatrix}
-0.5 \\
-0.5 \\
-0.5
\end{bmatrix} \text{, } \\

D & = \begin{bmatrix}
\frac{1}{4} & \frac{\sqrt{3}}{4} \\
\frac{1}{4} & -\frac{\sqrt{3}}{2} \\
- \frac{\sqrt{3}}{4} & 0
\end{bmatrix}, \\

U & = \begin{bmatrix}
1 & 0 \\
- \frac{1}{2} & \frac{\sqrt{3}}{2} \\
- \frac{1}{2} & - \frac{\sqrt{3}}{3}
\end{bmatrix}, b_U = \begin{bmatrix}
-0.3 \\
-0.3 \\
-0.3
\end{bmatrix} \text{, } \\


\end{align*}
$$

(all the parameters are as illustrated above), then if we were to randomly sample a whole bunch of 2-D points uniformly from $[-5, -5]$ to $[5, 5]$, i.e. 

```
num_samples = 10000
dim = 2
X = torch.rand(num_samples, dim) * (2 * 5) - 5
```

and send those points through the network $f$, and plot only those points that had a non-zero activation for feature 1 (i.e. $h_1 > 0$), we get:


<img src = "../../images/multi_layer_space/sampled_alpha_h1.png" alt="sampled_alpha_h1" width="400px"> 
*Only these points produce an activation in h1*

# Does This Scale?

Because we'd like to be able to do this algorithm for any number of layers, let's write out the necessary ingredients at each step of the way to figure out what our latent activation zones look like. This will also allow us to determine what the computational complexity is. We will compute what it takes to find the latent activation zone corresponding to just **one** end feature (in this case, it was $h_1$).

### Mapping out $\beta$-space

- Latent activation zone was a simple half-space defined by just 1 linear inequality ($U_1 \cdot x + b_{U, 1} > 0$)

### Mapping out $\alpha$-space, after having done $\beta$-space

- First, we have to propagate $\beta$-space back to $a$-space (not $\alpha$-space)! This corresponds to the truncated cube as shown above, which is made by the above constraint (linearly transformed by $W$), and the 3 $\text{ReLU}$ constraints. **This is `dim` + 1 contraints.**
- Then, we have to figure out how to map this to $\alpha$-space. This is difficult. When you think about what $\text{ReLU}$ does to a  `dim`-dimensional vector, there are basically `2 ** dim` different regimes you have to think about. For example, the regime where (pre-$\text{ReLU}$) `dim[0] < 0 and dim[1:] >= 0` (only the first element is negative) is different from the regime where `dim[-1] < 0 and dim[:-1] >= 0` (only the last element is negative). Of these regimes, only 2 regimes are ignorable: the one where all elements are negative (trivial), and the one where all elements are positive (un-encodable after $\text{ReLU}$). So, you have **`(2 ** dim) - 2` regimes, within which there could be an activation zone that is defined by up to `dim` + 1 constraints**. This is $O(2^\text{dim})$ computational complexity.

> This corresponds to the above example, where we expect `(2 ** 3) - 2 = 6` regions, but we only had 5, because the regime corresponding to $x \geq 0, z \geq 0$ did not have an activation zone.

### Mapping out to pre-$\alpha$-space: $\gamma$-space

Now with $O(2^\text{dim})$ activation zones in $\alpha$-space, you can imagine that if we were to do the same steps to back-propagate these activation zones back by 1 more MLP block (to a "pre-$\alpha$-space," which I'll just call $\gamma$-space). Each of the $O(2^\text{dim})$ activation zones would form a new set of constraints with $\text{ReLU}$, which would result in potentially another $O(2^\text{dim})$ latent activation zones in $\gamma$-space. You can see how this becomes $O(2^{2 \times \text{dim}})$ latent activation zones. **The number of zones you'd have to track is essentially $O(2^{L \times \text{dim}})$, where $L$ is the number of layers.**

## Unavoidably Exponential

You may wonder: why are tracking all these latent activation zones separately even though are (or at least seem to be) contiguous? Couldn't we combine some of them, or better yet, just record their vertices, and do away with the potentially exponential number of zones? The problem here is that every step of the back-propagation across spaces requires us to do projections to compute points of intersection. These points of intersection are easiest obtained by solving the system of linear inqualities that describe the region, and if you have a non-convex region, you can't have a consistent set of linear inequalities that describe the entire region. They must be thought of as the union of convex sub-regions.

Also, if you think about what $\text{ReLU}$ does, it splits every dimension of a latent space into 2 regimes, one silent one (for the negative values of the domain), and one linear one. If you have `dim` dimensions, then naturally you can have up to `2 ** dim` different regimes, which is why even simple non-linearities like $\text{ReLU}$ are so powerful. **You will never be able to side-step the problem of having $O(2^\text{dim})$ regions of space to reason about.**

# Intractible

Having walked through the technique one would use to back-propagate constraints on the output space all the way back to the output space, we observe how the number of constraints end up exploding, especially after just a small number of layers for a production model with many dimensions. Until a use-case comes by that piques my interest further, I would shelve this tool as not worth building.