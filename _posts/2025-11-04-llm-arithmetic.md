---
published: true
title: LLM Arithmetic
date: 2025-11-04 00:00:00 -500
categories: [mechanistic interpretability]
tags: [mechanistic interpretability]
math: true
---

There are thus far a few algorithms that researchers have discovered are being implicitly implemented by LLMs to be able to perform simple arithmetic. They include:
- The "Clock" algorithm ([Kantamneni and Tegmark](https://arxiv.org/html/2502.00873v1))
- Bag of Heuristics ([Nikankin et. al](https://arxiv.org/pdf/2410.21272), [Anthropic](https://transformer-circuits.pub/2025/attribution-graphs/methods.html#graphs-addition))
- Digit-wise and Carry-over Circuits ([Quirke and Barez](https://arxiv.org/pdf/2310.13121#:~:text=Further%2C%20while%20simple%20arithmetic%20tasks,for%20AI%20safety%20and%20alignment.))

**This post focuses on the rather interesting Clock algorithm,** which itself takes inspiration from the original Clock algorithm alluded to by Nanda et. al in [Progress Measures For Grokking Via Mechanistic Interpretability](https://openreview.net/pdf?id=9XFSbDPmdW), henceforth referred to as just the _Modulo Arithmetic Paper_.

# 1. Objective

The Clock algorithm is interesting because is admits an intuitively simple description: LLMs encode some periodic mechanism that conveniently captures the periodic nature of written numbers (digits go from 0 to 9 before wrapping around and repeating), and this periodic mechanism allows it to accurately perform simple or modulo addition. In the case of modulo arithmetic, the mental image this evokes is simple and beautiful; e.g. LLMs represent numbers on a circle much like a clock, and doing `(10 + 6) % 12 = 4`, much like how "10 o'clock + 6 hours = 4 o'clock."

Indeed, in both the Kantamneni and Nanda papers, the authors demonstrate proof that this circular representation of numbers actually exist in models specifically trained to perform simple / modulo arithmetic. Both papers also go on to use ablations, activation patching, and other methods to identify parts of the model that MUST be involved in the arithmetic computation, and essentially rely on modus tollens (if ablate out sine and cosine components, model performance goes to shit, therefore periodic structure must there) to numerically justify that the model must be using this periodic representational structure to perform the addition. However, they still don't quite fully paint a picture of how the model is "reading out" the answer. For example, what happens to the representation (both pre-attention and post-attention) of the `=` token (i.e. the prediction for the answer) when you change the prompt from `a + b =` to `a + d =` where `d` is greater than `a` by a known amount, and specifically how do the MLP embedding and unembedding matrices handle such a change to give a correct answer still? This is insufficiently useful in giving us a deep understanding for how the models are manipulating said periodic structures.

This post aims to re-construct the Modulo Arithmetic Paper with special focus on:
- The relationship between $W_E$ (the embedding matrix that maps vocabulary space to residual space) and $W_\text{in}$ (the up-projection matrix of the MLP block)
- What each attention head is doing
- The relationship between $W_E$ and / or $W_\text{in}$ and $W_L = W_U W_\text{down}$, where $W_\text{down}$ is the down-projection matrix of the MLP block, and $W_U$ is the unembedding matrix that maps residual space back to vocabulary

# 2. Model Task

To restate the paper's model task, a simple **1-layer Transformer** is trained to finish the statement: `{a} + {b} =` where `a` and `b` are numbers $\in [1, P]$, where $P$ is some prime, in this case $113$, and the answer is the mathematical answer to $a + b \mod P$.

<img src = "../../images/llm_arithmetic/nanda_intro.png" alt="Nanda Model" width="100%"> 
*Model Architecture and Illustration of Periodic Representation*

# 2.1. Exact Inputs and Outputs

The model is trained on simple input triplets of tokens: `(token_A, token_B, token_C)`, where `token_C` is ALWAYS token index `113`, which represents the `=` sign. `token_A` and `token_B` will simple be the index of the number they represent, i.e. token index `0` represents the number 0, token `42` represents the token 42, and so on.

> Sometimes I'll refer to `triplets` as `pairs` since only `token_A` and `token_B` are meaningfully variable.

The target label is calculated by `(token_A + token B) % P`, where `P = 113`. This will be matched against the `argmax` of the raw outputs of the model, which are `logits` of shape `(d_vocab,)`, where `d_vocab = P + 1 = 114` (remember the extra token due to the `=` token). The loss is hence `CrossEntropyLoss`.

# 2.2. Train and Test Data

Just like in the paper, we generate all `(token_A, token_B)` pairs and randomly obtain a 30% split for the train set and use the remainder as the test set.

# 3. Architecture

This image fully illustrates the architecture of the 1-layer transformer that we'll use to reproduce this paper:

<img src = "../../images/llm_arithmetic/grokking_architecture.png" alt="Grokking Architecture" width="100%"> 
*Architecture of 1-Layer Transformer We Use to Reproduce Paper*

Things to note:
- There is no layer norm
- There are no positional embeddings
- MLP activation function shall be $\text{ReLU}$

Both of the above intuitively don't play a crucial role in this narrow task, and were confirmed to be irrelevant in the paper.

# 4. Achieving 0 Test Loss

The first step is obviously to make the model learn (grok) modulo arithmetic. We know that this is achieved when the model achieves **0 test loss**, which it does:

<img src = "../../images/llm_arithmetic/achieving_0_loss.png" alt="Achieving 0 Test Loss" width="100%"> 
*Achieving 0 Test Loss*

Notice that the test accuracy hits 100% at about 4,000 epochs, but just for good measure, I let the model train till 10,000 epochs. This is just in case the regularization (`AdamW`) is still hard at work shrinking irrelevant weights (or weights dedicated only to memorization and not generalization) to 0; as you can see, the sum of squares of the `W_E` weight matrix still continues to decrease till about 6,000 steps.

# 5. Inspecting Periodicity

Before we can even try to tackle the objectives of this blog post, the next step is to confirm that there is some periodic mechanism in the weights, which we do by performing a Discrete Fourier Transform (DFT) on `W_E`, just as the authors have done. Also, because the authors hand-implemented DFT and left out some code, I'm including my code here for completeness.

<details>
<summary>Code of DFT</summary>

<div class="language-python highlighter-rouge">
<pre style="background-color: #f7f7f7; padding: 16px; border-radius: 6px; padding-left: 1.5rem; overflow-x: auto; font-family: 'SFMono-Regular,Menlo,Monaco,Consolas,monospace', Consolas, monospace; font-size: 76.5%;">

def inspect_periodic_nature(model: Transformer, do_DFT_by_hand: bool = False):
    """
    We want to see high-magnitude fourier components for:
    1.  W_E
    2.  W_L = W_U @ W_out
    """
    W_E = model.embed.W_E.detach().cpu()                    # shape (d_model, d_vocab)
    W_E = W_E[:,:-1]                                        # shape (d_model, P)
    # We expect periodicity over the vocab dimension.
    d_model, P = W_E.shape

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    cmap = plt.get_cmap('coolwarm', 7)

    if do_DFT_by_hand:
        fourier_basis = []

        # start with the DC freq
        fourier_basis.append(torch.ones(P) / np.sqrt(P))

        # All the pairs of sin cos
        for i in range(1, P // 2 + 1):
            fourier_basis.append(torch.cos(2 * torch.pi * torch.arange(P) * i/P))
            fourier_basis.append(torch.sin(2 * torch.pi * torch.arange(P) * i/P))
            fourier_basis[-2] /= fourier_basis[-2].norm()
            fourier_basis[-1] /= fourier_basis[-1].norm()

        fourier_basis = torch.stack(fourier_basis, dim=0)   # Shape (P, P), waves going along dim 1

        fourier_coeffs = W_E @ fourier_basis.T
        fourier_coeff_norms = fourier_coeffs.norm(dim=0)

        x_axis = np.linspace(1, P, P)
        x_ticks = [i for i in range(0, P, 10)]
        x_tick_labels = [i // 2 for i in range(0, P, 10)]

        colors = [cmap(2) if i % 2 == 0 else cmap(5) for i in range(P)]
        ax.bar(x_axis, fourier_coeff_norms, width=0.6, color=colors)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)
</pre>
</div>
</details>


<img src = "../../images/llm_arithmetic/W_E_fourier_coeff_norms.png" alt="W_E Fourier Coefficient Norms" width="100%"> 
*My W_E Fourier Coefficient Norms*

The expected characteristic is definitely present: **some frequencies have very high relative norms, indicating some sparse periodic structure.** However, it's still noticeably different from the result reported in The Modulo Arithmetic Paper:

<img src = "../../images/llm_arithmetic/W_E_fourier_coeff_norms_theirs.png" alt="Their W_E Fourier Coefficient Norms" width="100%"> 
*Their W_E Fourier Coefficient Norms*

In particular, these are the differences I note:
1. The original paper's "inactive" `W_E` coefficient norms are very small
2. The original paper's "active" `W_E` frequencies are very clean, whereas I only managed to produce 4 clean pairs in this run, with a few other frequencies being somewhat in-between in norm.
3. The frequencies that are "active" are different from the set of 5 that they cited in the paper.

My hunch for the first two points is that the weight decay works over the training process to slowly pare away frequencies that the model can afford to do without, and that I simply haven't allowed the model enough weight decay time to do that. This is somewhat supported by another run I did for only 5,000 epochs:

<img src = "../../images/llm_arithmetic/W_E_fourier_coeffs_5k.png" alt="My W_E Fourier Coefficient Norms (5k epochs)" width="100%"> 
*My W_E Fourier Coefficient Norms (5,000 epochs)*

We can see here that with fewer epochs, I have even more "active" frequencies, that the model has ostensibly not "cleaned up." As for point (3), that is not of any concern as the model doesn't *have* to learn any specific set of frequencies, so long as those frequencies are useful (what is "useful" can be explained multiple ways, but I'll just chalk it up to the explanation of constructive interference that was given in the paper; it's poorly explained, but I'll simply believe it as it's not the point of this post for now).

Just for good measure, let's try 20,000 epochs:

<img src = "../../images/llm_arithmetic/achieving_0_loss_20k.png" alt="Training Metrics (20k)" width="100%"> 
*Training Metrics (20,000 epochs)*

<img src = "../../images/llm_arithmetic/W_E_fourier_coeffs_20k.png" alt="My W_E Fourier Coefficient Norms (20k epochs)" width="100%"> 
*My W_E Fourier Coefficient Norms (20,000 epochs)*

You can see that there are fewer but cleaner frequency norm spikes now. One of my initial questions when reading the original paper was - why 5 circles? Clearly you just need 1 circle - why did the model settle on 5? I guess we now know that multiple local periodic structures form simultaneously, and the model eventually prefers the higher-signal periodic structures and pares away the redundant / lower signal structures; 5 was just an arbitrary number that the original authors happened to arrive at.

# 5.1. Looking at the Circular Embeddings (`W_E` Columns)

Now that we have a grokked model to work with (I'm just going to use the 20,000 epochs one), we can attempt to see if we can actually visualize these embeddings.

Given that:
- We know there's some periodic structure along the `d_vocab` dimension of `W_E`
- We know what the Fourier Coefficients for each frequency multiple (`k`) are

Let's try to visualize the embeddings in 2 chosen dimensions to see if the embeddings are indeed spherical. How may we choose the 2 dimensions? Well, if we computed our Fourier Coefficients Norms as such:

```
fourier_coeffs = W_E @ fourier_basis.T              # shape (d_model, P)
fourier_coeff_norms = fourier_coeffs.norm(dim=0)    # shape (P,)
```

Then, to visualize the circular structure for `k = 4`, we would simply take `fourier_coeffs[:, [1 + 2 * (k - 1), 2 + 2 * (k - 1)]]` (i.e. the columns that correspond to the sin and cos `d_model`-dimensional coefficients for `k = 4`) as the 2 basis vectors. Then, we would simply project the 113 `W_E` vectors onto these 2 basis vectors (normalized) and see what we get.

We get the circles, as expected:


<img src = "../../images/llm_arithmetic/found_the_circles.png" alt="W_E columns in circle spaces" width="100%"> 
*W_E columns in Fourier Coefficient Basis for k in {4, 32, 43} (20,000 epochs)*

> In case you were wondering how I color-coded the points to get such a continuous spectrum, I simply did this:
```
# We do milli_periods because cmaps can't give us decimal periods
milli_period = int(1000 * 113 / k)
cmap = plt.get_cmap('coolwarm', milli_period)
colors = [cmap(i * 1000 % milli_period) for i in range(P)]
```

Beautiful. And just so we clarify that this only happens for key frequencies, let's just randomly choose some other `k` values to plot the same for:

<img src = "../../images/llm_arithmetic/no_circles.png" alt="W_E columns in circle spaces" width="100%"> 
*W_E columns in Fourier Coefficient Basis for k in {3, 25, 50} (20,000 epochs)*

Notice that they are random and low-magnitude.

# 5.2. Same Thing (`W_L` Rows)

The paper mentions that they found the same key frequencies in $W_L$, where:

$$
W_L = W_U W_\text{down}
$$

So, let's do the same for `W_L`, which has shape `(d_vocab, d_mlp)`. Let's first inspect the Fourier Coefficient Norms:

<img src = "../../images/llm_arithmetic/W_L_fourier_coeffs_20k.png" alt="My W_L Fourier Coefficient Norms (20k epochs)" width="100%"> 
*My W_L Fourier Coefficient Norms (20,000 epochs)*

And plot our circles:

<img src = "../../images/llm_arithmetic/found_the_circles_W_L.png" alt="W_L rows in circle spaces" width="100%"> 
*W_L rows in Fourier Coefficient Basis for k in {4, 32, 43} (20,000 epochs)*

# 6. What Are The Attention Heads Doing?

The next step in figuring out how the model makes use of these circles is to figure out how the circles connect to the MLP block. Let's take a look at the attention maps for 6 samples pairs for all 4 heads:

<img src = "../../images/llm_arithmetic/rubbish_attn_maps.png" alt="Rubbish attention maps" width="500px"> 
*Attention Maps for 6 randomly sampled pairs*

Let's focus on the last row of each map since it tells you where the attention head is focusing on while processing the last token (`=`). It looks like rubbish. In particular, I was hoping that we'd see that some attention head has learnt to pay equal attention to the first two tokens, and that all other attention heads become useless, but we don't super see evidence of this.

So where do we go from here? Well, we know that:
- Attention is just something like the relative magnitude of all the $q \cdot k_i$, for some query vector $q$, where $i$ denotes token index.
- $q$ and $k$ are just linear projections of the input vector to the attention blocks (i.e. the vectors in the $\mathbb{R}^{128}$ space where we found circles; henceforth called "embedding space").

Therefore, if there's some periodicity in the embedding space, there MUST be some periodicity in the $q$ and $k$ spaces, which means there MUST be some periodicity in the attention maps. Let's fix one of the numbers (and vary the other) and plot the latest row of the attention maps for a large number of sample pairs:

<img src = "../../images/llm_arithmetic/periodic_attn_maps.png" alt="Periodic attention maps" width="500px"> 
*Attention Maps are actually Periodic!*

And we see here that the attention maps are indeed periodic! As a bonus, let's try to do DFT on these to get their primary frequencies.

<img src = "../../images/llm_arithmetic/attn_fourier_coeffs_20k.png" alt="Attention Fourier Coeff Norms" width="100%"> 
*Fourier Coefficient Norms of Attn Activations (Last Row) by Head*

You can see that the key frequencies 4, 32, and 43, are distributed over the heads. Some interesting observations include:
- Head 1 is most attuned to the frequency 4, which we can see from the attention maps
- Head 2 and Head 3 are most attuned to the frequencies 43 and 32 respectively, we we can also see from the attention maps
- Head 0 has high-norm coefficients in multiple frequencies, including several non-key frequencies, which interfere to give a non-sinusoidal pattern, which we can also see from the attention maps. It is unclear if the attenuation to multiple frequencies plays some crucial role, or if they're simply not yet sufficiently decayed.

Crucially, now's also a good time to point out that the attention patterns are not just periodic when varying `b`, but also when varying `a`. This makes sense because `a` and `b` are embedded using the sample `W_E` matrix, meaning that they both use the same circle structures in `W_E`. To see this, we can just visualize the attention score of `a` (or `b`, since they approximately sum to 1; I just arbitrarily chose `a`) in the last row, for all `(a, b)` pairs:

<img src = "../../images/llm_arithmetic/full_p_by_p_plot.png" alt="Full Attention Map" width="100%"> 
*Attention Maps are Periodic in 'a' and 'b'*

The original Modulo Arithmetic Paper also found such attention maps:

<img src = "../../images/llm_arithmetic/full_p_by_p_plot_og.png" alt="Full Attention Map" width="100%"> 
*Original Paper's Attention Maps are Periodic in 'a' and 'b' Too*

# 6.1. Periodic Attention on Periodic Embeddings

So we've observed that circular / periodic embeddings also mean:
- Periodic $k$, $q$, and $v$ vectors (since each of these are created via linear projections of the embeddings)
- Periodic attention patterns.

Remember that the attention values ($a_i$, where $i$ denotes token index) all sum to 1, which means that the attention output for token $i$ (call $z_i$), i.e.:

$$
\begin{align*}
z_i = \sum_{j \leq i} a_j * v_j
\end{align*}
$$

is a **convex combination** with periodic convex coefficients. In this setting where the attention (when processing the `=` token) is mostly on the first two tokens (`a` and `b` in the equation `a + b =`), we can roughly simplify this to:

$$
\begin{align*}
z_= = \alpha W_Vx_a + (1 - \alpha) W_V x_b
\end{align*}
$$

where $x_a$ and $x_b$ are the embeddings for tokens `a` and `b` respectively.

For the next step, I wish to see what a periodic convex combination of periodic features looks like. As of yet, we don't know what $W_V$ is, but we know that it is a simple linear projection, which means it won't *warp* the embedding space. I.E. if a bunch of embeddings formed a circle in the embedding space, applying $W_V$ to them may stretch / squeeze them, but won't *warp* them into a non-circle-looking shape like a bean, a knot, a square, or whatnot. Because a simple linear transformation won't fundamentally change the shape of the embeddings, and it is the shape that we're interested in, I'll just pretend $W_V$ is the identity matrix. Then, I can just plot $o$, given by:

$$
\begin{align*}
z_= = \alpha x_a + (1 - \alpha) x_b
\end{align*}
$$

for some periodic $\alpha$. Since Head 1 looks rather much like a simple sinusoidal curve, we'll just take $\alpha \sim \cos(2 \pi f)$ for some frequency $f$. For our plots, we will also fix `a` (hence $x_a$), and vary `b` with the same frequency. The actual value for Head 1's frequency is $4$, but for the sake of illustration, we'll make it $0.5$.

<div style="display: flex; justify-content: center;">
    <video width="100%" autoplay loop muted playsinline>
        <source src="../../images/llm_arithmetic/05_hz_diff_offsets.mov" type="video/mp4">
    </video>
</div>
<br/>

In fact, depending on how many wavelengths the attention coefficient $\alpha$ is offset (and I suspect each different value of `a` has a different offset), we get a whole family of curves!

<div style="display: flex; justify-content: center;">
    <video width="500px" autoplay loop muted playsinline>
        <source src="../../images/llm_arithmetic/05_family_of_curves.mov" type="video/mp4">
    </video>
</div>
<br/>

When the circle (embedding) frequency differs from the attention frequency, we also get different families of curves. Below, I illustrate what the attention output ($z$ or $v$) space looks like when we focus on the dimensions in which the periodicity is different from what the attention heads are attuned to.

<div style="display: flex; justify-content: center;">
    <video width="100%" autoplay loop muted playsinline>
        <source src="../../images/llm_arithmetic/family_of_curves.mov" type="video/mp4">
    </video>
</div>
<br/>

You will see the characteristic petal shapes in all combinations of embedding and attention frequencies. However, notice that when the attention frequency (alpha) is lower than the embedding frequency, you get petals that overlap, whereas when it's above, you get petals that don't overlap.

> A friend Jacob said they're reminiscent of trigonometric polar graphs; indeed, the construction of these curves are different but quite similar, and I wonder if there is some profound connection here.

# 6.2. Petals in Attention Outputs (z)?

The above curves are theoretical predictions for what the outputs $z$ look like when the attention values are periodic with some frequency (`alpha`) and the embeddings are periodic with some other frequency (`embedding freq`), when we fix token `a` and vary token `b`. Let's see if we can actually find these petal shapes in the actual $z$ values for `a = 70` (arbitrarily chosen) and `b in range(113)`.

## Example: Head 1 on 4 Hz Circle Subspace

The above section is a theoretical prediction on what the attention outputs should look like if we were to plot them all for all 113 pairs `(a = 70, b) for b in range(113)`. In visualizing the attention outputs $z$ (not $o$, i.e. just before applying $W_O$), we have two decisions:
- Which head's outputs to visualize?
- Which 2D subspace to visualize?

We'll pick the simplest option of looking at Head 1 (because its attention values are simply periodic in only 4 Hz), and visualize the 2D subspace corresponding to the 4 Hz circle in the embedding space. Now, what does this mean?

Remember that **to visualize the 4 Hz circle in the embedding space, we had to use the fourier coefficients to figure out 2 directions to use as our 2D axes**:

```python
k = 4
fourier_coeffs = W_E @ fourier_basis.T                              # shape (d_model, P)
basis_vecs = fourier_coeffs[:, [1 + 2 * (k - 1), 2 + 2 * (k - 1)]]  # shape (d_model, 2)
basis_vecs_norm = basis_vecs.norm(p=2, dim=0, keepdim=True)
basis_vecs /= basis_vecs_norm                                       # shape (d_model, 2)

circle_coords_to_visualize = W_E.T @ basis_vecs                     # shape (P, 2)
```

Because the embeddings are transformed via $W_V$ in the attention block, we want to follow these 2 directions through the transformation via $W_V$. The means answering the question: what 2 directions in the $v$ vectors (or $z$ vectors, since $z$ vectors are just weighted sums of $v$ vectors and hence $z$-space and $v$-space are equivalent) am I now interested in visualizing (that correspond to `basis_vecs` before the application of $W_V$)? Some linear algebra to figure this out:

```python
# This is the computation of v vectors in attn head i
V = W_E @ W_V_head_i                        # eqn shapes: (113, 32) = (113, 128) @ (128, 32)

# We would have wanted to visualize W_E @ basis_vecs.
# Now we only have V, so we have to figure out new_basis to extract from V,
# the equivalent of basis_vecs in W_E
W_E @ basis_vecs = V @ new_basis            # eqn shapes: (113, 128) @ (128, 2) = (113, 32) @ (32, 2)

# Substitute
W_E @ basis_vecs = W_E @ W_V_head_i @ new_basis

# Rearrange
basis_vecs       = W_V_head_i @ new_basis
new_basis        = pseudo_inverse(W_V_head_i) @ basis_vecs
                 = inverse(W_V_head_i.T @ W_V_head_i) @ W_V_head_i.T @ basis_vecs

# Coords to visualize
projected        = z_values @ new_basis     # shape (113, 2)
```

Above, we predicted that when the embedding circle frequency (4 Hz) is the same as the attention frequency (4 Hz), we'll get this:

<img src = "../../images/llm_arithmetic/prediction_4hz_4hz.png" alt="Prediction: 4Hz circle, 4Hz attention" width="500px"> 
*Periodicted z pattern in 4 Hz circle space, for 4 Hz attention frequency*

Let's see what actual embeddings we get:

<img src = "../../images/llm_arithmetic/actual_4hz_4hz.png" alt="Actual: 4Hz circle, 4Hz attention" width="500px"> 
*Actual z embeddings in 4 Hz circle space, for 4 Hz attention frequency*

## Example: Head 3 on 4 Hz Circle Subspace

Let's try another `(attention_head, subspace)` pair. Let's try head 3 (attention frequency of 32 Hz) on the 2D subspace corresponding to the 4 Hz embedding circle still,

Prediction:

<img src = "../../images/llm_arithmetic/prediction_4hz_32hz.png" alt="Prediction: 4Hz circle, 32Hz attention" width="500px"> 
*Periodicted z pattern in 4 Hz circle space, for 32 Hz attention frequency*

Actual:

<img src = "../../images/llm_arithmetic/actual_4hz_32hz.png" alt="Actual: 4Hz circle, 32Hz attention" width="100%"> 
*Actual z embeddings in 4 Hz circle space, for 32 Hz attention frequency*

## Example: Head 1 on 32 Hz Circle Subspace

Another one still!

Prediction:

<img src = "../../images/llm_arithmetic/prediction_32hz_4hz.png" alt="Prediction: 32Hz circle, 4Hz attention" width="500px"> 
*Periodicted z pattern in 32 Hz circle space, for 4 Hz attention frequency*

Actual:

<img src = "../../images/llm_arithmetic/actual_32hz_4hz.png" alt="Actual: 32Hz circle, 4Hz attention" width="100%"> 
*Actual z embeddings in 32 Hz circle space, for 4 Hz attention frequency*

I think this is pretty astoundingly beautiful! In particular, nobody has articulated the existence of these multi-lobed petal-like structures in embedding spaces of LLMs even if they've discovered the simple circles / periodic structures. These structures are also very surprising to me, because they don't seem very amenable to linear separation. Representations that sit on the perimeter of a simple circle are not surprising to me because every point can be linearly separated from the rest, and indeed that is what an MLP layer with $\text{ReLU}$ does (refer to ["Superposition - An Actual Image of Latent Spaces"](/posts/viewing-latent-spaces/) for illustrations of this), but the same cannot be said for points that sit on the perimeter of these petal shapes.


# 7. Petals Interfere to give Simple Circle

By extension, since the attention outputs ($o$) are merely linear projections of $z$ (i.e. $o = W_O \cdot z$), then these petal shapes should remain present in the $o$ space. For good measure, **for each attention head individually**, I went ahead and did similar visualizations for the head-specific $o$ vectors for all pairs `[(70, b) for b in range(113)]`, and saw the same petal structures.

However, visualizing head by head does **not** give the full picture of what's eventually going on in the $o$-space, because you are only considering each individual head's contribution to the $o$ vectors at any one time. Remember that all the $v$ vectors from each head are stacked, and $W_O$ is applied to that to transform it back to the $\mathbb{R}^{128}$ model space:

<img src = "../../images/llm_arithmetic/wo_action.png" alt="v vectors weighted sum of W_O columns" width="100%"> 
*v vectors get stacked, then get transformed by W_O*

The application of $W_O$ to the stacked $v$ vectors is equivalent to taking a weighted sum of the columns of $W_O$, where the coefficients are given by the values of the stacked $v$ vectors, so you can see that all the contributions of each attention head are summed together. The consequence of this is that it could well be possible for the contributions from head 2 (e.g.) to disrupt the petal structure contributed by head 1 (e.g.).

# 7.1. Empirical Evidence
Let's see how the different heads combine to give a final pattern.

## Zoom In: 4 Hz Circle Subspace

<img src = "../../images/llm_arithmetic/aggregation_of_petals_4hz.png" alt="Aggregation of Petals in 4 Hz" width="100%"> 
*o-vectors corresponding to subspace of 4 Hz Embedding Circle, head-wise and summed*

The petals structure seems to have disappeared in the aggregate $o$ vectors (red plot). But also, is it me or is there a new type of petal structure going on? Petals that look like when kids draw sunflowers (a bunch of conjoined semicircles forming a loop: &#x1F33C;)? That may merely be an artifact of the interpolation being too expressive / overfitting the points, just like in the attn head 1 plot. To be sure I'll just do a Fourier Decomposition and plot the norms of the coefficients for each coefficient again for the red points:

<img src = "../../images/llm_arithmetic/aggregation_of_petals_4hz_fourier.png" alt="Aggregation of Petals in 4 Hz" width="100%"> 
*Fourier Coefficient Norms for Projected 'o' vectors*

With such a dominant 4 Hz component, we are confident that the circle in the red plot is indeed a circle and not some complex petal-shaped curve. So, it's interesting that these complex petal-shaped curves have recombined to give a circle.

## 32 Hz Circle Subspace

The same happens for the 2D subspaces corresponding to the other embedding circles. Here are the plots for the 32 Hz embedding circle:

<img src = "../../images/llm_arithmetic/aggregation_of_petals_32hz.png" alt="Aggregation of Petals in 32 Hz" width="100%"> 
*o-vectors corresponding to subspace of 32 Hz Embedding Circle, head-wise and summed*

<img src = "../../images/llm_arithmetic/aggregation_of_petals_32hz_fourier.png" alt="Aggregation of Petals in 32 Hz" width="100%"> 
*Fourier Coefficient Norms for Projected 'o' vectors (32 Hz subspace)*

## 43 Hz Circle Subspace

And the 43 Hz embedding circle:

<img src = "../../images/llm_arithmetic/aggregation_of_petals_43hz.png" alt="Aggregation of Petals in 43 Hz" width="100%"> 
*o-vectors corresponding to subspace of 43 Hz Embedding Circle, head-wise and summed*

<img src = "../../images/llm_arithmetic/aggregation_of_petals_43hz_fourier.png" alt="Aggregation of Petals in 43 Hz" width="100%"> 
*Fourier Coefficient Norms for Projected 'o' vectors (43 Hz subspace)*

## Alternative Visualization

Now that we know that these 4 Hz, 32 Hz, and 43 Hz circles exist in the $o$ vectors, we can use another method to try and visualize them. It would simply be what we did at first with `W_E` and `W_L`: to run DFT on these $o$ vectors and to project them on the 2 directions given by the 128-dimensional $\sin$ and $\cos$ coefficients corresponding to each of these frequencies. I did them just to make sure that these circles were truly truly present in the $o$ vectors:

<img src = "../../images/llm_arithmetic/found_the_circles_o.png" alt="o vectors in circle spaces" width="100%"> 
*o vectors in Fourier Coefficient Basis for k in {4, 32, 43}*

I make no comment on why the 43 Hz Circle looks so stretched.

# 7.2. Why?

To state what we've observed so far in simple terms:
- We observed periodic (circular) embeddings
- Periodic embeddings necessarily lead to periodic attention coefficients
- Periodic attention on circular embeddings lead to complex periodic (multi-lobed) value embeddings
  - Each attention head has its own multi-lobed structure
  - Multi-lobed structures vary based on relationship between embedding circle and attention frequency
- These multi-lobed structures combine to give a simple single-lobed / circular structure in $o$ space again

**It would be nice if we could argue that:**
- The **main goal is to generate simple circles** in the $o$ space because they may be **particularly amenable to downstream readout functions** (MLP layer).
- Embeddings that approximate periodic functions like sines and cosines are especially amenable to composition (constructive interference) to give that desired simple circle. This is essentially the Fourier Decomposition argument again - something like: just like how sines and cosines of different magnitudes and frequencies can combine to give any arbitrary function, **multiple instances of the above petal structures (generated the way we generated them) of various orientations and frequency combinations can combine to give any arbitrary function**, and particularly, a simple circle.

The above would be a neat argument for why periodic embeddings are learnt, and why there are circles of MULTIPLE frequencies, not just one.

For now, this argumentation remains out of scope of this blog post and we will simply operate from the fact the petal structures just coalesce to form a simple circle again.

# 8. How Does MLP Use These Circles?

## Gotcha: Do The `W_E` and `o` Circles Coexist in the Same Subspace?

Between the attention and MLP blocks, the attention output ($o$) is added to the residual stream, which already contains the original embeddings (columns of `W_E`), which itself is a circle. The immediate question to ask is: did the newly added $o$ circle interfere with the `W_E` circle? The only way the 2 circles would meaningfully interact is if they lived in the same subspace. To figure out if any pairs of these basis vectors spanned similar subspaces, for all combination pairs of a set of $o$-derived basis vectors and `W_E`-derived basis vectors, I plotted the dihedral angle (the second angle in the results of `scipy.linalg.subspace_angle`) between them. The dihedral angle is small only if the subspaces are very similar.

<img src = "../../images/llm_arithmetic/dihedral_angles.png" alt="Dihedral Angles" width="700px"> 
*Dihedral Angles of all (W_E basis, o basis) pairs*

Ok, none of the pairs have a small dihedral angle, so we know that the $o$ vectors / circles are getting written into a different subspace than the ones that the original `W_E` circles live in.

## Guess: `W_up` Should Have 113 Vectors That Align With `o` Circles

My initial guess was that I should be able to find a set (or even 3) of 113 vectors that more or less align with the directions of these $o$ vectors that are arrange in a circle. The reasoning behind this is that each one of these `W_up` vectors should act as a detector for $o$ vectors in that direction and basically produce an activation in one of the 512 elements in the MLP output vector. The MLP output vector will hence have 113 (or 3 sets of 113) positions which each represent 1 number, and `W_down` can just permute them into the correct position such that the softmax will be able to decode the correct number. However, upon deeper thinking, this wasn't quite right, for these reasons:
- If this were true, we expect $W_L$ to have a large frequency 1 Hz component, which it doesn't. Instead, it has the same frequencies as before (4, 32, 43).
- This model actually doesn't have superposition (!!!)

## This Model Doesn't Have Superposition

In ["Superposition - An Actual Image of Latent Spaces"](/posts/viewing-latent-spaces/), I illustrate how `W_up`, `b_up`, and $\text{ReLU}$ can cleverly segment a subspace into 2 regimes (per MLP dimension) in order to silence all other features except the one we care about:

<img src = "../../images/opt_failure/latent_zone_intro_1.png" alt="Aggregation of Petals in 43 Hz" width="100%"> 
*o-vectors corresponding to subspace of 43 Hz Embedding Circle, head-wise and summed*

For example, in the above example, unless an embedding were found in the <span style="color: royalblue">mid-blue region</span>, the post-$\text{ReLU}$ value of that feature will be 0. So, even if you activated the <span style="color: lightsteelblue">light blue</span> or <span style="color: mediumblue">dark blue feature</span>, unless you STRONGLY activated it, you would still be outside of the <span style="color: royalblue">mid-blue region</span>, and hence would be zero-ed by $\text{ReLU}$. The orientation of the regions are controlled by `W_up` and `b_up`. In particular, the more selective (small) you want region $i$ to be, the more **negative** `b_up[i]` has to be.

The **above mechanism is how a model is able to store more features (6) than it has dimensions (2).** In such a setting, the model is said to exhibit superposition. However, **this toy model** that we trained here, after the Modulo Arithmetic Paper, **does not actually exhibit superposition.** Or at least, it doesn't need to exhibit superposition, because **there are only 113 meaningful features** the model has to learn (i.e. `['the answer is {c}' for c in range(113)]`), **which can be contained within the 128 dimensions** of the model (`d_model`).

And so, the MLP block can learn simple linear transformations to maintain the entire geometric structure of the circles WITHOUT making use of $\text{ReLU}$ at all. In such a setting, the model just has to learn **positive** `b_up` values in order to **shift all the features further into the positive orthant** (in the `d_mlp = 512`-dimensional MLP output space), **in order to avoid non-linear distortion by $\textbf{ReLU}$**, which happens at the value 0 for all dimensions.

To prove that this is happening, we visualize the `b_up` values (sorted) and observe that insufficient (way fewer than 113) of them are significantly negative. The negative `b_up` values are what makes a useful superposition of features possible, and since there are way too few of them, we know that this model is not relying on superposition to encode features.

<img src = "../../images/llm_arithmetic/b_up_sorted.png" alt="b_up sorted" width="100%"> 
*b_up values, sorted*

## `W_up` Columns Don't Line Up With Circle

And indeed, when we plot the 512 columns of `W_up` in the 2D subspaces that we expect those circles to live (given by the sets of Fourier Coefficients obtained from the $o$ vectors), we don't see any nice circles:

<img src = "../../images/llm_arithmetic/W_up_does_not_align.png" alt="b_up sorted" width="100%"> 
*b_up values, sorted*

## The MLP Layer Is Just An Affine Transformation

Following the argumentation that `d_model = 128` is already sufficient dimensions to encode the 113 features without superposition, the job of the MLP layer isn't to exploit $\text{ReLU}$ to arrange features in a superposition, but to actually shift the circles into the positive orthant such that they are NOT affected by $\text{ReLU}$. What this entails adding some offset (which can be done by `b_up`). Here's some visual intuition:

<img src = "../../images/llm_arithmetic/mlp_shift.png" alt="b_up sorted" width="100%"> 
*MLP just has to move Circle into Positive Orthant; ReLU not necessary*

As proof that the model doesn't have superposed features / need $\text{ReLU}$, I trained a variant of the toy model without the $\text{ReLU}$ activation function, and it managed to grok Modulo Arithmetic:


<img src = "../../images/llm_arithmetic/reluless_loss_curves.png" alt="ReLU-less loss curves" width="100%"> 
*Training Curves of ReLU-less Model Variant*

To reiterate, since the MLP block doesn't actually introduce any meaningful non-linearity to the feature space, we know that the circles are basically preserved through the MLP block. We can see this is true by visualizing the respective 2D circle spaces again (same as before, basis vectors are computed using Fourier Coefficients for these frequencies) and see that the circles are still there:

<img src = "../../images/llm_arithmetic/found_the_circles_mlp_acts.png" alt="Embeddings in Circle Spaces within MLP activations" width="100%">
*Circles are still there in MLP Activations*

## Decoder (`W_L`)

By now it should be pretty obvious that `W_L` (I'll call this "decoder") simply has to align itself with the circles found in the MLP activation space.

For example, the $65$-th row of `W_L`, or `W_L[65]` (remember that `W_L` has shape `(P, d_mlp)`, hence `W_L[65]` has shape `(d_mlp,)`):
- Is the decoder for the number $65$ (as in `"the answer is 65"`),
- So it has to align with the the number $65$'s embedding on the MLP-space circles. This means literally aligning itself parallel to $65$'s embeddings so that dot product is maximized.

Notice that if it were to just align with 1 of the circles (e.g. 4 Hz Circle) without caring about the others, it will NOT be able to properly target the $65$-th feature. This is because the circles are not at all "perfect" and there is a lack of linear separatability, so nearby features (that don't belong to the number $65$) could have larger magnitudes in the $65$ direction than the $65$ embedding itself. In the image below, we see that despite `W_L[65]` aligning perfectly to 65's embedding, it could be out-activated by other numbers like 93 and 8, which are further along the `W_L[65]` direction.

<img src = "../../images/llm_arithmetic/up_close_mlp.png" alt="up close MLP" width="600px">
*W_L[65] Perfectly Aligns with 65's Embedding, but is Activated More Strongly by 93, 8, and others*

So, the expectation here is that `W_L[65]` has to align with $65$'s MLP embedding with respect to all 3 circle spaces. 

> Because the circles have high frequency (> 1 Hz), they make several complete loops, and these complete loops are not "neat" and hence don't form linearly separable points that sit nicely on the perimeter of a perfect circle. From this, one may get the impression that because this is the case, THEREFORE the model NEEDS multiple circles of different frequencies to be able to disambiguate all numbers. I want to clarify that this is not a strong explanation for having multiple circles. **Firstly,** nothing inherently prevents these loops to be "neat" / "perfect." It's just that because the model is not relying totally on linear separability with respect to one circle, the model is not incentivized to nudge the embeddings to create that perfect circle. **Secondly,** having multiple frequencies is helpful in achieving circular structures (as opposed to petal-shaped structures) in the $o$-space (because multiple frequencies enable multiple petal structures to form, and it is the recombination of them that reconstructs a circular structure). This necessitates having multiple frequencies, hence multiple circles, which relaxes the constraint of linear separability on any one circle.