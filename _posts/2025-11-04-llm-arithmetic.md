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

The model is trained on simple input triplets of tokens: `(token_A, token_B, token_C)`, where `token_A` is ALWAYS token index `113`, which represents the `=` sign. `token_A` and `token_B` will simple be the index of the number they represent, i.e. token index `0` represents the number 0, token `42` represents the token 42, and so on.

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

# 5.1. Looking at the Circular Embeddings (W_E Columns)

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

# 5.2. Same Thing (W_L Rows)

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
