---
published: true
title: Automatic Reinforcement Unlearning
date: 2023-12-06 00:00:00 -500
categories: [research,statistics,interpretability]
tags: [mechanistic-interpretability]
math: true
---

# Introduction

This past year has really got me deep-diving into mechanistic interpretability research. I think it makes so much sense as a computational **science**, is very fundamental and generalizable (in the sense that when something better than LLMs appears, many of these techniques will likely still be transferrable), and more importantly, holds the key to identifying shortcomings and strengths of models in a tractible framework, that will allow us to develop better models accordingly. 

I've read a bunch of really awesome papers; those that have great intuition and present mathematically / methodologically feasible ways of peering under the hood of how models learn, represent information, and make decisions. They are too many to list, but 3 in particular have got me thinking about the idea of "Automatic Reinforcement Learning," that this post will explore. They are:

* [Discovering Latent Knowledge in Language Models without Supervision](https://www.lesswrong.com/posts/L4anhrxjv8j2yRKKp/how-discovering-latent-knowledge-in-language-models-without)
* [Toy Models of Superposition - Anthropic](https://transformer-circuits.pub/2022/toy_model/index.html)
* [Causal Scrubbing - Redwood Research](https://www.lesswrong.com/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing)

# Automatic Reinforcement Unlearning

So there is a problem that I learned about first from Collin Burns et. al in *Discovering Latent Knowledge in Language Models without Supervision*. TLDR of the problem: **sometimes, models will *know* something, but not tell you that which it knows.** This can already happen with present-day language models, and can arise due to reasons such as uncareful RLHF. For example (hypothetically), GPT-4 is a nice model; it probably won't tell you that your cat in the picture that you gave it is really ugly. Perhaps it'll tell you that "there seems to be possible improvements in lighting," even though it may know that this picture falls in a subspace of "ugly." Similar things can happen for violent content - perhaps you're asking for a way to defend yourself in a situation, of which you give lots of violent detail, but the model recognizes it to be a subspace of violent requests that it should not comply with. It likely could provide useful suggestions, but maybe it chooses not to comply with the request as it contains violent content. The question, then, is **how do you elicit this latent knowledge that the model is not explicitly giving out**? The paper presents an interesting approach based on logical predicates, which I'll do a quick summary of, because it motivates my approach.

# Logical Consistencies within Model Embeddings

Burns asserts that there must be black box-function (call $p_\theta : v \to y$, where $v$ is a vector and $y \in [0, 1]$) that can process a model's activations for two pairs of inputs (we'll linearize each set of activations so that we have the pair of vectors $\{\phi(x^+), \phi(x^-)\}$), where the pair is logically opposite to each other, such that the function yields opposite results. For example, for a phrase "Is this cat ugly?", we have the pairs:

* $x^+$: "Is this cat ugly? Yes."
* $x^-$: "Is this cat ugly? No."

The black-box function hence should be approximately such that:

$$
p_\theta(\phi(x^+)) \approx 1 - p_\theta(\phi(x^-))$
$$

The authors also took care to ensure that the degenerate solution of $p_\theta(\phi(x^+)) = 1 - p_\theta(\phi(x^-)) = 0.5$ is avoided by minimizing the following loss:

$$
\sum \begin{cases}
\mathcal{L}_\text{consistency} = \left( p_\theta (\phi(x^+)) - (1 - p_\theta (\phi(x^-)))\right) ^ 2 \\
\mathcal{L}_\text{confidence} = \min\left\{ p_\theta(\phi(x^+)), p_\theta(\phi(x^-)) \right\}^2
\end{cases}
$$

There is definitely an idea that a model is able to encode "yes-ness" and "no-ness" to the model, which leads to this new approach that I propose below.

# Fine-Tuning introduces Bias

RLHF fine-tuning can be seen as introducing a systematic bias to a model, such that the model activations for an input datapoint, if seen as an embedding ($\phi(x)$), will be pushed to a half-space (or intersection of many half-spaces, or linear hull, etc.) that basically tells the model to “play nice” or “avoid responding” or whatever reinforced behavior, if it falls under the category of undesirable prompts.

In mathematical parlance, we can somewhat hand-wavily say that there is relatively large bias in the component direction(s) (we’ll just call this vector $\rho$) that lead to these undesirable half-spaces.

* Let $X^+$ denote the samples (prompt-response pairs) that RLHF taught to reinforce.
* Let $X^-$ denote the samples (prompt-response pairs) that RLHF taught to NOT do.

Within $\{\phi(x) \mid x \in X^- \}$, we expect $\phi(x)$ to be pushed to the “undesirable behavior” zone. So, I hypothesis that bias along $\rho$ will be large. Specifically:

$$
\frac{\mathbb{E}[\phi(x) \cdot \rho]}{\text{Var}(\phi(x) \cdot \rho)}
$$

will be relatively large, compared to all the other components. Likewise for $\{\phi(x) \mid x \in X^+\}$. 

# Automatic Bias Unlearning by Projecting It Out

Also possibly described as unlearning the fine-tuned behavior, the procedure I’m interested in investigating is:

1. Perform PCA ($U\Sigma V^\top$) on the embeddings of a toy model (on $X^+ \cup X^-$) that has been RLHF fine-tuned as described.
2. For each component $\rho \in U$, we compute the average of the above bias to variance ratio across $x \in X^+$ and $x \in X^-$. Call this $\omega_\rho$.
3. Project $\omega_\rho$ out of each $\phi(x)$ and use that for downstream tasks. Evaluate on downstream tasks. We expect to see that the specific behavior that was reinforced by RLHF fine-tuning has deteriorated. This can be done via:
    
$$
\hat{\phi}(x) = \left[I - \frac{1}{d}\omega_\rho \omega_\rho^\top \right] \phi(x)
$$
    

Although this seems conceptually plausible, it depends on important assumptions, including the assumptions of linearity of model activation embeddings, which can only be empirically verified (we have reason to believe it holds though, from other papers, such as the Anthropic one above!). I would be interested to see how this deterioration would look like, and to see if there are implications on methods to jailbreak a model without having access to **all** the weights, but perhaps only some.