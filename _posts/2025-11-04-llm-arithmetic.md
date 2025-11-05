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

**This post focuses on the rather interesting Clock algorithm,** which itself takes inspiration from the original Clock algorithm alluded to by Nanda et. al in [Progress Measures For Grokking Via Mechanistic Interpretability](https://openreview.net/pdf?id=9XFSbDPmdW), henceforth referred to as just the _Modular Arithmetic Paper_.

# 1. Objective

The Clock algorithm is interesting because is admits an intuitively simple description: LLMs encode some periodic mechanism that conveniently captures the periodic nature of written numbers (digits go from 0 to 9 before wrapping around and repeating), and this periodic mechanism allows it to accurately perform simple or modular addition. In the case of modular arithmetic, the mental image this evokes is simple and beautiful; e.g. LLMs represent numbers on a circle much like a clock, and doing `(10 + 6) % 12 = 4`, much like how "10 o'clock + 6 hours = 4 o'clock."

Indeed, in both the Kantamneni and Nanda papers, the authors demonstrate proof that this circular representation of numbers actually exist in models specifically trained to perform simple / modular arithmetic. Both papers also go on to use ablations, activation patching, and other methods to identify parts of the model that MUST be involved in the arithmetic computation, and essentially rely on modus tollens (if ablate out sine and cosine components, model performance goes to shit, therefore periodic structure must there) to numerically justify that the model must be using this periodic representational structure to perform the addition. However, they still don't quite fully paint a picture of how the model is "reading out" the answer. For example, what happens to the representation (both pre-attention and post-attention) of the `=` token (i.e. the prediction for the answer) when you change the prompt from `a + b =` to `a + d =` where `d` is greater than `a` by a known amount, and specifically how do the MLP embedding and unembedding matrices handle such a change to give a correct answer still? This is insufficiently useful in giving us a deep understanding for how the models are manipulating said periodic structures.

This post aims to re-construct the Modular Arithmetic Paper with special focus on:
- The relationship between $W_E$ (the embedding matrix that maps vocabulary space to residual space) and $W_\text{in}$ (the up-projection matrix of the MLP block)
- What each attention head is doing
- The relationship between $W_E$ and / or $W_\text{in}$ and $W_L = W_U W_\text{out}$, where $W_\text{out}$ is the down-projection matrix of the MLP block, and $W_U$ is the unembedding matrix that maps residual space back to vocabulary

# 2. Model Task

To restate the paper's model task, a simple **1-layer Transformer** is trained to finish the statement: `{a} + {b} =` where `a` and `b` are numbers $\in [1, P]$, where $P$ is some prime, in this case $113$, and the answer is the mathematical answer to $a + b \mod P$.

<img src = "../../images/llm_arithmetic/nanda_intro.png" alt="Nanda Model" width="100%"> 
*Model Architecture and Illustration of Periodic Representation*