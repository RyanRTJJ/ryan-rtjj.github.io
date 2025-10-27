---
published: false
title: Arithmetic Circuits
date: 2025-10-02 00:00:00 -500
categories: [statistics]
tags: [statistics]
math: true
---

There are thus far a few algorithms that researchers have discovered are being implicitly implemented by LLMs to be able to perform simple arithmetic. They include:
- The "Clock" algorithm ([Kantamneni and Tegmark](https://arxiv.org/html/2502.00873v1))
- Bag of Heuristics ([Nikankin et. al](https://arxiv.org/pdf/2410.21272), [Anthropic](https://transformer-circuits.pub/2025/attribution-graphs/methods.html#graphs-addition))
- Digit-wise and Carry-over Circuits ([Quirke and Barez](https://arxiv.org/pdf/2310.13121#:~:text=Further%2C%20while%20simple%20arithmetic%20tasks,for%20AI%20safety%20and%20alignment.))

There are other papers that have found more results, but for the most part, I find it sufficiently helpful to explore these 3 main mechanisms, as they are sufficiently different and represent good sample algorithms from the spectrum of vibe arithmetic. In this post, I wish to look at LLM-learnt solutions for the toy problem of arithmetic from the lenses of feature geometry and computational complexity. In particular:
- How do seemingly complex "rotational" algorithms like the "Clock" algorithm fit into the view that ReLU models merely learn a manifold (a superposition of half-space pairs) that is piece-wise linear (more details in ["Superposition - An Actual Image of Latent Spaces"](/posts/viewing-latent-spaces/)), where each "piece" of latent space corresponds to an activation of a subset of features?
- What are the limitations of each algorithm? Have they truly learnt a sufficiently expressive algorithm for arithmetic? Can it scale up to any number of digits?
- If they can't, why not, and what can we do to enable such algorithms?

# 1. Algorithm Summaries

## 1.1. The Clock Algorithm

Performance and Descriptions
Authors analyze GPT-J 6B (GeLU), Pythia-6.9B (GeLU), and Llama3.1-8B (SwiGLU).