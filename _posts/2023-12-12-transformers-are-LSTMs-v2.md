---
published: true
title: Transformers are LSTMs v2
date: 2023-12-12 00:00:00 -500
categories: [research,ML,time-series]
tags: [transformers,LSTMs]
math: true
---

So I recently got into a bit of an argument with 2 friends while driving in the car to get dinner. It went a little bit like this:

Me: I think LSTMs are definitely the precursor to transformers, minus all the parallelization.

Henry: What?!

Me: Attention definitely came from transformers.

Raghav: I think if you told anyone who hasn't thought insanely deeply and seen the weird connection, that LSTMs are transformers, they'd think you're an idiot.

Henry: You can definitely do LSTMs with attention.

The rest was irrelevant banter, but TLDR: I still believe LSTMs are **an** early form of attention. I'll just drop this graphic that I made here and let you draw the connections. I also wrote a [medium article on an LSTM deep-dive](https://medium.com/analytics-vidhya/lstms-explained-a-complete-technically-accurate-conceptual-guide-with-keras-2a650327e8f2) years ago so I'm pretty convinced that I know a thing or two about LSTMs.

<img src = "../../images/LSTMs_Attention.png" alt="LSTMs and Attention" width="100%"><br/>

I'm fully aware of all the different types of attention exist (here's a [really nice post by one of my favorite technical writers, Lilian Weng, that summarizes them](https://lilianweng.github.io/posts/2018-06-24-attention/)) and LSTMs don't fit squarely into any of those boxes, **BUT**, the purpose and general mechanism of all these gates still very much are those of attention mechanisms.