---
published: true
title: Complexity Series (3 / 3) - Bag Of Heuristics
date: 2026-03-04 00:00:00 -500
categories: [mechanistic interpretability]
tags: [mechanistic interpretability]
math: true
---

This is the third and final installment of the "Complexity Series," where I endeavor to argue that there are certain classes of problems (simple arithmetic being one of them) that Transformer architectures will **never** be able to solve.

In my previous two posts, I outlined that we have thus far discovered a few algorithms that are being implicitly implemented by LLMs to perform arithmetic (regular or modulo):
- The "Clock" algorithm ([Kantamneni and Tegmark](https://arxiv.org/html/2502.00873v1)), covered in my first post of the series, ["LLM Arithmetic"](/posts/llm-arithmetic)
- Digit-wise and Carry-over Circuits ([Quirke and Barez](https://arxiv.org/pdf/2310.13121#:~:text=Further%2C%20while%20simple%20arithmetic%20tasks,for%20AI%20safety%20and%20alignment.)), covered in the second post of the series, ["Carry-Over Circuits"](/posts/carry-over-circuits)
- Bag of Heuristics ([Nikankin et. al](https://arxiv.org/pdf/2410.21272), [Anthropic](https://transformer-circuits.pub/2025/attribution-graphs/methods.html#graphs-addition))

These represent a good coverage of classes of algorithms that LLMs have learnt.

# Summary of Previous Posts
In ["LLM Arithmetic"](/posts/llm-arithmetic), I did a deep dive to explain how numbers are encoded and features on a "number circle" (as opposed to number line), and uncover how these circular structures are transformed into petal-shaped structures and recombined to form circles that rotate as the operands of a modulo arithmetic expression change. This is the most mechanistically descriptive blog post in the series that illustrates how useful number features can be learnt and how mathematically sound computations based on them can occur. However, this algorithm learnt by the LLM is limited in that the number vocabulary it has doesn't allow its ability to perform modulo arithmetic to generalize to any triple of numbers (`({a} + {b}) % {P}`).

Building on the understanding that LLMs can learn features that can perform constrained but mathematically sound operations (e.g. `({single_digit_number} + {another_single_digit_number}) % 10 = ?`), I then, in ["Carry-Over Circuits"](/posts/carry-over-circuits), talk about the discovery of another algorithm that makes use of such features to perform general arithmetic of 2 numbers the traditional way we humans were taught to do so - adding digits column-by-column, carrying 1 over whenever necessary. The main takeaway of this paper is that there are several key features, such as "Make Sum 9", "Make / Use Carry 1", that are computed by dedicated attention heads, and used as feature bigrams or trigrams in yet other attention heads to figure out what the correct digit should be for the sum, in that position. At the end of that post, I also illustrate that this algorithm is severely limited by how many attention heads across all layers that it has. The length of the numbers that it can perform arithmetic for is linearly proportional to the total number of attention heads it has. Oftentimes, even this is a severely overestimated upperbound because hitting this upperbound would require all attention heads to be used for arithmetic, and an LLM in general has other non-arithmetic fish to fry.

In this post, I will do a high-level sketch of yet another algorithm that LLMs have learnt to perform arithmetic. Similar to ["Carry-Over Circuits"](/posts/carry-over-circuits), it is essentially learning a bunch of features / heuristics (akin to "Use / Make Sum 9" and "Use / Make Carry 1"), and then combining (logical `AND`, or if you wanna get fancy, "constructively interfering") them to arrive at the correct answer. This was introduced in [Nikankin et. al](https://arxiv.org/pdf/2410.21272), but I will mainly be building upon the example offered by [Anthropic](https://transformer-circuits.pub/2025/attribution-graphs/methods.html#graphs-addition), simply because there is a great diagram provided. This **post is the least mechanistic** due to the last 2 posts; for a deeper dive into how features look like and work, I recommend reading the last two posts. The **main insight of this post is how LLMs can learn any finite number of arbitrary heuristics that need not be principled (but are still mathematically sound)**, in order to perform arithmetic. In practice, this is indeed how they implement arithmetic, but since these are arbitrary heuristics, there can only be **a finite number of them, so this algorithm is unable to perform arithmetic on any and all pairs of numbers**.

# Features and the Arithmetic Circuit

<img src = "../../images/bag_of_heuristics/anthropic_circuit.png" alt="36_plus_59" width="100%"> 
*[Anthropic](https://transformer-circuits.pub/2025/attribution-graphs/methods.html#graphs-addition): Computed Features and How They're Used to Compute 36 + 59*

These "features" that they've learnt are essentially Crosscoder (a variant of Sparse Auto-Encoders) features, that they've then labeled with interpretations based on what operand(s) they activate highly on. The key things to note here are that:
- There are input features that are purely a function of the input operand (e.g. the operand `36` activates the "inputs near `30`" feature, as well as the "is `36`" feature, as well as the `_6` or "ends-with-`6`" feature).
- There are features that add the "addition" meaning to the second input operand (`59`), i.e. these features light up when the model recognizes that it needs to "add `~57`" or "add to something a number that ends with `9`".
- There are many heuristic features related to the computed result, e.g. there's a "sum is `~92`" feature that seems to be the result of a circuit that recognizes that two numbers around `40` and `50` are being added, there's a modulo arithmetic circuit that recognizes that `_6` (feature) + `_9` (feature) should give you `_5` (feature), and these heuristics constructively interfere (yet another small circuit) to give `95`.

Notice that these features are entirely arbitrary. Examples:
- The `~30` feature could have well been a `35 ~ 40` feature, there's no mathematical reason that a `~30` feature is necessary or better than a `35 ~ 40` feature, other than the fact that the arabic number system involves *writing* (this is linguistic, not mathematical) numbers in base-10, and is hence perhaps a more convenient for LLMs to learn.
- The `~40 + ~50` feature could similarly have been a `~35 + ~60` feature.

Notice also that these features are highly specific, and doesn't necessarily cover all numbers. Examples
- The `_6 + _9` and `sum = _5` features are modulo-10 features, but the `sum = _95` feature is a modulo-100 feature. What about modulo-1000?
- The `sum ~92` feature is also highly specific to `92`, but what about literally any other number?

And finally, note also that this **circuit is unidirectional**, in that each feature exists at a certain place (a certain layer / Crosscoder), is used ONCE in the forward pass, and is never reused. This means that like before, even if a very exhaustive set of features has been learnt, the **length of the numbers** on which the model can perform arithmetic **still scales linearly with the total number of features or "circuit gates"** (my own term for places where features are combined and synthesized; usually attention heads).

# Downfalls of Transformers

Though these 3 algorithms that we've reviewed are not *provably* **entirely descriptive** of all the ways in which LLMs could perform arithmetic, I argue that the similarity / coherence between the algorithms discovered suggest that all arithmetic-performing algorithms learnt by LLMs look somewhat like that, in that they have many similar ingredients (heuristics), representations, and composition of these heuristics into unidirectional circuits.

The original goal if this Complexity Series was to argue that there exist class(es) of problems that Transformers can never one-shot (no "Thinking"). A useful