---
published: true
title: Patterning
date: 2026-04-16 00:00:00 -500
categories: [mechanistic interpretability]
tags: [mechanistic interpretability]
math: true
---

I recently came across [this paper](https://arxiv.org/pdf/2601.13548) by George Wang and Daniel Murfet at Timeaus that introduces the idea of "patterning" as the "dual problem" to mechanistic interpretability - i.e. if mech interp is about reverse engineering model internals, then patterning is the complementary question of how to influence training in order to produce a certain kind of model internals. I think it's really interesting, but it strikes me that their paper could be more aggressively pared down to the essentials in order to really gain an intuition of how patterning works / is possible, so this is what this post will be about.

# Introduction

First, they introduce a posterior of trained model weights $w$ given some dataset $D_n$ of size $n$:

$$
\begin{align*}
p(w \mid D_n)
\end{align*}
$$

These weights $w$ are post-training weights, so $p(w \mid D_n)$ is implicitly also dependent on the model architecture and optimization parameters / algorithms. They further note that **the shape of $p(w \mid D_n)$ reveals internal structural information of the model.** For example, if there are redundancies in weights / circuitry, then perturbation of some weights will not matter so much to model performance, and hence there are groups of degenerate ("equivalent") weight configurations. Another example of degeneracy is when a perturbation in a weight can be compensated for via coordinated perturbations in other weights.

Then, they claim that "structural information can be extracted by computing expectation values" of observables $\phi_i$ which one can design:

$$
\begin{align*}
\mu_i^n = \int \phi_i (w) p(w \mid D_n) dw
\end{align*}
$$

$\phi_i$ could be a linear probe, an SAE, the "local learning coefficient $\lambda$" (introduced [later](#learning-coefficient)), "susceptibilities $\chi$" (and another!), whatever - functions of the free parameter space of $w$. One can then collate these observable means into a **vector of "structural coordinates"**:

$$
\begin{align*}
\mu^n = \begin{bmatrix}
\mu_1^n \\
\mu_2^n \\
\vdots \\
\end{bmatrix}
\end{align*}
$$

# Affordance

This paper is about studying the **"affordances"** (&#x1F644;) of these structural coordinates. What does that mean? Your guess is as good as mine.

First, the authors try to generalize $\mu^n$ to $\mu^\infty$, which they compute using the "annealed posterior":

$$
\begin{align*}
p(w \mid D_\infty) \propto \frac{\varphi(w)}{\exp (n \beta L(w))}
\end{align*}
$$

where:

$$
\begin{align*}
L(w) = \mathbb{E}_q [\ell(w)]
\end{align*}
$$

is the "population loss", where:
- $q$ is the data distribution (introducing $q$ makes sense since we're trying to generalize $D_n$ to $D_\infty$).
- I assume $\beta$ is some hyperparameter.
- $\varphi$ is hthe **prior density** of $w$.

## Matrix of Susceptibilities

Previously, we introduced $q$ as the data distribution, but now they say "let **$h$ denote a vector of hyperparameters governing the data distribution** (for instance, **mixture weights** [(???)] over a baseline distribution and a **collection of probe distributions** [(???)])". Then:

$$
\begin{align*}
d \mu^\infty = \chi \cdot dh
\end{align*}
$$

where $d$ refers to the total differential (i.e. first order approximation of $\delta$), and so **$\chi$ is the Jacobian of $\mu^\infty$ w.r.t $h$.** This is precisely the matrix of susceptibilities:

$$
\begin{align*}
\text{ } \\
\chi = 
\begin{bmatrix} \frac{\delta \mu_1}{\delta h_1} & \frac{\delta \mu_1}{\delta h_2} & \cdots & \frac{\delta \mu_1}{\delta h_k} \\
\frac{\delta \mu_2}{\delta h_1} & \frac{\delta \mu_2}{\delta h_2} & \cdots & \frac{\delta \mu_2}{\delta h_k} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\delta \mu_i}{\delta h_1} & \frac{\delta \mu_i}{\delta h_2} & \cdots & \frac{\delta \mu_i}{\delta h_k}
\end{bmatrix}
\end{align*}
$$

They continue: "the $ik$-entry $\chi_{ik}$ measures how the **expectation of observable $\phi_i$** [(a.k.a. $\mu_i$, yes)] responds to an infinitesimal shift of the data distribution toward probe $k$" (which I've denoted $h_k$). Note that we still don't know what these probes are like.

## The Fundamental Equation of Patterning

So then, the natural question is, given that we would like to observe a target change in the observables $\mu^\infty$, what's the minimum-norm intervention in the data distribution? It is just a rearrangement of the above equation:

$$
\begin{align*}
(d h)_\text{opt} = \chi^\dagger \cdot (d \mu^\infty)_\text{target}
\end{align*}
$$

where $\chi^\dagger$ is the Moore-Penrose pseudoinverse. This, they call the **Fundamental Equation of Patterning**.

## Producing Perceptible Changes

Summarizing from the last section, assuming that your observables $\phi$ and data hyperparameters $h$ work / are well designed (i.e. susceptibilities must be sensitive to internal structure), then $\chi$ is useful in telling you how to change $h$ in order to produce desired changes in $\phi$ (or $\mu^\infty$).

For example, [Baker et. al](https://arxiv.org/pdf/2504.18274) found that "PC2 of the susceptibility matrix couples induction patterns in the data (roughly speaking, the right singular vector $v_2$) to the induction circuit in the weights (the left singular vector $u_2$)," and so up or down-weighting tokens with high $v_2$ magnitudes can accelerate or delay induction circuit formation. The unspoken part is that the design of $\phi$ and $h$ are crucial here, so we'll just have to see how Wang and Murfet demonstrate this later on.

Another example is that if you know that there are multiple plausible internal structures that can solve the same problem (roughly, achieve the same small loss), you can use Patterning to select one of them in particular to be learnt. Wang and Murfet demonstrate this using a model trained on a bracket classification task.

# Singular Learning Theory (SLT)

So far, we've mentioned that a key observable is the Local Learning Coefficient (LLC) $\lambda$, but we haven't defined that. This necessitates an introduction to SLT, for it:
- Defines the LLC
- Provides mathematical framework ("annealed posteriors", "free energy") for susceptibilities.

## Define "Singular" Statistical Model

A statistical model is simply something that models probability distributions

$$
\begin{align*}
p(y \mid x, w)
\end{align*}
$$

by maximizing this (or some proxy, e.g. $\mathbb{E} p(y)$) w.r.t to $w$. A statistical model becomes **singular** when roughly, there are multiple solutions, or multiple ways to produce the same distribution. Neural networks are quite obviously singular.

> **Note**: Another commonly cited explanation is when the Fisher Information (a Hessian) is degenerate (non-invertible; i.e. there exists some direction along which the likelihood has no curvature) at some $w$. But I find this to be insufficiently strict as a requirement, for zero curvature doesn't mean that gradient of likelihood is 0, i.e. doesn't mean that there are multiple parameter solutions to get the same probability distribution. However, the opposite is a quite a strong prerequisite: a model is definitely *regular* if the Fisher information matrix is positive definite.

## Learning Coefficient

Consider a statistical model:

$$
\begin{align*}
p(y \mid x, w)
\end{align*}
$$

with parameters $w \in W \subseteq \mathbb{R}^d$, which prior density $\varphi(w)$. Suppose also that the dataset $D_n = \\{ (x_i, y_i) \\}_{i=1}^n$ is drawn i.i.d. from a true distribution $q(x, y)$.

And define the empirical loss:

$$
\begin{align*}
L_n (w) = - \frac{1}{n} \sum_{I=1}^n \log p(y_i \mid x_i, w)$
\end{align*}
$$

(basically just cross entropy loss), and the population loss:

$$
\begin{align*}
L(w) = - \mathbb{E}_{q(x, y)} \left[ \log p(y \mid x, w) \right]
\end{align*}
$$

Define the **volume**:

$$
\begin{align*}
\text{vol}(\epsilon) = \int_{L(w) < L(w^\star) + \epsilon} \varphi (w)
\end{align*}
$$

where $w^\star$ is the global minimizer of $L$. Intuitively speaking, this is the probability mass (according to $\phi$) over all $w$ that produce "small enough" $L(w)$. The learning coefficient is then a measure of how fast this volume shrinks:

$$
\lambda = - \lim_{\epsilon \rightarrow 0^+} \log_2 \left[ \frac{\text{vol}\left( \frac{\epsilon}{2}\right)}{ \text{vol} (\epsilon)} \right]
$$

> **Note**: There is probably heavy theory behind the construction using $\log_2$ and the claim that in regular models (where $w^\star$ is unique and the Hessian is non-degenerate), $\lambda = d/2$ where $d$ is the parameter dimension, but this is not relevant for our discussion.

A high $\lambda$ means that from $\epsilon \rightarrow \epsilon / 2$, the volume decreases by relatively much, while a low $\lambda$ means that the volume doesn't decrease by that much. I think of this by visualizing a highly pointy function versus a relatively smooth / bowl-shaped one, and noticing that the relative sizes of the 2 cylinder volumes are much more comparable for the low $\lambda$ function:

<img src = "../../images/patterning/high_lc_low_lc.png" alt="Pointy versus Flat" width="100%"> 
*Pointier (high LC) versus Bowl / Flatter (low LC). [0, 0] is the minimizer; vertical axis is $\varphi$.*

So basically, $\lambda$ is conceptually similar to the Fisher Information $\mathcal{I}(w)$. With the learning coefficient defined, the LLC is just a local version of that. And since there are assumed to be multiple local minima, you also have to specify which local minimum $w^\star$ it is, so it is denoted as $\lambda (w^\star)$.


## Posterior $p(w \mid D_n)$

The next concept introduced is the Bayesian posterior probability of a small neighborhood $\mathcal{U}$ around a local minimum $w^\star$:

$$
\begin{align*}
p_n(\mathcal{U}) = \frac{Z_n(\mathcal{U})}{Z_n(\mathcal{W})}
\end{align*}
$$

where

$$
\begin{align*}
Z_n(\mathcal{U}) = \int_{\mathcal{U}} \exp \left( -n L_n(w)\right) \varphi (w) dw
\end{align*}
$$

### Derivation

Why is this correct? Starting from Bayes' theorem, we have:

$$
\begin{align*}
p(w \mid D_n) \propto p(D_n \mid w) \varphi (w)
\end{align*}
$$

Playing with $\exp$, $\log$, and substituting in the definition of empirical loss $L_n(w)$ defined above, the **likelihood of i.i.d. data** is:

$$
\begin{align*}
p(D_n \mid w) & = \prod_{i = 1}^n p(y_i \mid x_i, w) \\ 
& = \exp \left( \sum_{i=1}^n \log p( y_i \mid x_i, w) \right) \\
& = \exp \left( -n L_n(w) \right)
\end{align*}
$$

So we have **the posterior equals:**

$$
\begin{align*}
p_n(\mathcal{U}) = \frac{\int_{\mathcal{U}} \exp \left( -n L_n(w) \right) \varphi(w)}{Z_n(\mathcal{W})}
\end{align*}
$$

where the denominator is just a normalizing constant.

## Free Energy

They next introduce the "Local Free Energy $F_n(\mathcal{U})$":

$$
\begin{align*}
F_n(\mathcal{U}) = - \log Z_n(\mathcal{U})
\end{align*}
$$

This is a measure of how much posterior mass is in the neighborhood $\mathcal{U}$:
- Large posterior mass in $\mathcal{U} \Longleftrightarrow$ low $F_n(\mathcal{U})$
- Small posterior mass in $\mathcal{U} \Longleftrightarrow$ large $F_n(\mathcal{U})$

They then cite the asymptotic ($n \rightarrow \infty$) expansion from Watanabe, 2009:

$$
\begin{align*}
F_n(\mathcal{U}) = n L_n(w^\star) + \lambda (w^\star) \log n - (m(w^\star) - 1)\log \log n + O_p(1)
\end{align*}
$$

where "$m(w^\star)$ is the ***multiplicity***, a secondary geometric invariant."

### Derivation

Lmao I dunno mate Claude gave me crap. I'm willing to just say this is a result from Watanabe and move on.

## Bayesian Inference Prefers Lower Free Energy

The paper compares two local minima $w_A^\star$ and $w_B^\star$ with neighborhoods $\mathcal{U}$ and $\mathcal{V}$ via this difference of free energy:

$$
\begin{align*}
\log \frac{p_n (\mathcal{U})}{p_n (\mathcal{V})} & = F_n(\mathcal{V}) - F_n(\mathcal{U}) \\
& = \Delta L_n \cdot n + \Delta \lambda \cdot \log n + O_p (\log \log n)
\end{align*}
$$

And they note that when the 2 local minima achieve the same loss, then the difference in free energy (or ratio of posteriors) depends entirely on $\Delta \lambda$. **They then make the claim that "among equal-loss solutions, the posterior favors those with lower LLC."** I.E. they mean that the **posterior for the lower LLC optimum is actually higher than that of the higher LLC optimum**. 

How is it possible that a local optimum can be favored over another despite producing equal loss? My understanding is that the heavy lifting is done by choosing the prior distribution $\varphi (w)$ ($\lambda(w)$ does depends on this afterall), and by relating the posterior to $\lambda(w)$. It remains to be seen how this is chosen well.

## Estimating the LLC

Since the LLC is the ratio of volumes, and these volumes are integrals of the prior $\varphi$ over parameter space and have no analytical solution, they use prior work to estimate the LLC:

$$
\begin{align*}
\hat{\lambda} (w^\star) & = n \beta \left[ \mathbb{E}^\beta_{w \mid w^\star, \gamma} \left[ L_n(w) \right] - L_n(w^\star) \right]
\end{align*}
$$

where the "expectation is w.r.t a localized tempered posterior" ([Watanabe, 2013](https://arxiv.org/pdf/1208.6338)):

$$
\begin{align*}
p(w; w^\star, \beta, \gamma) & \propto \exp \left\{ -n \beta L_n(w) - \frac{\gamma}{2} \|w - w^\star \|^2  \right\}
\end{align*}
$$

where these are the hyperparameters:
- $\beta$ is the inverse temperature
- $\gamma$ is the localization strength

