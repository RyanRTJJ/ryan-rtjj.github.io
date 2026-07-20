---
published: true
title: Unified RL
date: 2024-06-30 00:00:00 -500
categories: [reinforcement learning]
tags: [reinforcement learning]
math: true
---

All matrix math here will follow pytorch conventions. I.E. applying a matrix $M$ to a vector $x$ will look like $xM$ rather than $Mx$. Further applying a matrix $W$ will hence look like $xMW$ rather than $WMx$.

# 1. Intro

## 1.1. Definitions

### Markov State ($S$)

An information state ($S_t$) is considered Markov iff it contains all useful information from the history:

$$
P\left(S_{t + 1} \mid S_t \right) = P \left(S_{t + 1} \mid S_1, \cdots, S_{t}\right)
$$

Markov states are desirable because agents are generally expected to be long-lived and take very frequent actions. The analogy here is with LLMs. LLM states are not exactly markov, you have to keep the entire KV-cache, and that's your "history". The could in theory be markov, but the residual stream would just have to be much larger to contain the expressivity of the KV cache.

### Markov Decision Process (MDP)

A process where the agent observes the full environment state.

### Partially Observable MDP (POMDP)

A process where the agent indirectly observes the environment state (i.e. there are hidden variables controlling environment behavior). E.g. stock trader only sees stock prices, but doesn't know driving forces.

### Policy ($\pi$)

A function from state $\rightarrow$ actions. There are:
- Determistic Policies
- Stochastic Policies

### Value Function ($V$)

"Expected total future reward".

$$
v_\pi(s) = \mathbb{E}_\pi \left[ R_t + \gamma R_{t + 1} + \gamma^2 R_{t + 2} + \cdots \mid S_t = s \right]
$$

### Model

Something that predicts what the environment will do next. Typically this includes:

A Transition Model / table ($\mathcal{P}$)

$$
\mathcal{P}_{s \rightarrow s'}^a = P \left[ S' = s' \mid S = s, A = s \right]
$$

A Reward Model (this below is rather straightforward)

$$
\mathcal{R}_s^a = \mathbb{E} \left[ R \mid S = s, A = a \right]
$$

**A model is not required. Many techniques are model-free.**

## 1.2. Types of RL Agents

The "type" basically describes which of the key components `{policy, value_function, model}` the agent uses.

- Value Based: No explicit policy (e.g. "greedy"); uses Value Function
- Policy Based: No explicit value function, just a policy table (e.g. "if at this square in a maze, go right").
- Actor Critic: Has both value function and policy.
- Model Free: orthogonally from Value Function / Policy, we choose to ignore trying to build out a model of how the environment works.

<style>
  .rl-venn { margin: 1.5rem 0; text-align: center; }
  .rl-venn svg { width: 100%; height: auto; max-width: 520px; }
  .rl-venn .rl-circle { fill-opacity: 0.41; stroke-opacity: 0.71; stroke-width: 2.5; }
  .rl-venn .rl-name { font-weight: 700; font-size: 16px; }
  .rl-venn .rl-region { font-weight: 600; font-size: 14.5px; fill: var(--text-color, #1f2328); }
  .rl-venn figcaption { font-size: 0.85rem; opacity: 0.8; margin-top: 0.6rem; max-width: 540px; margin-inline: auto; }
</style>

<figure class="rl-venn" markdown="0">
  <svg viewBox="0 0 560 520" role="img" aria-label="Venn diagram of RL agent types: Policy, Value Function, and Model.">
    <!-- The three circles -->
    <circle class="rl-circle" cx="200" cy="200" r="135" fill="#d75344" stroke="#d75344"/>
    <circle class="rl-circle" cx="360" cy="200" r="135" fill="#5978e3" stroke="#5978e3"/>
    <circle class="rl-circle" cx="280" cy="345" r="135" fill="#dddddd" stroke="#888888" style="stroke-opacity: 1;"/>

    <!-- Circle names -->
    <text class="rl-name" x="138" y="135" text-anchor="middle" fill="#d75344">
      <tspan x="138" dy="0">Value</tspan>
      <tspan x="138" dy="18">Function</tspan>
    </text>
    <text class="rl-name" x="422" y="140" text-anchor="middle" fill="#5978e3">Policy</text>
    <text class="rl-name" x="280" y="420" text-anchor="middle" fill="#888888">Model</text>

    <!-- Region labels -->
    <text class="rl-region" x="200" y="200" text-anchor="middle">Value-Based</text>
    <text class="rl-region" x="360" y="200" text-anchor="middle">Policy-Based</text>
    <text class="rl-region" x="280" y="158" text-anchor="middle">Actor-Critic</text>
    <text class="rl-region" x="280" y="332" text-anchor="middle">Model-Based</text>
  </svg>
</figure>

## 1.3 Types of RL Problems

RL:
- Multi-rollout (agent gets to learn from unlimited rollouts) in same environment
- Multi-rollout in changing environment (structure of environment is same)
- Single-rollout

# 2. Introducing MDPs

Most RL problems can be converted to MDPs. Let's take a look at an MDP where we happen to have a fully observable environment.

> As it turns out, having a fully observable environment is not make-or-break, as partially observable RL problems can be converted to POMDPs, which can be converted to MDPs. 

## 2.1. Model / Transition table ($\mathcal{P}$)

Since we have a fully observable environment, then we can build a transition matrix that models the probabilities we believe each state ($s$) will lead to a successive state ($s'$).

$$
\begin{align*}
\because S' &= S \mathcal{P} \\ 
\therefore \mathcal{P} &= \begin{bmatrix}
\mathcal{P}_{s_1 \rightarrow s_1} & \cdots & \mathcal{P}_{s_1 \rightarrow s_n} \\
\vdots & \ddots & \vdots \\
\mathcal{P}_{s_n \rightarrow s_1} & \cdots & \mathcal{P}_{s_n \rightarrow s_n}
\end{bmatrix}
\end{align*}
$$

> Each row of $\mathcal{P}$ sums to 1.

In an MDP where there are various actions, each action $a \in \mathcal{A}$ will yield one of these 2D matrices, which we call $\mathcal{P}^a$.

## 2.2. Markov Reward Process

A Markov Reward Process is simply an "Markov Chain / rollout with rewards", i.e. a rollout of an MDP where there are no chosen actions. Everything is automatic. This can be written / parameterized as $\langle \mathcal{S}, \mathcal{P}, \mathcal{R}, \gamma \rangle$, where:
- $\mathcal{S}$ is the set of states
- $\mathcal{P}$ is the transition table
- $$\mathcal{R}_s$$ is a reward function (e.g. $$R_t = \mathbb{E} [R_{t + 1} \mid S_t = s]$$). **Note:**
  - $R_{t + 1}$ is the reward for taking actions at time $t$
  - Conversely, $R_t$ is the reward for simply having arrived at the current state at time $t$.
- $\gamma \in [0, 1]$ is the discount factor. We do this because:
  - We have uncertainty about our forecasting ability
  - To keep the math bounded. You can't have infinite returns.

### Return

The "return" $G_t$ is the total discounted reward from time-step $t$, computed as you are ABOUT TO make the next step 

$$
\begin{align*}
G_t &= R_{t + 1} + \gamma R_{t + 2} + \gamma^2 R_{t + 3} + \cdots \\
&= \sum_{k = 0}^\infty \gamma^k R_{k + t + 1}
\end{align*}
$$

### Value Function

$$
v(s) = \mathbb{E}[G_t \mid S_t = s]
$$

This is a statement of fact. Its a higher-level abstraction that does not concern itself with how $G_t$ was gotten (i.e. what policy). When we concern ourselves with the "decision"-making part of "MDP", we will decide on some policy to follow, but the value function is still conceptually this.

Because $v(s)$ is an expectation, this implies that you have to sample rollouts so that you have multiple rollouts and returns to take the expectation of.

### Bellman Equation for $v$ in MRP

The Bellman Equation is just a recursive formulation of $v$:

$$
\begin{align*}
v(s) &= \mathbb{E}[G_t \mid S_t = s] \\
&= \mathbb{E}[R_{t + 1} + \gamma (R_{t + 2} + \gamma R_{t + 3} + \cdots) \mid S_t = s] \\
&= \mathbb{E}[R_{t + 1} + v(S_{t + 1}) \mid S_t = s]
\end{align*}
$$

**Factoring in $\mathcal{P}$**:

$$
\begin{align*}
v(s) = \mathcal{R}_s + \gamma \mathcal{P}_{[s,:]} \cdot
\begin{bmatrix}
v(s_1) \\
\vdots \\
v(s_n)
\end{bmatrix}
\end{align*}
$$

Visually, we have:

<figure class="rl-tree" markdown="0">
  <svg viewBox="0 0 730 220" role="img" aria-label="State s fans out to successor states s_1 through s_n.">
    <defs>
      <marker id="rl-arrowhead-mrp" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" fill="#b0b0b0"/>
      </marker>
    </defs>

    <!-- Edges: state s -> successor states -->
    <path class="rl-arrow" d="M279,110 C366,110 366,65 452,65" marker-end="url(#rl-arrowhead-mrp)"/>
    <path class="rl-arrow" d="M279,110 C366,110 366,155 452,155" marker-end="url(#rl-arrowhead-mrp)"/>

    <!-- Edge labels -->
    <text class="rl-edge" x="330" y="80" text-anchor="middle"><tspan class="rl-var">P</tspan><tspan class="rl-sub" baseline-shift="sub"><tspan class="rl-var">s</tspan>→<tspan class="rl-var">s</tspan><tspan class="rl-sub" baseline-shift="sub">1</tspan></tspan></text>
    <text class="rl-edge" x="330" y="148" text-anchor="middle"><tspan class="rl-var">P</tspan><tspan class="rl-sub" baseline-shift="sub"><tspan class="rl-var">s</tspan>→<tspan class="rl-var">s</tspan><tspan class="rl-sub" baseline-shift="sub">n</tspan></tspan></text>

    <!-- Root state node: s (cx=268, cx_succ=463, midpoint=365=730/2) -->
    <circle class="rl-node" cx="268" cy="110" r="11" style="fill: #d65344; stroke: #d65344;"/>
    <text class="rl-label" x="268" y="141" text-anchor="middle"><tspan class="rl-var">s</tspan></text>

    <!-- Successor state nodes -->
    <circle class="rl-node" cx="463" cy="65" r="11"/>
    <text class="rl-label" x="463" y="41" text-anchor="middle"><tspan class="rl-var">s</tspan><tspan class="rl-sub" baseline-shift="sub">1</tspan></text>
    <text class="rl-dots" x="463" y="113" text-anchor="middle">⋮</text>
    <circle class="rl-node" cx="463" cy="155" r="11"/>
    <text class="rl-label" x="463" y="183" text-anchor="middle"><tspan class="rl-var">s</tspan><tspan class="rl-sub" baseline-shift="sub">n</tspan></text>
  </svg>
</figure>

### Bellman in Matrix Form

Let:

$$
\begin{align*}
v = 
\begin{bmatrix}
v(s_1) \\
\vdots \\
v(s_n)
\end{bmatrix}, \text{ and } \mathcal{R} = \begin{bmatrix}
\mathcal{R}_{s_1} \\
\vdots \\
\mathcal{R}_{s_n}
\end{bmatrix}
\end{align*}
$$

Then

$$
\begin{align*}
v & = \mathcal{R} + \gamma \mathcal{P} v
\end{align*}
$$

This means there exists a **closed form solution** for $v$, which is simply:

$$
\begin{align*}
v & = (I - \gamma \mathcal{P})^{-1} \mathcal{R}
\end{align*}
$$

The only thing is that the inverse has $O(n^3)$ computational complexity, so it's not typically practical for large MRPs. So in practice, there exist a few different ways to solve this that are all variants of Dynamic Programming:
- Simple Dynamic Programming
- Monte-Carlo Evaluation
- Temporal-Difference Learning

## 2.3. Markov Decision Process

An MDP is the same as an MRP, just that we now have agency (i.e. a set of actions to choose from). It is represented as $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$, where:
- $\mathcal{A}$ is the set of actions.
- $$\mathcal{R}_{s}^a$$ is the reward for being at state $$s$$ and deciding to take action $$a$$. Sometimes interchangable with $$R_{t + 1}$$.

### Policy

A policy $\pi$ is just a distribution over possible actions to take, given a state:

$$
\begin{align*}
\pi(a \mid s) & = P\left[ A_t = a \mid S_t = s \right]
\end{align*}
$$

### Value Function

For an MRP, we had:

$$
v(s) = \mathbb{E}[G_t \mid S_t = s]
$$

This does not have any sense of action-taking baked into it. The process just unfolds by itself without any choice of actions. Now, we have a policy that tells us how to take action. $v$ becomes a distribution (because $\pi$ is a distribution over actions), and we can then take the expectation of it.

### Bellman Equation for $v$ in MDP

For an MRP, $v$ has the Bellman Equation:

$$
\begin{align*}
v & = \mathcal{R} + \gamma \mathcal{P} v
\end{align*}
$$

If we were to add in the action-taking steps, we arrive at this decision tree:

<figure class="rl-tree" markdown="0">
  <svg viewBox="0 0 730 340" role="img" aria-label="Decision tree: state s leads to three action nodes; each action fans out into successor states.">
    <defs>
      <marker id="rl-arrowhead-mdp" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" fill="#b0b0b0"/>
      </marker>
    </defs>

    <!-- Edges: state s -> action nodes -->
    <path class="rl-arrow" d="M63,215 C148,215 148,110 234,110" marker-end="url(#rl-arrowhead-mdp)"/>
    <path class="rl-arrow" d="M63,215 C148,215 148,215 234,215" marker-end="url(#rl-arrowhead-mdp)"/>
    <path class="rl-arrow" d="M63,215 C148,215 148,320 234,320" marker-end="url(#rl-arrowhead-mdp)"/>

    <!-- Edges: action node 1 -> successor states -->
    <path class="rl-arrow" d="M258,110 C344,110 344,70 429,70" marker-end="url(#rl-arrowhead-mdp)"/>
    <path class="rl-arrow" d="M258,110 C344,110 344,150 429,150" marker-end="url(#rl-arrowhead-mdp)"/>

    <!-- Edges: action nodes 2 & 3 -> "..." (collapsed successor states) -->
    <path class="rl-arrow" d="M258,215 C341,215 341,215 424,215" marker-end="url(#rl-arrowhead-mdp)"/>
    <path class="rl-arrow" d="M258,320 C341,320 341,320 424,320" marker-end="url(#rl-arrowhead-mdp)"/>

    <!-- Edges: successor state s_1 -> next action nodes -->
    <path class="rl-arrow" d="M453,70 C538,70 538,30 624,30" marker-end="url(#rl-arrowhead-mdp)"/>
    <path class="rl-arrow" d="M453,70 C538,70 538,70 624,70" marker-end="url(#rl-arrowhead-mdp)"/>
    <path class="rl-arrow" d="M453,70 C538,70 538,110 624,110" marker-end="url(#rl-arrowhead-mdp)"/>

    <!-- Edges: successor state s_n -> "..." (collapsed next actions) -->
    <path class="rl-arrow" d="M453,150 C536,150 536,150 619,150" marker-end="url(#rl-arrowhead-mdp)"/>

    <!-- Edge labels: R_s^{a_i} -->
    <text class="rl-edge" x="140" y="140" text-anchor="middle"><tspan class="rl-var">R</tspan><tspan class="rl-sub rl-var" baseline-shift="sub">s</tspan><tspan class="rl-sup" baseline-shift="super" dx="-4"><tspan class="rl-var">a</tspan><tspan class="rl-sub" baseline-shift="sub">1</tspan></tspan></text>
    <text class="rl-edge" x="150" y="200" text-anchor="middle"><tspan class="rl-var">R</tspan><tspan class="rl-sub rl-var" baseline-shift="sub">s</tspan><tspan class="rl-sup" baseline-shift="super" dx="-4"><tspan class="rl-var">a</tspan><tspan class="rl-sub" baseline-shift="sub">2</tspan></tspan></text>
    <text class="rl-edge" x="160" y="260" text-anchor="middle"><tspan class="rl-var">R</tspan><tspan class="rl-sub rl-var" baseline-shift="sub">s</tspan><tspan class="rl-sup" baseline-shift="super" dx="-4"><tspan class="rl-var">a</tspan><tspan class="rl-sub" baseline-shift="sub">3</tspan></tspan></text>

    <!-- Edge labels from s_1 to next action nodes -->
    <text class="rl-edge" x="590" y="16" text-anchor="middle"><tspan class="rl-var">R</tspan><tspan class="rl-sub" baseline-shift="sub"><tspan class="rl-var">s</tspan><tspan class="rl-sub" baseline-shift="sub">1</tspan></tspan><tspan class="rl-sup" baseline-shift="super" dx="-9"><tspan class="rl-var">a</tspan><tspan class="rl-sub" baseline-shift="sub">1</tspan></tspan></text>
    <text class="rl-edge" x="590" y="54" text-anchor="middle"><tspan class="rl-var">R</tspan><tspan class="rl-sub" baseline-shift="sub"><tspan class="rl-var">s</tspan><tspan class="rl-sub" baseline-shift="sub">1</tspan></tspan><tspan class="rl-sup" baseline-shift="super" dx="-9"><tspan class="rl-var">a</tspan><tspan class="rl-sub" baseline-shift="sub">2</tspan></tspan></text>
    <text class="rl-edge" x="590" y="92" text-anchor="middle"><tspan class="rl-var">R</tspan><tspan class="rl-sub" baseline-shift="sub"><tspan class="rl-var">s</tspan><tspan class="rl-sub" baseline-shift="sub">1</tspan></tspan><tspan class="rl-sup" baseline-shift="super" dx="-9"><tspan class="rl-var">a</tspan><tspan class="rl-sub" baseline-shift="sub">3</tspan></tspan></text>

    <!-- Root state node: s -->
    <circle class="rl-node" cx="50" cy="215" r="11" style="fill: #d65344; stroke: #d65344;"/>
    <text class="rl-label" x="50" y="246" text-anchor="middle"><tspan class="rl-var">s</tspan></text>

    <!-- Action nodes (all light blue via CSS default) -->
    <circle class="rl-action" cx="245" cy="110" r="11"/>
    <text class="rl-label" x="245" y="88" text-anchor="middle"><tspan class="rl-var">a</tspan><tspan class="rl-sub" baseline-shift="sub">1</tspan></text>
    <circle class="rl-action" cx="245" cy="215" r="11"/>
    <text class="rl-label" x="245" y="193" text-anchor="middle"><tspan class="rl-var">a</tspan><tspan class="rl-sub" baseline-shift="sub">2</tspan></text>
    <circle class="rl-action" cx="245" cy="320" r="11"/>
    <text class="rl-label" x="245" y="298" text-anchor="middle"><tspan class="rl-var">a</tspan><tspan class="rl-sub" baseline-shift="sub">3</tspan></text>

    <!-- Successor state nodes (from action 1) -->
    <circle class="rl-node" cx="440" cy="70" r="11"/>
    <text class="rl-label" x="440" y="46" text-anchor="middle"><tspan class="rl-var">s</tspan><tspan class="rl-sub" baseline-shift="sub">1</tspan></text>
    <text class="rl-dots" x="440" y="114" text-anchor="middle">⋮</text>
    <circle class="rl-node" cx="440" cy="150" r="11"/>
    <text class="rl-label" x="440" y="178" text-anchor="middle"><tspan class="rl-var">s</tspan><tspan class="rl-sub" baseline-shift="sub">n</tspan></text>

    <!-- Collapsed successor states for actions 2 & 3 -->
    <text class="rl-dots" x="440" y="219" text-anchor="middle">...</text>
    <text class="rl-dots" x="440" y="324" text-anchor="middle">...</text>

    <!-- Next action nodes (from successor state s_1) -->
    <circle class="rl-action" cx="635" cy="30" r="11"/>
    <text class="rl-label" x="646" y="16" text-anchor="start"><tspan class="rl-var">a</tspan><tspan class="rl-sub" baseline-shift="sub">1</tspan></text>
    <circle class="rl-action" cx="635" cy="70" r="11"/>
    <text class="rl-label" x="646" y="56" text-anchor="start"><tspan class="rl-var">a</tspan><tspan class="rl-sub" baseline-shift="sub">2</tspan></text>
    <circle class="rl-action" cx="635" cy="110" r="11"/>
    <text class="rl-label" x="646" y="96" text-anchor="start"><tspan class="rl-var">a</tspan><tspan class="rl-sub" baseline-shift="sub">3</tspan></text>

    <!-- Collapsed next actions for s_n -->
    <text class="rl-dots" x="635" y="154" text-anchor="middle">...</text>
  </svg>
</figure>

We are at state <span style="background-color: #d65344; color: white; border-radius: 999px; padding: 0.1em 0.5em; display: inline-block; line-height: 1.4;">$\boldsymbol{s}$</span>. From there, we sample a next action <span style="background-color: #c0d3f5; border-radius: 999px; padding: 0.1em 0.5em; display: inline-block; line-height: 1.4;">$$A_{t + 1}$$</span>, which transitions into an unknown new state <span style="background-color: #f1cab6; border-radius: 999px; padding: 0.1em 0.5em; display: inline-block; line-height: 1.4;">$$S_{t + 1}$$</span>, and so on. So, the expectation becomes:

$$
\begin{align*}
v_\pi(s) &= \sum_{a \in \mathcal{A}} \pi(a \mid s)  \left( \mathcal{R}_s^a + \gamma \mathcal{P}_{[s,:]}^a \cdot v_\pi \right)
\end{align*}
$$

### Action-Value Function ($q$)

So far, we've talked about only the state value function $v$. Now, we can introduce the action-value function $q(s, a)$:

$$
\begin{align*}
q_\pi (s, a) &= \mathbb{E}_\pi [G_t \mid S_t = s, A_t = a]
\end{align*}
$$

### Bellman Equation for $q$

$$
\begin{align*}
q_\pi(s, a) &= \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a] \\
&= \mathbb{E}_\pi[R_{t + 1} + \gamma (R_{t + 2} + \gamma R_{t + 3} + \cdots) \mid S_t = s, A_t = a] \\
&= \mathbb{E}_\pi[R_{t + 1} + \gamma q_\pi(S_{t + 1}, A_{t + 1}) \mid S_t = s, A_t = a] 
\end{align*}
$$

But we're not actually done yet, because we're being imprecise about what $$R_{t + 1}$$, $$S_{t + 1}$$, and $$A_{t + 1}$$ are.

<style>
  @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:ital@0;1&display=swap');
  .rl-tree {
    margin: 1.5rem 0; text-align: center;
    --rl-state-color: #f1cab6;
    --rl-action-color: #c0d3f5;
    --rl-node-r: 11px;
  }
  .rl-tree svg { width: 100%; height: auto; max-width: 680px; font-family: "Roboto Mono", ui-monospace, "SFMono-Regular", Menlo, monospace; }
  .rl-tree .rl-node { fill: var(--rl-state-color); stroke: var(--rl-state-color); stroke-width: 2.5; r: var(--rl-node-r); }
  .rl-tree .rl-action { fill: var(--rl-action-color); stroke: var(--rl-action-color); stroke-width: 1.5; r: var(--rl-node-r); }
  .rl-tree .rl-arrow { stroke: #b0b0b0; stroke-width: 2; fill: none; }
  .rl-tree .rl-label { font-weight: 700; font-size: 18px; fill: var(--text-color, #1f2328); }
  .rl-tree .rl-edge { font-size: 17px; fill: var(--text-color, #1f2328); }
  .rl-tree .rl-var { font-style: italic; }
  .rl-tree .rl-sub { font-size: 0.72em; }
  .rl-tree .rl-sup { font-size: 0.72em; }
  .rl-tree .rl-dots { font-size: 14px; font-weight: 700; fill: var(--text-color, #1f2328); }
  .rl-tree figcaption { font-size: 0.85rem; opacity: 0.8; margin-top: 0.6rem; max-width: 560px; margin-inline: auto; }
</style>

<figure class="rl-tree" markdown="0">
  <svg viewBox="0 0 730 340" role="img" aria-label="Decision tree: state s leads to three action nodes; the first action fans out into successor states; the first successor state fans out again into the next layer of action nodes.">
    <defs>
      <marker id="rl-arrowhead" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" fill="#b0b0b0"/>
      </marker>
      <marker id="rl-arrowhead-black" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" fill="black"/>
      </marker>
      <marker id="rl-arrowhead-verylight" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" fill="#dcdcdc"/>
      </marker>
    </defs>

    <!-- Edges: state s -> action nodes -->
    <path class="rl-arrow" d="M63,215 C148,215 148,215 234,215" style="stroke: #dcdcdc;" marker-end="url(#rl-arrowhead-verylight)"/>
    <path class="rl-arrow" d="M63,215 C148,215 148,320 234,320" style="stroke: #dcdcdc;" marker-end="url(#rl-arrowhead-verylight)"/>
    <path class="rl-arrow" d="M63,215 C148,215 148,110 234,110" style="stroke: black;" marker-end="url(#rl-arrowhead-black)"/>

    <!-- Edges: action node 1 -> successor states -->
    <path class="rl-arrow" d="M258,110 C344,110 344,70 429,70" marker-end="url(#rl-arrowhead)"/>
    <path class="rl-arrow" d="M258,110 C344,110 344,150 429,150" marker-end="url(#rl-arrowhead)"/>

    <!-- Edges: action nodes 2 & 3 -> "..." (collapsed successor states) -->
    <path class="rl-arrow" d="M258,215 C341,215 341,215 424,215" style="stroke: #dcdcdc;" marker-end="url(#rl-arrowhead-verylight)"/>
    <path class="rl-arrow" d="M258,320 C341,320 341,320 424,320" style="stroke: #dcdcdc;" marker-end="url(#rl-arrowhead-verylight)"/>

    <!-- Edges: successor state s_1 -> next action nodes -->
    <path class="rl-arrow" d="M453,70 C538,70 538,30 624,30" marker-end="url(#rl-arrowhead)"/>
    <path class="rl-arrow" d="M453,70 C538,70 538,70 624,70" marker-end="url(#rl-arrowhead)"/>
    <path class="rl-arrow" d="M453,70 C538,70 538,110 624,110" marker-end="url(#rl-arrowhead)"/>

    <!-- Edges: successor state s_n -> "..." (collapsed next actions) -->
    <path class="rl-arrow" d="M453,150 C536,150 536,150 619,150" marker-end="url(#rl-arrowhead)"/>

    <!-- Edge labels: a_i + R_{t+1} -->
    <text class="rl-edge" x="140" y="140" text-anchor="middle"><tspan class="rl-var">R</tspan><tspan class="rl-sub rl-var" baseline-shift="sub">s</tspan><tspan class="rl-sup" baseline-shift="super" dx="-4"><tspan class="rl-var">a</tspan><tspan class="rl-sub" baseline-shift="sub">1</tspan></tspan></text>
    <text class="rl-edge" x="150" y="200" text-anchor="middle"><tspan class="rl-var">R</tspan><tspan class="rl-sub rl-var" baseline-shift="sub">s</tspan><tspan class="rl-sup" baseline-shift="super" dx="-4"><tspan class="rl-var">a</tspan><tspan class="rl-sub" baseline-shift="sub">2</tspan></tspan></text>
    <text class="rl-edge" x="160" y="260" text-anchor="middle"><tspan class="rl-var">R</tspan><tspan class="rl-sub rl-var" baseline-shift="sub">s</tspan><tspan class="rl-sup" baseline-shift="super" dx="-4"><tspan class="rl-var">a</tspan><tspan class="rl-sub" baseline-shift="sub">3</tspan></tspan></text>

    <!-- Edge labels from s_1 to next action nodes -->
    <text class="rl-edge" x="590" y="16" text-anchor="middle"><tspan class="rl-var">R</tspan><tspan class="rl-sub" baseline-shift="sub"><tspan class="rl-var">s</tspan><tspan class="rl-sub" baseline-shift="sub">1</tspan></tspan><tspan class="rl-sup" baseline-shift="super" dx="-9"><tspan class="rl-var">a</tspan><tspan class="rl-sub" baseline-shift="sub">1</tspan></tspan></text>
    <text class="rl-edge" x="590" y="54" text-anchor="middle"><tspan class="rl-var">R</tspan><tspan class="rl-sub" baseline-shift="sub"><tspan class="rl-var">s</tspan><tspan class="rl-sub" baseline-shift="sub">1</tspan></tspan><tspan class="rl-sup" baseline-shift="super" dx="-9"><tspan class="rl-var">a</tspan><tspan class="rl-sub" baseline-shift="sub">2</tspan></tspan></text>
    <text class="rl-edge" x="590" y="92" text-anchor="middle"><tspan class="rl-var">R</tspan><tspan class="rl-sub" baseline-shift="sub"><tspan class="rl-var">s</tspan><tspan class="rl-sub" baseline-shift="sub">1</tspan></tspan><tspan class="rl-sup" baseline-shift="super" dx="-9"><tspan class="rl-var">a</tspan><tspan class="rl-sub" baseline-shift="sub">3</tspan></tspan></text>

    <!-- Root state node: s -->
    <circle class="rl-node" cx="50" cy="215" r="11" style="fill: #d65344; stroke: #d65344;"/>
    <text class="rl-label" x="50" y="246" text-anchor="middle"><tspan class="rl-var">s</tspan></text>

    <!-- Action nodes (from s) -->
    <circle class="rl-action" cx="245" cy="110" r="11" style="fill: #5977e3; stroke: #5977e3;"/>
    <text class="rl-label" x="245" y="88" text-anchor="middle"><tspan class="rl-var">a</tspan><tspan class="rl-sub" baseline-shift="sub">1</tspan></text>
    <circle class="rl-action" cx="245" cy="215" r="11" style="fill: #dcdcdc; stroke: #dcdcdc;"/>
    <text class="rl-label" x="245" y="193" text-anchor="middle"><tspan class="rl-var">a</tspan><tspan class="rl-sub" baseline-shift="sub">2</tspan></text>
    <circle class="rl-action" cx="245" cy="320" r="11" style="fill: #dcdcdc; stroke: #dcdcdc;"/>
    <text class="rl-label" x="245" y="298" text-anchor="middle"><tspan class="rl-var">a</tspan><tspan class="rl-sub" baseline-shift="sub">3</tspan></text>

    <!-- Successor state nodes (from action 1) -->
    <circle class="rl-node" cx="440" cy="70" r="11"/>
    <text class="rl-label" x="440" y="46" text-anchor="middle"><tspan class="rl-var">s</tspan><tspan class="rl-sub" baseline-shift="sub">1</tspan></text>
    <text class="rl-dots" x="440" y="114" text-anchor="middle">⋮</text>
    <circle class="rl-node" cx="440" cy="150" r="11"/>
    <text class="rl-label" x="440" y="178" text-anchor="middle"><tspan class="rl-var">s</tspan><tspan class="rl-sub" baseline-shift="sub">n</tspan></text>

    <!-- Collapsed successor states for actions 2 & 3 -->
    <text class="rl-dots" x="440" y="219" text-anchor="middle">...</text>
    <text class="rl-dots" x="440" y="324" text-anchor="middle">...</text>

    <!-- Next action nodes (from successor state s_1) -->
    <circle class="rl-action" cx="635" cy="30" r="11"/>
    <text class="rl-label" x="646" y="16" text-anchor="start"><tspan class="rl-var">a</tspan><tspan class="rl-sub" baseline-shift="sub">1</tspan></text>
    <circle class="rl-action" cx="635" cy="70" r="11"/>
    <text class="rl-label" x="646" y="56" text-anchor="start"><tspan class="rl-var">a</tspan><tspan class="rl-sub" baseline-shift="sub">2</tspan></text>
    <circle class="rl-action" cx="635" cy="110" r="11"/>
    <text class="rl-label" x="646" y="96" text-anchor="start"><tspan class="rl-var">a</tspan><tspan class="rl-sub" baseline-shift="sub">3</tspan></text>

    <!-- Collapsed next actions for s_n -->
    <text class="rl-dots" x="635" y="154" text-anchor="middle">...</text>
  </svg>
  <figcaption>The bolded s and a_1 nodes are the specified operands in q(s, a). Everything else after is unknown.</figcaption>
</figure>

As you can see from the above decision tree diagram, we only know what <span style="background-color: #d65344; color: white; border-radius: 999px; padding: 0.1em 0.5em; display: inline-block; line-height: 1.4;">$\boldsymbol{s}$</span> and <span style="background-color: #5977e3; color: white; border-radius: 999px; padding: 0.1em 0.5em; display: inline-block; line-height: 1.4;">$a$</span> are in $$q_\pi(s, a)$$, hence $$R_{t + 1} = R_s^{a = a_1}$$. The continuation of the path from there is a random variable, hence <span style="background-color: #f1cab6; border-radius: 999px; padding: 0.1em 0.5em; display: inline-block; line-height: 1.4;">$$S_{t + 1}$$</span> and <span style="background-color: #c0d3f5; border-radius: 999px; padding: 0.1em 0.5em; display: inline-block; line-height: 1.4;">$$A_{t + 1}$$</span> are random variables. To fill in this piece of information to make the Bellman Equation more precise, we have:


$$
\begin{align*}
q_\pi(s, a) &= R_s^a + \gamma  \sum_{s' \in \mathcal{S}} \mathcal{P}_{s \rightarrow s'}^a \underbrace{\sum_{a' \in \mathcal{A}} \pi(a' \mid s') q_\pi(s', a')}_{\textstyle = v_\pi(s')}
\end{align*}
$$


### Connection between $q$ and $v$

By now, we can see that $q$ and $v$ are just two sides of the same coin - one allows you to specify the next $a$, one does not.

# 3. Solving the Planning Problem

## 3.1. Optimizing $v$, $q$, and $\pi$

In sequential decision making problems, we want to maximize $v$ / $q$. We will simply optimize $q$ because $q$ is a more useful formulation of value.

$$
\begin{align*}
q_\star (s, a) &= \max_{\pi} q_\pi(s, a)
\end{align*}
$$

### Optimal Policy $\pi_\star$

A policy $\pi$ is better than another policy $\pi'$ if:

$$
\begin{align*}
v_\pi (s) \geq v_{\pi'} (s), \forall s
\end{align*}
$$

There is a very important Theorem about MDPs: **For any MDP, there exists an optimal policy $$\pi_\star$$ that is better than or equal to all other policies $$\pi$$.** This is amazing because it means we don't have to reason about where in an MDP a policy might be better or worse than another.

### Optimal Action-Value Function Q*

The optimal (deterministic) policy can be found by maximizing $q_\star$, and the optimal is hence a greedy one:

$$
\begin{align*}
\pi_\star (a \mid s) & = 
\begin{cases}
1 & \text{if } a = \operatorname*{argmax}\limits_{a} q_\star(s, a) \\
0 & \text{otherwise}
\end{cases}
\end{align*}
$$

### Bellman Optimality Equation for $v_\star$

Previously, we arrived at the Bellman Equation for $v$ in a general MDP as:

$$
\begin{align*}
v_\pi(s) &= \sum_{a \in \mathcal{A}} \pi(a \mid s)  \left( \mathcal{R}_s^a + \gamma \mathcal{P}_{[s,:]}^a \cdot v_\pi \right)
\end{align*}
$$

Notice that in a deterministic optimal policy, our actions are greedy. There's no variability in the actions we choose - they are always the value-maximizing action, so we have:

$$
\begin{align*}
v_\star(s) &= \max_a  \left( \mathcal{R}_s^a + \gamma \mathcal{P}_{[s,:]}^a \cdot v_\star \right)
\end{align*}
$$

### Bellman Optimality Equation for $q_\star$

Similarly, we previously arrived at this Bellman Equation for $q$ in a general MDP:

$$
\begin{align*}
q_\pi(s, a) &= R_s^a + \gamma  \sum_{s' \in \mathcal{S}} \mathcal{P}_{s \rightarrow s'}^a \underbrace{\sum_{a' \in \mathcal{A}} \pi(a' \mid s') q_\pi(s', a')}_{\textstyle = v_\pi(s')}
\end{align*}
$$

Factoring in the action greediness, we have:

$$
\begin{align*}
q_\star(s, a) &= R_s^a + \gamma  \sum_{s' \in \mathcal{S}} \mathcal{P}_{s \rightarrow s'}^a \underbrace{\max_a q_\star(s', a')}_{\textstyle = v_\star(s')}
\end{align*}
$$

## 3.2. Solving Bellman Optimality Equations

In an MRP, we saw that we could solve the Bellman Equation by simple linear algebra, because in an MRP, we simply took expectations over actions and transitions - everything was a linear equation. In an MDP where you have a greedy policy, the greediness introduces a non-linearity, so there no longer exists a closed form solution (not that the closed form solution would have been practical anyway).

So, we have to use iterative methods, including:
- Value Iteration
- Policy Iteration
- Q-Learning
- SARSA

## 3.3. Policy Iteration

### Policy Evaluation

Given an MDP $$\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$$, and a policy $\pi$, we can evaluate the policy (calculate $v_\pi$). Since this works for any policy, even non-greedy ones, we'll use the Bellman **expectation** equation instead of the optimality equation:

$$
\begin{align*}
v_\pi(s) &= \sum_{a \in \mathcal{A}} \pi(a \mid s)  \left( \mathcal{R}_s^a + \gamma \mathcal{P}_{[s,:]}^a \cdot v_\pi \right)
\end{align*}
$$

The formulation here already directly tells us how to update our $v$ values:

$$
\begin{align*}
v_{k + 1}(s) &= \sum_{a \in \mathcal{A}} \pi(a \mid s)  \left( \mathcal{R}_s^a + \gamma \mathcal{P}_{[s,:]}^a \cdot v_{k} \right)
\end{align*}
$$

What we do is just to traverse all the $v$ values and update all of them per timestep ($k$).

#### Example:

Here's the setup: a 🦆 (our agent) lives on a 4&times;4 grid and wants to reach a checkered flag 🏁. The two flag cells are **terminal** — the episode ends once the duck steps on either. On every other cell the duck can move up / down / left / right (moves off the edge leave it in place), and each step costs a reward of &minus;1. Under a uniform-random policy, how good is each cell?

<style>
  .rl-grid.rl-grid-static .rl-cell { background: #ededed; font-size: clamp(19px, 7vw, 29px); }
  .rl-grid.rl-grid-static .rl-board { position: relative; }
  .rl-policy-arrows { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; overflow: visible; }
  .rl-policy-arrows path { stroke: #5977e3; stroke-width: 3.5; fill: none; stroke-linecap: round; }
</style>

<figure class="rl-grid rl-grid-static" markdown="0">
  <div class="rl-board" role="img" aria-label="4x4 gridworld: checkered flags at the top-left and bottom-right corners, a duck in the second row second column.">
    <div class="rl-cell rl-terminal">🏁</div>
    <div class="rl-cell"></div>
    <div class="rl-cell"></div>
    <div class="rl-cell"></div>
    <div class="rl-cell"></div>
    <div class="rl-cell">🦆</div>
    <div class="rl-cell"></div>
    <div class="rl-cell"></div>
    <div class="rl-cell"></div>
    <div class="rl-cell"></div>
    <div class="rl-cell"></div>
    <div class="rl-cell"></div>
    <div class="rl-cell"></div>
    <div class="rl-cell"></div>
    <div class="rl-cell"></div>
    <div class="rl-cell rl-terminal">🏁</div>
    <!-- policy arrows: from the duck (cell 5, center 150,150) across each gap into cells 1/4/6/9 -->
    <svg class="rl-policy-arrows" viewBox="0 0 400 400" preserveAspectRatio="xMidYMid meet" aria-hidden="true">
      <defs>
        <marker id="rl-duck-arrow" markerUnits="userSpaceOnUse" viewBox="0 0 10 7" refX="9" refY="3.5" markerWidth="10" markerHeight="7" orient="auto-start-reverse" overflow="visible">
          <path d="M0,0 L10,3.5 L0,7 z" fill="#5977e3" stroke="none"/>
        </marker>
      </defs>
      <path d="M150,120 L150,56"  marker-end="url(#rl-duck-arrow)"/> <!-- up    -> cell 1 -->
      <path d="M150,180 L150,244" marker-end="url(#rl-duck-arrow)"/> <!-- down  -> cell 9 -->
      <path d="M120,150 L56,150"  marker-end="url(#rl-duck-arrow)"/> <!-- left  -> cell 4 -->
      <path d="M180,150 L244,150" marker-end="url(#rl-duck-arrow)"/> <!-- right -> cell 6 -->
    </svg>
  </div>
  <figcaption>🏁 = terminal goal states; 🦆 = the agent.<br>
  Reward: &minus;1.0 per step<br>
  &gamma;: 1.0<br>
  Transition Prob: 1.0<br>
  &pi;: uniform-random</figcaption>
</figure>

Policy Evaluation in action:

<style>
  .rl-grid { margin: 1.5rem 0; text-align: center; font-family: "Roboto Mono", ui-monospace, "SFMono-Regular", Menlo, monospace; }
  .rl-grid .rl-board { display: grid; grid-template-columns: repeat(4, 1fr); gap: 3px; width: 100%; max-width: 320px; margin: 0 auto; }
  .rl-grid .rl-cell { aspect-ratio: 1; display: flex; align-items: center; justify-content: center; border-radius: 4px; font-size: 14px; font-weight: 600; color: #1f2328; /* NO transition: colors snap per timestep */ }
  .rl-grid .rl-cell.rl-terminal { box-shadow: inset 0 0 0 2.5px #1f2328; }
  .rl-grid .rl-controls { display: flex; align-items: center; justify-content: center; gap: 0.6rem; margin-top: 0.85rem; flex-wrap: wrap; }
  .rl-grid button { font: inherit; font-size: 13px; padding: 0.25em 0.7em; border-radius: 6px; border: 1px solid #b0b0b0; background: var(--card-bg, #fff); color: var(--text-color, #1f2328); cursor: pointer; }
  .rl-grid button:hover { border-color: #888; }
  .rl-grid input[type="range"] { width: 150px; accent-color: #d75344; }
  .rl-grid .rl-k { font-size: 13px; min-width: 4em; text-align: left; }
  .rl-grid figcaption { font-size: 0.85rem; opacity: 0.8; margin-top: 0.7rem; max-width: 470px; margin-inline: auto; }
</style>

<figure class="rl-grid" markdown="0">
  <div id="rl-grid-board" class="rl-board" role="img" aria-label="Animated 4x4 gridworld value function during iterative policy evaluation."></div>
  <div class="rl-controls">
    <button id="rl-grid-play" type="button">&#10074;&#10074;</button>
    <input id="rl-grid-slider" type="range" min="0" max="0" value="0" step="1" aria-label="timestep k">
    <span id="rl-grid-k" class="rl-k">k = 0</span>
  </div>
  <figcaption>Value Iteration: v<sub>k</sub> slowly converges to v<sub>&pi;</sub></figcaption>
</figure>

<script>
(function () {
  // ---- coolwarm[30:]  (matplotlib get_cmap('coolwarm', 61)[30:]): white -> deep red ----
  const PALETTE = ['#dddddd','#e1dad7','#e5d8d1','#e9d5ca','#ecd2c4','#efcebd','#f2cbb7','#f4c6b0','#f5c2aa','#f6bda3','#f7b89c','#f7b295','#f7ad8f','#f7a788','#f5a081','#f49a7b','#f29374','#f08c6e','#ee8468','#eb7d61','#e7755b','#e46d55','#e0654f','#db5c4a','#d75344','#d24a3f','#cc403a','#c63535','#c12a30','#ba172b','#b40426'];

  // ---- gridworld setup ----
  const N = 4, GAMMA = 1, REWARD = -1;
  const TERM = new Set([0, N * N - 1]);          // top-left & bottom-right are terminal
  const idx = (r, c) => r * N + c;

  // one synchronous sweep of the Bellman expectation backup (uniform-random policy)
  function sweep(v) {
    const nv = v.slice();
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
      const s = idx(r, c);
      if (TERM.has(s)) { nv[s] = 0; continue; }
      let newV = 0;
      for (const [nr, nc] of [[r - 1, c], [r + 1, c], [r, c - 1], [r, c + 1]]) {
        const withinBounds = nr >= 0 && nr < N && nc >= 0 && nc < N;
        const sp = withinBounds ? idx(nr, nc) : s;        // off-grid moves stay in place
        newV += 0.25 * (REWARD + GAMMA * v[sp]);
      }
      nv[s] = newV;
    }
    return nv;
  }

  // ---- precompute frames: v_0, v_1, ... until convergence ----
  const frames = [new Array(N * N).fill(0)];
  for (let k = 0; k < 300; k++) {
    const prev = frames[frames.length - 1];
    const nv = sweep(prev);
    frames.push(nv);
    let maxDelta = 0;
    for (let i = 0; i < nv.length; i++) maxDelta = Math.max(maxDelta, Math.abs(nv[i] - prev[i]));
    if (maxDelta < 1e-2) break;
  }

  // fixed global min -> stable color scale across all timesteps (0 = white, vmin = deepest red)
  let vmin = 0;
  for (const f of frames) for (const x of f) vmin = Math.min(vmin, x);
  const palIndex = v => Math.round((vmin < 0 ? Math.min(1, v / vmin) : 0) * (PALETTE.length - 1));

  // ---- build cells once ----
  const board = document.getElementById('rl-grid-board');
  const cells = [];
  for (let s = 0; s < N * N; s++) {
    const d = document.createElement('div');
    d.className = 'rl-cell' + (TERM.has(s) ? ' rl-terminal' : '');
    board.appendChild(d);
    cells.push(d);
  }

  const playBtn = document.getElementById('rl-grid-play');
  const slider  = document.getElementById('rl-grid-slider');
  const kLabel  = document.getElementById('rl-grid-k');
  slider.max = String(frames.length - 1);

  let cur = 0, timer = null, holdTimer = null;
  const STEP_MS = 650, HOLD_MS = 1500;

  function render(k) {
    cur = k;
    const f = frames[k];
    for (let i = 0; i < cells.length; i++) {
      const v = f[i];
      const pi = palIndex(v);
      cells[i].style.backgroundColor = PALETTE[pi];
      cells[i].style.color = pi >= 18 ? '#ffffff' : '#1f2328';
      cells[i].textContent = v === 0 ? '0' : v.toFixed(1);
    }
    slider.value = String(k);
    kLabel.textContent = 'k = ' + k;
  }

  function pause() { clearInterval(timer); timer = null; clearTimeout(holdTimer); playBtn.innerHTML = '&#9654;'; }
  function play()  { if (timer) return; playBtn.innerHTML = '&#10074;&#10074;'; timer = setInterval(tick, STEP_MS); }

  function tick() {
    if (cur >= frames.length - 1) {                 // reached convergence: hold, then loop
      clearInterval(timer); timer = null;
      holdTimer = setTimeout(() => { render(0); play(); }, HOLD_MS);
      return;
    }
    render(cur + 1);
  }

  playBtn.addEventListener('click', () => { (timer || holdTimer) ? pause() : (render(cur >= frames.length - 1 ? 0 : cur), play()); });
  slider.addEventListener('input', () => { pause(); render(Number(slider.value)); });

  render(0);
  play();                                            // autoplay on load
})();
</script>

### Policy Improvement

So now, having evaluated the policy to arrive at converged values of each state, we can use those values to infer a policy that is optimal (w.r.t to these values). In the 4x4 grid example, the optimal policy should just tell us to proceed in the direction of most reward (least red neighbor).

<style>
  /* --- converged-value grid + greedy-policy arrows (reuses .rl-grid base styling) --- */
  .rl-policy-grid .rl-cell { font-size: 11px; }                     /* smaller value labels */
  .rl-policy-grid .rl-board { position: relative; }                 /* anchor for the arrow overlay */
  .rl-policy-grid .rl-greedy-arrows { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; overflow: visible; }
  .rl-policy-grid .rl-greedy-arrows marker { overflow: visible; }                                                     /* keep arrow-head corners from clipping */
  .rl-policy-grid .rl-greedy-arrows > path { stroke: #1f2328; stroke-width: 2; fill: none; stroke-linecap: round; }  /* shafts */
  .rl-policy-grid .rl-greedy-arrows marker path { fill: #1f2328; stroke: none; }                                      /* arrow heads */
</style>

<figure class="rl-grid rl-policy-grid" markdown="0">
  <div id="rl-policy-board" class="rl-board" role="img" aria-label="Converged 4x4 gridworld values with black arrows pointing from each cell to its best (least-negative) neighbour."></div>
  <figcaption>Greedy policy w.r.t. the converged v<sub>&pi;</sub></figcaption>
</figure>

### Policy Iteration = Evaluation + Improvement

Ping-ponging between Evaluation and Improvement gives us Policy Iteration. Crucially, you don't have to run the Evaluation step to convergence each time. You can see that even with 5 time-steps of Evaluation, we can already sufficiently update $v$ such that we can meaningfully improve on the policy (which happens to already be optimal):

<figure class="rl-grid rl-policy-grid" markdown="0">
  <div id="rl-ts5-board" class="rl-board" role="img" aria-label="4x4 gridworld values after 5 sweeps of policy evaluation, with greedy-policy arrows."></div>
  <figcaption>Greedy policy w.r.t. v<sub>5</sub></figcaption>
</figure>

The reason we can do partial evaluations and still update the policy is because of the powerful **theorem that this method of evaluation and improvement [will always converge to the optimal deterministic policy](https://youtu.be/Nd1-UUMVfz4?list=PLzuuYNsE1EZAXYR4FJ75jcJseBmo4KQ9-&t=1998).** The proof for this is simply the argument along the lines:

$$q_\pi (s, s(\pi')) = \max_{a \in \mathcal{A}} q_\pi (s, a) \geq \left(q_\pi (s, \pi(s)) = v_\pi (s) \right)$$

English: if we acted greedily for this 1 step from $s$ but acted according to $\pi$ from there-on, we will do at least as well as before. If the LHS and RHS are the same, then at least for the state, $\pi$ is optimal, and we have satisfied the Bellman Optimality Equation. The intuition for why this improvement will always lead to the optimal policy is non-obvious to me, but I find it helpful to remember that all states are updated. This isn't the case of true Reinforcement Learning problems where we only have information on states that we visit (sample) and can hence be stuck in some local optimum.

#### Implications for Implementation

Stopping Conditions for Evaluation Step - we can:
- Updates for values are within some $\epsilon$
- Simply stop after $K$ iterations
- Extreme: stop after $K = 1$. This is **Value Iteration**

## 3.4. Value Iteration

The crux of Value Iteration is the same as in Policy Iteration: **repeatedly using the Bellman Optimality (not Expectation this time) Equation** to update the value scores for every state. The key differences are:
- **You don't have an explicit policy to follow** - everything subsequent state can be chosen greedily during Evaluation. In the above Policy Iteration example, we followed the uniform-random policy and applied the Bellman Expectation Equation during Evaluation.
- For each round of value updates, you do **only 1 sweep** over all the states ($K = 1$)

In practice, you almost always us Value Iteration, unless there is an explicit policy to follow. Value Iteration can be thought of as the special case of Policy Iteration where your policy is simply greedy.

## 3.5. Implementation Tricks

Because the convergence conditions for Policy / Value Iteration are (1) greedy update and (2) you continually update all states (no states left behind), you can speed up the training process without affecting convergence possibility by:

- Doing In-Place updates of values
- Parallelizing updates over all states
- Asynchronously sampling and updating states
- Sampling rollouts and update only information seen during the rollout

These are all techniques used in large RL problems, where state spaces can be huge (and hence updating all of them always can sometimes be a waste of time), and some techniques also introduce stochasticity, which can be useful at times.

# 4. Model-Free Planning

In the previous methods where we could use variants of Dynamic Programming (Policy Iteration, Value Iteration) to solve MDPs, we could do this because:
- Transition Dynamics were known (we had the model $\mathcal{P}$). In our examples, the transitions were all deterministic, but the methods would have worked too for stochastic transitions.
- The Rewards $\mathcal{R}$ were known as well.
- The problem was not large (millions of states or smaller)

In the real world, either they may be trivial (deterministic), or more crucially, we may not have $\mathcal{P}$ or $\mathcal{R}$. However, what matters is that we are able to estimate the expectation of these. Sampling trajectories allows us to still do so. Goodbye models.

## 4.1. Sampling Full Episodes: Monte-Carlo RL

Suppose you have a simple trajectory that looks like this:

<style>
  .rlhop-wrap { display: flex; flex-direction: column; align-items: center; margin: 1.5rem 0; }
  .rlhop-stage { position: relative; }
  .rlhop-board { display: grid; grid-template-columns: repeat(3, 80px); grid-template-rows: repeat(3, 80px); gap: 6px; }
  .rlhop-cell { border-radius: 10px; background: #ededed; position: relative; }  /* ring cells get coolwarm colours via JS */
  .rlhop-cell.rlhop-hollow { background: transparent; }                 /* hollow center -> 8 cells */
  .rlhop-ring { position: absolute; box-sizing: border-box; border: 2px solid #1f2328; background: transparent; pointer-events: none; }  /* one hollow rounded square per visit */
  .rlhop-overlay { position: absolute; inset: 0; pointer-events: none; }
  .rlhop-shadow { position: absolute; top: 0; left: 0; width: 42px; height: 14px; margin: -7px 0 0 -21px; border-radius: 50%; background: rgba(0, 0, 0, 0.28); filter: blur(1px); will-change: transform, opacity; }
  .rlhop-agent { position: absolute; top: 0; left: 0; width: 52px; height: 52px; margin: -26px 0 0 -26px; display: flex; align-items: center; justify-content: center; font-size: 2.3rem; line-height: 1; will-change: transform; }
  .rlhop-controls { display: flex; gap: 0.6rem; margin-top: 1rem; }
  .rlhop-btn { padding: 0.4rem 1rem; font-size: 0.85rem; font-weight: 600; border: 1px solid var(--main-border-color, #d0d3d6); border-radius: 8px; background: transparent; color: inherit; cursor: pointer; }
  .rlhop-btn:hover { background: var(--card-hover-bg, #e6e8eb); }
</style>

<div id="rl-mc-hop" class="rlhop-wrap" markdown="0">
  <div class="rlhop-stage">
    <div class="rlhop-board"></div>
    <div class="rlhop-overlay">
      <div class="rlhop-shadow"></div>
      <div class="rlhop-agent">🦆</div>
    </div>
  </div>
  <div class="rlhop-controls">
    <button class="rlhop-btn rlhop-prev" type="button" aria-label="step back">&lt;</button>
    <button class="rlhop-btn rlhop-toggle" type="button">Pause</button>
    <button class="rlhop-btn rlhop-next" type="button" aria-label="step forward">&gt;</button>
  </div>
</div>

<script>
(function () {
  "use strict";
  const root = document.getElementById('rl-mc-hop');
  if (!root || root.dataset.init) return;
  root.dataset.init = '1';

  const boardEl   = root.querySelector('.rlhop-board');
  const agentEl   = root.querySelector('.rlhop-agent');
  const shadowEl  = root.querySelector('.rlhop-shadow');
  const toggleEl  = root.querySelector('.rlhop-toggle');
  const prevEl    = root.querySelector('.rlhop-prev');
  const nextEl    = root.querySelector('.rlhop-next');

  // Grid geometry — must match the CSS above.
  const N = 3, CELL = 80, GAP = 6, STRIDE = CELL + GAP;
  const HOLLOW = 1 * N + 1;                 // center cell (1,1): hollow, never stepped on

  // Hop tuning (same parabolic-arc logic as agent-hop.html).
  const HOP_APEX = 62;   // px the arc peaks above the straight line between centres
  const HOP_MS   = 420;  // duration of one hop
  const LAND_MS  = 180;  // pause after landing on a cell
  const END_MS   = 900;  // pause back at the start before looping

  // Build the 3x3 cells; the centre is hollow.
  const cells = [];
  for (let i = 0; i < N * N; i++) {
    const cell = document.createElement('div');
    cell.className = 'rlhop-cell' + (i === HOLLOW ? ' rlhop-hollow' : '');
    boardEl.appendChild(cell);
    cells.push(cell);
  }
  const cellAt  = (r, c) => cells[r * N + c];
  const centerX = c => c * STRIDE + CELL / 2;
  const centerY = r => r * STRIDE + CELL / 2;

  // Static per-cell colour: the 8 ring cells in clockwise order get coolwarm(12)[2:10].
  const RING   = [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2], [2, 1], [2, 0], [1, 0]];  // TL,T,TR,R,BR,B,BL,L
  const COLORS = ['#7598f6', '#95b7ff', '#b4cdfa', '#d1dae9', '#e8d6cc', '#f5c1a8', '#f6a384', '#ea7c61'];
  RING.forEach(([r, c], k) => { cellAt(r, c).style.background = COLORS[k]; });

  // Ring path: clockwise from top-left, once around back to top-left, then on to top-right.
  const CORNERS = [[0, 0], [0, 2], [2, 2], [2, 0], [0, 0], [0, 2]];
  function buildPath(corners) {
    const path = [corners[0].slice()];
    for (let i = 0; i < corners.length - 1; i++) {
      let [r, c] = corners[i];
      const [tr, tc] = corners[i + 1];
      while (r !== tr || c !== tc) {
        r += Math.sign(tr - r);
        c += Math.sign(tc - c);
        path.push([r, c]);
      }
    }
    return path;
  }
  const PATH = buildPath(CORNERS);   // 11 stops: TL,T,TR,R,BR,B,BL,L,TL,T,TR (ends at TR)
  const LAST = PATH.length - 1;

  // Draw the agent + ground shadow with a parabolic lift.
  function draw(x, y, lift, h) {
    agentEl.style.transform = 'translate(' + x + 'px, ' + (y - lift) + 'px)';
    const s = 1 - 0.45 * h;                                  // shadow shrinks as it rises
    shadowEl.style.transform = 'translate(' + x + 'px, ' + y + 'px) scale(' + s + ')';
    shadowEl.style.opacity = (0.35 - 0.22 * h).toFixed(3);
  }
  const place = i => { const [r, c] = PATH[i]; draw(centerX(c), centerY(r), 0, 0); };

  // Visit indicator: each landing stamps one more concentric hollow rounded square
  // inside the cell (nested inward), so N visits => N rings. Same shape as the cell.
  const CELL_R = 10, RING_START = 3, RING_STEP = 4;     // px: cell radius, first inset, inset per extra ring
  function stampVisit(r, c) {
    const cell = cellAt(r, c);
    const inset = RING_START + cell.querySelectorAll('.rlhop-ring').length * RING_STEP;
    const ring = document.createElement('div');
    ring.className = 'rlhop-ring';
    ring.style.inset = inset + 'px';
    ring.style.borderRadius = Math.max(2, CELL_R - inset) + 'px';
    ring.style.borderColor = 'white';
    ring.style.borderWidth = 0.5;
    cell.appendChild(ring);
  }
  const clearRings = () => cells.forEach(cell =>
    cell.querySelectorAll('.rlhop-ring').forEach(ring => ring.remove()));
  // Rings are a pure function of progress: re-derive from PATH[0..upto] so stepping
  // backward removes rings just as naturally as stepping forward adds them.
  function renderRings(upto) { clearRings(); for (let i = 0; i <= upto; i++) stampVisit(...PATH[i]); }

  let idx, phase, phaseT, paused, lastTs;
  function show(i) { idx = i; phase = 'land'; phaseT = 0; renderRings(idx); place(idx); }   // snap to stop i
  function start() { show(0); }

  function frame(ts) {
    const dt = lastTs == null ? 0 : ts - lastTs;
    lastTs = ts;
    if (!paused) {
      phaseT += dt;
      if (phase === 'hop') {
        const t = Math.min(phaseT / HOP_MS, 1);
        const [r0, c0] = PATH[idx], [r1, c1] = PATH[idx + 1];
        const x = centerX(c0) + (centerX(c1) - centerX(c0)) * t;
        const y = centerY(r0) + (centerY(r1) - centerY(r0)) * t;
        const h = 4 * t * (1 - t);                           // 0 at takeoff/landing, 1 at apex (t=1/2)
        draw(x, y, HOP_APEX * h, h);
        if (t >= 1) { idx += 1; renderRings(idx); phase = 'land'; phaseT = 0; }
      } else {                                               // resting on a cell
        place(idx);
        if (phaseT >= (idx === LAST ? END_MS : LAND_MS)) {
          if (idx === LAST) start(); else { phase = 'hop'; phaseT = 0; }
        }
      }
    }
    requestAnimationFrame(frame);
  }

  const setPaused = p => { paused = p; toggleEl.textContent = p ? 'Play' : 'Pause'; };
  // "<" / ">" pause playback and snap one stop back / forward, wrapping around the loop.
  const step = dir => { setPaused(true); const n = PATH.length; show(((idx + dir) % n + n) % n); };

  toggleEl.addEventListener('click', () => setPaused(!paused));
  prevEl.addEventListener('click', () => step(-1));
  nextEl.addEventListener('click', () => step(+1));

  paused = false; lastTs = null;
  start();
  requestAnimationFrame(frame);          // kick off
})();
</script>

There are two variants of Monto-Carlo Policy Evaluation (computing $v_\pi(s)$).
1. "First-Visit": We only use information collected at each state the **first time** that state is visited. That means the information collected at repeated visits (when the cell above is annotated by a second inset ring) is discarded.
2. "Every-Visit": We use all information at **every visit**.

It is not clear which is better, but intuitively, I would think "Every-Visit" makes more sense, because of the Markov Property assumption. Moving forward, I'll use this paradigm.

### 4.1.1. Collecting Episodes, Computing Empirical Means

Our goal here is to estimate $$v_\pi(s)$$, which is an expectation of returns:

$$
\begin{align*}
v_\pi(s) &= \mathbb{E}_\pi[G_t \mid S_t = s]
\end{align*}
$$

So let's walk through the above example to demonstrate the procedure for doing so. In the above example, we have this episode:

<style>
  .rltraj-wrap { display: flex; align-items: center; justify-content: center; flex-wrap: nowrap; gap: 3px; max-width: 100%; overflow-x: auto; margin: 1.5rem 0; padding-bottom: 4px; }
  .rltraj-sq { position: relative; flex: 0 0 auto; width: 34px; height: 34px; border-radius: 6px; }
  .rltraj-ring { position: absolute; box-sizing: border-box; border: 0.5px solid white; background: transparent; pointer-events: none; }
  .rltraj-arrow { flex: 0 0 auto; color: var(--text-color, #1f2328); font-size: 0.9rem; line-height: 1; }
</style>

<div id="rl-traj-line" class="rltraj-wrap" markdown="0"></div>

<script>
(function () {
  "use strict";
  const root = document.getElementById('rl-traj-line');
  if (!root || root.dataset.init) return;
  root.dataset.init = '1';

  // Same coolwarm(12)[2:10] palette and white concentric-ring style as the grid above.
  const COLORS = ['#7598f6', '#95b7ff', '#b4cdfa', '#d1dae9', '#e8d6cc', '#f5c1a8', '#f6a384', '#ea7c61'];
  // Unrolled trajectory: each entry is the visited cell (index into COLORS), in visit order.
  // TL,T,TR,R,BR,B,BL,L, TL,T,TR  -> the 11-stop path, ending at TR.
  const SEQ = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2];

  const SQ_R = 6, RING_START = 2, RING_STEP = 3;     // px: square radius, first ring inset, inset per extra ring

  function makeSquare(colorIdx, visits) {            // colored square with `visits` concentric hollow rings
    const sq = document.createElement('div');
    sq.className = 'rltraj-sq';
    sq.style.background = COLORS[colorIdx];
    for (let k = 0; k < visits; k++) {
      const inset = RING_START + k * RING_STEP;
      const ring = document.createElement('div');
      ring.className = 'rltraj-ring';
      ring.style.inset = inset + 'px';
      ring.style.borderRadius = Math.max(1, SQ_R - inset) + 'px';
      sq.appendChild(ring);
    }
    return sq;
  }

  const arrow = () => {
    const a = document.createElement('span');
    a.className = 'rltraj-arrow';
    a.innerHTML = '&rarr;';
    return a;
  };

  // Lay the stops out in a line; ring count = cumulative visits of that cell so far.
  const seen = new Array(COLORS.length).fill(0);
  SEQ.forEach((colorIdx, i) => {
    seen[colorIdx] += 1;
    if (i > 0) root.appendChild(arrow());
    root.appendChild(makeSquare(colorIdx, seen[colorIdx]));
  });
})();
</script>

If this were the start of the learning process, for the states that are visited once, we simply set $v(s_t) = G_t$:

<div id="rl-vg-line" markdown="0"></div>

<script>
(function () {
  "use strict";
  const root = document.getElementById('rl-vg-line');
  if (!root || root.dataset.init) return;
  root.dataset.init = '1';

  /* ===== Layout knobs — everything below is derived from these ===== */
  const SQ = 34, GAP = 20, PAD_L = 50;        // square size, gap between squares, left padding
  const STRIDE = SQ + GAP;                    // square-to-square pitch  <-- the spacing variable
  const SQ_R = 6, RING_INSET = 2, RING_STEP = 3;
  const ARC_DROP = 20;                        // how far the transition arcs bow below the row
  const BRACE_R = 6, BRACE_PAD = 14;          // brace corner radius; overhang past first/last reward

  // vertical bands, each stacked under the previous
  const Y_ROW   = SQ;                         // arcs start at the row's bottom edge
  const Y_ARC   = Y_ROW + ARC_DROP;           // arc control point
  const Y_REW   = Y_ARC + 6;                  // reward-label baseline
  const Y_BRACE = Y_REW + 12;                 // brace top edge
  const Y_LABEL = Y_BRACE + 2 * BRACE_R + 15; // v(s_t)= / G_t baseline

  /* ===== Data: the unrolled trajectory ===== */
  // coolwarm(12)[2:10] in clockwise ring order (same palette as the grid above).
  const COLORS = ['#7598f6', '#95b7ff', '#b4cdfa', '#d1dae9', '#e8d6cc', '#f5c1a8', '#f6a384', '#ea7c61'];
  const FULL = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2];   // full 11-stop path (index into COLORS)
  const START = 5;                                   // render from the 6th square (1-indexed) on
  const seen = new Array(COLORS.length).fill(0);
  const squares = [];                                // [{ color, rings }] for the displayed tail
  FULL.forEach((c, i) => { seen[c] += 1; if (i >= START) squares.push({ color: COLORS[c], rings: seen[c] }); });
  const N = squares.length;

  /* ===== Geometry helpers (all positions derived from STRIDE) ===== */
  const sqLeft   = i => PAD_L + i * STRIDE;
  const sqRight  = i => sqLeft(i) + SQ;
  const gapMid   = i => (sqRight(i) + sqLeft(i + 1)) / 2;   // centre of the i-th transition
  const braceX0  = gapMid(0) - BRACE_PAD;
  const braceX1  = gapMid(N - 2) + BRACE_PAD;
  const braceMid = (braceX0 + braceX1) / 2;

  /* ===== SVG element factory ===== */
  const NS = 'http://www.w3.org/2000/svg';
  const el = (tag, attrs = {}, kids = []) => {
    const node = document.createElementNS(NS, tag);
    for (const k in attrs) node.setAttribute(k, attrs[k]);
    for (const c of kids) node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    return node;
  };
  const sup = s => el('tspan', { 'baseline-shift': 'super', 'font-size': '9' }, [s]);
  const sub = s => el('tspan', { 'baseline-shift': 'sub',   'font-size': '10' }, [s]);
  const it  = s => el('tspan', { 'font-style': 'italic' }, [s]);

  /* ===== Builders, one per layer ===== */
  function squareEls(i, { color, rings }) {
    const out = [el('rect', { x: sqLeft(i), y: 0, width: SQ, height: SQ, rx: SQ_R, fill: color })];
    for (let k = 0; k < rings; k++) {                 // k concentric hollow rings, nested inward
      const inset = RING_INSET + k * RING_STEP;
      out.push(el('rect', {
        x: sqLeft(i) + inset, y: inset, width: SQ - 2 * inset, height: SQ - 2 * inset,
        rx: Math.max(1, SQ_R - inset), fill: 'none', stroke: 'white', 'stroke-width': 1,
      }));
    }
    return out;
  }
  const arcEl = i => el('path', {                     // transition arc from sq i down to sq i+1
    d: `M${sqRight(i)},${Y_ROW} Q${gapMid(i)},${Y_ARC} ${sqLeft(i + 1)},${Y_ROW}`,
    'marker-end': 'url(#rlvg-archead)',
  });
  function rewardEl(i) {                              // γ^i R_{t+i+1}, centred under the i-th arc
    const kids = [];
    if (i >= 1) kids.push('γ');
    if (i >= 2) kids.push(sup(String(i)));
    kids.push(it('R'), sub('t+' + (i + 1)));
    return el('text', { x: gapMid(i), y: Y_REW, 'text-anchor': 'middle' }, kids);
  }
  const bracePath = (x0, x1, y, r) => {              // upward-opening underbrace, nub at centre
    const xm = (x0 + x1) / 2;
    return `M${x0},${y} q0,${r} ${r},${r} L${xm - r},${y + r} q${r},0 ${r},${r}`
         + ` q0,${-r} ${r},${-r} L${x1 - r},${y + r} q${r},0 ${r},${-r}`;
  };

  /* ===== Assemble ===== */
  const W = sqRight(N - 1), H = Y_LABEL + 6;
  const svg = el('svg', {
    width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: 'img',
    'aria-label': 'Unrolled trajectory tail (stops 6-11): reward terms under transition arcs, under-braced as G_t = v(s_t).',
    style: "display:block; margin:1.5rem auto; overflow:visible;"
         + "font-family:'Roboto Mono',ui-monospace,'SFMono-Regular',Menlo,monospace;"
         + "color:var(--text-color,#1f2328);",
  });

  // arrowhead marker (triangle inset 1px inside a padded box -> corners never clip)
  svg.appendChild(el('defs', {}, [
    el('marker', {
      id: 'rlvg-archead', markerUnits: 'userSpaceOnUse', viewBox: '0 0 12 12',
      refX: 11, refY: 6, markerWidth: 8, markerHeight: 8, orient: 'auto', overflow: 'visible',
    }, [el('path', { d: 'M1,1 L11,6 L1,11 z', fill: 'currentColor', stroke: 'none' })]),
  ]));

  squares.forEach((sq, i) => squareEls(i, sq).forEach(e => svg.appendChild(e)));

  const arcs = el('g', { stroke: 'currentColor', fill: 'none', 'stroke-width': 1.2 });
  for (let i = 0; i < N - 1; i++) arcs.appendChild(arcEl(i));
  svg.appendChild(arcs);

  const rewards = el('g', { 'font-size': 13, fill: 'currentColor' });
  for (let i = 0; i < N - 1; i++) rewards.appendChild(rewardEl(i));
  svg.appendChild(rewards);

  svg.appendChild(el('path', {
    d: bracePath(braceX0, braceX1, Y_BRACE, BRACE_R),
    stroke: 'currentColor', fill: 'none', 'stroke-width': 1.2,
  }));

  // v(s_t) =   — s_t is coloured like the first square (it labels that square)
  const sColor = squares[0].color;
  const stEl = el('tspan', { fill: sColor }, [it('s'), sub('t')]);   // keep a handle so we can measure it
  svg.appendChild(el('text', { x: braceX0 - 4, y: Y_LABEL, 'text-anchor': 'end', 'font-size': 13, fill: 'currentColor' },
    [it('v'), '(', stEl, ') =']));
  svg.appendChild(el('text', { x: braceMid, y: Y_LABEL, 'text-anchor': 'middle', 'font-size': 13, fill: 'currentColor' },
    [it('G'), sub('t')]));

  // Leader line from s_t up to the bottom-centre of the first square. We measure s_t's
  // actual position with getBBox (no hardcoded glyph offset), re-running once the web font loads.
  const lead = el('line', { stroke: sColor, 'stroke-width': 1.2, 'stroke-dasharray': '2 2' });
  svg.appendChild(lead);
  const placeLead = () => {
    const b = stEl.getBBox();
    lead.setAttribute('x1', b.x + b.width / 2);
    lead.setAttribute('y1', b.y);                       // top of s_t
    lead.setAttribute('x2', sqLeft(0) + SQ / 2);        // bottom-centre of the first square
    lead.setAttribute('y2', Y_ROW);
  };

  root.appendChild(svg);
  placeLead();                                          // measure now...
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(placeLead);  // ...and again after the font settles
})();
</script>

However, if it weren't the start of the learning process, there may already be several prior episodes where we saw the same <span style="color: #f5c1a8"><b>state</b></span> featured:

#### Previous Episodes:

<style>
  .rl-ep { display: block; margin: 1rem 0; max-width: 100%; height: auto; overflow: visible;
           color: var(--text-color, #1f2328); }  /* arrows inherit the page font, like the reference chain */
  .rl-ep-glow { filter: drop-shadow(0 0 3px #ea7c61) drop-shadow(0 0 6px #ea7c61); }  /* halo on the highlighted COLORS[5] squares */
</style>

<div id="rl-eps" markdown="0"></div>

<script>
(function () {
  "use strict";
  const host = document.getElementById('rl-eps');
  if (!host || host.dataset.init) return;
  host.dataset.init = '1';

  // coolwarm(12)[2:10] — same palette as every other diagram.
  const COLORS = ['#7598f6', '#95b7ff', '#b4cdfa', '#d1dae9', '#e8d6cc', '#f5c1a8', '#f6a384', '#ea7c61'];
  const EPISODES = [
    [0, 1, 2, 3, 4, 5, 6],
    [0, 1, 2, 3, 4, 5, 6, 7, 6, 7],
    [0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4],
  ];
  const HILITE = 5;   // the state whose return G we brace (first COLORS[5])

  /* ===== shared helpers ===== */
  const NS = 'http://www.w3.org/2000/svg';
  const el = (tag, attrs = {}, kids = []) => {
    const node = document.createElementNS(NS, tag);
    for (const k in attrs) node.setAttribute(k, attrs[k]);
    for (const c of kids) node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    return node;
  };
  const bracePath = (x0, x1, y, r) => {              // upward-opening underbrace, nub at centre
    const xm = (x0 + x1) / 2;
    return `M${x0},${y} q0,${r} ${r},${r} L${xm - r},${y + r} q${r},0 ${r},${r}`
         + ` q0,${-r} ${r},${-r} L${x1 - r},${y + r} q${r},0 ${r},${-r}`;
  };

  /* ===== layout knobs — shared by every chain; positions all derive from STRIDE ===== */
  const SQ = 30, GAP = 18, PAD = 2, STRIDE = SQ + GAP;
  const SQ_R = 6, BRACE_R = 6;
  const Y_ROW = SQ, Y_ARROW = SQ / 2;                // arrows sit at the row's mid-height
  const Y_BRACE = Y_ROW + 10;                        // top of the first brace
  const LABEL_DROP = 2 * BRACE_R + 15;               // brace top -> its "G" baseline
  const BLOCK = LABEL_DROP - 6;                       // vertical pitch between stacked braces

  const sqLeft  = i => PAD + i * STRIDE;
  const sqRight = i => sqLeft(i) + SQ;
  const gapMid  = i => (sqRight(i) + sqLeft(i + 1)) / 2;   // centre of the i-th transition

  function renderChain(seq) {
    const n = seq.length;
    const occ = seq.reduce((a, c, i) => (c === HILITE ? a.concat(i) : a), []);  // every highlighted state
    const W = sqRight(n - 1) + PAD;
    const H = Y_BRACE + (occ.length - 1) * BLOCK + LABEL_DROP + 6;              // room for all stacked braces

    const svg = el('svg', { class: 'rl-ep', width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: 'img',
      'aria-label': 'Episode trajectory; one return G is braced from each visit of the highlighted state to the end.' });

    // squares (the highlighted COLORS[5] state gets a glow)
    seq.forEach((c, i) => {
      const r = el('rect', { x: sqLeft(i), y: 0, width: SQ, height: SQ, rx: SQ_R, fill: COLORS[c] });
      if (c === HILITE) r.setAttribute('class', 'rl-ep-glow');
      svg.appendChild(r);
    });

    // → arrows between adjacent squares (page font + size to match the reference chain)
    for (let i = 0; i < n - 1; i++)
      svg.appendChild(el('text', { x: gapMid(i), y: Y_ARROW, 'text-anchor': 'middle',
        'dominant-baseline': 'central', 'font-size': 14, fill: 'currentColor' }, ['→']));

    // one underbrace + G per visit of the highlighted state: from that visit to the end, stacked downward
    const MONO = "'Roboto Mono', ui-monospace, 'SFMono-Regular', Menlo, monospace";
    occ.forEach((idx, j) => {
      const x0 = gapMid(idx), x1 = sqRight(n - 1), yTop = Y_BRACE + j * BLOCK;
      svg.appendChild(el('path', { d: bracePath(x0, x1, yTop, BRACE_R),
        stroke: 'currentColor', fill: 'none', 'stroke-width': 1.2 }));
      svg.appendChild(el('text', { x: (x0 + x1) / 2, y: yTop + LABEL_DROP, 'text-anchor': 'middle',
        'font-family': MONO, 'font-size': 14, 'font-style': 'italic', fill: 'currentColor' }, ['G']));
    });

    host.appendChild(svg);
  }

  EPISODES.forEach(renderChain);
})();
</script>

So this means that you have to keep a running count of how many times you've updated $v(s)$ for each $s$, so that you can scale updates properly such that $v(s)$ always reflects the mean. Specifically, this means that everytime you see an episode suffix beginning with state $s$, you do:
1. $n(s) \leftarrow n(s) + 1$
2. $v(s) \leftarrow v(s) + \frac{1}{n(s)}(G - v(s))$

In non-stationary problems, it can sometimes also be useful to just prioritize more recent values and let old values decay, so we just do:
- $v(s) \leftarrow v(s) + \alpha(G - v(s))$

## 4.2. Sampling Actions: Temporal-Difference RL (TD Learning)

TD-Learning is conceptually the same as MC Learning: it's model-free, so we use sampling to try and estimate the mean value of states. The one key difference is that MC uses full trajectories / episodes, while TD-Learning uses current value estimates of successor states, just like in the Bellman Equation for MDPs. This means that TD-Learning can also be used in non-terminating environments, whereas MC can only be applied in episodic environments.

### 4.2.1. TD(0)

$$
\begin{align*}
V(S_t) \leftarrow V(S_t) + \alpha (\underbrace{\underbrace{R_{t + 1} + \gamma V(S_{t+1})}_{\textstyle \text{TD target}} - V(S_t)}_{\textstyle \delta_t = \text{TD error}})
\end{align*}
$$

Pictorially:

<div id="rl-td-chain" markdown="0"></div>

<script>
(function () {
  "use strict";
  const host = document.getElementById('rl-td-chain');
  if (!host || host.dataset.init) return;
  host.dataset.init = '1';

  // COLORS[0..6] then a tail of greys stepping toward white. The one-step TD(0) backup is
  // drawn on the S_t -> S_{t+1} transition; the greys are the rest of the episode TD never rolls out.
  const FILLS = ['#7598f6', '#95b7ff', '#b4cdfa', '#d1dae9', '#e8d6cc', '#f5c1a8', '#f6a384',
                 '#e6e6e6', '#f0f0f0', '#fafafa'];
  const HILITE = '#f5c1a8';   // COLORS[5] = S_t (glows)
  const TD = 5;               // transition 5 -> 6 is the one-step backup

  const NS = 'http://www.w3.org/2000/svg';
  const el = (tag, attrs = {}, kids = []) => {
    const node = document.createElementNS(NS, tag);
    for (const k in attrs) node.setAttribute(k, attrs[k]);
    for (const c of kids) node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    return node;
  };
  const MONO = "'Roboto Mono', ui-monospace, 'SFMono-Regular', Menlo, monospace";
  const it  = s => el('tspan', { 'font-style': 'italic' }, [s]);
  const sub = (s, sz) => el('tspan', { 'baseline-shift': 'sub', 'font-size': sz || 9 }, [s]);

  // layout — vertical bands stacked under the row
  const SQ = 30, GAP = 18, PAD = 2, STRIDE = SQ + GAP, SQ_R = 6, ARC_DROP = 16;
  const Y_ROW = SQ;                    // square bottom (arc + leaders start here)
  const Y_ARC = Y_ROW + ARC_DROP;      // arc control point
  const Y_REW = Y_ARC + 5;             // R_{t+1} baseline (just below the arc)
  const Y_VLABEL = Y_REW + 22;         // V(...) baseline
  const sqLeft  = i => PAD + i * STRIDE;
  const sqRight = i => sqLeft(i) + SQ;
  const sqMid   = i => sqLeft(i) + SQ / 2;
  const gapMid  = i => (sqRight(i) + sqLeft(i + 1)) / 2;

  const n = FILLS.length, W = sqRight(n - 1) + PAD, H = Y_VLABEL + 8;
  const svg = el('svg', { class: 'rl-ep', width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: 'img',
    'aria-label': 'TD(0): one-step backup R_{t+1} from S_t to S_{t+1}, with the rest of the episode greyed out.' });

  // arrowhead marker for the curved TD arc (triangle inset inside a padded box -> no clipping)
  svg.appendChild(el('defs', {}, [
    el('marker', { id: 'rltd-archead', markerUnits: 'userSpaceOnUse', viewBox: '0 0 12 12',
      refX: 11, refY: 6, markerWidth: 8, markerHeight: 8, orient: 'auto', overflow: 'visible' },
      [el('path', { d: 'M1,1 L11,6 L1,11 z', fill: 'currentColor', stroke: 'none' })]),
  ]));

  // squares (S_t glows)
  FILLS.forEach((f, i) => {
    const r = el('rect', { x: sqLeft(i), y: 0, width: SQ, height: SQ, rx: SQ_R, fill: f });
    if (f === HILITE) r.setAttribute('class', 'rl-ep-glow');
    svg.appendChild(r);
  });

  // straight → for every transition except the TD step (curved arc) and the grey tail
  // (no arrows between grey squares — i.e. transitions starting at the first grey, i >= n-3)
  for (let i = 0; i < n - 1; i++) {
    if (i === TD || i >= n - 3) continue;
    svg.appendChild(el('text', { x: gapMid(i), y: SQ / 2, 'text-anchor': 'middle',
      'dominant-baseline': 'central', 'font-size': 14, fill: 'currentColor' }, ['→']));
  }

  // curved arc on S_t -> S_{t+1}, labelled R_{t+1}
  svg.appendChild(el('path', { d: `M${sqRight(TD)},${Y_ROW} Q${gapMid(TD)},${Y_ARC} ${sqLeft(TD + 1)},${Y_ROW}`,
    stroke: 'currentColor', fill: 'none', 'stroke-width': 1.2, 'marker-end': 'url(#rltd-archead)' }));
  svg.appendChild(el('text', { x: gapMid(TD), y: Y_REW, 'text-anchor': 'middle', 'font-family': MONO,
    'font-size': 12, fill: 'currentColor' }, [it('R'), sub('t+1', 8)]));

  // V(S_t) under [5] and V(S_{t+1}) under [6]; each has a dashed leader in its square's colour
  function vLabel(i, subText) {
    const cx = sqMid(i), col = FILLS[i];
    svg.appendChild(el('line', { x1: cx, y1: Y_ROW, x2: cx, y2: Y_VLABEL - 12,
      stroke: col, 'stroke-width': 1.2, 'stroke-dasharray': '2 2' }));
    const state = el('tspan', { fill: col }, [it('S'), sub(subText, 8)]);   // S_t coloured like its square
    svg.appendChild(el('text', { x: cx, y: Y_VLABEL, 'text-anchor': 'middle', 'font-family': MONO,
    'font-weight': 700,
    'font-size': 11, fill: 'currentColor' }, [it('V'), '(', state, ')']));
  }
  vLabel(TD, 't');
  vLabel(TD + 1, 't+1');

  // "no need complete trajectory" superposed over the grey tail (white halo for legibility)
  const gL = sqLeft(n - 3), gR = sqRight(n - 1), gcx = (gL + gR) / 2;
  svg.appendChild(el('text', { x: gcx, y: 19, 'text-anchor': 'middle', 'font-style': 'italic',
    'font-size': 11, fill: '#444', stroke: '#fff', 'stroke-width': 2.5, 'paint-order': 'stroke',
    'stroke-linejoin': 'round' }, [
      el('tspan', { x: gcx }, ['no need complete trajectory']),
    ]));

  host.appendChild(svg);
})();
</script>

## 4.3. MC vs TD

- MC has better convergence properties (it will converge to the true values even with function approximators for $V$)
- MC does not assume the Markov property, whereas TD does (TD implicitly builds an MDP)
- TD is usually much more efficient

## 4.4. MC vs TD vs DP

So far we've seen Dynamic Programming (Policy / Value Iteration), MC Learning, and TD Learning. They differ along 2 axes:
- Sampling vs no sampling ("full backups")
- Shallow vs Deep backups

<svg width="500" height="296" viewBox="0 0 500 296" role="img"
     aria-label="Two-axis chart (transposed): horizontal = backup depth (shallow to deep), vertical = backup width (sample to full). TD = sample+shallow, MC = sample+deep, DP = full+shallow, Exhaustive Search = full+deep."
     style="display:block; margin:1.5rem auto; max-width:100%; height:auto; overflow:visible;
            font-family:'Roboto Mono',ui-monospace,'SFMono-Regular',Menlo,monospace;
            color:var(--text-color,#1f2328);"
     markdown="0">
  <defs>
    <!-- orient=auto-start-reverse lets one marker serve both ends of a double-headed arrow -->
    <marker id="rl2ax-head" markerUnits="userSpaceOnUse" viewBox="0 0 12 12"
            refX="11" refY="6" markerWidth="8" markerHeight="8" orient="auto-start-reverse" overflow="visible">
      <path d="M1,1 L11,6 L1,11 z" fill="#9aa0a6" stroke="none"/>
    </marker>
  </defs>

  <!-- two crossing double-ended axes (no border / grid) -->
  <g stroke="#9aa0a6" stroke-width="1.2" marker-start="url(#rl2ax-head)" marker-end="url(#rl2ax-head)">
    <line x1="60"  y1="152" x2="440" y2="152"/>   <!-- depth:  Shallow <-> Deep        -->
    <line x1="250" y1="44"  x2="250" y2="260"/>   <!-- width:  Sample <-> Full backups -->
  </g>

  <!-- axis-end labels -->
  <g font-size="12" fill="#6b7280">
    <text x="52"  y="156" text-anchor="end">Shallow</text>
    <text x="448" y="156" text-anchor="start">Deep</text>
    <text x="250" y="34"  text-anchor="middle">Full backups</text>
    <text x="250" y="278" text-anchor="middle">Sample backups</text>
  </g>

  <!-- methods, one per quadrant (flipped vertically: full backups now on top) -->
  <g text-anchor="middle" font-size="14" font-weight="700" fill="currentColor">
    <text x="120" y="99">DP</text>                  <!-- full   + shallow -->
    <text x="380" y="99">Exhaustive Search</text>  <!-- full   + deep    -->
    <text x="120" y="215">TD</text>                 <!-- sample + shallow -->
    <text x="380" y="215">MC</text>                 <!-- sample + deep    -->
  </g>
</svg>

## 4.5. TD($\lambda$)

TD($\lambda$) is a way of interpolating between TD(0) and MC. $\lambda$ specifies how many real steps to take, before imputing the rest with current estimates of $V$. You can imagine that TD($\infty$) is equivalent to MC.

### $\lambda$-return

There is a concept known as $\lambda$-return, where we try to take the geometric mean between the returns computed via TD(0), TD(1), ..., TD($\infty$). This does not seem immediately useful, so I'm skipping this.

# 5. How To Sample

In the previous section, we arrived at a clear idea of how we may use sampled episodes / trajectories to update our $v$ values. In this section, we'll explore **how** those episodes are sampled. After all, in real RL problems, the experience we get is only what the agent has actually done.

## 5.1. The Problem with $V$.

Previously, we already **had** the episode / partial episode samples. We already knew which state $$S_{t + 1}$$ followed the current state $$S_t$$. When we're **generating** these trajectories, we need to know how to act. Most of the time, our action will be greedy, which means we need to take:


$$
\begin{align*}
a = \operatorname*{argmax}\limits_{a \in \mathcal{A}} \left(R_{s + 1} + \gamma \mathbb{E} [V_{s + 1}]\right)
\end{align*}
$$


The problem with this is that $$\mathbb{E} [V_{s + 1}]$$ requires a model: 


$$
\begin{align*}
\mathbb{E} [V_{s + 1}] = \mathcal{P}_{[s:]}^a \cdot v
\end{align*}
$$

And we don't have a model! So to solve this, we use $Q$ instead. This way, we can offload the expectation part into $Q(s, a)$.

## 5.2. Explore vs Exploit

The next idea is leaving room for exploration. If you only exploit the best action you've seen so far, you're susceptible to never knowing any better option. Therefore, we leave some room for exploration in the policy:


$$
\begin{align*}
\pi(a \mid s) &= \begin{cases}
\operatorname*{argmax}\limits_{a \in \mathcal{A}} Q(s, a) & \text{ with prob } 1 - \epsilon \\
\text{random } & \text{ with prob } \epsilon
\end{cases}
\end{align*}
$$


> Epsilon-greedy Q values and policy still do guarantee improvements (since Epsilon-greedy is just a mixture of greedy and uniform-random; the uniform-random bit improves iff $q$ values improves, which it does, since the other part is greedy)

### Implementation Notes

In practice, we slowly decay $\epsilon \rightarrow 0$, so something like $\epsilon = 1/k$ where $k$ is the number of steps, or $\epsilon = \frac{1 + L}{k + L}$ where $L$ is some large integer.

## 5.3. Online TD-Learning: SARSA

Previously, we saw that TD-Learning has this update rule:


$$
\begin{align*}
V(S_t) \leftarrow V(S_t) + \alpha (\underbrace{\underbrace{R_{t + 1} + \gamma V(S_{t+1})}_{\textstyle \text{TD target}} - V(S_t)}_{\textstyle \delta_t = \text{TD error}})
\end{align*}
$$


The above requires us to know $$S_t, R_{t + 1}, S_{t + 1}$$. In replacing $V$ with $Q$, we now have:


$$
\begin{align*}
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha (\underbrace{\underbrace{R_{t + 1} + \gamma Q(S_{t+1}, A_{t + 1})}_{\textstyle \text{TD target}} - Q(S_t, A_{t})}_{\textstyle \delta_t = \text{TD error}})
\end{align*}
$$


Observe that this requires us to know $$S_t, A_t, R_{t + 1}, S_{t + 1}, A_{t + 1}$$, hence the name SARSA. And similar to TD-$\lambda$, we can also interpolate between TD and MC in the $Q$ context by deciding how many actual steps to sample; this is known as "SARSA($\lambda$)".

## 5.4. Off-Policy Learning

What if you wanted to evaluate your $V$ or $Q$ values according to $\pi$, but the episodes you have were generated while following some other behavior policy $\mu$? This is mostly a question of how to use already-existing data (from other robots, humans, etc.).

### 5.4.1. Importance Sampling

The problem with this is that obviously the distributions $\pi$ and $\mu$ are not the same, which means that your expectations are also not the same. Here, we introduce the idea of **Importance Sampling**, which applies scaling factors to the terms in the TD Update Equation so morph $$\mathbb{E}_{\sim \mu}$$ into $$\mathbb{E}_{\sim \pi}$$:

$$
\begin{align*}
V (S_t) \leftarrow V (S_{t + 1}) + \alpha \left( \frac{\pi (A_t \mid S_t)}{\mu (A_t \mid S_t)} \left( R_{t + 1} + \gamma V(S_{t + 1}) \right) - V(S_t) \right)
\end{align*}
$$

But obviously you may not know what $\mu(A_t \mid S_t)$ is. If you just found a dataset of trajectories lying around, you definitely won't know what $\mu$ was.

### 5.4.2. Q-Learning

So we don't use Importance Sampling, we use Q-Learning (a variant of SARSA)!

Unfortunately, Q-Learning doesn't address the idea of how we can use found data generated from an unknown policy. I suspect you can use classic TD-learning to learn your $V$ values and back out a policy from there that you can then use to warm-start Q-learning on your own generated data, but this is out of scope.

What Q-Learning does is:
1. We sample an action $$A_{t + 1} \sim \mu(A_t \mid S_t)$$ to actually take while exploring
2. We also sample an action $$A' \sim \pi(A_t \mid S_t)$$ to use to do our Q updates.

<style>
  .rlt-agent { font-size: 30px; pointer-events: none; will-change: transform; }
</style>

<figure class="rl-tree" id="rl-traj" markdown="0">
  <svg viewBox="0 -60 850 400" role="img" aria-label="Animation: a robot follows a trajectory from state S_t through action A_{t+1} to state S_{t+1}, where a duck (target policy A') joins the robot (behaviour policy A_{t+1}).">
    <defs>
      <marker id="rlt-mk-grey" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" fill="#b0b0b0"/>
      </marker>
      <marker id="rlt-mk-e1" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
        <path id="rlt-e1-head" d="M0,0 L10,5 L0,10 z" fill="#b0b0b0"/>
      </marker>
      <marker id="rlt-mk-e2" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
        <path id="rlt-e2-head" d="M0,0 L10,5 L0,10 z" fill="#b0b0b0"/>
      </marker>
      <marker id="rlt-mk-e3" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
        <path id="rlt-e3-head" d="M0,0 L10,5 L0,10 z" fill="#b0b0b0"/>
      </marker>
      <marker id="rlt-mk-e4" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
        <path id="rlt-e4-head" d="M0,0 L10,5 L0,10 z" fill="#b0b0b0"/>
      </marker>
    </defs>

    <!-- Edges: S_t -> action nodes (first one is the traversed/active edge) -->
    <path id="rlt-e1" class="rl-arrow" d="M63,215 C148,215 148,110 234,110" marker-end="url(#rlt-mk-e1)"/>
    <path class="rl-arrow" d="M63,215 C148,215 148,320 234,320" marker-end="url(#rlt-mk-grey)"/>

    <!-- Edges: A_{t+1} -> successor states (first one is the traversed/active edge) -->
    <path id="rlt-e2" class="rl-arrow" d="M258,110 C344,110 344,70 429,70" marker-end="url(#rlt-mk-e2)"/>
    <path class="rl-arrow" d="M258,110 C344,110 344,150 429,150" marker-end="url(#rlt-mk-grey)"/>

    <!-- Edge: muted action node -> collapsed successors -->
    <path class="rl-arrow" d="M258,320 C341,320 341,320 424,320" marker-end="url(#rlt-mk-grey)"/>

    <!-- Edges: S_{t+1} -> next action nodes (A', A_{t+1}); both blacken on the split -->
    <path id="rlt-e3" class="rl-arrow" d="M453,70 C538,70 538,30 624,30" marker-end="url(#rlt-mk-e3)"/>
    <path id="rlt-e4" class="rl-arrow" d="M453,70 C538,70 538,110 624,110" marker-end="url(#rlt-mk-e4)"/>

    <!-- Edge: s_n -> collapsed next actions -->
    <path class="rl-arrow" d="M453,150 C536,150 536,150 619,150" marker-end="url(#rlt-mk-grey)"/>

    <!-- Reward label on the S_t -> A_{t+1} edge -->
    <text id="rlt-lbl-r" class="rl-edge" x="138" y="146" text-anchor="middle" style="opacity:0"><tspan class="rl-var">R</tspan><tspan class="rl-sub" baseline-shift="sub">t+1</tspan></text>

    <!-- Root state node: S_t (dark red) -->
    <circle class="rl-node" cx="50" cy="215" r="11" style="fill:#d65344;stroke:#d65344"/>
    <text class="rl-label" x="50" y="246" text-anchor="middle"><tspan class="rl-var">S</tspan><tspan class="rl-sub" baseline-shift="sub">t</tspan></text>

    <!-- Action nodes from S_t (a1 = A_{t+1} animates to dark blue; the other is muted) -->
    <circle id="rlt-n-a1" class="rl-action" cx="245" cy="110" r="11"/>
    <text id="rlt-lbl-a1" class="rl-label" x="245" y="138" text-anchor="middle" style="opacity:0"><tspan class="rl-var">A</tspan><tspan class="rl-sub" baseline-shift="sub">t</tspan></text>
    <circle class="rl-action" cx="245" cy="320" r="11"/>

    <!-- Successor state nodes (s1 = S_{t+1} animates to dark red) -->
    <circle id="rlt-n-s1" class="rl-node" cx="440" cy="70" r="11"/>
    <text id="rlt-lbl-s1" class="rl-label" x="440" y="96" text-anchor="middle" style="opacity:0"><tspan class="rl-var">S</tspan><tspan class="rl-sub" baseline-shift="sub">t+1</tspan></text>
    <circle class="rl-node" cx="440" cy="150" r="11"/>

    <!-- Collapsed successors for the muted action node -->
    <text class="rl-dots" x="440" y="324" text-anchor="middle">...</text>

    <!-- Next action nodes from S_{t+1}: A' (duck, -> yellow) and A_{t+1} (robot, -> dark blue), centred on S_{t+1} -->
    <circle id="rlt-n-n1" class="rl-action" cx="635" cy="30" r="11"/>
    <text id="rlt-lbl-n1" class="rl-label" x="650" y="34" text-anchor="start" style="opacity:0"><tspan class="rl-var">A</tspan>&prime;<tspan dx="6" style="font-weight:400">(<tspan class="rl-var">&pi;</tspan> chose)</tspan></text>
    <circle id="rlt-n-n2" class="rl-action" cx="635" cy="110" r="11"/>
    <text id="rlt-lbl-n2" class="rl-label" x="650" y="114" text-anchor="start" style="opacity:0"><tspan class="rl-var">A</tspan><tspan class="rl-sub" baseline-shift="sub">t+1</tspan><tspan dx="6" style="font-weight:400">(<tspan class="rl-var">&mu;</tspan> chose)</tspan></text>

    <!-- Collapsed next actions for s_n -->
    <text class="rl-dots" x="635" y="154" text-anchor="middle">...</text>

    <!-- Agents (painted last so they sit on top of the nodes) -->
    <text id="rlt-robot" class="rlt-agent" x="50" y="189" text-anchor="middle" dominant-baseline="central">🤖</text>
    <text id="rlt-duck" class="rlt-agent" x="50" y="189" text-anchor="middle" dominant-baseline="central" style="opacity:0">🦆</text>
  </svg>
  <figcaption>🤖 follows the behaviour policy <span class="rl-var">&mu;</span> (sampling <span class="rl-var">A</span><sub>t+1</sub>); 🦆 marks the target policy <span class="rl-var">&pi;</span> (sampling <span class="rl-var">A</span>&prime;). The black path is the trajectory actually taken.</figcaption>
</figure>

This allows us to explore off-$\pi$ while updating our $Q$ values according to $\pi$.

$$
\begin{align*}
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left(R_{t + 1} + \gamma Q(S_{t + 1}, A') - Q(S_t, A_t) \right)
\end{align*}
$$


<script>
(function () {
  "use strict";
  const fig = document.getElementById('rl-traj');
  if (!fig || fig.dataset.init) return;
  fig.dataset.init = '1';

  const $ = sel => fig.querySelector(sel);
  const robot = $('#rlt-robot'), duck = $('#rlt-duck');
  const nA1 = $('#rlt-n-a1'), nS1 = $('#rlt-n-s1'), nN1 = $('#rlt-n-n1'), nN2 = $('#rlt-n-n2');
  const e1 = $('#rlt-e1'), e2 = $('#rlt-e2'), e3 = $('#rlt-e3'), e4 = $('#rlt-e4');
  const e1h = $('#rlt-e1-head'), e2h = $('#rlt-e2-head'), e3h = $('#rlt-e3-head'), e4h = $('#rlt-e4-head');
  const lblR = $('#rlt-lbl-r'), lblA1 = $('#rlt-lbl-a1'), lblS1 = $('#rlt-lbl-s1');
  const lblN1 = $('#rlt-lbl-n1'), lblN2 = $('#rlt-lbl-n2');

  // node centres (SVG user units, matching the static markup above)
  const ROOT = [50, 215], A1 = [245, 110], S1 = [440, 70], N1 = [635, 30], N2 = [635, 110];
  const PERCH = 30;                          // emoji sits this far above a node, perched on top of it
  const OFF = 15;                            // half-gap between the perched robot/duck pair
  const perch = n => [n[0], n[1] - PERCH];
  const pROOT = perch(ROOT), pA1 = perch(A1), pS1 = perch(S1), pN1 = perch(N1), pN2 = perch(N2);
  const pS1L = [pS1[0] - OFF, pS1[1]], pS1R = [pS1[0] + OFF, pS1[1]];

  const lp = (a, b, p) => a + (b - a) * p;
  const at = (el, x, y) => { el.setAttribute('x', x); el.setAttribute('y', y); };
  const clamp = p => Math.max(0, Math.min(1, p));

  // colour ramps: muted (light) -> active (dark)
  const BLUE_L = [192, 211, 245], BLUE_D = [89, 119, 227];   // #c0d3f5 -> #5977e3
  const RED_L  = [241, 202, 182], RED_D  = [214, 83, 68];    // #f1cab6 -> #d65344
  const YELLOW = [255, 200, 50];                             // A' highlight (#ffc832)
  const GREY   = [176, 176, 176], DARK   = [31, 35, 40];     // #b0b0b0 -> ~black
  const mix = (a, b, p) => `rgb(${Math.round(lp(a[0], b[0], p))},${Math.round(lp(a[1], b[1], p))},${Math.round(lp(a[2], b[2], p))})`;
  const paintNode = (el, l, d, p) => { const c = mix(l, d, p); el.style.fill = c; el.style.stroke = c; };
  const paintEdge = (path, head, p) => { const c = mix(GREY, DARK, p); path.style.stroke = c; head.style.fill = c; };

  // ----- timeline (ms) -----
  const HOLD0 = 600, HOP = 700, PAUSE = 350, SHIFT = 600, SETTLE = 350, SPLIT = 850, FINAL = 1800;
  const APEX = 42, DROP = 30;
  const t = {};
  t.hold0  = HOLD0;                  // hover above S_t
  t.hop1   = t.hold0 + HOP;          // robot: hover -> A_{t+1}
  t.pause1 = t.hop1 + PAUSE;         // dwell on A_{t+1}
  t.hop2   = t.pause1 + HOP;         // robot: A_{t+1} -> S_{t+1}
  t.pause2 = t.hop2 + PAUSE;         // dwell on S_{t+1}
  t.shift  = t.pause2 + SHIFT;       // duck drops in, robot slides aside
  t.settle = t.shift + SETTLE;       // hold as a centred pair
  t.split  = t.settle + SPLIT;       // duck -> A', robot -> A_{t+1}
  const TOTAL = t.split + FINAL;     // hold the final frame

  function update(e) {
    // ---- progressive reveals (cumulative; each loop resets cleanly because p=0 below t.* ) ----
    const p1 = e < t.hold0  ? 0 : clamp((e - t.hold0)  / HOP);   // A_{t+1} node + edge1 + R_{t+1}
    paintNode(nA1, BLUE_L, BLUE_D, p1);
    paintEdge(e1, e1h, p1);
    lblR.style.opacity = p1; lblA1.style.opacity = p1;

    const p2 = e < t.pause1 ? 0 : clamp((e - t.pause1) / HOP);   // S_{t+1} node + edge2
    paintNode(nS1, RED_L, RED_D, p2);
    paintEdge(e2, e2h, p2);
    lblS1.style.opacity = p2;

    const p4 = e < t.settle ? 0 : clamp((e - t.settle) / SPLIT); // split: reveal A'/A_{t+1}, colour their nodes + edges
    const p4c = Math.min(1, p4 * 1.6);
    lblN1.style.opacity = p4c;
    lblN2.style.opacity = p4c;
    paintEdge(e4, e4h, p4c);                 // S_{t+1} -> A_{t+1} edge blackens
    paintNode(nN1, BLUE_L, YELLOW, p4c);     // A'      -> yellow   (target policy pi)
    paintNode(nN2, BLUE_L, BLUE_D, p4c);     // A_{t+1} -> dark blue (behaviour policy mu)

    // ---- agent choreography (the emoji always perches on top of its node) ----
    duck.style.opacity = 0;
    if (e < t.hold0) {                                           // perched on top of S_t
      at(robot, pROOT[0], pROOT[1]);
    } else if (e < t.hop1) {                                     // hop onto A_{t+1}
      const p = (e - t.hold0) / HOP;
      at(robot, lp(pROOT[0], pA1[0], p), lp(pROOT[1], pA1[1], p) - APEX * 4 * p * (1 - p));
    } else if (e < t.pause1) {
      at(robot, pA1[0], pA1[1]);
    } else if (e < t.hop2) {                                     // hop onto S_{t+1}
      const p = (e - t.pause1) / HOP;
      at(robot, lp(pA1[0], pS1[0], p), lp(pA1[1], pS1[1], p) - APEX * 4 * p * (1 - p));
    } else if (e < t.pause2) {
      at(robot, pS1[0], pS1[1]);
    } else if (e < t.shift) {                                    // robot slides aside; duck fades in + drops
      const p = (e - t.pause2) / SHIFT;
      at(robot, lp(pS1[0], pS1R[0], p), pS1[1]);
      duck.style.opacity = p; at(duck, lp(pS1[0], pS1L[0], p), pS1L[1] - DROP * (1 - p));
    } else if (e < t.settle) {                                   // perched pair on S_{t+1} (robot right, duck left)
      at(robot, pS1R[0], pS1R[1]); duck.style.opacity = 1; at(duck, pS1L[0], pS1L[1]);
    } else if (e < t.split) {                                    // duck hops to A', robot hops to A_{t+1}
      const p = (e - t.settle) / SPLIT;
      duck.style.opacity = 1;
      at(duck,  lp(pS1L[0], pN1[0], p), lp(pS1L[1], pN1[1], p) - APEX * 4 * p * (1 - p));
      at(robot, lp(pS1R[0], pN2[0], p), lp(pS1R[1], pN2[1], p) - APEX * 4 * p * (1 - p));
    } else {                                                     // hold the final frame
      duck.style.opacity = 1; at(duck, pN1[0], pN1[1]); at(robot, pN2[0], pN2[1]);
    }
  }

  let startTs = null;
  function frame(ts) {
    if (startTs == null) startTs = ts;
    update((ts - startTs) % TOTAL);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
</script>

# 6. Function Approximators

Sometimes, RL problems can be huge, or states can be continuous. So what do we do? Use function approximators! This section will only cover deep networks since they are universal function approximators.

## 6.1. Formulation

Suppose our state $S$ is a vector (hence continuous), then the formulation is simply that $V$ is a deep network that outputs the estimate of $$V_\pi(S)$$.

Our TD updates used to be this:

$$
\begin{align*}
V(S_t) \leftarrow V(S_t) + \alpha (\underbrace{\underbrace{R_{t + 1} + \gamma V(S_{t+1})}_{\textstyle \text{TD target}} - V(S_t)}_{\textstyle \delta_t = \text{TD error}})
\end{align*}
$$

Everything is conceptually the same as before (we update towards the TD target), except that now:
- We use $\hat{V}_w$ instead of $V$ to denote that the value is just an estimate parameterized by $w$ (the weights of the deep network)
- We update $w$, which in turn updates the value generated by $V$

In ML parlance, we have:

#### Error

Something like MSE:

$$
\begin{align*}
\frac{1}{2}
\left(
    R_{t + 1} + \gamma \hat{V}_w(S_{t+1}) - \hat{V}_w(S_t)
)
\right)^2
\end{align*}
$$

#### Gradient

$$
\begin{align*}
\left(
    R_{t + 1} + \gamma \hat{V}_w(S_{t+1}) - \hat{V}_w(S_t)
)
\right) \nabla(- \hat{V}_w(s_t))
\end{align*}
$$

#### Update

Negative Gradient $\times$ learning rate:

$$
\begin{align*}
\Delta w = \alpha
\left(
    R_{t + 1} + \gamma \hat{V}_w(S_{t+1}) - \hat{V}_w(S_t)
)
\right) \nabla(\hat{V}_w(s_t))
\end{align*}
$$

A similar thing can be done for $Q$, but for $Q$, you have several options for the shape of $\hat{Q}$:
1. $Q: \mathbb{R}^\text{state dim} \rightarrow \mathbb{R}^\text{action dim}$ (useful for discrete action space)
2. $Q: \mathbb{R}^\text{state dim + action dim} \rightarrow \mathbb{R}^1$

## 6.2. Convergence & Efficiency

As you can see, not only are our $\hat{V}_w$ values changing, the way we compute those $\hat{V}_w$ values (i.e. $w$) is also changing (moving target). This can cause convergence troubles. Moreover, doing gradient descent after every sampled SARSA tuple is inefficient. Here's where we solve both problems with Deep-Q Networks.

### 6.2.1. Deep-Q Networks (DQN)

In the below code, you'll see:
- We have `q_net_target`, which is updated much less frequently than `q_net`. This solves the problem where $\hat{Q}_w$ is always moving (moving target problem).
- `q_net_target` is updated softly (interpolated between `q_net` and `q_net_target`).

#### Implementation
```python
def soft_update_q_target(q_net, q_net_target, tau):
    """
    Desc:                   Architecture-agnostic helper function to copy weights.

    @param q_net:           (type[nn.Module]) The source net
    @param q_net_target:    (type[nn.Module]) The dest net
    """
    for param, target_param in zip(q_net.parameters(), q_net_target.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

q_net = QNet(...parameters)
q_net_target = copy.deepcopy(q_net)

for _ in range(num_train_episodes):
    episode_reward = 0.0
    s, _ = env.reset()
    for _ in range(max_steps_per_episode):
        # Step 1) we sample a transition to add to replay buffer.
        a = select_action_eps_greedy(env, q_net, s, eps)
        s_next, r = env.step(a)

        episode_reward += r
        replay_buffer.append((s, a, r, s_next)) # we will compute a_next on the fly

        # Step 2) sample a mini batch and fit the Q network
        if len(replay_buffer) < batch_size:
            continue

        batch_s, batch_a, batch_s_next, batch_r, batch_terminal = \
            sample_experience_batch(replay_buffer, batch_size)

        # Compute Q targets (in this case q_net goes from state_dim to action_dim)
        batch_s_next_q_vals = q_net_target(batch_s_next)
        batch_s_next_q_greedy = torch.max(batch_s_next_q_vals, axis=-1).values
        batch_targets = torch.where(
            batch_terminal,
            batch_r,
            batch_r + gamma * batch_s_next_q_greedy
        )

        # Update
        q_net.train(True)
        batch_s_q_vals = q_net(batch_s)[torch.arange(batch_size), batch_a]
        loss = F.mse_loss(batch_s_q_vals, batch_targets.detach())
        q_net_opt.zero_grad()
        loss.backward()
        q_net_opt.step()

        # Update q_net_target and evaluate
        if total_steps % q_target_update_freq == 0:
            soft_update_q_target(q_net, q_net_target, tau)

```

# 7. Policy Gradient

So far, everything we've seen revolves around Value-based methods, but Value-based methods may lose out to Policy-based methods because:
- Value-based methods require us to compute $V$ or $Q$ for our state, which may be much more tedious than a policy. For example, in pong, the value of a given state may be hard to predict, but the policy is simple: if ball is above your paddle, go up, and vice versa.
- Computing $$a = \operatorname*{argmax}\limits_{a} q_\star(s, a)$$ in particular may also be hard because $\mathcal{A}$ may be huge.
- Policy-based methods allow you to specify explicit, stochastic policies that are more general than $\epsilon$-greedy.

Sections 7.1. to 7.5. basically speed through what metrics we might use for a policy, what naive methods there are, what better methods there are, and how we might implement those.

## 7.1. Policy Objective Functions

Given some policy $\pi_\theta$ parameterized by weights $\theta$, we want to optimize the policy. We first have to define how to measure the quality of $\pi_\theta$. For that, we have several options, including:
- Using the average **start state value**
- Using the average **state value**
- Using the average **reward per time-step**

In the above, "average" is w.r.t to the stationary distribution of the Markov chain for $\pi_\theta$.

## 7.2. Policy Gradients by Finite Distances

One way of estimating a policy gradient w.r.t to $\pi_\theta$ is by perturbing $\theta$ by small amounts. Below shows how you would perturb $\theta$ in the $k$-th dimension:

$$
\begin{align*}
\frac{\partial J(\theta)}{\partial \theta_k} \approx \frac{J(\theta + \epsilon \textbf{e}_k) - J(\theta)}{\epsilon}
\end{align*}
$$

## 7.3. Analytical Policy Gradients

There's also an analytical way of arriving at a policy gradient, and that's by treating the above metrics ($v$, $r$, $Q$, etc.) as constants w.r.t to $\theta$ (which they are, so I'm not sure why we had to do finite distances in the first place). To do this, I'll introduce a rearrangement of $$\nabla_{\theta} \pi_\theta (s, a)$$ and a **score function**:

$$
\begin{align*}
\nabla_\theta \pi_\theta (s, a) &= \pi_\theta (s, a) \frac{\nabla_\theta \pi_\theta (s, a)}{\pi_\theta (s, a)} \\
&= \pi_\theta (s, a) \underbrace{\nabla_\theta \log \pi_\theta (s, a)}_{\textstyle \text{score function}}
\end{align*}
$$

This formulation is convenient for us because the RHS has a $$\pi_\theta (s, a)$$ factor that helps us take expectations of $$\nabla_\theta \pi_\theta (s, a)$$.


### Example: One-Step MDPs

Let's see the above convenience at play in the case of a one-step MDP, starting in state $$s \sim d(s)$$. Suppose the objective is this:

$$
\begin{align*}
J(\theta) = \mathbb{E}_{\pi_\theta} [r]
\end{align*}
$$

Then plugging in the score function reformulation:

$$
\begin{align*}
J(\theta) &= \sum_{s \in \mathcal{S}} d(s) \sum_{a \in \mathcal{A}} \pi_\theta \left( s, a \right) \mathcal{R}_{s, a} \\
\Rightarrow \nabla_\theta J(\theta) &= \sum_{s \in \mathcal{S}} d(s) \sum_{a \in \mathcal{A}} \pi_\theta \left( s, a \right) \nabla_\theta \log \pi_\theta (s, a) \mathcal{R}_{s, a} \\
&= \mathbb{E}_{\pi_\theta} [ \nabla_\theta \log \pi_\theta (s, a) r]
\end{align*}
$$

## 7.4. Policy Gradient Theorem

The policy gradient theorem simply states that for any of the earlier[ Policy Objective Functions](#71-policy-objective-functions) we mentioned, the above equation for the policy objective function holds true, just that we use $Q$ instead of $r$:

$$
\begin{align*}
\nabla_\theta J(\theta) &= \mathbb{E}_{\pi_\theta} [ \nabla_\theta \log \pi_\theta (s, a) Q_{\pi_\theta}(s, a)]
\end{align*}
$$

> NOTE: This is something I would just memorize, without remembering the (even if just intuitive) derivation

### Advantage Function

In fact, just like how we used $Q$ instead of $r$, we can also just plug in $A$, as in the **Advantage Function**:

$$
\begin{align*}
A_\pi (s, a) &= Q_\pi(s, a) - V_\pi (s)
\end{align*}
$$

Alright, so we have $V$, $Q$, and now we have $A$. They all look like variants of the same thing, and indeed, in the end, the learnt policy (assuming some form of greedy) is still the same. But the advantage function is useful because it has lower variance.

## 7.5. Implementation: Monte-Carlo Policy Gradient (REINFORCE)

In the above equation, you'll notice that there's a $$\mathbb{E}_{\pi_\theta}$$. But as with every iterative algorithm, we don't actually have to compute that. We just naturally keep an estimate of it through sampling. Here's a rough algorithm:

```python
def REINFORCE(policy, learning_rate):
    policy.theta.init()
    for episode in episode_bank:
        for t in range(0, len(episode)):
            s, a, G_t = get_at_timestep(t, episode)
            loss = -learning_rate * torch.log(policy(s, a)) * G_t
            loss.backward()
```

This algorithm has a number of names - Monte-Carlo Policy Gradient, REINFORCE, [Vanilla Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/vpg.html#pseudocode), etc..

## 7.6. Actor-Critic Methods

REINFORCE is a Monte-Carlo method, meaning that we sampled full trajectories and used ground-truth returns $$G_t$$ as our value function. We can also turn this into a shallow version (i.e. introduce bootstrapping) by introducing our friendly old learnt $$Q_w (s,a)$$. Note that we denote $Q$'s weights as $w$ and $\pi$'s weights as $\theta$. Then, our policy gradient becomes:

$$
\begin{align*}
\mathbb{E} \left[ \nabla_\theta \log \pi_\theta (s, a) Q_w (s, a) \right]
\end{align*}
$$

### Implementation

This means that we can apply pretty much the same algorithms (SARSA / DQN; they're all the same save for some tricks here and there):

```python
def Q_action_critic(env, policy, Q, policy_lr, Q_lr):
    policy.theta.init()
    Q.w.init()
    a = policy.sample_action(env.s)

    for _ in range(however_many_steps):
        s_next, r = env.step(a)
        a_next = policy.sample_action(s_next)

        TD_error = r + Q_lr * Q(s_next, a_next) - Q(s, a)
        policy_grad = -policy_lr * torch.log(policy(s, a)) * Q(s, a)

        TD_error.backward()
        policy_grad.backward()

        a = a_next
        s = s_next
```

You can use all of the same tricks in the $Q$ part of this implementation (e.g. use 2 $Q$ networks, use soft-updates, etc.).

# 8. Beyond Vanilla Policy Gradient

Since I'd like for us to achieve an understanding of RL sufficiently to extend it to LLM training, where PPO and GRPO are common, I will cover them here.

## 8.1. Proximal Policy Optimization

A 2017 method developed by John Schulman based on his earlier algorithm, Trust Region Policy Optimzation (TRPO). Both PPO and TRPO are motivated by the same question - how big of an update step can you take to improve performance without overstepping and causing performance collapse? For that, we have to introduce a new objective function:

$$
\begin{align*}
L (s, a, \theta_k, \theta) &= \min \left( \frac{\pi_\theta (a \mid s)}{\pi_{\theta_k} (a \mid s)} A_{\pi_{\theta_k}} (s, a), \text{clip}\left(  \frac{\pi_\theta (a \mid s)}{\pi_{\theta_k} (a \mid s)} A_{\pi_{\theta_k}}, 1 - \epsilon, 1 + \epsilon \right) A_{\pi_{\theta_k}} (s, a) \right)
\end{align*}
$$

where $\theta_k$ is the old policy and $\theta$ is the updated policy.

<script>
(function () {
  "use strict";

  /* ============================================================
   * 1. Gridworld definition  (single source of truth)
   * ============================================================ */
  const N = 4, GAMMA = 1, REWARD = -1;
  const TERMINALS = new Set([0, N * N - 1]);          // top-left & bottom-right

  const toRC     = s => [Math.floor(s / N), s % N];
  const toId     = (r, c) => r * N + c;
  const inBounds = (r, c) => r >= 0 && r < N && c >= 0 && c < N;

  // The four moves. This one table drives BOTH the dynamics and the on-screen
  // arrow directions (dr = row delta = screen-y, dc = col delta = screen-x).
  const MOVES = [
    { name: "up",    dr: -1, dc:  0 },
    { name: "down",  dr:  1, dc:  0 },
    { name: "left",  dr:  0, dc: -1 },
    { name: "right", dr:  0, dc:  1 },
  ];

  // In-bounds neighbours of s: [{ move, state }]
  function neighbours(s) {
    const [r, c] = toRC(s);
    return MOVES
      .map(m => ({ move: m, r: r + m.dr, c: c + m.dc }))
      .filter(n => inBounds(n.r, n.c))
      .map(n => ({ move: n.move, state: toId(n.r, n.c) }));
  }

  /* ============================================================
   * 2. Policy evaluation -> converged value function v_pi
   * ============================================================ */
  const moveResult = (s, m) => {                       // off-grid move = stay put
    const [r, c] = toRC(s);
    return inBounds(r + m.dr, c + m.dc) ? toId(r + m.dr, c + m.dc) : s;
  };

  function sweep(v) {                                  // one synchronous Bellman backup
    const next = v.slice();
    for (let s = 0; s < N * N; s++) {
      if (TERMINALS.has(s)) { next[s] = 0; continue; }
      next[s] = MOVES.reduce(
        (acc, m) => acc + (REWARD + GAMMA * v[moveResult(s, m)]) / MOVES.length, 0);
    }
    return next;
  }

  function evaluatePolicy() {                          // iterate to (near-exact) convergence
    let v = new Array(N * N).fill(0);
    for (let k = 0; k < 1000; k++) {
      const next = sweep(v);
      const delta = Math.max(...next.map((x, i) => Math.abs(x - v[i])));
      v = next;
      if (delta < 1e-9) break;
    }
    return v;
  }

  /* ============================================================
   * 3. Greedy policy: best (least-negative) neighbour(s), ties included
   * ============================================================ */
  const TIE_EPS = 1e-6;
  function greedyMoves(v, s) {
    const adj  = neighbours(s);
    const best = Math.max(...adj.map(n => v[n.state]));
    return adj.filter(n => v[n.state] >= best - TIE_EPS).map(n => n.move);
  }

  /* ============================================================
   * 4. Colour scale (identical mapping to the animation above)
   * ============================================================ */
  const PALETTE = ['#dddddd','#e1dad7','#e5d8d1','#e9d5ca','#ecd2c4','#efcebd','#f2cbb7','#f4c6b0','#f5c2aa','#f6bda3','#f7b89c','#f7b295','#f7ad8f','#f7a788','#f5a081','#f49a7b','#f29374','#f08c6e','#ee8468','#eb7d61','#e7755b','#e46d55','#e0654f','#db5c4a','#d75344','#d24a3f','#cc403a','#c63535','#c12a30','#ba172b','#b40426'];
  const palIndex = (v, vmin) => Math.round((vmin < 0 ? Math.min(1, v / vmin) : 0) * (PALETTE.length - 1));

  /* ============================================================
   * 5. Render the grid cells (value text + background colour)
   * ============================================================ */
  function renderCells(board, v, vmin) {
    for (let s = 0; s < N * N; s++) {
      const cell = document.createElement("div");
      cell.className = "rl-cell" + (TERMINALS.has(s) ? " rl-terminal" : "");
      const i = palIndex(v[s], vmin);
      cell.style.backgroundColor = PALETTE[i];
      cell.style.color = i >= 18 ? "#ffffff" : "#1f2328";
      cell.textContent = v[s] === 0 ? "0" : v[s].toFixed(1);
      board.appendChild(cell);
    }
  }

  /* ============================================================
   * 6. Arrow overlay  (one black arrow per greedy move)
   * ============================================================ */
  const VB = 400, CELL = VB / N, HALF = CELL / 2;      // 100 units per cell
  const START = 25, END = 70;                          // shaft endpoints, in units from cell centre
  const SVG_NS = "http://www.w3.org/2000/svg";

  const svgEl = (tag, attrs) => {
    const el = document.createElementNS(SVG_NS, tag);
    for (const k in attrs) el.setAttribute(k, attrs[k]);
    return el;
  };
  const centre = s => { const [r, c] = toRC(s); return [c * CELL + HALF, r * CELL + HALF]; };

  function arrowPath(s, m) {                           // shaft from near s's centre across the gap toward the neighbour
    const [cx, cy] = centre(s);
    return `M${cx + m.dc * START},${cy + m.dr * START} L${cx + m.dc * END},${cy + m.dr * END}`;
  }

  const arrowMarker = (markerId) => {                  // sharp black arrowhead
    // 10-long x 10-wide triangle inset by 1 unit inside a padded 12x12 box, so no corner
    // sits on the viewBox boundary -> never clipped, independent of marker overflow handling.
    const marker = svgEl("marker", { id: markerId, markerUnits: "userSpaceOnUse",
      viewBox: "0 0 12 12", refX: 11, refY: 6, markerWidth: 12, markerHeight: 12,
      orient: "auto-start-reverse", overflow: "visible" });
    marker.appendChild(svgEl("path", { d: "M1,1 L11,6 L1,11 z" }));
    return marker;
  };

  function buildArrowLayer(v, markerId) {
    const svg = svgEl("svg", { class: "rl-greedy-arrows", viewBox: `0 0 ${VB} ${VB}`,
      preserveAspectRatio: "xMidYMid meet", "aria-hidden": "true" });
    const defs = svgEl("defs", {});
    defs.appendChild(arrowMarker(markerId));
    svg.appendChild(defs);
    for (let s = 0; s < N * N; s++) {
      if (TERMINALS.has(s)) continue;
      for (const m of greedyMoves(v, s))
        svg.appendChild(svgEl("path", { d: arrowPath(s, m), "marker-end": `url(#${markerId})` }));
    }
    return svg;
  }

  /* ============================================================
   * 7. Reusable mount: render any value function as cells + greedy arrows
   * ============================================================ */
  const valuesAfter = k => {                            // v_k: k synchronous sweeps from v_0 = 0
    let v = new Array(N * N).fill(0);
    for (let i = 0; i < k; i++) v = sweep(v);
    return v;
  };

  function mountGrid(boardId, v, vmin) {
    const board = document.getElementById(boardId);
    if (!board) return;
    renderCells(board, v, vmin);
    board.appendChild(buildArrowLayer(v, boardId + "-arrowhead"));   // unique marker id per grid
  }

  /* ============================================================
   * 8. Draw the grids (same vmin => identical colour scale across both)
   * ============================================================ */
  const converged = evaluatePolicy();
  const vmin = Math.min(...converged);                  // deepest red = most negative value
  mountGrid("rl-policy-board", converged,      vmin);   // converged v_pi + greedy policy
  mountGrid("rl-ts5-board",    valuesAfter(5), vmin);   // ...the very same, but at timestep k = 5
})();
</script>

