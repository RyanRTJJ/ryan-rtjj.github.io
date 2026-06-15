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

## 1.1 Definitions

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

## 1.2 Types of RL Agents

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

# 2. MDPs

Most RL problems can be converted to MDPs. Let's take a look at an MDP where we happen to have a fully observable environment.

> As it turns out, having a fully observable environment is not make-or-break, as partially observable RL problems can be converted to POMDPs, which can be converted to MDPs. 

## 2.1 Model / Transition table ($\mathcal{P}$)

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

## 2.2 Markov Reward Process

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

## 2.3 Markov Decision Process

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

