# Review 1 — `_posts/2026-07-28-algozoo.md`, the $M_{4,3}$ section

I read the whole post and worked through the $M_{4,3}$ section carefully — pulling frames from all six videos in that section, and re-deriving the model numerically from the $W_{hi}$ and $W_{hh}$ published in the post so I could check the geometry rather than eyeball it.

**One caveat up front:** the post never gives $W_{oh}$, and I couldn't find it in the repo. So where I needed a readout I fit the *best possible* linear readout to the final hidden states. That turns out to be more useful than the real one (see §4), and it reproduces the post's per-ordering accuracy table to within ~0.3pp, so it's clearly close to what the trained model does.

---

## 1. The symmetry noticed three times is one symmetry, and it explains almost everything

The post observes the mirror structure three separate times — "dims `(0,1)` are a mirror of `(3,2)`", "columns `(0,1)` are a flip of `(3,2)`", "rows `(0,1)` are a flip of `(3,2)`" — but treats them as separate curiosities. They're one fact. Let $J$ be the permutation $0\leftrightarrow3,\ 1\leftrightarrow2$. Then, numerically:

$$J W_{hi} = -W_{hi} \quad (\text{err } 0.017), \qquad J W_{hh} J = W_{hh} \quad (\text{max err } 0.019)$$

Since $J$ is a permutation it commutes with ReLU, and these two facts give an exact **equivariance of the whole recurrence**:

$$h_t(-x) = J\,h_t(x)$$

Verified on the real forward pass — mean relative error 1.0%, which is just the weights not being exactly symmetric.

This is the real reason for the "3 major decision groups." For $n=3$, 2nd-argmax *is the median*, and the median's **index** is invariant under $x \mapsto -x$ even though negation reverses the ordering. So `0>2>1` and `1>2>0` aren't grouped because they "happen to share an answer" — they are literally the same point of hidden space up to $J$. The model has discovered and hard-coded the negation symmetry of the task.

The suspicion at line 188 that the "forward/reverse ordering feature" story was "an opinionated guess" was right, but the alternative the post settles on (they share an answer) is weaker than what's actually going on: sharing an answer is a *consequence* of the symmetry, not the cause.

This is also what the paired dimensions *are*: dims `(0,1)` and dims `(3,2)` are the two $J$-images of a single 2D computation. Not two features — one feature, stored twice.

## 2. Answering the open question: why the blocks must be rank 2

Line 273: "there must be some reason the model needs the 2 dimensionality." There is, and it's forced by the task.

$M_{2,2}$ only ever needs to carry **one** scalar past the first step ($x_0$), because the final comparison is $x_0$ vs $x_1$. For median-of-3 you must, after seeing $x_0,x_1$, still be able to answer three different questions about $x_2$: is it above both, below both, or between? That requires carrying **both order statistics** $\max(x_0,x_1)$ and $\min(x_0,x_1)$ — a genuinely 2-dimensional sufficient statistic. $\mathrm{sign}(x_1-x_0)$ alone is not enough.

And that is exactly what $h_1$ carries. Linear regression from $h_1$:

| target | $R^2$ |
|---|---|
| $\max(x_0,x_1)$ | 0.9987 |
| $\min(x_0,x_1)$ | 0.9988 |
| $x_0$ | 0.9997 |
| $x_1$ | 0.9997 |

So the ReLU-1 state is a near-lossless encoding of the *pair*: which of the two pairs of dims is alive tells you the sign of $x_1-x_0$, and the 2D position within that pair recovers both values. The wedge isn't a smeared-out difference feature — it's a faithful 2D chart of $(x_0, x_1)$.

The cost is visible in the conditioning. The four $2\times2$ blocks of $W_{hh}$:

| block | singular values | cond |
|---|---|---|
| `(0,1)<-(0,1)` | 4.026, 0.244 | 16.5 |
| `(0,1)<-(3,2)` | 4.892, 0.204 | 24.0 |
| `(3,2)<-(0,1)` | 4.909, 0.204 | 24.1 |
| `(3,2)<-(3,2)` | 4.025, 0.242 | 16.7 |

All four are invertible, but the second direction is carried at ~5% of the gain of the first. That's the quantitative version of "thin wedge" — the second order statistic is stored, but at a 20:1 disadvantage. That anisotropy is where the fragility comes from.

## 3. The readout is effectively 1-dimensional

The blockquote at line 322 asks whether the symmetry cuts the effective rank roughly in half. It does, exactly, and it can be proven rather than estimated.

For the model to respect the task symmetry, the logits must be unchanged under $x \mapsto -x$, i.e. $W_{oh}J = W_{oh}$. The fitted readout satisfies this to 3 decimals (col 0 ≈ col 3, col 1 ≈ col 2). Given $W_{oh}J = W_{oh}$:

$$W_{oh}h = W_{oh}\,\tfrac{h + Jh}{2}$$

so the logits depend on $h$ **only** through the two $J$-invariant coordinates $s = h_0 + h_3$ and $u = h_1 + h_2$. A readout using only those two features gets 97.73% vs 97.83% for the full 4D one — the other two dimensions are worth 0.1pp.

Both $s,u \ge 0$, so the entire model is a 3-way classification of a ray in the **first quadrant of a plane**, decided purely by $\theta = \operatorname{atan2}(u, s)$:

| $\theta$ | predicted |
|---|---|
| $< 37.1°$ | 2 |
| $37.1° - 49.9°$ | 0 |
| $> 49.9°$ | 1 |

And the true classes occupy:

| answer | $\theta$ range |
|---|---|
| 0 | 37.0° – 49.9° (0.5–99.5 pct) |
| 1 | 50.4° – 90° (5–95 pct) |
| 2 | 0° – 36.7° (5–95 pct) |

Answer 0 gets a **12.8° wedge** wall-to-wall between the other two, and fills it almost exactly. That's the whole model: 4 neurons, 20 parameters, one angle.

## 4. The thumbs are the whole error budget, and $W_{oh}$ is not to blame

The best achievable linear readout on the final hidden state gets **97.83%** (97.47% if a bias is allowed — a bias actively hurts, as it must, since the net is positively homogeneous). The measured model gets ~97.7% on the same distribution.

That's the strongest form of the section's claim: the thumbs are not a $W_{oh}$ placement failure. **No** choice of $W_{oh}$ recovers those points. The information is destroyed at ReLU-2.

Where exactly? Classify final states by their ReLU activation pattern:

| pattern | frequency | error rate | share of all errors |
|---|---|---|---|
| `(1,1,0,0)` | 41.2% | 0.7% | 13% |
| `(0,0,1,1)` | 40.9% | 0.6% | 12% |
| `(1,0,1,0)` | 4.4% | **12.6%** | 25% |
| `(0,1,0,1)` | 4.4% | **13.6%** | 27% |

The two "mixed" patterns — one coordinate alive from *each* pair, which is not a state either clean cone can produce — are 8.8% of inputs and carry **52% of the errors** at a 14% error rate. Those are the thumbs, identified combinatorially rather than visually.

## 5. Where the thumbs come from — and it's something already spotted in the post

At line 277 and again at 364 the post notes, twice, "some outliers along the axis lines, not just at the origin," and moves on. Those outliers are the causal origin of the thumbs.

A ReLU-1 state lands on an axis line iff $x_0 \approx x_1$: median $|x_1-x_0|/\text{spread}$ is 0.090 for on-axis points vs 0.822 for off-axis, and $P(\text{on-axis} \mid |x_1-x_0|/\text{spread} < 0.05) = 0.94$ vs $0.03$ for $>0.2$. A near-tie pushes the sum onto the ReLU boundary, one coordinate of the live pair gets clipped, and the state leaves the 2D wedge for its 1D edge — where the second order statistic no longer exists. Step 3 then propagates that degenerate state into the mixed pattern, which is the thumb.

So: **axis outlier at ReLU-1 → mixed pattern at ReLU-2 → thumb → error.** The three things noticed separately are one causal chain.

## 6. A full description of the failure set (the post's own criterion #2)

The network has no biases, so it's positively homogeneous — verified $h(2x) = 2h(x)$ exactly. Combined with the $J$-symmetry, the error set is a **cone in $\mathbb{R}^3$, symmetric under negation**. Two scale-invariant coordinates fully describe it. With $a > b > c$ the sorted values:

$$r = \frac{b-c}{a-c} \in (0,1), \qquad m = \frac{(a+c)/2}{a-c}$$

Error rate on that grid (all orderings pooled, 400k samples):

|  | $\|m\| > 2$ | $\|m\| \le 2$ |
|---|---|---|
| $r < 0.1$ or $r > 0.9$ | 0.34 – 0.47 | 0.05 – 0.14 |
| $0.1 \le r \le 0.9$ | 0.09 – 0.19 | **0.000** |

In the bulk — median not near-tied with either neighbour, triple not far-offset from zero — the error count is *literally zero* out of ~250k samples. Every error is in one of two thin slabs:

- **$r \to 0$ or $r \to 1$**: the median nearly ties one of its neighbours. $\text{tie} = \min(r, 1-r) < 0.01$ → 48.7% error rate; $\text{tie} \ge 0.2$ → 0.47%.
- **$|m| > 2$**: the triple sits far from the origin relative to its own spread. This is pure bias-lessness — the model is scale-invariant but *not* shift-invariant, while the task is both. Shifting $\mathcal{N}(0,1)$ inputs by $+5$ drops accuracy from 98.5% to **81.0%**.

And the failure is not a benign near-miss. On an error, the model predicts the index of the **argmax** 53% of the time and the **argmin** 47% — against a 1.2% base rate for predicting the argmax. It never softly confuses the median for its tied neighbour; it collapses to reporting an extremum. The thumb is a state that has lost the median entirely.

## 7. The 98.5% vs 97.7% gap is a distribution mismatch

The post flags ARC's 98.5% against the measured ~97.7% without resolving it. It's the input distribution:

| distribution | accuracy |
|---|---|
| $\mathcal{U}(-20,20)$ | 0.9773 |
| $\mathcal{U}(-1,1)$ | 0.9775 |
| $\mathcal{N}(0,1)$ | **0.9850** |
| $\mathcal{U}(0,1)$ | 0.9538 |

$\mathcal{N}(0,1)$ reproduces 98.5% on the nose. Gaussian samples are less likely to be far-offset from zero in units of their own spread, so the $|m| > 2$ slab is under-sampled. Worth a sentence, since it also makes the point that quoting an accuracy for this model without quoting a distribution is meaningless.

---

## Smaller things noticed while reading

**Substantive:**

- Line 396: *"they literally inhabit the same space, so they **have** to be linearly separable"* — this reads as the opposite of the argument. Meaning is presumably that they *cannot* be, or that they *would have to be* for the model to work.
- Line 271: describing the diagonal as "an $x_1 - x_0$ feature" undersells it — per §2, the wedge carries both values, not the difference. The difference framing is what makes the rank-2 question look mysterious.
- Line 261: "the embedding of the next number in each input, $W_{hh} x_1$" — should be $W_{hi} x_1$ (the legend two lines down has it right).
- Line 354 heading: "Add $W_hh x_0 + W_{hi} x_1$" — should be $W_{hh}W_{hi}x_0$, and `W_hh` is missing braces so it renders as $W_hh$.

**Typos:**

- Line 132: "having a **non-zero $h_2$**" appears twice for the two opposite cases; the second should be $h_1$ (or whichever index is intended).
- Line 277: "points where $x_1 > x_0$ and $x_0 < x_1$ respectively" — the two conditions are identical; the second should be $x_0 > x_1$.
- Line 149 "device" → "devise"; line 259 "negaitve"; lines 352 and 358 "analagous".
- Lines 269 and 275 both carry `id="fig-relu-1-pre"`; the anchor on line 271 will resolve to the first. Line 275's should probably be `fig-relu-1-post`.
- Lines 318 and 398 have alt text copied from other figures (`relu_2_pre_cat_1_3` on a `_post` image).
- The architecture section defines $W^{oh}$, but Step 6 of the $M_{2,2}$ walkthrough uses $W_{ho}$ throughout (lines 134, 143, 145).

**Unused assets** sitting in `images/algozoo/` that the post never references: `W_oh_sim_matrix.png`, `wrong_logits_and_halfspaces.png`, `error_and_halfspaces_1_2.mp4`, `zoom_into_red_orange.mp4`. Possibly intentional, but the first two look directly relevant to the Inaccuracy section.

---

## Reproduction

Analysis scripts (frame extraction + all numbers above) are in the session scratchpad:

```
/private/tmp/claude-501/-Users-ryantan-Documents-Life-ryan-rtjj-github-io/5fcf70c0-ff85-4c09-b678-23bfdce76feb/scratchpad/
  vidframes.sh   # ffmpeg frame extraction: vidframes.sh <video> [outdir] [n]
  analyze.py     # J-symmetry, block structure, singular values (§1, §2)
  fwd.py         # forward pass, best linear readout, per-ordering accuracy (§4)
  sym.py         # J-symmetrised 2D readout, theta sectors (§3)
  more.py        # order-statistic recoverability, ReLU-pattern attribution (§2, §4)
  errset.py      # scale-invariant (r, m) error grid (§6)
  tie.py         # near-tie -> axis outlier -> thumb chain (§5)
  dist.py        # accuracy by input distribution (§7)
```

All of them hardcode $W_{hi}$ and $W_{hh}$ from the post; none needs the checkpoint.
