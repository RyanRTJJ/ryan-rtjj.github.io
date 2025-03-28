---
published: true
title: Optimization Failure
date: 2024-11-04 00:00:00 -500
categories: [mechanistic interpretability]
tags: [mechanistic interpretability]
math: true
---

# Introduction

In one of their smaller papers - ["Superposition, Memorization, and Double Descent"](https://transformer-circuits.pub/2023/toy-double-descent/), Anthropic investigated how model features were learnt over time under different dataset size regimes. They trained a toy ReLU autoencoder (a model that compresses an input vector from higher dimensionality to a lower dimensional **latent space** via weight matrix $W$, and then tries to reconstruct it by applying $W^\top$, adding a bias $b$, and finally, $\text{ReLU}$-ing it). The **main claim of the paper is that at lower dataset sizes, the model prefers to represent training examples ("memorization"), while at higher dataset sizes, the model prefers to represent features ("generalization").**

But! **This post will focus on the representation of features / examples**. In particular, we see the beautiful geometric arrangement of examples, where each example is assigned a direction, and the directions are as spaced out as possible in the latent space:

> In this post, I will treat "examples" and "features" as interchangeable, because I will be using a toy dataset where there will be exactly 1 example of each feature; in that sense, examples are just instances of features.

<img src = "../../images/opt_failure/anthropic_training_hidden_vecs.png" alt="Anthropic training set hidden embeddings">
*Each vector corresponds to a training example in latent space, over a range of dataset sizes (Anthropic)*

Intuitively, this all makes sense - the model wants to represent what it learns with as little interference as possible, so it tries to space everything out as much as possible in its latent space, but appealing to this intuition is way too anthropomorphic and doesn't give us a good enough picture of **why there's this regular geometric arrangement, and when / how this regular geometry is broken**. 

In particular, there are is a spin-off work / question that was mentioned at the end of the Anthropic post, which caught my eye:

<img src = "../../images/opt_failure/anthropic_opt_failure_post.png" alt="Anthropic Optimization Failure post" width="700px">
*Screenshot of an interesting follow-up work cited by Anthropic*

This post is also mentioned as the "Two Circle" Phenomenon in Anthropic's [Circuits Updates - May 2023](https://transformer-circuits.pub/2023/may-update/index.html#simple-factorization). There are 3 claims here that I think are faulty:
* The models are organizing points on two circles of different radii. I question both the claims of **"two"** and **"circles"**, explicitly because it does not seem like two is a special number, nor that features are in general arranged as circles, despite how reminiscent of electron / nuclear energy levels this is.
* The models "fail" to merge the multiple circles - it seems to me that if the model indeed differentiated features into multiple "energy levels" such as these, it is not so much that the model wants to merge the multiple circles, but that the multiple circles is a desirable property.
* "To ensure points on the smaller circle can be linearly selected, these solutions require a large angular gap at the interface between points on different circles". In general, this seems plausible, but offers very little specificity on what constitutes a large enough phase angle, how linearly separable is considered linearly separable enough, and so on. 

TLDR, I find this spinoff question **particularly interesting because there are obvious unknowns**. Not to shade the researchers (I consider them to be some of the best, field-defining interpretability researchers), but because they only offer intuitive explanations without any justification, for such behavior that could have many intuitively plausible explanations, these claims are particularly feature. In any case, uncovering the mechanical forces behind the geometric arrangement of features and what causes the nice geometry to break in "Optimization Failure" will be illuminating in understanding why models represent features the way they do.

# The Questions
Based on their very short write-up above, these questions seem natural to investigate:
- Why do points organize themselves into a circle in the ideal case? By circle, I mean with equal magnitude and equally spaced apart.
- When do points not organize themselves into a circle?
- What are key features of the "Optimization Failure" solution? Is it having multiple circles of different radii? How linearly separable must each point be from the rest of the points?

This post will turn into a bit of a rabbit-hole deep-dive, but I hope to put in my most lucid insights that I have managed to hallucinate over the past months, that I believe will be crucial in attaining a good intuition of how models represent features.

# The Problem Set-Up

### Anthropic's Problem Set-Up

I'll do a very quick summary of Anthropic's Toy Model set up, and only for the variant I'm interested in. As mentioned, they define a toy ReLU autoencoder that tries to reconstruct $x$, call the reconstruction $y$, by doing the following:

$$
y = \text{ReLU}(W^\top W x + b)
$$

Where $x \in \mathbb{R}^n, W \in \mathbb{R}^{m \times n}$. $n$ refers to the data's dimensionality (**input dimensionality**) and is much higher than $m$, which is the **latent dimensionality** (aka "hidden dimensionality").

There is also a **sparsity** level of $S = 0.999$, which means that every entry of each data sample has a $0.999$ chance of being $0$, and being uniformly sampled from $(0, 1]$ otherwise. After sampling, these data vectors are **normalized to have magnitude 1**.

Lastly, they define their dataset to have $T$ examples, which is varied from very small ($3$) to very large (million, or infinity, since they generate new data on the fly).

### Our Problem Set-Up

We stick to the same toy model, but focus on the case of $n > 2, m = 2$. We will not have a sparsity parameter, but simply have all data samples be perfectly sparse (1-hot) vectors, and lastly, because we are not investigating memorization vs generalization, we will not have a variable ($T$) number of samples, but just have 1 example per feature (1-hot vector). Essentially, our dataset is the Identity Matrix.

# A More Intuitive Description

Let's break down $y = \text{ReLU}(W^\top W x + b)$ and get a more intuitive understanding of each of these parts. After all, all linear layers basically follow this form, so an understanding of this will be very powerful.

1. $Wx$, also known as $h$, is the latent vector. It is a smaller-dimensional (2D) representation of the data. 
> **Purpose**: The point of compressing dimensionality is important in learning because it forces the model to learn only the most generalizable features in its training data. If it had unlimited representational dimensions, it would simply memorize the training data. This is not very relevant to our discussion here.

    <img src = "../../images/opt_failure/pca_like_2d.png" alt="PCA-like 2D" width="100%">
    *How one might represent a 3D dataset in 2D space by projecting it down onto a plane*

2. $W^\top W x$, aka $W^\top h$, is the decoded vector. The decoding operation (applying $W^\top$) simply takes the compressed 2D representation ($h$) and projects it out to the higher dimensional space (same as the input space). Note that even though $W^\top h$ lives in $\mathbb{R}^n$ (a high dimensional space), its effective dimensionality is that of $h$ (2D), since no matter how you linearly transform (stretch & scale) a 2D set of points, you can never make it more than 2D. **Note**: In real neural networks, the decoding function is not simply a linear function like $W^\top$, but some more expressive non-linear function. The decoding function represents the construction of output features (or in this case, the reconstruction of input features, because the model is trained to reconstruct its inputs) from a more compressed representation ($h$).

3. $\text{ReLU}$, the most exciting part, is the non-linear function that is magically able to extract more dimensions of information than exists in the latent space (2). We will see how this works in a while. Mechanically, $\text{ReLU}$ simply takes the max of $0$ and its input value. It's useful to think of this as clamping values to $> 0$:

    <img src = "../../images/opt_failure/relu_profile.png" alt="Graph of ReLU" width="350px">
    *Graph of the ReLU function*

4. $b$, the bias. The bias is, in my opinion, the most deceptive part of the linear layer. Intuitively, it's simple: adding a bias simply shifts everything by an offset. $(W^\top W x + b)$ is basically $W^\top W x$, but $b$ distance away from the origin. But, this bias is incredibly powerful and has the potential to disrupt many assumptions of feature geometry. For example, if a model learns to represent features linearly from the origin, such as the days of the week in this following image I got from [here](https://arxiv.org/html/2405.14860v2):

    <img src = "../../images/opt_failure/week.png" alt="Days of the week features" width="350px"> 
    *Features of each day of the week, Engels et. al*
    
    Adding a bias would suddenly shift this circle of features to be, say, in the positive quadrant, rather than centered about the origin. Suddenly, the "Sunday" feature will be in the same line of sight as the "Wednesday" feature, which would make the features NOT linear relative to the origin! **Purpose**: we will discuss the purpose(s) of the bias in this post.

**Why is this toy model worth our time?** It captures basically all of the properties of actual linear layers in deep models. I'll just list 2 key reasons that this toy model forms a good surrogate for large model components:
- This toy model is basically a "Linear Layer" (although the name has "Linear" in it, in deep-learning parlance, it just means one linear transformation followed by a non-linear activation function like ReLU, hence it's not linear), which are the basic building blocks of ALL deep models
- This toy model implements compression and decompression, which are key behaviors of ALL deep models

All of the local behaviors of deep models can basically be studied with this toy model.

# ReLU: Free Dimensions, Somewhat

So, in the world of linear transformations, one can never extract more dimensions of information than the minimum that was available at any point. For example, in the above illustration that I'll paste below, we transformed a 3D dataset into a 2D one. But, given the 2D points <span style="color: green">(**green**)</span>, you can never reconstruct the original 3D points, because you don't know how far along the dotted lines they originally were away from the plane.

<img src = "../../images/opt_failure/pca_like_2d.png" alt="PCA-like 2D" width="100%">
*How one might represent a 3D dataset in 2D space by projecting it down onto a plane*

It follows that when you've compressed a $10,000$-dimensional vector to $2$ dimensions, like Anthropic did in the problem set-up, you would not be able to construct the original vector. **However, throw in a non-linearity and some sparsity in your data points and you can actually reconstruct the original training data fairly well, sometimes without any training loss! How does this happen? Let's take a look.**
> This discussion will pertain **solely** to the $\text{ReLU}$ non-linearity. The intuition will transfer somewhat to other activation functions, but activation functions have other kinds of inductive biases that will likely cause the representation of features that models end up learning to be different. This is out of the scope of this discussion

We start by considering the dataset. The dataset is incredibly sparse ($S = 0.999$), meaning that on average, about $10$ entries out of the $10,000$ in a single data vector are non-zero. To keep things simple, **I will use a dataset where each data sample has exactly $1$ active (non-zero) entry**. That's what Anthropic effectively did anyway (by adjusting sparsity such that the $\mathbb{E} (\text{num non-zero features per data sample}) = 1$), in the paper where they originally discovered the beautiful geometry (I still remember Zac Hatfield-Dodds going "whaaaaaat why are there pentagons here!!") - [Toy Models of Superpositions](https://transformer-circuits.pub/2022/toy_model/index.html). This means that instead of having vectors that point in random direction in space, you essentially have the ***standard basis vectors (axial lines)***: 

<img src = "../../images/opt_failure/figure_axis_vectors.png" alt="PCA-like 2D" width="500px">
*The three axial vectors in 3D space*

I know that we are operating in high-dimensional input space (e.g. $\mathbb{R}^{10,000}$), but we can only focus on the dimensions where there is data with non-zero entries in that dimension. Suppose we have only 3 data points, then we can basically only look at the 3 dimensions where the data points are active in. Let's look at how these 3 'data archetypes' can be represented in a 2D latent space. There are 2 explanations:

## Explanation 1: The effective dimensionality of the points is low (2D)

In this case, this is correct. You can observe that the 3 data points (the axial vectors) form a plane (a "2D simplex"). Since the effective dimensionality of the data is 2, then of course we can represent it without loss of information in 2D. However, it only works in the case of trying to represent $n$-dimensional data in a $(n-1)$-dimensional latent space. In general, $n$ points in $n$-dimensional space will form an $(n-1)$-dimensional simplex. If your latent space is coincidentally also $(n-1)$-dimensional or bigger, this is fine and dandy, but if your latent space is smaller than $(n-1)$ dimensions, then it will not be able to capture your $(n-1)$-dimensional object perfectly, and this explanation does not suffice.

## Explanation 2: The ReLU Hypercube

I'm not quite sure how to phrase this as an explanation, but I'll explain the intuition that makes it super clear. I mentioned that a helpful way of thinking about what ReLU does is that it clamps values to be $\geq 0$, i.e. that it snaps all points to the edges of the positive half-space (1D) / quadrant (2D) / octant (3D), and so on; I'll call the high-dimensional generalization of this the "ReLU hypercube," because why not. Let's see how this clamping visually looks in 1 to 3 dimensions:

<img src = "../../images/opt_failure/relu_1d_2d_3d.png" alt="ReLU clamping" width="600px">
*Before (left) and after applying ReLU (right)*

Notice how, because ReLU effectively zeros out all negative values, many points are snapped against the positive edges of the ReLU hypercube. This means that a lot of the post-ReLU points are scalar multiples of the axial vectors. Visually, you can see this happening by observing that the axis lines are particularly packed with data points in the right column. This also means that ReLU is particularly good at extracting axial vectors, which is what breaks the rotational invariance ("symmetry") that is otherwise present in purely linear systems - the axial vectors are hence "privileged," to borrow the term from Anthropic.

Also, because we are working in the ideal case where the data points are axis-aligned, this is particularly convenient for us, because the type of points that ReLU loves to output (axis-aligned), is exactly the type of points that we are looking to reconstruct. This may look like a beautiful coincidence, and you may think that in real models, you can't expect axis-aligned vectors or features like this, but the reality is the these things are actually super common! By the end of this section, hopefully the intuition behind how ReLU gives us free dimensions is clear, and it'll become obvious why learning such axis-aligned feature vectors is beneficial to the model, and is hence what happens in real models.

Because ReLU sort of "wraps" your data points around the ReLU hypercube at the origin, this mechanism has the effect of giving you "free dimensions." In particular, if you wanted to be able to construct a dataset that comprises axial vectors, you could represent them in your latent space (that is, pre-ReLU) in such a way where the first axial vector's ($e_1$) <span style="color: purple">**latent representation ($h(e_1)$)**</span> will be snapped onto <span style="color: yellowgreen">$e_1$</span> by ReLU, the second axial vector's ($e_2$) <span style="color: purple">**latent representation ($h(e_2)$)**</span> will be snapped onto <span style="color: yellowgreen">$e_2$</span> by ReLU, and so on. To achieve this, you simply have to achieve the following criteria, which I'll call the **"unique single positive value"** criteria:

***The latent representation of $e_i$ must have a positive $i$-th element and all other elements be non-positive.***

In 2D, you can achieve such criteria by putting all your datapoints' latent representations on a line that extends into the $(\leq 0, +)$ and $(+, \leq 0)$ region. The <span style="color: purple">**purple dashed line**</span> illustrates such a line; the <span style="color: yellowgreen">**green points**</span> are the ReLU-ed versions of each of the datapoints' latent representations.

<img src = "../../images/opt_failure/relu_cube_2d.png" alt="ReLU clamping 2D" width="350px">
*Data latents (purple) being ReLU-ed onto the ReLU hypercube*

If you looked from the under-side of the line (from the bottom-left to the top-right), you would see the axial vectors as the feature vectors (entries of $W$, since $W \in \mathbb{R}^{1 \times 2}$).

In 3D, there is such a similar way of arranging your latent representations, fulfilling the same criteria:

<img src = "../../images/opt_failure/relu_regions.png" alt="ReLU clamping" width="100%">
*Data latents (on the plane) being ReLU-ed onto the ReLU hypercube in 3D*

Note that points in the bluish purple regions of the plane (the latent space) fulfill this criteria and will get snapped onto the axial vectors (edges of ReLU hypercube), while points in the red regions of the plane do not, and will get snapped onto the faces of the ReLU hypercube, as opposed to the edges. **Note** too how the 3 axial vectors form that trigonal planar arrangement when viewed from the underside of the plane (right diagram), and these are precisely the columns of $W \in \mathbb{R}^{2 \times 3}$ in [Anthropic's Toy ReLU autoencoder](https://transformer-circuits.pub/2022/toy_model/index.html) that was trained to reconstruct 3D points with a 2D latent space:

<img src = "../../images/opt_failure/anthropic_trigonal_planar.png" alt="Anthropic's W columns" width="350px">
*Anthropic's ReLU autoencoder features (rows of W), image from their paper. Note that m in this image represents n in this post.*

Even though we (and Anthropic) achieve the geometrically perfect solution (the 3 vectors are equally spaced apart), in reality, the 3 feature vectors do **not** have to be equally spaced apart, because the plane has some wiggleroom to pivot at various different angles about the origin, such that the cube appears to be rotated at various angles when viewed from the underside of the plane. Any mildly rotated solution will work, so long as the axial vectors still fall inside their respective purple zones on the plane. If you're trying to be able to extract more axial vectors from a low-dimensional latent space, the amount of wiggle room decreases, just because the regions of space that satisfy the single unique positive value criteria decreases. For example, when we try to represent the four $4$-dimensional axial lines on a 2D plane such that $\text{ReLU}$ can snap them back onto their respective axes, we find that the regions of space that satisfies the criteria has no area (no wiggleroom at all!) - it is just a set of 4 lines, that correspond to the origin-bound edges of the 4D hyper-cube shown below (yes, this is what 4D hyper-cube looks like when projected onto a 2D surface):

<img src = "../../images/opt_failure/splat_4d_perfect.png" alt="Four 4D axial vectors embedded into 2D space, + some rotation" width="350px">
*Four 4D axial vectors embedded into 2D space, + some rotation*


**This is precisely the job of $W$: their columns act as feature vectors, while their rows act as the span of the $m$-dimensional hyperplane in $n$-dimensional space.**

> Note that this description of $W$ is specific to this context where the compression weights $W$ are the same as the decoding weights ($W^\top$). In reality, they are commonly different, and so you have something like $y = \text{ReLU} (MWx + b)$, hence it would be that the rows of $M$ act as feature vectors.

# The role of $W$

If you don't have a very intuitive understanding of linear transformations, then what I wrote above about the "job of $W$" probably sounds very unobvious, so I'll quickly illustrate the connection between the plane, the ReLU hypercube, and $W$.

We've established that to get the free dimensions via the ReLU Hypercube explanation requires you to rotate your plane (an $m$-dimensional object, in this case, 2) in $n$-dimensional space properly such that when you view the axis lines from the underside of the plane, they fall into their respective sectors of the plane that fulfill the criteria of having 1 positive entry and all else being non-positive. In linear algebra, applying a matrix to a vector (e.g. $Mx$) simply re-interprets $x$ to be not in the standard axes, but in the axes denoted by the columns of $W$. There will be a section on this from my linear algebra primer I'll release in the future, but for now, here is a quick image of what I mean:

<img src = "../../images/opt_failure/change_of_basis_ortho.png" alt="Change of basis orthogonal" width="100%">
*Left: an example vector x (purple). Right: Mx, where the maroon and orange lines are the columns of M*

If $M$ were a tall matrix (more rows than columns, i.e. each column is tall), then the changed basis simply just lives in higher-dimensional space. If $M$ were a wide matrix (more columns than rows, i.e. each column is short), then in addition to changing your basis, you're also reducing the dimensionality of $x$, because $x$ is going from a `num_columns`-dimensional vector to a $Mx$, a `num_rows`-dimensional vector.

So if we have $x \rightarrow Wx$, this means that the 2-dimensional columns of $W$ represent the basis vectors of the latent space (a 2D plane) in a 2D world. If we further have $Wx \rightarrow W^\top W x$, this means that the 3-dimensional columns of $W^\top$ represent the basis vectors of the latent space (same 2D plane) in a ***3D*** world. **This is why the rows of $W$ act as the span of the $m$-dimensional hyperplane in $n$-dimensional space.**

Further, notice that because applying $W$ to $x$ is a reinterpretation of $x$'s entries as values in each of the columns of $W$, this is a remapping of $x$'s features from the standard basis world (e.g. first entry $=$ first feature $= [1,0,0]$), to the $W$ basis world (e.g. first feature is now represented via $W$'s first column - a 2D vector, instead of $[1,0,0]$). **This is why the columns of $W$ act as feature vectors.**

Now then, it becomes obvious that $W$'s role over training is just to adjust it values properly such that it defines a good $m$-dimensional hyperplane in $n$-dimensional space, such that ReLU can recover all the free dimensions that it needs to perfectly reconstruct the $n$-dimensional vectors with only an $m$-dimensional intermediate representation.


# Limit of number of Free Dimensions

I find this intuition very clear: if you pivot a plane about the $n$-dimensional hypercube at the origin and look at the hypercube from the underside of the plane, you can always find a way to rotate the plane about that pivot point until you can see all the axial vectors (edges of the hypercube). But does this mean that you can always partition the plane into at least $n$ sectors that will correctly snap to their corresponding axial vectors after $\text{ReLU}$ (i.e. that fulfill the above criteria)? NO! 

**In general, the combined powers of $W$ and $\text{ReLU}$ can only perfectly reconstruct up to $2m$-dimensional axial vectors from an $m$-dimensional latent space.** This is because you cannot select a set of $(n > 2m)$ $n$-dimensional vectors, where they satisfy the requirement of 1 positive element in a unique position and all else being non-positive, while having them all be on the same $m$-dimensional object. This means that, **for example, a 2-dimensional latent space is expressive enough to reconstruct up to 4 dimensions (if we only care about the axial vectors)**, a 3-dimensional latent space can reconstruct up to 6 dimensions, and so on. This is great, but it's definitely not infinite.

<img src = "../../images/opt_failure/splat_4d_perfect.png" alt="Four 4D axial vectors embedded into 2D space, + some rotation" width="350px">
*Repeat: four 4D axial vectors embedded into 2D space, + some rotation*

> While I am quite confident that this upper bound is correct, and can be achieved with vector sets similar to $[1, -1, 0, 0], [-1, 1, 0, 0], [0, 0, 1, -1], [0, 0, -1, 1]$ (in 4D), I have no proof. It would be satisfying to have a proof, and I would greatly appreciate one.

>**Actually...:** A friend came by to prove this wrong. This is only true in the case of $m = 2$, but he had a pretty stunning postulate: if you have an $s$-hot dataset, you need a $(s + 1)$-dimensional latent space, + bias, for perfect reconstruction. David - wanna provide a proof here?

So, $W$ and $\text{ReLU}$ are definitely not sufficient to account for what Anthropic found - "in fact, the ReLU output model can memorize any number of orthonormal training examples using $T$-gons if given infinite floating-point precision," where $T$ represents the size of the dataset. The memorized training examples have these latent representations:

<img src = "../../images/opt_failure/anthropic_infinite_vectors.png" alt="Memorized Training Examples" width="100%">
*Latent vectors of datasets as T increases (under a certain threshold limit), image from their paper*

So how on earth does this happen? This is where the bias $b$ comes in. But before that, let's take a quick look at what happens when we try to memorize more than $2m$ $m$-dimensional axial vectors with only an $m$-dimensional latent space.

# Regular Geometry is Desired When $n > 2m$

When $n > 2m$, the model will still try to learn a $W$ matrix whose columns are geometrically regular. In this case, we'll examine $n = 5, m = 2$. **Aside**: if you tried to project an $n$-dimensional hypercube down to the $m$-dimensional hyperplane in a way that mapped the $n$-dimensional axial lines to the columns of $W$, the $n$-dimensional hypercube would look VERY regular (which is special; any randomly sampled orientation of a hypercube is exceedingly likely to have a 2D projection that looks very asymmetrical). I trained such an autoencoder and got the following columns of $W$ (and plotted the corresponding projection of the implied 5D hypercube):

<img src = "../../images/opt_failure/splat_5d_perfect.png" alt="Five 5D axial vectors embedded into 2D space, + some rotation" width="400px">
*Five 5D axial vectors embedded into 2D space, + some rotation*

These five 2D feature vectors, when projected back out to 5D space via $W^\top$, (i.e. the columns of $W^\top W$), do not fulfill the unique positive value criteria; each of them has more than 1 positive entry. As such, $\text{ReLU}$ is unable to snap these 5D feature vectors cleanly back to the axial lines. The reconstruction is **not** perfect and has some loss. In particular, the extraneous / "wrong" positive entries of each column of $W^\top W$ will contribute some error.

Lastly, let's address the intuition for why $W$ is still pressured to space these feature vectors out and achieve that regular geometry. There are some big caveats here that could be the cause(s) of optimization failure; take everything in the next 2 paragraphs at face-value. Remember that the Toy ReLU Autoencoder was trained using the Mean Squared Error (MSE) loss function, which is quadratic:

$$
\mathcal{L}(W, X) = \frac{1}{|X|} \sum_{x \in X} (y - x)^2
$$

This means that the greater the extraneous positive entries, the quadratically greater the loss. This makes it prefer evenly spaced out columns of $W$, and here's why. Because angular space is finite, preferentially spacing out any pair of feature vectors will necessarily mean that other pairs of feature vectors will be squeezed closer together. The additional loss incurred by the squeezed vectors will weigh quadratically as much as the loss "saved" by the spaced out vectors, resulting in greater overall loss. **This is why $W$ still faces optimization pressure to distribute the error evenly to all feature vectors and achieve that geometric regularity.** This is standard behavior whenever you use some sort of convex loss function, which MSE is, but I thought I would just point it out.

### Caveat 1: Loss is not actually quadratic, only convex

Actually, loss here is not quadratic with respect to the angle between the vectors, but a composition of the quadratic MSE Function and the cosine function, which ends up being a sinusoidal function:

$$
\begin{align*}
\mathcal{L}\left(\theta, x^{(1)}, x^{(2)}\right) &= \left(\max(0, \cos(\theta))\right)^2 \\
&= \frac{1}{2}\left(\cos(2\theta) + 1 \right)
\end{align*}
$$

This is not quadratic, but it is still convex near $\theta = \pm \frac{\pi}{2}$.

<img src = "../../images/opt_failure/loss_over_theta.png" alt="Loss over theta" width="100%">
*Loss over angle between 2 vectors*

### Caveat 2: Loss is only convex near $\theta = \pm \frac{\pi}{2}$

This means that the system faces different optimization preferences when the feature vectors (columns of $W$) are in the range $\frac{-\pi}{2} < \theta < \frac{\pi}{2}$. We will investigate more later.

# The role of $b$

By now, we can observe 2 things:
- Because of the optimization pressure to achieve geometric regularity, the representations of these axial vectors in the 2D latent space mimic the spokes of a wheel (generalizable to higher-dimensional hyperspheres).
- Trying to represent more than $2m$ axial vectors in an $m$-dimensional latent space will violate the unique positive value criteria.

Previously, I said that the combined powers of $W$ and $\text{ReLU}$ can only allow us to hydrate $2m$ $n$-dimensional axial lines from an $m$-dimensional latent space. The intuition for that is basically that in an $m$-dimensional latent space, you can only fit $2m$ orthogonal pairs of antipodal lines. If your vectors are forced to be less than $90^\circ$ apart from each other, then you get some interference. But, it turns out, that $b$ can basically absorb all that interference by trading off some "sensitivity". Let's see how that works. Much of the proof was worked out by Henighan in this [Google Colab Notebook](https://colab.research.google.com/drive/1AREdeODhgsQ_ukqPKnhWQ4bijVqTa4eW?usp=sharing#scrollTo=Or1plIMKqUts).

The intuition thus far was for the latent space (a plane in all our examples so far) to be pivoted at the origin, but it doesn't have to be pivoted at the origin. In particular, you could add a negative bias in all dimensions such that you can shift the latent representations away from the origin, into the region of space where all dimensions are negative:

<img src = "../../images/opt_failure/feat_wheel.png" alt="Feature wheel with bias" width="400px">
*Feature representations in latent space, with bias*

This isn't super useful, because if all your features were represented in the negative regions of space, $\text{ReLU}$ would come along and zero everything; you have no information. However, you can shift the feature wheel in a way such that 1 feature vector is in a region of space with 1 positive dimension, while the rest of the feature vectors are in a region of space of non-negatives:

<img src = "../../images/opt_failure/shifted_feat_wheel.png" alt="Shifted Feature wheel" width="100%">

With just 3 vectors, you may think that the shifting is unnecessary, because even with the wheel of features pivoted at the origin ($0$ bias), the 3 feature vectors already fulfilled the unique positive value criteria. However, if we had more than 3 vectors, e.g. we have 8 $8$-dimensional axial vectors projected down onto a 2D wheel similarly, then the shifting is necessary to ensure that each one of those unique positive value criteria-fulfilling regions only has 1 feature vector.

<img src = "../../images/opt_failure/shifted_feat_wheel_crowded.png" alt="Shifted Feature wheel crowded" width="100%">

In the above image, we can only see that 3 feature vectors are in the correct regions that make them fulfill the unique positive value criteria, but that is merely because we are unable to visualize an 8-dimensional space. If we could plot the above wheel in an 8D space as opposed to a 3D one, we would see that all 8 feature vectors are in their correct respective regions.

## Can this really generalize to higher dimensions?

The last paragraph above is not intuively obvious because we can't visualize more than 3 dimensions, so we can fall back on analysis to see that this really works. Here is where I will liberally quote the work of Henighan, in that Colab Notebook. This section is more mathematical analysis and less intuition, so feel free to skip it.

Henighan starts of by declaring the dataset $X$ the identity matrix, without loss of generality. This works, because thus far we've been assuming that the dataset is just points that align with the axis lines. This means that the data points (the rows of $X$) are one-hot, with their activated element being in unique positions. The identity matrix is just a neat way of arranging these rows, with the $i$-th row being active in the $i$-th position. This makes the objective of the ReLU toy model to reconstruct the identity matrix:

$$
\begin{align*}
y & = \text{ReLU} \left( X W^\top W + b \right)\\
\Rightarrow I &= \text{ReLU} \left( W^\top W + b \right)
\end{align*}
$$

This means that for this to work, the diagonal entries of $(W^\top W + b)$ need to be the ONLY entries $> 0$; hence the goal becomes to find $W$ and $b$ such that this is true.

He starts off with an a guess for $W$, based on the assumption that the columns (2D vectors) of $W$ will represent vectors that are spaced out evenly over a circle:

$$
\begin{align*}
W = \begin{bmatrix} \cos(0) & \cos(\phi) & \cdots & \cos(-\phi) \\
\sin(0) & \sin(\phi) & \cdots & \sin(-\phi)
\end{bmatrix} \text{, where } \phi = \frac{2 \pi}{\text{num examples / features}}
\end{align*}
$$

This guess turns out to work in most cases, because of my above explanation on how the partially convex loss function causes $W$ to distribute the total $2 \pi$ radians in a circle over each consecutive pair of vectors as evenly as possible.

And so, $W^\top W$ looks like this, after a bunch of massaging with the trigonometric identities:

$$
\begin{align*}
W^\top W =
\begin{bmatrix}
\cos(0) & \cos(\phi) & \cos(2\phi) & \cdots & \cos(-\phi) \\
\cos(-\phi) & \cos(0) & \cos(\phi) & \cdots & \cos(-2\phi) \\
\cos(-2\phi) & \cos(-\phi) & \cos(0) & \cdots & \cos(-3\phi) \\
\cos(-3\phi) & \cos(-2\phi) & \cos(-\phi) & \cdots & \cos(-4\phi) \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
\cos(2\phi) & \cos(3\phi) & \cos(4\phi) & \cdots & \cos(\phi) \\
\cos(\phi) & \cos(2\phi) & \cos(3\phi) & \cdots & \cos(0) \\
\end{bmatrix}
\end{align*}
$$

But since that doesn't look like much, let's look at the matrix in visual form (if you had $10$ 10D vectors now):

<img src = "../../images/opt_failure/WTW_visual.png" alt="WTW visual" width="400px">
*A visual depiction of W.T @ W*

I've highlighted the maximum value of each row (column too) in the above image. Remember that because $Y = \text{ReLU}(W^\top W X + b)$, and $X = I$, each row here represent the pre-ReLU, pre-bias representations of each data point. From here, it's obvious that **you can just subtract the second-highest value of each row (i.e. $\cos(\phi)$) from every single value, and every row will just have ONE value that is $> 0$ - the blue entries, a.k.a. the diagonal. This explanation works for any $n$, so it will indeed generalize to any $n$.**

One small nitpick is that Anthropic said that this set up of embedding $n$-dimensional vectors in $m=2$-dimensional latent space can memorize "**any** number of examples [or features]", but it's obvious here that it's not really **any** number of features; it's just any number that is $\leq n$, because once you have $n + 1$, by the pigeonhole argument, there will be at least 1 unique single value criteria-fulfilling hyperquadrant with more than 1 example / feature vector in it. That said, if you allowed $n$ to increase with your number of examples / features, then it really is infinite. This is where the added requirement of having infinite floating point precision also makes sense, because as $\phi \rightarrow 0$, the difference between $\cos(0) = 1$ and $\cos(\phi)$ also tends to $0$.

## Cost of absorbing interference: decrease sensitivity

By adjusting $b$ to shift the feature wheel in such a manner, you also notice that the length of the part of each vector that is in the correct unique single value criteria-fulfilling hyperquadrant is also shortened. Visually, you can observe this by noticing that the part of the "positive z" vector that is $> 0$ is only the tip of the vector, and not the full length of it. Same goes for the "positive x" and "positive y" vectors. This means that **for that feature to be reconstructable, the original input's value in that feature needs to be more than a certain threshold**. A small value of the feature will not be reconstructable because its representation in the latent space is $< 0$ in all dimensions. I find this intuitive - the **bias is commonly explained as a threshold that a neuron needs to be activated beyond in order to make the neuron "fire"**. If it's too weak, the neuron simply won't fire.

# Does the Optimization Failure solution pursue linear separability?

Finally, we have come to the point where we have sufficient intuition to begin investigating Optimzation Failure. The first question we shall investigate is as written in the title of this section. I'm particularly interested in this question because Olah and Henighan claim that in the case of Optimization Failure, "in order for points on the small circle to be linearly separated from points on the large one, there must be a large angular gap."

In so doing, I think the natural first step is to see what regions in latent space ($h = Wx$) correspond to the axial lines, post-ReLU. I found it useful to plot what I call **"latent zones"**:

<img src = "../../images/opt_failure/latent_zone_intro_1.png" alt="Latent zone intro" width="100%">

In the above example, I trained a bunch of $n=6, m=2$ autoencoders, and selected one where the model learnt to perfectly reconstruct all features. On the left, you can see the column vectors of $W$ arranged into a perfect circle. On the right, I plotted the regions of the latent space (2-dimensional) that correspond to an activation of the $i$-th feature. For example, in order to activate (reconstruct) the <span style="color: darksalmon">**orange feature**</span> in the reconstructed vector, the hidden vector computed by the model needs to be in the <span style="color: darksalmon">**orange region**</span>.

> **Plotting quirks:** Notice that the latent space seems to be limited to a hexagon, rather than the entire 2D subspace. This is merely a quirk of my plotting logic; since the non-zero entries of our input vectors (the standard basis vectors) are all $1$'s, it is sufficient to plot a portion of the latent space that contains all the columns of $W$ in 2D (equivalent: all the columns of $WI$ in $n$-dimensional latent space). Such a portion of space is simply given by the outline drawn by enumerating the columns of $W$ in a clockwise fashion. 

> Occasionally, you will also see that there is a kink in the shape, so instead of expecting a nice hexagon (or heptagon) for 7 points, you may observe an "indented pac-man". This is simply because there is a point that is so near to the origin (the mouth of the pac-man) that connecting all the 7 points results in a pac-man shape.

To fully appreciate why the activation zones are as such, I illustrate the latent space (a planar hexagon) in a 3D space:

<img src = "../../images/opt_failure/latent_zone_intro_2.png" alt="Latent zone intro 2" width="500px">

I've also highlighted the <span style="color: royalblue">activation zone of the $z$-th feature</span>, which is simply the part of the latent space with <span style="color: royalblue">$z > 0$</span>. It is now also obvious that **the $z$-th value of the bias is the lever which controls where that blue sliver starts**. A $b_z$ that is more negative means that the latent hexagon is pushed down further along the $z$-axis, meaning that the sliver of the hexagon that satisfies $z > 0$ starts further up the hexagon. Conversely, a less negative $b_z$ means that the hexagon will be situated further up the $z$-axis, meaning that the sliver of the hexagon that satisfies $z > 0$ will start nearer the bottom of the hexagon.

I'd also like to point out that the amount of reconstruction a feature gets is hence determined by how much the feature vector extends into its corresponding latent zone:

<img src = "../../images/opt_failure/illustrate_activation.png" alt="Amount of activation must be equivalent to reconstruction of the feature" width="500px">
*Amount of activation must be equivalent to reconstruction of the feature*

So, in response to whether or not optimization pursues linear separability, the answer is **yes, but there's a lot more nuance to it than is let on by Olah's and Henighan's illustration**, which I paste below again:

<img src = "../../images/opt_failure/olah_henighan_2.png" alt="Olah Henighan 2" width="600px">

The dotted line linearly separates the one short feature from the rest, yes, but in a rather abitrary way, and doesn't illustrate how much linear separation is enough, and the point at which linear separation starts to count. Applying our intuition of latent zones and activation amount, we can draw in the real line of separation, which is also where the latent zones start. I've drawn one for a <span style="color: indianred">short feature</span> and one for a <span style="color: blue">long feature</span>:

<img src = "../../images/opt_failure/olah_henighan_3.png" alt="Olah Henighan 3" width="300px">

> You may notice that the activation amount for the long feature seems much less than the activation amount for the short feature - doesn't this mean that the reconstructed value for the <span style="color: blue">long feature</span> is less than the reconstructed value of the <span style="color: indianred">short feature</span>? E.g., if distance annotated by the red double-ended arrow correspond to a reconstructed value of $1$, wouldn't the distanced annotated by the blue arrow correspond to a value of much less than $1$? This is a great observation, but keep in mind that the reconstructed value of feathre $i$ is merely the extent to which feature vector $i$ is in the latent zone, in ***the $i$-th dimension***. In a perfectly symmetric latent set-up, the latent plane is tilted equally in all dimensions, but in this asymmetric set-up, the latent plane is tilted much more in the dimensions of the <span style="color: blue">long features</span> (i.e. much more parallel to the <span style="color: blue">long features</span>). You will get a more complete picture of "tilt" in the coming sections.

# Optimization Failure Mode: Symmetric Failure
There are only a few ways optimization can fail for this toy model. In Olah's and Henighan's screenshot, the model is obviously asymmetrical and turns out, lots of model latents become asymmetrical in Optimization Failure, but it seems to me that all failed models have at some point passed through the "Symmetric Failure" mode before turning asymmetrical. Let's take a look at what this "Symmetric Failure" mode looks like.

<img src = "../../images/opt_failure/symmetric_failure_1.png" alt="Symmetric Failure 1" width="100%">
*n = 7, m = 2, trained for 25,000 epochs using AdamW*

Symmetric Failure happens when the model fails to learn some / a <span style="color: mediumblue">feature</span> (does not represent it **at all**), but does a perfect job of representing the rest (the other 6 features have equal magnitude and are spaced apart equally).

> We now observe the pac-man shape as opposed to a regular hexagon. As explained above, because I draw the latent shape by simply connecting the 7 feature vectors, the unlearnt feature that is close to the origin causes there to be that pac-man mouth.

The key things to note are that:
* For the features that were properly learnt (call them features $1$ to $6$), we see the characteristic activation sliver in the latent space.
* For the  <span style="color: mediumblue">feature (feature $7$) that was not properly learnt</span>, that feature activates everywhere - the whole latent shape is  <span style="color: mediumblue">blue</span>. This is because this feature's bias value is positive, causing the entire latent shape to have a positive value in the $7$-th dimension.
* Observing the training process, the fact that this model's internal features stayed like since about step 3,000 means that this is a local optimum.

We know what the perfect solution looks like. The latent shape should be a regular heptagon, and the bias should be such that it pushes the latent shape in the dense negative direction ($[-1, -1, ..., -1]$). It should reside in the completely negative hyper-quadrant except for the 7 activation slivers that have positive values in their respective dimensions. In order to get from the current state of the Symmetric Failure model to this perfect solution, 2 things need to happen:
* $b_7$ needs to decrease and eventually become negative - this pushes the entire latent shape down into the negative region of the $7$-th dimension (so that it will stop activating everywhere).
* $W_7$ ($7$-th column of $W$) needs to become non-$0$ (and eventually take its own unique direction, but to learn a non-$0$ value is the first step) - this **introduces a tilt** (more on "tilt" later) in the latent shape such that only a sliver of the shape will be positive in the $7$-th dimension.

Why can't the model do either of these? Let's take begin to describe this local optimum and perturb it to see how strong of a local optimum this is.

> **It's not AdamW's fault**: At some point, I suspected that AdamW may have been the culprit, because AdamW has weight decay, which is a form of regularization. I then trained the autoencoder using Adam without any regularization, but the same optimization failure occurred.

## What is $b_7$? Why can't it decrease?

The first step is to solidify our intuition of $b_7$ and its relation to the position of the latent shape relative to the dim-$7$ axis, and of $W_7$ and its relation to the idea of "tilt". This calls for more illustrations. Since I can't illustrate in more than 3 dimensions, I'll just illustrate in 3D. The nice thing about linear algebra is that the dimensions are all linearly independent, and we can just visualize any subset of dimensions without incurring some error from other dimensions. Hence, we'll visualize our latent shape in dimensions 1, 2, and 7:

<img src = "../../images/opt_failure/disc_simple_1.png" alt="Dimensions 1, 2, 7, simple" width="100%">
*The 1st, 2nd, and 7th dimension values of the latent shape (7-th dimension is enlarged)*

I've drawn in the bias as well and highlighted <span style="color: mediumblue">$b_7$ (the value of the bias in the 7-th dimension)</span>. The entire hexagonal shape is also shaded blue to remind us that the entire shape is activated ($> 0$) in the 7-th dimension. You may notice that the hexagon seems squished - this is simply due to the fact that visualizing only 3 of the 7 dimensions results in this kind of "perspectival" squishing; the same thing happens when you try to take a purely side-profile look at a square piece of paper that's half-tilted to face the sky - it doesn't look square.

The important thing to note here is that <span style="color: mediumblue">$b_7$</span> is precisely the offset of the hexagonal plane with respect to the <span style="color: mediumblue">dim-$7$ axis</span>. If you increase <span style="color: mediumblue">$b_7$</span>, the plane moves up, and if you decrease it, the plane moves down.

Since <span style="color: mediumblue">$W_7 = 0$</span>, the only thing that can impact the reconstruction of the 7-th feature is $b_7$:

$$
\begin{align*}
\hat{y} & = \text{ReLU} \left( W^\top W x + b\right) \\
\Longrightarrow \hat{y}_7 & = \text{ReLU} \left( W_7^\top W x + b_7 \right) \\
& = \text{ReLU} (b_7) \text{, because } W_7 = 0
\end{align*}
$$

Remember that the model wants to be able to reconstruct a $1$ for the <span style="color: mediumblue">$7$-th feature</span> of the $7$-th data sample, and a $0$ for the <span style="color: mediumblue">$7$-th feature</span> of the other 6 data samples. If you had only 1 variable (<span style="color: mediumblue">$b_7$</span>) to try and do this as well as possible, your best option is really to just encode the mean. And if your measure of error is Mean Square Error (MSE), then that the mean is really your solution. This is rather intuitive, and is also a standard result in Ordinary Least Squares (OLS) Regression because that also aims to minimize MSE. We expect <span style="color: mediumblue">$b_7 = 1/n = 1/7$</span> to hence be a local optimum, and from my experiments, it is indeed observed that <span style="color: mediumblue">$b_7 = 1/n = 1/7$</span>.

## What is $W_7$? Why can't it increase?

In the perfect optimization case, <span style="color: mediumblue">$W_7$</span> is but one of the spokes of the feature wheel, and encodes the 7-th feature. In this case, it is $0$ and has failed to encode the 7-th feature. Turns out, this is also a local optimum; in this Symmetric Failure setup, <span style="color: mediumblue">$W_7$</span> prefers to remain $0$. To understand why this is so, we have to flesh out the idea of "tilt" that I briefly mentioned above.

Here, we go back to $W^\top W$ and compare how it should ideally look, versus how it looks like in this Symmetric Failure case:

<img src = "../../images/opt_failure/ideal_vs_unideal_WTW.png" alt="Ideal vs Unideal WTW" width="100%">
*Left: ideal W.T @ W (all 7 features learnt). Right: Symmetric Failure W.T @ W (only 6 features learnt)*

The key observation is that the <span style="color: mediumblue">$7$-th row and column are $0$</span>. This means that the latent shape (in the 7D reconstruction space) does not span the 7-th dimension. This means that **the latent shape has no tilt with respect to the 7-th dimension.**

> **Help:** Do we call this the quotient space (Latent Plane / $e_7$)?

All of the $W_i$ introduce some tilt to the latent plane with respect to their own dimensions. If <span style="color: mediumblue">$W_7$</span> were to learn some non-zero value, it would introduce a tilt in the 7-th dimension. The simple way to see this is that if <span style="color: mediumblue">$W_7 \neq 0$</span>, then the <span style="color: mediumblue">$7$-th row and column</span> will be various non-$0$ values, which means that the plane given by $W^\top W$ will have some translation in the 7-th dimension, $\Longrightarrow$ tilt.

### How tilt is generated

The first thing to note about creating tilt with <span style="color: mediumblue">$W_7$</span>, is that the tilt will favor the <span style="color: mediumblue">$W_7$ direction</span>. This means that the further along <span style="color: mediumblue">$W_7$</span> you go on the latent plane, the more positive the reconstructed value in the 7-th dimension gets. Simple math to sketch it out:

$$
\begin{align*}
\hat{y}_7 & = \text{ReLU} \left( W_7^\top W x + b_7 \right) \\
\end{align*}
$$

Because $Wx$ is the hidden vector (i.e. "somewhere on the latent plane"), if $h = Wx$ has positive interference with <span style="color: mediumblue">$W_7$</span> (i.e. "along the positive <span style="color: mediumblue">$W_7$ direction"</span>), $W_7^\top Wx$ will be positive. The more $h = Wx$ lies along the positive <span style="color: mediumblue">$W_7$ direction</span>, the greater the value of $W_7^\top W x$, and the greater the value of $\hat{y}_7$.

Relatedly, the second thing to note is that the other features $W_i$ that lie on the same side of <span style="color: mediumblue">$W_7$</span> will also tilt towards a larger value in the 7-th dimension, while the other feathres $W_i$ on the other side of <span style="color: mediumblue">$W_7$</span> will tilt towards a smaller value in the 7-th dimension. To illustrate, let's suppose <span style="color: mediumblue">$W_7$</span> is a mini version of <span style="color: mediumslateblue">$W_6$</span>, i.e. <span style="color: mediumblue">$W_7$</span> $ = a \cdot $<span style="color: mediumslateblue">$W_6$</span> for small $a$:

<img src = "../../images/opt_failure/disc_tilted_W6.png" alt="W7 = 0.2 * W6" width="100%">
*Tilt introduced when W7 = 0.06 * W6*

For a tiny value of $a = 0.06$, we introduced a tiny tilt to the latent hexagon. **We limit the tilt such that the entire latent hexagon is still positive in <span style="color: mediumblue">dim-$7$</span>, for our initial analysis.** Both <span style="color: mediumblue">$W_7$</span> and <span style="color: mediumslateblue">$W_6$</span> are now tilting towards the positive direction along the <span style="color: mediumblue">dim-$7$</span> axis. To be more complete, features <span style="color: lightskyblue">$W_5$</span>, <span style="color: mediumslateblue">$W_6$</span>, and <span style="color: firebrick">$W_1$</span> are on the same side of the tilt as <span style="color: mediumblue">$W_7$</span> and are now more positive in the <span style="color: mediumblue">$7$-th dimension</span>, while features <span style="color: darksalmon">$W_2$</span>, <span style="color: peachpuff">$W_3$</span>, and <span style="color: darkgrey">$W_4$</span> are on the other side of the tilt, and have less positive values in the <span style="color: mediumblue">$7$-th dimension</span>. 

Remember that correct value that we would like the model to reconstruct in the <span style="color: mediumblue">$7$-th dimension</span> is $0$ for the first 6 data samples, because each of the first 6 data samples are not supposed to have an activated <span style="color: mediumblue">$7$-th feature</span>. This means that because <span style="color: lightskyblue">$W_5$</span>, <span style="color: mediumslateblue">$W_6$</span>, and <span style="color: firebrick">$W_1$</span> now have a more positive value (further from $0$) in the <span style="color: mediumblue">$7$-th dimension</span>, they are more WRONG, whereas because <span style="color: darksalmon">$W_2$</span>, <span style="color: peachpuff">$W_3$</span>, and <span style="color: darkgrey">$W_4$</span> now have values closer to $0$ in the <span style="color: mediumblue">$7$-th dimension</span>, they are now more CORRECT.

You may think that there is symmetry here - the total additional error incurred (in terms of L2 distance) by the newly upward-tilted features is equal to the error reduced by the newly downward-tilted features - so the overall "correctness" of the untilted model and the tilted model is still the same, but this is not the case. What's happening here is that because our loss (MSE) is convex (specifically quadratic), the additional error incurred by the upward-tilted features is larger than the error reduced by the newly downward-tilted features. We can sketch out a quick proof by looking at the loss changes incurred by <span style="color: mediumslateblue">$W_6$</span> and its mirror, <span style="color: peachpuff">$W_3$</span>:

$$
\begin{align*}
\text{Old Loss} &= \text{Old Loss}_{W_6} + \text{Old Loss}_{W_3} \\
& = 2 b_7^2 \\
\text{New Loss} &= \text{New Loss}_{W_6} + \text{New Loss}_{W_3} \\ 
& = (b_7 + \Delta)^2 + (b_7 - \Delta)^2 \\
& = 2 b_7^2 + 2 \Delta^2 > 2 b_7^2
\end{align*}
$$

This is simply Jensen's Inequality at work. Again, convexity gives us the standard result of L2-regression (OLS regression) preferring mean values. **The consequence: any tilt will incur more loss and the model will learn to correct it by preferring to set <span style="color: mediumblue">$W_7$</span> back to $0$.** You can also calculate the gradients of the loss of each of the reconstructed data samples $\hat{y}^{(i \neq 7)}$ with respect to <span style="color: mediumblue">$W_7$</span> and observe that it will always point in the <span style="color: mediumblue">$W_7$</span> direction, which means that the update will be in the direction of <span style="color: mediumblue">$-W_7$</span>.

### When can the gradient favor more tilt after Symmetric Failure has happened?

In the previous section, we showed that by considering the other 6 features (i.e. $W_{i \neq 7}$), less tilt is always desired. However, we did not consider the error reduced by the newly learnt, non-$0$ value of <span style="color: mediumblue">$W_7$</span>. In particular, we are interested to see if the error reduced for <span style="color: mediumblue">feature 7</span> can compensate for the overall increase in error for the other 6 features caused by the tilt, and if so, under what circumstances. Because if there are scenarios where it can compensate for the tilt - and of course there must be, how else can Perfect Optimization happen? - then we will have understood a mechanism for avoiding Symmetric Failure well. Be warned, a lot of this is just banging out the math.

<span style="color: mediumaquamarine">**[Skip to the last paragraph of this section (just before "Asymmetric Recovery") if you don't want to follow the dry math]**</span>

Let's look at the gradient of the loss of the reconstruction of data sample <span style="color: mediumblue">$x_7$</span> w.r.t <span style="color: mediumblue">$W_7$</span>:

$$
\begin{align*}
\frac{\partial \mathcal{L} (x_7)}{\partial W_7} & = \left( \hat{y}^{(7)} - y^{(7)}\right) \cdot \mathbb{1} \left\{ W^\top W x_7 + b > 0 \right\} \cdot \frac{\partial (W^\top W x_7 + b) }{\partial W_7} \\
& = \left( \hat{y}^{(7)} - y^{(7)}\right) \cdot \mathbb{1} \left\{ W^\top W_7 + b > 0 \right\} \cdot \frac{\partial (W^\top W_7 + b) }{\partial W_7} \text{, because } x_7 = e_7\\

& = \left( \hat{y}^{(7)} - y^{(7)}\right) \cdot 
\begin{bmatrix}
0 \\ 0 \\ \vdots \\ 0 \\ \vdots \\ 1
\end{bmatrix}
\cdot 
\begin{bmatrix}
\text{don't care} \\
\text{don't care} \\
\vdots \\
\text{don't care} \\
\vdots \\
\frac{\partial (W_7^\top W_7 + b_7) }{\partial W_7}
\end{bmatrix} \text{, because the other 6 features don't activate} \\

& = 
\begin{bmatrix}
\text{don't care} \\
\text{don't care} \\
\vdots \\
\text{don't care} \\
\vdots \\
\left( \text{ReLU} \left( W_7^\top W_7 + b_7\right) - 1\right) 
\end{bmatrix}
\cdot 
\begin{bmatrix}
\text{don't care} \\
\text{don't care} \\
\vdots \\
\text{don't care} \\
\vdots \\
2W_7
\end{bmatrix} \\

& = 2 \left( \text{ReLU} \left( \|W_7\|_2^2 + b_7\right) - 1\right) \cdot W_7 \\
& = 2 \left( \|W_7\|_2^2 + b_7 - 1\right) \cdot W_7 \text{ (can drop ReLU as entire latent shape is > 0 in dim 7)}
\end{align*} \\

$$

And let's also look at the gradient of the loss of the reconstruction of another data sample (not <span style="color: mediumblue">$x_7$</span>), call it sample $x_j$ (where $j \neq 7$) w.r.t <span style="color: mediumblue">$W_7$</span>:

$$
\begin{align*}
\frac{\partial \mathcal{L} (x_j)}{\partial W_7} & = \left( \hat{y}^{(j)} - y^{(j)}\right) \cdot \mathbb{1} \left\{ W^\top W x_j + b > 0 \right\} \cdot \frac{\partial (W^\top W x_j + b) }{\partial W_7} \\
& = \cdots \\
& = \left( \text{ReLU} \left( W_7^\top W_j + b_7\right) - 0\right) \cdot W_j \\
& = \left( W_7^\top W_j + b_7\right) \cdot W_j \text{ (likewise, can drop ReLU)}
\end{align*}
$$

The sum of these gradients from all 6 $x_{j \neq 7}$ is hence:

$$
\begin{align*}
\sum_{j \neq 7} \frac{\partial \mathcal{L} (x_j)}{\partial W_7} & = \sum_j \left( W_7^\top W_j + b_7\right) \cdot W_j \\
\end{align*}
$$

There's actually a rather neat way to simplify this, and it hinges on the fact that the features $W_{j \neq 7}$ are arranged in a symmetric fashion, where the <span style="color: mediumblue">$W_7$</span> line is the line of symmetry. This means that all the components of gradients that are orthogonal to the <span style="color: mediumblue">$W_7$</span> direction will all cancel each other out. For example, if <span style="color: mediumblue">$W_7$</span> is "front" to us, then the left-ward direction of <span style="color: mediumslateblue">$\frac{\mathcal{L} x_6}{W_7}$</span> will cancel out the right-ward direction of <span style="color: firebrick">$\frac{\mathcal{L} x_1}{W_7}$</span>. Likewise, the left-ward direction of <span style="color: darkgrey">$\frac{\mathcal{L} x_4}{W_7}$</span> will cancel out the right-ward direction of <span style="color: darksalmon">$\frac{\mathcal{L} x_2}{W_7}$</span>. This means that we only have to consider the "front-back" <span style="color: mediumblue">($W_7$)</span> component of all these gradients, i.e. we can work with only the projections of these gradients onto <span style="color: mediumblue">$W_7$</span>.

$$
\begin{align*}
\sum_{j \neq 7} \frac{\partial \mathcal{L} (x_j)}{\partial W_7} & = \sum_{j \neq 7} \left( W_7^\top W_j + b_7 \right) \cdot \hat{W}_7^\top W_j \cdot \hat{W}_7 \text{, by projecting } W_j \text{ to } W_7  \\
& = \sum_{j \neq 7} \left\{ \left(W_7^\top W_j\right) \cdot \hat{W}_7^\top W_j + b_7 \cdot \hat{W}_7^\top W_j \right\} \cdot \hat{W}_7 \\
& = \sum_{j \neq 7} \left \{ \|W_7\|_2 \left( \hat{W}_7^\top W_j\right)^2 + b_7 \cdot \hat{W}_7^\top W_j\right\} \cdot \hat{W_7} \\
& = \hat{W}_7 \cdot \sum_{j \neq 7} \left\{ \|W_7\|_2 \left( \hat{W}_7^\top W_j\right)^2 \right\} + \hat{W}_7  \cdot \sum_{j \neq 7} \left\{ b_7 \cdot \hat{W}_7^\top W_j\right\} \\
& = \hat{W}_7 \cdot \sum_{j \neq 7} \left\{ \|W_7\|_2 \left( \hat{W}_7^\top W_j\right)^2 \right\} \text{, because all } W_j \text{ cancel out} \\
& = W_7 \cdot \sum_{j \neq 7} \left( \hat{W}_7^\top W_j\right)^2
\end{align*}
$$

I find this simplified form appallingly... simple; great because it significantly simplifies our calculations. So in order for us to favor tilt, the update has to be in the <span style="color: mediumblue">$W_7$</span> direction, which means the gradient has to be in the <span style="color: mediumblue">$-W_7$</span> direction, which means that:

$$
\begin{align*}
\frac{\partial \mathcal{L} (x_7)}{\partial W_7} + \sum_{j \neq 7} \frac{\partial \mathcal{L} (x_j)}{\partial W_7} & = (< 0) \cdot W_7 \\
2 \left( \|W_7\|_2^2 + b_7 - 1\right) \cdot W_7 + W_7 \cdot \sum_{j \neq 7} \left( \hat{W}_7^\top W_j\right)^2 & = (< 0) \cdot W_7 \\
2 \left( \|W_7\|_2^2 + b_7 - 1\right) + \sum_{j \neq 7} \left( \hat{W}_7^\top W_j\right)^2 & < 0 \\
2 \left[ 1 - \left( \|W_7\|_2^2 + b_7 \right) \right] & > \sum_{j \neq 7} \left( \hat{W}_7^\top W_j\right)^2
\end{align*} \\
$$

Without having to write out too much further math, we can see that this is impossible for any non-0 value of $W_7$. How? We simply observe that:
* Both sides are quadratic in $W_7$, which means the overall shape of $LHS - RHS$ is also quadratic.
* Since we are analyzing the regime where the entire latent hexagon is positive in <span style="color: mediumblue">dimension $7$</span>, that also means that <span style="color: mediumblue">$b_7 > 0$</span>. That means the maxmum value of $LHS$ is just $2$. In particular, to maximize the $LHS$, we want to minimize <span style="color: mediumblue">$\|W_7\|_2$</span>, and to minimize the $RHS$, we also want to minimize <span style="color: mediumblue">$\|W_7\|_2$</span>. **Both sides will contribute optimization pressure to force <span style="color: mediumblue">$W_7$</span> to $0$.**

### What if Symmetric Failure and $b_7 = 0$?

Very briefly, we can extend our intuition from the previous section to make conclusions for the case of Symmetric Failure and $b_7 = 0$. Most notably, this means that the latent hexagon is pivoted on the dim-$7 = 0$ hyperplane, such that if there is a tilt, half of the latent hexagon will tilt upwards, while the other half will tilt under dim-$7 = 0$. 

<img src = "../../images/opt_failure/disc_zero_dim7.png" alt="disc zero dim7" width="100%">
*Tilted latent hexagon at b7 = 0*


This is significant because the half that is in the negative region (in dim-$7$) will be "deactivated" by ReLU (i.e. set to $0$). This impacts $\mathcal{L}(x_{j \neq 7})$, where $j$ are the indices of the features that are negative in dim-$7$, because they are now $0$. This means that the model now has no loss for samples <span style="color: darksalmon">$x_2$</span>, <span style="color: peachpuff">$x_3$</span>, and <span style="color: darkgrey">$x_4$</span>, hence these samples no longer exert any optimization pressure to "untilt" the latent hexagon, but the other samples still exert that same untilting optimization pressure. The above calculation will still hold, just that there are now only half the number of $W_j$'s.

So now, **we conclude that when <span style="color: mediumblue">$b_7 \geq 0$</span> and we're in the Symmetric Failure case**, we're stuck. Constrained to  <span style="color: mediumblue">$b_7 \geq 0$</span>, **this is the best we can do.**

# Asymmetric Recovery

We have since found that once we've reached Symmetric Failure, assuming that the learnt $W_i$'s are all symmetric (equal magnitude and equally spaced apart in a circle), and $b_\text{unlearnt} \geq 0$, we're kind of stuck. Let's move on to the next mode of behavior and see what happens when the symmetric constraint on $W_i$'s is broken, and how that might happen in the first place.

To quickly describe, **Asymmetric Recovery** happens when we reach Symmetric Failure, but at some point, the $W_\text{learnt}$'s begin to skew such that one side has much larger features than the other, the $b_\text{learnt}$'s begin to become much more negative, and then $W_\text{unlearnt}$ and $b_\text{learnt}$ finally get learnt, achieving a 0-loss model.


<img src = "../../images/opt_failure/asymmetric_failure_1.png" alt="Symmetric Failure Reached" width="100%">
*Symmetric Failure Reached*

<img src = "../../images/opt_failure/asymmetric_failure_2.png" alt="Skew begins to show" width="100%">
*Skew begins to show*

<img src = "../../images/opt_failure/asymmetric_failure_3.png" alt="W7 increasing and b7 decreasing" width="100%">
*W7 increasing and b7 decreasing*

<img src = "../../images/opt_failure/asymmetric_failure_4.png" alt="Asymmetric Recovery achieved" width="100%">
*Asymmetric Recovery achieved*

> Note that the color mappings between these 'training progress' plots and the latent visualization plots are different, simply because I can't tell ahead of time (without excessively more code to write) which feature will be the unlearnt feature, and hence which feature to color dark blue.

Let's try to make sense of what's going on at the skew stage. The skew stage is characterized by elongating learnt feature vectors $W_\text{learnt}$ as well as $b_\text{learnt}$ values that get increasingly negative (except for the initial increase for a number of that). The fact that these 2 behaviors happen simultaneously makes sense to me.

<img src = "../../images/opt_failure/asymmetric_recovery_skew_stage.png" alt="Skew stage" width="100%">
*Skew stage*

If you look at the right view of the latent shape, you can realize that, for example, as <span style="color: lightskyblue">$b_5$</span> gets increasingly negative, the latent shape will shift in the negative direction in <span style="color: lightskyblue">dim-$5$</span>. This means that to preserve the activation sliver for <span style="color: lightskyblue">feature $5$</span>, <span style="color: lightskyblue">$W_5$</span> has to be longer such that it can still reach the positive region in <span style="color: lightskyblue">dim-$5$</span>.

Remember that the goal of the reconstruction of each feature is to have the model output a $1$ whenever that feature is supposed to be active. This affects the extent to which $W_i$ must protrude into the positive region of dim-$i$, because:

$$
y_i = \text{ReLU} (W_i^\top W_i - b_i)
$$

Because <span style="color: mediumslateblue">$b_6$</span> and <span style="color: firebrick">$b_1$</span> are less negative (smaller magnitude), <span style="color: mediumslateblue">$W_6$</span> and <span style="color: firebrick">$W_1$</span> are also shorter. This asymmetry is important because it actually helps us favor the forces that generate the tilt in <span style="color: mediumblue">dim-$7$</span>. Remember that the sum of all the data sample errors w.r.t to <span style="color: mediumblue">$W_7$</span> is:

$$
\begin{align*}
\sum_{j \neq 7} \frac{\partial \mathcal{L} (x_j)}{\partial W_7} & = \sum_j \left( W_7^\top W_j + b_7\right) \cdot W_j \\
\end{align*}
$$

In the Symmetric Failure case, we could appeal to arguments of symmetry to say that the total uptilt introduced is equal to the total downtilt introduced, but because of convexity, bla-bla-bla, the total gradient points in the <span style="color: mediumblue">$W_7$</span> direction, which makes the update favor untilting. However, now, the vectors on the opposite side of <span style="color: mediumblue">$W_7$</span> (negative $W_7^\top W_j$) are significantly longer than those on the same side of <span style="color: mediumblue">$W_7$</span> (positive $W_7^\top W_j$). This overcomes the tendency towards untilting induced by the convexity of MSE, and allows the total gradient to point in the <span style="color: mediumblue">$-W_7$</span> direction, and for the update to favor tilting! Another way of phrasing is that **the imbalance of the learnt vectors causes the local optimum of <span style="color: mediumblue">$W_7$</span> to shift away from $0$.**

At some point, the plane tilts so much that some of the features $W_{j \neq 7}$ begin to dip into the negative <span style="color: mediumblue">dim-$7$</span> region. This means that the loss incurred by those corresponding data samples become $0$.

<img src = "../../images/opt_failure/asymmetric_recovery_half_zero.png" alt="Some features (W3, W4) achieve 0 loss in dim-7" width="100%">
*Some features (W3, W4) achieve 0 loss in dim-7*

This is not convenient for us because we have fewer contributors to the tilting force. At some point, all the learnt features $W_{j \neq 7}$ on the *other side* of <span style="color: mediumblue">$W_7$</span> (i.e. negative dot product with <span style="color: mediumblue">$W_7$</span>) will be in the negative <span style="color: mediumblue">dim-$7$</span> region, generating no more loss. The only features generating loss left will be the features on the same side of <span style="color: mediumblue">$W_7$</span> (<span style="color: mediumslateblue">$W_6$</span>, <span style="color: firebrick">$W_1$</span>), and <span style="color: mediumblue">$W_7$</span> itself. Except for <span style="color: mediumblue">$W_7$</span>, the other features will generate an untilting force, which will easily outweigh the tilting force generated by <span style="color: mediumblue">$W_7$</span>. This initially seemed like it would be a problem, but turns out, by the time this would happen, <span style="color: mediumblue">$W_7$</span> would already have such significant magnitude that the added performance of being able to encode the <span style="color: mediumblue">$7$-th feature</span> would outweigh the slight increase in loss of the features <span style="color: mediumslateblue">$6$</span> and <span style="color: firebrick">$1$</span>.

## How does one side magically start to elongate?

If we consider only the 6 learnt features, we have already achieved a 0-loss solution; the 6 features are equally spaced out on a wheel, and have more or less the same magnitude. The fact that there is a "skew" that is being introduced, and that the skew always "sweeps" the learnt feature vectors to the opposite side of the unlearnt feature (<span style="color: mediumblue">$W_7$</span>), can only mean that this is due to either <span style="color: mediumblue">$W_7$</span> or <span style="color: mediumblue">$b_7$</span>.

### Small asymmetry in Symmetric Failure?

It is a fact that the 6 learnt features are never truly symmetric, because gradient descent is a numerical method; it is merely almost exactly symmetric. Some learnt features are slightly shorter than others, but the model can compensate by adjusting the biases and the angles between the features accordingly to compensate. Let's look at how this happens by deliberately introducing some asymmetry.

Suppose we introduce a perturbation such that <span style="color: mediumslateblue">$W_6$</span> and <span style="color: firebrick">$W_1$</span> are now both a tiny bit shorter than the other $W_{j \neq 7}$. Now we'll have some loss, because the activations for features <span style="color: mediumslateblue">$6$</span> and <span style="color: firebrick">$1$</span> are not as strong, and there's interference. This triggers a series of updates that will cascade the loss from one side (the side of <span style="color: mediumslateblue">$W_6$</span> and <span style="color: firebrick">$W_1$</span>) the other side of the latent space:
1. Because of insufficient activation of features <span style="color: mediumslateblue">$6$</span> and <span style="color: firebrick">$1$</span>, the model would update itself and try to **elongate <span style="color: mediumslateblue">$W_6$</span> and <span style="color: firebrick">$W_1$</span> and decrease <span style="color: mediumslateblue">$b_6$</span> and <span style="color: firebrick">$b_1$</span>**. 
2. Because features <span style="color: mediumslateblue">$6$</span> and <span style="color: firebrick">$1$</span> are also now interfering with each other, the update will also want to **push <span style="color: mediumslateblue">$W_6$</span> and <span style="color: firebrick">$W_1$</span> away from each other**.
3. This causes features <span style="color: mediumslateblue">$6$</span> and <span style="color: firebrick">$1$</span> to now interfere with features features <span style="color: lightskyblue">$5$</span> and <span style="color: darksalmon">$2$</span> respectively, so the same updates of pushing <span style="color: lightskyblue">$W_5$</span> and <span style="color: darksalmon">$W_2$</span> away (move to the right side) and outward (elongate) will happen.
4. By the same mechanism, these updates will cascade all the way to all the features on the opposite of the latent space.

Pictorially:

<img src = "../../images/opt_failure/construct_asymmetry_1.png" alt="constructing asymmetry" width="100%">
*Constructing a slightly asymmetric, 0 loss solution*

This type of "skewing" of the feature vectors towards one side would also happen if instead of <span style="color: mediumslateblue">$W_6$</span> and <span style="color: firebrick">$W_1$</span> being slightly shorter, <span style="color: mediumslateblue">$b_6$</span> and <span style="color: firebrick">$b_1$</span> were more positive (because the same effects of insufficient activation and interference would happen). Here's one such "slightly asymmetric" solution - notice how the feature vectors seem to be more splayed out on the left side, and more elongated on the right side, as if there's a repulsor on the left and an attractor on the right:

<img src = "../../images/opt_failure/construct_asymmetry_2.png" alt="constructing asymmetry" width="100%">
*One such slightly asymmetric, 0 loss solution*

We've now observed that any small perturbation in weights near the Symmetric Failure scenario results a series of small updates (skewing features to one side) to quickly re-establish 0 loss. This is how the model compensates for small asymmetries to arrive at a new loss-less equilibrium. **Though it's great that the model can recover like this, there's unfortunately no "run-away" or self-reinforcing mechanism that would cause the skewing action to keep happening to the extent that it does in "Asymmetric Recovery"** - what gives?

## Small asymmetry favoring tilt
In the previous section, we examined what happens if there are small asymmetries in the learnt features, and we considered **only** the learnt features and their corresponding training data samples. However, **if we added the unlearnt vector back in, we find that this generates the run-away sweeping effect we are looking for!** Let's start with a Symmetric Failure (with the above subtle asymmetry) set of weights. Note that:
* <span style="color: mediumslateblue">$W_6$</span> and <span style="color: firebrick">$W_1$</span> are ever so SLIGHTLY shorter, and correspondingly, <span style="color: mediumslateblue">$b_6$</span> and <span style="color: firebrick">$b_1$</span> are more positive (around $\approx -0.988$, while the rest are $-1.02 \sim -1.03$). 
* <span style="color: darksalmon">$W_2$</span> and <span style="color: lightskyblue">$W_5$</span> are now tilted SLIGHTLY to the right side (**important!**)
* This set of weights has $\approx 0$ loss on the 6 examples for which the learnt features correspond to.

<img src = "../../images/opt_failure/created_asymmetric_7.png" alt="Example 'Symmetric Failure' with some asymmetry" width="400px">
*Example 'Symmetric Failure' with some asymmetry*

**The first thing to note that having asymmetry in the 6 learnt vectors induces a new optimum on <span style="color: mediumblue">$W_7$</span> that is not $0$ anymore.** Previously, when the 6 learnt vectors formed a perfectly symmetric hexagon, we could say: if the latent hexagon tilted one way, half of the predictions will have a larger value ($\approx \frac{1}{7} + \Delta$) for the <span style="color: mediumblue">$7$-th logit</span>, while half of the predictions will have a smaller value ($\approx \frac{1}{7} - \Delta$), but since loss (MSE) is convex, the additional loss incurred by the larger values will outweigh the loss savings of the smaller values, hence the system will want to remove tilt.

In this case, notice that:
* There are more vectors on 1 side (<span style="color: darksalmon">$W_2$</span>, <span style="color: peachpuff">$W_3$</span>, <span style="color: darkgrey">$W_4$</span>,  <span style="color: lightskyblue">$W_5$</span>) than the other (<span style="color: mediumslateblue">$W_6$</span>, <span style="color: firebrick">$W_1$</span>). So, if the tilt is such that the more populous side moves down in <span style="color: mediumblue">dim-$7$</span>, there are more contributors to the loss savings.
* The vectors on the left side (<span style="color: mediumslateblue">$W_6$</span>, <span style="color: firebrick">$W_1$</span>) are shorter, which means that the $\Delta$ along  <span style="color: mediumblue">dim-$7$</span> incurred by the tilt on each of these 2 vectors is going to be slightly less than those of their opposite vectors.

**These two factors contribute to the loss having a local optimum** at <span style="color: mediumblue">$W_7 = \varepsilon \cdot (W_6 + W_1)$</span>, i.e. **still very near $0$, but shifted somewhere in the <span style="color: mediumslateblue">$W_6$</span> and <span style="color: firebrick">$W_1$</span> direction**. I empirically confirmed this by freezing all the learnt weights and letting <span style="color: mediumblue">$W_7$</span> settle at its local optimum.

The second to note is that **having a non-zero <span style="color: mediumblue">$W_7$</span> now means that the learnt vectors will also want to update in the opposite side of <span style="color: mediumblue">$W_7$</span>**:

$$
\begin{align*}
\frac{\partial \mathcal{L}(x_7)}{\delta W_{j \neq 7}} & = (y_7 - x_7) \cdot \mathbb{1} \left\{ W^\top W_j + b > 0 \right\} \cdot \frac{\partial (W^\top W_j + b) }{\partial W_j} \\
& = 
\begin{bmatrix}
0 \\ 0 \\ \vdots \\ 0 \\ \vdots \\ W^\top_7 W_j + b_7
\end{bmatrix}
\cdot 
\begin{bmatrix}
\text{don't care} \\
\text{don't care} \\
\vdots \\
\text{don't care} \\
\vdots \\
\frac{\partial (W_7^\top W_j + b_7) }{\partial W_j}
\end{bmatrix} \text{ (the other 6 features are correctly reconstructed)} \\
&= (W_7^\top W_j + b_7) \cdot W_7

\end{align*}
$$

This means that there are 3 forces at play (first 2 described in previous section):
1. (Due to asymmetry of learnt vectors) insufficient activation for learnt features (updates will favor elongating feature vectors)
2. (Due to asymmetry of learnt vectors) inteference between learnt features (updates will favor splaying out and sweeping vectors to one side)
3. (Due to <span style="color: mediumblue">$W_7 \neq 0$</span>) all other features will update in favor of the <span style="color: mediumblue">$-W_7$</span> direction (i.e. also sweep towards <span style="color: mediumblue">$-W_7$</span>)
I've illustrated what the weight updates look like when we start from this set of weights:
<video width="100%" controls style="box-shadow: none; -webkit-box-shadow: none;">
  <source src="../../images/opt_failure/asymmetric_updates.mov" type="video/mp4">
</video>

> The lengths of the updates can be misleading since I've normalized them such that the biggest update has magnitude 1. In particular, it is not that the updates (`learning_rate * - gradient`) of $W_{j \neq 7}$ are getting larger, but that the dominant update (<span style="color: mediumblue">$W_7$</span>) is decreasing very rapidly, so the others look like they are increasing rapidly in comparison.

We know that the first 2 forces will result in a small amount of splaying / sweeping, but because this splaying / sweeping creates an imbalance (the opposite side of <span style="color: mediumblue">$W_7$</span> is now more populous and longer), this shifts the local optimum of <span style="color: mediumblue">$W_7$</span> even more towards the positive <span style="color: mediumblue">$W_7$</span> direction. This means that the third force, which is function of <span style="color: mediumblue">$W_7$</span>, will increase (since <span style="color: mediumblue">$W_7$</span> is now larger). This causes more sweep, which shifts the local optimum of <span style="color: mediumblue">$W_7$</span> yet again, so on and so forth. This, is the run-away tilting effect we are looking for. Eventually, the skew in vectors is so large that the local optimum for <span style="color: mediumblue">$W_7$</span> basically becomes so positive that it over-activates the <span style="color: mediumblue">$7$-th logit</span> and causes <span style="color: mediumblue">$b_7$</span> to very quickly decrease into being negative, achieving no loss on all data samples (Asymmetric Recovery).

<img src = "../../images/opt_failure/asymmetric_recovery_achieved.png" alt="Asymmetric Recovery Achieved" width="100%">
*Asymmetric Recovery Achieved (color coding of vectors is different)*

## Asymmetric Recovery is Chaotic
The real kicker here is that this sort of recovery can happen not only in the subtly asymmetric case, but in the Symmetric Failure case as well - after all, the probability that the 6 learnt vectors ($W_{j \neq 7}$) in the Symmetric Failure case are perfectly symmetric is $0$. so there is **always** some subtle asymmetry. In reality, because our learning algorithm (gradient descent) is iterative, this recovery doesn't happen very much though, because it relies on the learning rate and the specific configuration of asymmetry (which vectors are too long / short, by how much, etc.) to be in a sweet spot. **The closer you are to perfect Symmetry Failure, the more susceptible asymmetric recoverability is to chaotic behavior (critical)**, because gradient descent steps are very likely to change the confluence of factors that give rise to the above 3 forces. Perhaps you took too large a step and now <span style="color: mediumblue">$W_7$</span> has flipped direction, causing the third force to also flip, or perhaps an update to <span style="color: mediumslateblue">$W_6$</span> and <span style="color: firebrick">$W_1$</span> pushed them outwards too far and the local optimum for <span style="color: mediumblue">$W_7$</span> gets set closer to $0$. Perhaps the asymmetry wasn't so fortuitous to begin with (e.g. $W_{j \neq 7}$ were alternatingly short and long). This explains why Asymmetric Recovery is very rare (anecdotally, I trained 100 randomly seeded such auto-encoders and only got 1).
> In fact, this is general. Knowing whether a model is going to exhibit some behavior (converge, diverge, achieve Asymmetric Recovery) is in general hard to predict for iterative algorithms. I found this blogpost / preprint by Jascha Sohl-Dickstein on [the fractality of neural network training (when plotting whether training converges vs diverges over learning rates)](https://sohl-dickstein.github.io/2024/02/12/fractal.html) to make a lot of sense. His illustrations of the chaotic patterns (it seems more merely chaotic than fractal to me) are particularly relevant to what's going on here. (And of course, I later discovered that he works at Anthropic... another reason to love their research).

The reason I focused on this special case of having 2 adjacent features perturbed is that this is basically rigging the weights such that asymmetries line up with each other (perturbed features are on the same side, and also the same side as <span style="color: mediumblue">$W_7$</span>). It hence offers a good shot of achieving Asymmetric Failure (it is not too near critical points). 

And because seeing is believing, to see the chaotic behavior in our case, we can construct a similar plot of how much splaying / sweeping happens (I'm most interested in this phenomenon because it is a crucial prerequisite to Asymmetric Recovery) over 2 variables: the magnitude of the perturbation to both <span style="color: mediumslateblue">$b_6$</span> and <span style="color: firebrick">$b_1$</span>, and learning rate. The sum of the magnitudes of all the $W_j$ are the metric I plot here as a measure of how much splaying / sweeping there is:

<img src = "../../images/opt_failure/chaos_exp_10000_with_annotations.png" alt="Asymmetric Recovery Achieved" width="100%">
*Interpret this as 'how much splay / sweep occurs with a certain perturbation introduced to b6 & b1 and at a certain learning rate?'*

We notice two things from this plot:
- The amount of splaying / sweeping generally increases with learning rate, but starts to get patchy beyond a certain point. This indicates that the ability of a model to achieve splaying / sweeping / Asymmetric Recovery isn't just a question of whether there's a *sufficiently high* learning rate, but rather, if the learning rate in a sweet spot that depends on all other factors (which features are too long / short, which biases are too large / small, by how much, and so on). **This demonstrates the chaotic behavior that we postulate.** I'd like to point out, though, that this seems like very much equivalent to how doing standard Stochastic Gradient Descent (SGD) on a simple convex surface can either converge (learning rate is sufficiently small) or diverge (learning rate too large); in this case, **perhaps our problem isn't chaotic**, but that it's so complex that it effectively requires us to search over an exponential combination of factors - not chaotic, **but impractical.**
- At a fixed amount of perturbation to <span style="color: mediumslateblue">$b_6$</span> and <span style="color: firebrick">$b_1$</span>, the amount of splaying / sweeping is a function of learning rate. Notice also that at basically 0 perturbation to <span style="color: mediumslateblue">$b_6$</span> and <span style="color: firebrick">$b_1$</span>, we still see some amount of splaying / sweeping. This reiterates just how much influence SGD has on optimization failure as an iterative algorithm. 

# Bringing It All Together

So far, we have:
- Illustrated what the latent space of our model looks like, and how there are latent zones, the sizes of which are determined by the need to have "sufficient activation", which
- Answers the question of what is meant by "Optimization pursues linear separability," and answers the question of why shorter features need to be more widely spaced apart.
- Understood why Optimization Failure (Symmetric Failure) happens - the local optimality of Symmetric Failure - but that
- Small fortuitous asymmetries can cause a runaway splaying / sweeping of learnt features to happen, which is a prerequisite for Asymmetric Recovery,
- which happens when there is so much splaying / sweeping that the local optimum for the unlearnt feature is sufficiently far away from $0$ that it's basically "learnt"

# Avoiding / Correcting Optimization Failure

So then, what next? The ideal next step is obviously to prevent Optimization Failure altogether, but seeing that this phenomenon is chaotic (or at least "non-polynomially complex"), this is basically impossible. The best we can do is to reason probabilistically about the best conditions that we can try to create to avoid or correct Optimization Failure, and this, Anthropic has basically begun to do. In this next section, I will examine their [neuron resampling method](https://transformer-circuits.pub/2023/monosemantic-features#appendix-autoencoder-resampling) to see how well they work, how we may reason theoretically about it, and if we can do better.