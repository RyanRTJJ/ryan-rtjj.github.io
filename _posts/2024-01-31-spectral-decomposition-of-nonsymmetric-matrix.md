---
published: no
title: Spectral Decomposition of Non-Symmetric Matrix
date: 2024-01-31 00:00:00 -500
categories: [statistics]
tags: [statistics]
math: true
---

So I recently had to revisit some of my work that used Common Spatial Pattern (the covariance orthogonalization technique I wrote about [here](https://amagibaba.com/posts/common-spatial-pattern/) - totally non-necessary to read that for understanding this post), and so of course I had to revisit what it meant to take an eigendecomposition of something. I believe I speak for most people when I say that the eigendecomposition makes the most sense in context of the singular value decomposition (SVD):

$$
\begin{align*}
A & = U \Sigma V^\top \\
A^\top A & = Q \Lambda Q^{-1} \\ 
& = Q \Lambda Q^\top \\ 
& = U \Sigma ^2 V^\top 
\end{align*}
$$

Here's the intuition for the relationship between eigendecomposition and SVD: **ALL** matrices have an SVD but only full-rank square matrices have an eigendecomposition. In this case, the eigendecomposition is precisely equal to its SVD. So in a sense, the eigendecomposition is a special case of the SVD. The SVD also expresses a matrix in terms of its directions (singular vectors) of descending column-wise covariance (singular values), and all its singular vectors are orthogonal to each other, which gives us a great geometric picture of being able to decompose the covariances into non-interfering parts (directions). So, from the SVD, we achieve three intuitive insights:
1. We can find the set of basis vectors (singular vectors $u_i$ and $v^\top_i$) that are aligned with the directions of greatest covariance in the matrix. 
2. We can find the multiplicative factor (singular values $\sigma_i$) that is akin to the standard deviation along these basis vectors.
3. Because all matrices can be expressed as $A = U \Sigma V^\top$, and since $U$ and $V$ are orthogonal and can be understood as rotations, all matrices (which are just linear transformations when applied to an operand) can be understand as a sequence of "rotate, stretch, rotate."

This all hinges on the fact that $U$ and $V^\top$ are orthogonal matrices. However, $U$ and $V^\top$ are the left and right eigenvectors that come from the eigendecomposition of $A^\top A$, and in general, the left and right eigenvectors of a full-rank square matrix are **NOT orthogonal**! In the case of SVD, we are taking the eigendecomposition of $A^\top A$, which is a real symmetric positive-definite matrix, which guarantees orthogonal eigenvectors, which allows us to get the intuitive geometric insights as listed above. However, this really begs the question - why is $Q$ in the eigendecomposition in general not orthogonal, and how does a non-orthogonal set of eigenvectors look like?! If they are not orthogonal, this totally breaks the intuition of the eigenvectors being a set of basis vectors that I carry from the land of SVD into eigendecomposition!

# Contents
1. Schur Triangularization Theorem

<img src = "../../images/schur/cuboid_dot_A.png" alt="LSTMs and Attention" width="100%"><br/>

I'm fully aware of all the different types of attention exist (here's a [really nice post by one of my favorite technical writers, Lilian Weng, that summarizes them](https://lilianweng.github.io/posts/2018-06-24-attention/)) and LSTMs don't fit squarely into any of those boxes, **BUT**, the purpose and general mechanism of all these gates still very much are those of attention mechanisms.