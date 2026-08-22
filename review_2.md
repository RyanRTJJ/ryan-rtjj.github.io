# Review: Crucial Properties of $W_{hi}$ and $W_{hh}$

## 1. The input direction is not an eigenvector of the recurrent transformation

The relevant $2\times2$ submatrices of $W_{hh}$ do not have the corresponding $W_{hi}$ embedding direction as an eigenvector:

$$
W_{hh,(0,1),(0,1)}W_{hi,(0,1)}
\not=
\lambda W_{hi,(0,1)}
\qquad\text{for every }\lambda.
$$

If they did, then the contributions from $x_0$ and $x_1$ would remain on one line before the second ReLU. Their non-collinearity gives the pair $(x_0,x_1)$ two independent geometric directions.

## 2. The recurrent transformation approximately flips a nearby direction

Although $W_{hi,(0,1)}$ is not an exact eigenvector, there must be a nearby direction $v$ that is approximately flipped by the recurrent transformation:

$$
W_{hh,(0,1),(0,1)}W_{hi,(0,1)}
\approx
\lambda v,
\qquad
\lambda\approx-1,
\qquad
v\approx W_{hi,(0,1)}.
$$

This creates the approximate $x_1-x_0$ component while preserving the non-collinear remainder identified in section 1, which supplies the second geometric direction needed for the two-dimensional pre-ReLU regions.

## 3. The dominant direction of `relu_1_pre` lies in the ReLU-preserving orthant

The model must preserve not only the difference $x_1-x_0$, but also large changes in which $x_0$ and $x_1$ move together. Those inputs lie near the diagonal $x_1=x_0$. If the image of this high-variance direction did not lie in the positive orthant of the active pair, ReLU would set one coordinate to zero over a substantial part of the input region and destroy that information. Therefore the dominant principal direction of `relu_1_pre` must point approximately along the positive diagonal.

Along the line $x_1=x_0$, the relevant pair of preactivations is proportional to

$$
W_{hh,(0,1),(0,1)}W_{hi,(0,1)}
+W_{hi,(0,1)}.
$$

For the positive ray, $x_1=x_0>0$, this vector must point approximately along the positive diagonal:

$$
\begin{bmatrix}1\\1\end{bmatrix}.
$$

For the sign-mirrored pathway, the corresponding vector must satisfy the same positive-orthant condition after the coordinate reversal encoded by $W_{hi}$ and $W_{hh}$.

$$
W_{hh,(3,2),(3,2)}W_{hi,(3,2)}+W_{hi,(3,2)}
\quad\text{points approximately along}\quad
\begin{bmatrix}1\\1\end{bmatrix}.
$$

Otherwise ReLU would silence part of the comparison information in one of the two sign-symmetric pathways.

## 4. The two recurrent pathways are mirror images

The rows and columns of $W_{hh}$ corresponding to dimensions $(0,1)$ and $(3,2)$ are approximately reversed copies of one another. Together with the corresponding sign reversal in $W_{hi}$, this gives:

- the $(0,1)$ pathway a positive-orthant ReLU-surviving region;
- the mirrored $(3,2)$ pathway the corresponding positive-orthant region after coordinate reversal, appearing as the negative-orthant mirror in the original coordinate convention;
- both positive and negative values of each input a surviving representation.

## 5. The abstract relationship between $\ell$ and $W_{hi}$

Let $\ell$ be the unit vector defining the red–brown separation axis. The two independent requirements are

$$
\ell^\top W_{hi}>0
$$

and

$$
\ell^\top W_{hh}^{2}W_{hi}\approx0.
$$

The first fixes the orientation of the $x_2$ displacement along $\ell$. The second removes the $x_0$ contribution after two recurrent transformations. The relation involving $\ell^\top W_{hh}W_{hi}$ concerns the scale and location of the $x_2=x_1$ boundary, not an additional structural relationship between $\ell$ and $W_{hi}$.
