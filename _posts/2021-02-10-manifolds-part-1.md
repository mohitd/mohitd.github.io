---
layout: post
title: "Manifolds - Part 1"
excerpt: "As the first in a multi-part series, I'll introduce manifolds and discuss how vectors, dual vectors, and tensors work in a flat, Euclidean space."
comments: true
---

Manifolds! This might be an esoteric word you've heard in the most arcane of contexts, but manifolds are really interesting structures that are useful in a variety of fields like mathematics (obviously!), physics, robotics, computer vision, chemistry, and computer graphics. My own motivation to study manifolds stems from general relativity. In that context, spacetime is defined as a 4D manifold with 3 spatial components and 1 temporal component. Almost every interesting structure that arises in general relativity is a result of the manifold structure (specifically the metric). The goal of this post is to introduce the machinery that is the Riemannian manifold!

Manifolds are a fairly large topic so here's an overview of the big picture that we'll be discussing:

1. Vectors, dual vectors, and general tensors in a flat, Euclidean space.
2. Construction of manifolds and curved spaces
3. Vectors, dual vectors, and general tensors again, but on a manifold this time
4. Geodesics and Curvature.

In the interest of accessibility, I'll assume you're comfortable with multivariable calulus (differentiation, integration, parametrization of curves) and linear algebra (vector spaces, determinants, bases, linear transforms). Although some of my examples do include physics, I won't assume any prior knowledge.

# Introduction

To motivate our discussion consider airplane pilots charting a course on the Earth, e.g., a sphere in 3 dimensions. The paths they take between two cities do not look like straight-line paths on a paper map. We know that the shortest distance between any two points in a Euclidean space is a line; so why don't pilots chart courses that are straight lines on the map? This is because the Earth is a *curved surface*; it's not like a flat Euclidean space. As we'll prove later, the shortest path between any two points on a sphere is the **great circle**, i.e., the cirle along the surface of the Earth such that the center of the cirlce is the center of the Earth. So pilots are using this fact to chart courses to take the least amount of time and fuel to get from one city to another.

Another way to convince yourself that a sphere is a curved surface is to consider what happens to neighboring parallel lines. Suppose we consider two adjacent longitudes at the equator. Place two people next to each other at the equator and ask them to point North. They'll start off parallel since they're neighboring, but, as they move towards the North pole, the directions they're pointing in will intersect until they end up at the North pole pointing in different directions! This kind of scenario can never happen in a Euclidean space: lines that start out parallel will forever stay parallel in a Euclidean space. So clearly, a sphere, intrinsically, is not a Euclidean space! However, the fact that we could have two "close enough" people initially parallel suggests that for "close enough" distances, we can think of the sphere as being like a flat Euclidean space.

This is exactly the intuition behind a **manifold**! Informally, it's a kind of space that is locally flat/Euclidean, but, globally, it might have a more complicated, non-Euclidean structure. Our goal is to understand these structures by re-inventing the things we know how to do in Euclidean space, namely differentiation and integration, on a manifold.

As we've learned in multivariable calculus, most of the interesting things we do in Euclidean space involve *vectors* so I want to start close to there. However, instead of just dealing with ordinary vectors, we're going to upgrade them to **tensors**, which can be thought of as a generalization of vectors, i.e., vector is a special kind of tensor. Working with tensors versus sticking with vectors won't seem immediately useful until we discuss how to construct a manifold, but, in general, we can think of tensors as being the most basic, fundamental object in geometry.

Tensors might be something you've heard of, especially in machine learning (e.g., Tensorflow). In that context, tensors are taken to be multidimensional arrays. This definition works in that context, but it forgoes the geometric properties of a tensor, which are critial to the appeal of using them. It also conflates the components of a tensor with the abstract, geometric object that is the tensor itself. (That being said, we can sometimes interpret the multidimensional arrays in machine learning as being geometric transformations between spaces, ending at the space where our training data is linearly separable; but that's a topic for a different time 😉.)

In addition to being a basic building block in geometry, another reason we like tensors is because *a tensor equation is true in all coordinate systems!* As you can imagine, this is an incredibly useful fact: if we have a tensor equation, we can work in whatever coordinate system we want and, if expressed correctly, end up with the right answer. When working with manifolds, or just curved coordinates, e.g., polar coordinates, in general, there isn't always a canonical set of coordinates like in Euclidean space (e.g., if I write the vector $\begin{bmatrix}1 & 1 & 1\end{bmatrix}^T$, you know exactly what I mean and can visualize it in your head). Tensors allow us to write equations and work with different quantities in a coordinate-free way. (After all, I don't want to write this post a hundred times for a hundred different coordinate systems!) For non-mathematical (and sometimes even mathematical) uses, at the end of the day, we'll have to pick coordinates to fully understand or implement the structure we're working with, but we can work out all of the theory independent of coordinates in case we change our mind.

So I've motivated why we're starting with tensors and up why we like them, but I haven't actually given a useful definition or construction of one, although I've given an example of a not-so-good definition for our purposes. We know tensors should have the same properties, or at least more general properties, as vectors since they're a generalization of them; however, tensors are also comprised of another kind of geometric object we're going to discuss called a *dual vector* (often shortened to just "duals") or *covariant vector*/*covector*. In a Euclidean space, we don't need to discuss duals, but we lose that convenience in a non-Euclidean space. As we'll see, every vector space implies the existence of a corresponding dual vector space so it's more powerful to have tensors comprise of both vectors and duals since they're separate-but-complementary, objects in a non-Euclidean space. Furthermore, we know from linear algebra that vectors transform in a very specific way: with a transformation matrix. Similarly, we want tensors to also have this property since it's essential for tensor equations to look the same in all coordinates.

Putting all of these pieces together, we arrive at our definition of a **tensor**: *a multilinear map constructed from vectors and dual vectors that obey the tensor transformation law*. This is also perhaps not the most useful definition at this point, but we're going to dissect each piece and initially work in the Euclidean space we're all very comfortable with.

# Vectors

Let's first review vectors in plain Euclidean space. There are a couple of definitions people refer to when thinking of a vector. Probably the most common is a displacement arrow with a magnitude and direction that can be slid around the space. This isn't a bad definition for some uses, but it's not a very good one for our case. As a counterexample, suppose we have curvy coordinates (like polar coordinates!) on a flat piece of paper. What does a vector look like in this space? Is the arrow straight? Or is it curved along the coordinates?

![Vectors](/images/manifolds-part-1/tangent-space.svg "Vectors")

<small>In polar coordinates, does a vector look like the image on the left where it is straight in a curved coordinate system or is it curved along the curved coordinates, like the image on the right?</small>

The vector being curved doesn't really match with the "displacement" notion. To get around this problem, let's define vectors only at a point $p$. In fact, let's take all possible magnitudes and directions at the point $p$ and construct the **tangent space** $T_p$ of vectors. This circumvents our problem with curved coordinates by only defining vectors in the tangent space *at a point*. Gone are the days we can think of vectors as displacements or sliding around a space! The tangent space is an honest vector space, but we'll prove this more formally later on.

As a reminder, a vector space is a collection of objects such that they can be linearly scaled and added to get another element in the collection. Mathematically, $U$ and $V$ are elements of a vector space and $a,b\in\mathbb{R}$, then

$$
\begin{equation}
(a+b)(U+V) = aU + bU + aV + bV
\label{eq:vector_space}
\end{equation}
$$

There are a few other properties, like existence of an identity element, but I think Equation $\eqref{eq:vector_space}$ is the most important. So these vectors live in an honest vector space called the tangent space (the "tangent" in the name will become apparent later). Any abstract vector can be decomposed into a set of components and basis vectors. These **basis vectors** must (1) have the same dimensionality of the vector space, (2) be linearly independent, (3) and span the space.

1. If the basis vectors weren't of the same dimensionality of the space, then we couldn't decompose an arbitrary vector into these basis vectors because vector addition of two vectors of different dimensionality is ill-defined.
2. **Linearly independent** means we can't write one basis vector as a linear combination of the others. If we could, then why would we need that one in the first place? If our basis were $u, v, w$ and $w=u+v$, then everywhere we use $w$, we could just use $u+v$; $w$ is redundant.
3. **Spanning the space** means that every vector in the space can be written as a linear combination of the basis vectors. If they didn't span the space, then there would exist vectors in our space that we couldn't construct! For any vector space, there are an infinite number of bases we could select. In a Euclidean space, we usually stick with the canonical basis.

$$
\begin{Bmatrix} \begin{bmatrix}1\\ 0\\ 0\\ \vdots\\ 0\end{bmatrix}, \begin{bmatrix}0\\ 1\\ 0\\ \vdots\\ 0\end{bmatrix}, \cdots, \begin{bmatrix}0\\ 0\\ 0\\ \vdots\\ 1\end{bmatrix}\end{Bmatrix}
$$

In general, on a manifold, there usually isn't such a convenient basis everywhere. Instead, let's assume we have some arbitrary basis vectors 
$\hat{e}_{(\mu)}$ where $\mu=0,\cdots,n$ is an index that iterates over the dimensionality of our space $n$. (I'm using Sean Carroll's notation where the parentheses in the subscript denote a *set* of vectors.) Then we can decompose a vector into components and basis vectors.

$$
\begin{equation}
V = V^\mu \hat{e}_{(\mu)} = V^0\hat{e}_{(0)} + V^1\hat{e}_{(1)} + \cdots + V^n\hat{e}_{(n)}
\end{equation}
$$

where $V$ is the abstract vector and $V^\mu$ are the **components**. There's a bit of notation to unpack here. The superscripts aren't exponents, but indices. There's an importance to index placement, i.e., upper versus lower, but we won't fully see that until we discuss dual vectors. We're using Einstein summation convention where we sum over repeated upper and lower indices. (Einstein himself claimed that this summation convention was one of his most important contributions!)

![Abstract vector](/images/manifolds-part-1/abstract-vector.svg "Abstract vector")

<small>A vector is an geometric object that exists independent of coordinates. We can impose a basis and look at the components of the vector in that basis. Changing bases changes components, but the abstract, geometric object is left unchanged.</small>

It's important to distuinguish the vector from its components. The vector is a geometrical object that is independent of any coordinates or basis. However, the *components* of the vector are dependent on the choice of basis. In linear algebra, we learned we can transform the components of vectors with a linear transform by multiplying by a matrix. We can express the same transformation law in our new summation convention like this.

$$
\begin{equation}
V^{\mu} = \Lambda_{\mu'}^{\mu} V^{\mu'}
\label{eq:vector_transform_law}
\end{equation}
$$

where $\Lambda_{\mu'}^{\mu}$ is the linear transformation matrix. We're representing different coordinates with a primed index $\mu'$ rather than a primed variable to emphasize that the geometric vector is still the same but the coordinates are transformed. The other notational thing about tensor equations is that the upper and lower indices on each side of the equation must match. In the above case, notice how the summed out index $\mu'$ and free index $\mu$ match up. The right-hand-side has no $\mu'$ because it's a dummy variable that's being summed over; the left-hand-side has an upper $\mu$ to match the one in $\Lambda_{\mu'}^{\mu}$. This is also a really useful tool to catch mistakes in equations: the indices don't match!

So Equation $\eqref{eq:vector_transform_law}$ allows us to change coordinates to get new vector components. This lets us work in a more convenient basis for computations, then convert our answer to whichever basis we need. But there's another way to transform the vector! Remember that the components are a function of the basis so changing the basis imposes a change in components! But how does the basis transform? We can derive the transformation law by using the property that an abstract vector $V$ is invariant under a coordinate change and relate the components and basis.

$$
\begin{align*}
V = V^{\mu}\hat{e}_{(\mu)} &= V^{\mu'}\hat{e}_{(\mu')}\\
\Lambda_{\mu'}^{\mu} V^{\mu'} \hat{e}_{(\mu)}&= V^{\mu'}\hat{e}_{(\mu')}\tag*{Apply Equation \eqref{eq:vector_transform_law}}\\
\end{align*}
$$

But $V^{\mu'}$ is arbitrary so we can get rid of it.

$$
\Lambda_{\mu'}^{\mu} \hat{e}_{(\mu)} = \hat{e}_{(\mu')}\\
$$

Now to solve for the $\mu$ basis in terms of the $\mu'$ basis, we need to multiply by the inverse matrix $\Lambda_\mu^{\mu'}$, which is still a valid linear transform. The resulting transformation law for basis vectors can be written as the following. 

$$
\begin{equation}
\hat{e}_{(\mu)} = \Lambda_\mu^{\mu'}\hat{e}_{(\mu')}
\end{equation}
$$

Notice the indices are in the right place! To transform the basis, we have the multiply by the *inverse* of the matrix used to transform the components. Here's another way to express that these matrices are inverses.

$$
\Lambda_{\nu'}^{\mu}\Lambda_{\rho}^{\nu'} = \delta_\rho^\mu\\
$$

where $\delta_\mu^\nu$ is the Kronecker Delta that is equal to 1 if $\mu=\nu$ and 0 otherwise. (This is the Einstein summation convention equivalent of the linear algebra definition of inverse: $AA^{-1}=A^{-1}A=I$ where $I$ is the identity matrix.)

To review, we have transformation laws for the vector components and the basis vectors.

$$
\begin{align}
V^{\mu} &= \Lambda_{\mu'}^{\mu} V^{\mu'}\\
\hat{e}_{(\mu)} &= \Lambda_\mu^{\mu'}\hat{e}_{(\mu')}
\end{align}
$$

*Vector components transform in the opposite way as the basis vectors*! In other words, doubling the basis vectors halves the components. Historically, since vector components transform with $\Lambda_{\mu'}^{\mu}$, they are sometimes called **contravariant vectors**. Nowadays, we just call them vectors with upper indices.

Let's look at a numerical example of these transformation laws. Suppose we have a vector $\begin{bmatrix}1 & 1\end{bmatrix}^T$ in the canonical Cartesian basis. Now let's double the basis and see what happens to the components; this operation corresponds to applying the following transformation matrix to the basis vectors.

$$
\Lambda_{\mu'}^{\mu} = \begin{bmatrix}
2 & 0\\
0 & 2
\end{bmatrix}
$$

Try this out for yourself. Apply this matrix to each canonical basis vector and check the result is twice the length. With some linear algebra (or MATLAB/numpy), the inverse matrix to apply to the components is the following.

$$
\Lambda_{\mu}^{\mu'} = \begin{bmatrix}
\frac{1}{2} & 0\\
0 & \frac{1}{2}
\end{bmatrix}
$$

Indeed doubling the basis vectors halves the components: the basis vectors and vector components transform in the opposite way!


![Basis vectors](/images/manifolds-part-1/basis-vectors.svg "Basis vectors")

<small>When we double the basis vectors, the components are halved because they transform inversely to each other.</small>

A slightly more abstract example that we'll see all the time is a vector tangent to a curve. Suppose we have a parameterized curve $x^\mu(\lambda) : \mathbb{R}\to M$ where $\lambda$ is the parameter and $M$ is the manifold. (Note that the definition of a curve in a space $V$ is a function $\gamma : \mathbb{R}\to V$.) Einstein convention used here means we have a function for each component of the space $x^0(\lambda), x^1(\lambda), \cdots, x^n(\lambda)$. Then we can take a derivative with respect to $\lambda$ to get the tangent-to-the-curve vector $\frac{\mathrm{d}x^\mu(\lambda)}{\mathrm{d}\lambda}$.

# Dual Vectors

In the original definition of tensors I gave, there was another kind of object I said tensors were comprised of: dual vectors. Pedogogically, I've found that duals are difficult to motivate without starting off with a definition or flimsy motivation, but I'll do my best to try to draw on what we've learned so far.

We saw that the transformation law for vectors means that doubling the basis vectors halves the components. The natural question arises: "are there geometric objects such that doubling the basis vectors double their components?" It turns out there are! And, in fact, these objects are the second part of constructing tensors: dual vectors!

To discuss dual vectors, I'll start by saying a good way to understand structures in mathematics is to look at maps between them. In our specific case, we can try to understand our vector space $V$ better by looking at linear maps from it to the reals $\{ \omega : V\to \mathbb{R}\}$. In other words, a linear map $\omega$ *acts on* a vector to produce a scalar $\omega(V)\in\mathbb{R}$. This itself creates a new kind of space *dual* to the tangent space called the **cotangent space** $T_p^\*$ at a point $p$. It's constructed from all possible linear maps from the corresponding vector space to the reals. As it turns out, this is also a vector space! (Remember that many things in mathematics form a vector space; "vector space" is really a misnomer since plenty of things obey the properties required of a vector space besides conventional vectors.) If we have two linear maps $\omega$ and $\eta$ in the cotangent space and $a,b\in\mathbb{R}$, then

$$
(a+b)(\omega+\eta)(V) = a\omega(V) + b\omega(V) + a\eta(V) + b\eta(V)
$$

Since these functions are linear, we can express them as a collection of numbers. In linear algebra, we learned all linear operators and functions can be expressed as a "matrix times the vector input". Since the input here is a vector and the output a scalar, duals can be thought of to be *row vectors*. We'll circle back to this interpretation soon, but the key point is that we can represent duals in the same way we represent vectors: as components in a basis.

The basis for the cotangent space is defined to be $\hat{\varepsilon}^{(\nu)}$ such that the following property holds.

$$
\hat{\varepsilon}^{(\nu)}(\hat{e}_{(\mu)})\triangleq\delta_{\mu}^{\nu}
$$

Therefore, we can write a general dual vector as a combination of components and basis duals.

$$
\begin{equation}
\omega = \omega_\mu\hat{\varepsilon}^{(\mu)}
\end{equation}
$$

As we've discussed before, we can act a dual vector on a vector to get a scalar.

$$
\begin{align*}
\omega(V) &= \omega_\mu\hat{\varepsilon}^{(\mu)}(V^\nu\hat{e}_{(\nu)})\\
&= \omega_\mu V^\nu \hat{\varepsilon}^{(\mu)}(\hat{e}_{(\nu)})\\
&= \omega_\mu V^\nu \delta_\nu^\mu\\
&= \omega_\mu V^\mu\in\mathbb{R}
\end{align*}
$$

One good intuition to note is that applying the Kronecker Delta essentially "replaces" indices. We can think of the third line as "applying" the Kronecker Delta to $V^\nu$ to swap $\nu$ with $\mu$.

Similar to vectors, we can derive the transformation laws for dual vectors; specifically, we can use the index notation to our advantage to figure out the right matrices.

$$
\begin{align}
\omega_{\mu} &= \Lambda_{\mu}^{\mu'}\omega_{\mu'}\\
\hat{\varepsilon}^{(\mu)} &= \Lambda_{\mu'}^{\mu}\hat{\varepsilon}^{(\mu')}\\
\end{align}
$$

Notice that the dual components transform using the same matrix as the basis vectors $\Lambda_{\mu}^{\mu'}$. For this reason, historically, they are sometimes called **covariant vectors** or **covectors** for short. Nowadays, we just call them vectors with lower indices. (I'll use duals and covectors interchangeably.)

We've discussed the theory of dual vectors but I haven't given you a geometric description or picture of one yet. If we visualize vectors as arrows, we can visualize a dual as a stack of oriented lines/hyperplanes!

![Basis duals](/images/manifolds-part-1/basis-duals.svg "Basis duals")

<small>The top row shows the canonical basis $x$ and $y$ duals. The bottom row shows the $x$ and $y$ basis vectors with the dual basis as well.</small>

To act a dual on a vector to produce a scalar, we simply count how many lines the vector pierces and that gets us our scalar. As with the transformation law, if we double the basis vectors, the dual's components are also doubled. Graphically, this corresponds to the stack of lines getting more dense so the vector pierces more lines.

![Dual action](/images/manifolds-part-1/dual-action.svg "Dual action")

<small>To figure out the components of a dual, act the basis vectors on it. The dual pictured above has the components $\begin{bmatrix}1 & 1\end{bmatrix}$.</small>

Yet another way to think about duals is algebriacally: we can think of dual vectors as being row vectors while vectors are column vectors. Only in Cartesian coordinates can we simply transpose one to get the other. In general, a column vector and a row vector are two fundamentally different objects: they transform differently! (After we introduce the metric tensor, we can use it to convert freely between vectors and duals, but, since the components aren't usually identity, the conversion usually modifies the components.) Let's look at an algebraic example: the dual with components 1 and 1 would be written as the row vector $\begin{bmatrix} 1 & 1\end{bmatrix}$. Acting a dual on a vector then becomes matrix multiplication.

$$
\begin{align*}
\begin{bmatrix}1 & 1\end{bmatrix}\begin{bmatrix}1\\0\end{bmatrix} &= 1\\
\begin{bmatrix}1 & 1\end{bmatrix}\begin{bmatrix}0\\1\end{bmatrix} &= 1
\end{align*}
$$

A more important example of a dual vector is the gradient! In multivariable calculus, we learned the gradient of a scalar function $f$ produces a vector field $\nabla f$. However, the gradient is really a dual vector because of the way it transforms! Suppose we have a scalar function $f$, then we can define the gradient with the following notation:

$$
\mathrm{d}f = \frac{\partial f}{\partial x^\mu}\hat{\varepsilon}^{(\mu)}
$$

(Coarsely, upper indices in the denominator become lower indices.) There's a much deeper meaning to $\mathrm{d}f$: $\mathrm{d}$ is an exterior derivative operator that promotes the function $f$ from a *0-form* to a *1-form*. We'll discuss more about differential forms when we re-invent duals on a manifold. Getting back to why the gradient $\mathrm{d}f$ is a dual and not a vector, let's apply a transformation to change coords from $x^\{\mu'}$ to $x^\mu$. The components must transform like the following to preserve index notation:

$$
\begin{align*}
(\mathrm{d}f)_{\mu} &= \frac{\partial f}{\partial x^\mu} = \Lambda_{\mu}^{\mu'}\frac{\partial f}{\partial x^{\mu'}}\\
&= \partial_{\mu}f = \Lambda_{\mu}^{\mu'}\partial_{\mu'}f\\
\end{align*}
$$

Notice that this is exactly how the components of a dual transform, with $\Lambda_{\mu}^{\mu'}$! I've also introduced a new notational shorthand that we'll frequently use: $\partial_\mu f = \frac{\partial f}{\partial x^\mu}$.

# Tensors

With vectors and duals covered, we can revisit our definition of tensors: *a multilinear map constructed from vectors and dual vectors that obey the tensor transformation law*. This makes a bit more sense now, but we're going to fill in the gaps. With duals, we thought of them as linear functions that sent elements of our vector space to the reals; with tensors, we can think of them as multilinear functions, i.e., linear in each argument, that sends multiple vectors and duals to the reals. The **rank** of a tensor tells us how many of each the tensor takes: a rank $(k, l)$ tensor maps $k$ duals and $l$ vectors to the reals:

$$
T : \underbrace{T^*_p\times\cdots\times T^*_p}_{k}\times \underbrace{T_p\times\cdots\times T_p}_{l}\to\mathbb{R}
$$

To see the multilinearity property, consider a rank $(1,1)$ tensor $T(\omega, V)$ and some scalars $a,b,c,d\in\mathbb{R}$:

$$
\begin{align*}
T(a\omega + b\eta, V) &= aT(\omega, V) + bT(\eta, V)\\
T(\omega, aU + bV) &= aT(\omega, U) + bT(\omega, V)
\end{align*}
$$

which we can write more compactly as:

$$
T(a\omega + b\eta, cU + dV) = acT(\omega, U) + adT(\omega, V) + bcT(\eta, U) + bdT(\eta, V)$
$$

Thus, the entire tensor itself is linear since linear combinations of already linear things like vectors and duals produce linear things like tensors. Therefore, we should be able to decompose a tensor into its components in a particular basis. But how do we construct a basis? Well we know the bases for vector and dual spaces and tensors are comprised of both so we need to somehow "combine" the bases into a single one. The operation we need is the **tensor product** $\otimes$, which allows us build higher-rank tensors from lower-rank ones. Regarding ranks, the tensor product has the following property: $(k, l)\otimes(m,n)\to(k+m,l+n)$. Therefore, we can construct the basis for higher-rank tensor by taking the tensor product of the basis vectors and basis duals.

$$
\hat{e}_{(\mu_1)}\otimes\cdots\otimes\hat{e}_{(\mu_k)}\otimes\hat{\varepsilon}^{(\nu_1)}\otimes\cdots\otimes\hat{\varepsilon}^{(\nu_l)}
$$

To get a better understanding of the tensor product, let's once again look at the algebriac interpretation of the basis vectors and basis duals. For simplicity, let's use the canonical Cartesian basis vectors and basis duals, and, as an example, suppose we want to construct a rank $(1, 1)$-tensor, $\Lambda^\mu_{\mu'}$ for instance! We know this is a matrix so our basis for this should also be matrices. We'll take the tensor product of the basis vectors $\hat{e}_{(\mu)}$ and basis duals $\hat{\varepsilon}^{(\nu)}$ while treating them as column and row vectors. Doing this for each combination, we get the following basis for a $(1,1)$-tensor in a canonical basis.

$$
\begin{align*}
\hat{e}_{(0)}\otimes\hat{\varepsilon}^{(0)} &\to \begin{bmatrix}1\\0\end{bmatrix}\begin{bmatrix}1 & 0\end{bmatrix} = \begin{bmatrix}1 & 0\\0 & 0\end{bmatrix}\\
\hat{e}_{(0)}\otimes\hat{\varepsilon}^{(1)} &\to \begin{bmatrix}1\\0\end{bmatrix}\begin{bmatrix}0 & 1\end{bmatrix} = \begin{bmatrix}0 & 1\\0 & 0\end{bmatrix}\\
\hat{e}_{(1)}\otimes\hat{\varepsilon}^{(0)} &\to \begin{bmatrix}0\\1\end{bmatrix}\begin{bmatrix}1 & 0\end{bmatrix} = \begin{bmatrix}0 & 0\\1 & 0\end{bmatrix}\\
\hat{e}_{(1)}\otimes\hat{\varepsilon}^{(1)} &\to \begin{bmatrix}0\\1\end{bmatrix}\begin{bmatrix}0 & 1\end{bmatrix} = \begin{bmatrix}0 & 0\\0 & 1\end{bmatrix}\\
\end{align*}
$$

The resulting basis looks just like a canonical basis for $2\times 2$ matrices! Indeed we can take any matrix and write it as a scalar times one of these "basis matrices".

Now let's go back to our abstract basis and write out our tensor components in the abstract basis

$$
T = T^{\mu_1\cdots\mu_k}_{\nu_1\cdots\nu_l}\hat{e}_{(\mu_1)}\otimes\cdots\otimes\hat{e}_{(\mu_k)}\otimes\hat{\varepsilon}^{(\nu_1)}\otimes\cdots\otimes\hat{\varepsilon}^{(\nu_l)}
$$

Since tensors are comprised of vectors and duals, they transform like $(k, l)$ duals and vectors with $k+l$ transformation matrices.

$$
T^{\mu_1'\cdots\mu_k'}_{\nu_1'\cdots\nu_l'} = \Lambda^{\mu_1'}_{\mu_1}\cdots\Lambda^{\mu_k'}_{\mu_k}\Lambda^{\nu_1}_{\nu_1'}\cdots\Lambda^{\nu_l}_{\nu_l'}T^{\mu_1\cdots\mu_k}_{\nu_1\cdots\nu_l}
$$

Notice all of the indices are in the right place! Summation convention makes it really easy to verify if we've made a mistake or not. All of the operations on tensors we're going to cover really amount to keeping careful track of our indices. 

## The Metric Tensor 

Before discussing tensor operations, I want to introduce the most important tensor: the **metric tensor** $g$! It allows us to compute distances and angles in arbitrary coordinates. Specifically, the inner product in an arbitrary space is written in terms of the metric tensor:

$$
g(U, V) = g_{\mu\nu}U^\mu V^\nu
$$

where $g_{\mu\nu}$ are the components of the metric tensor. Notice the metric tensor is a $(0, 2)$-tensor so it takes two vectors to produce a scalar. In a Euclidean space, $g_{\mu\nu}=\delta_{\mu\nu}$ so the inner product simply the component-wise product $U_\nu V^\nu$, which is exactly how the dot product that we're familiar with is defined. In general spaces, however, the metric tensor is often not even constant and changes based on where we are in the space. We'll see an example of this with polar coordinates shortly.

In addition to representing the metric as a function, remember that we can also represent it by it's components in a basis, particularly two basis duals in this case:

$$
g = g_{\mu\nu}\hat{\varepsilon}^{(\mu)}\otimes\hat{\varepsilon}^{(\nu)}
$$

For the metric tensor, there's another way to express the components that is a bit more canonical: as a **line element**. For example, consider the line element of Cartesian coordinates.

$$
ds^2 = dx^2 + dy^2
$$

Notice that the nonzero components of the metric tensor in Cartesian coordinates are simply $1$ and so are the coefficients on $dx^2$ and $dy^2$. The zero components represent the $0$ coefficients on $dxdy$ and $dydx$, which is why there are no cross-terms.

For now, we can think of the line element as being an infinitesimal displacement, but there's actually a deeper meaning. $ds^2$ is just a symbol, but something like $dx^2$ is secretly the bilinear differential form $\mathrm{d}x\otimes\mathrm{d}x$ which is the exterior derivative $\mathrm{d}$ applied to the 0-form $x$. Anyways, in this notation, let's try to find a corresponding line element for polar coordinates and then write the metric tensor components.

We can start by writing the Cartesian coordinates $(x,y)$ in terms of the polar coordinates $(r,\theta)$:

$$
\begin{align*}
x &= r\cos\theta\\
y &= r\sin\theta
\end{align*}
$$

Now we take the total derivative $df = \frac{\partial f}{\partial x^\mu}dx^\mu$ of both sizes. (Note this is also a differential form!)

$$
\begin{align*}
dx &= \cos\theta\,dr - r\sin\theta\,d\theta\\
dy &= \sin\theta\,dr + r\cos\theta\,d\theta
\end{align*}
$$

Then we can "square" both sizes, being careful not to commute $dxdy$ and $dydx$ for good practice. As a side note, the metric tensor is indeed symmetric, but we haven't defined that just yet.

$$
\begin{align*}
dx^2 &= \cos^2\theta\,dr^2 - 2r\cos\theta\sin\theta(drd\theta + d\theta dr) + r^2\sin^2\theta\,d\theta^2\\
dy^2 &= \sin^2\theta\,dr^2 + 2r\sin\theta\cos\theta(drd\theta + d\theta dr) + r^2\cos^2\theta\,d\theta^2
\end{align*}
$$

Now we can add them and cancel the cross terms with $drd\theta + d\theta dr$.

$$
\begin{align*}
ds^2 &= dx^2 + dy^2\\
&= \cos^2\theta\,dr^2 + r^2\sin^2\theta\,d\theta^2 + \sin^2\theta\,dr^2 + r^2\cos^2\theta\,d\theta^2\\
&= (\cos^2\theta + \sin^2\theta)dr^2 + r^2(\sin^2\theta + \cos^2\theta)d\theta^2\tag*{Group like terms}\\
&= dr^2 + r^2\,d\theta^2\tag*{$\sin^2\theta + \cos^2\theta = 1$}
\end{align*}
$$

Now we can read the components of the metric tensor from the coefficients on $dr^2$ and $d\theta^2$:

$$
g_{\mu\nu} = \begin{bmatrix}1 & 0\\ 0 & r^2\end{bmatrix}
$$

Notice we're arranging the components in a matrix since the metric tensor is a $(0, 2)$-tensor. So it seems the metric in polar coordinates isn't constant and *does* depend on where we are in the space. This makes sense because, as we move farther away from the origin in polar coordinates, the arc length between any two angles increases. In fact, if we treat $ds^2$ as an infinitesimal displacement and keep the radius fixed, i.e., $dr^2=0$, then we get $ds^2=r^2 d\theta^2\to s=r \theta$ which is exactly the arc length formula! (I'm being a little loose with the notation, but the point still remains.)

## Tensor Operations 

Now that we've discussed the metric tensor, we can start looking at different operations we can perform on tensors. I want to cover these operations now rather than when they're needed so that we don't need to digress too often when we start to use them. One of the most important operations is *raising and lowering indices*: in other words, we're converting between upper and lower indices and converting between vectors and duals. We do this via the metric tensor. We'll need a small bit of additional machinery if we want to raise indices: the **inverse metric tensor**. Given a metric tensor, there exists an inverse $g^{\mu\nu}$ with two upper indices defined as the following.

$$
g^{\mu\lambda}g_{\lambda\nu} = g_{\nu\lambda}g^{\lambda\mu}=\delta^\mu_\nu
$$

The combination of these two lets us raise and lower indices. For example, we can convert between vectors and duals.

$$
\begin{align*}
V^\mu &= g^{\mu\nu}\omega_\nu\\
\omega_\nu &= g_{\mu\nu}V^\mu\\
\end{align*}
$$

This operation explains why we don't distinguish between vectors and duals in Euclidean space: since $g_{\mu\nu}=\delta_{\mu\nu}$, the components are the same, and we can simply transpose between column and row vectors. Of course, we can raise and lower arbitrary indices on arbitrary tensors for as many indices as we want.

$$
\begin{align*}
T^{\alpha\beta}_\mu g_{\alpha\rho} &= T^{\beta}_{\mu\rho}\\
T^{\alpha\beta\mu\nu} g_{\alpha\rho} g_{\beta\sigma} &= T^{\mu\nu}_{\rho\sigma}\\
T^{\alpha\beta}_{\mu\nu}g^{\mu\rho}g_{\alpha\sigma} &= T^{\rho\beta}_{\sigma\nu}
\end{align*}
$$

Notice that in the last equation, the ordering of the indices doesn't change. Another tensor operation is called **contraction**, which maps $(k,l)$ tensors to $(k-1, l-1)$ tensors by summing over an upper and lower index:

$$
T^\mu_\nu\to T^\mu_\mu = T\in\mathbb{R}
$$

Contractions are *only* defined for one upper and one lower; we can't contract two lower or two upper indices. Algebraically, if we represent $T^\mu_\nu$ as a matrix, then contraction is the same as taking the trace. As with raising and lowering, we can contract arbitrary indices from arbitrary tensors as long as we keep the ordering the same:

$$
T^{\alpha\mu\beta}_{\mu\rho} = S^{\alpha\beta}_\rho
$$

If we did want to raise or lower two upper or two lower indices, we'd have to first use the metric tensor to lower or raise one of them, then contract.

The final tensor operation we're going to look at for the moment is symmetrization and anti-symmetrization. We say a tensor is **symmetric in its first two indices** if $T_{\mu\nu\alpha\beta}=T_{\nu\mu\alpha\beta}$. A tensor is just **symmetric** if all pairs are symmetric. We've seen a symmetric tensor: the metric tensor! The metric tensor is symmetric $g_{\mu\nu}=g_{\nu\mu}$. This is apparent from the inner product being symmetric (although the latter is actually a collolary of the former).

On the other hand, we say a tensor is **anti-symmetric in its first two indices** if $T_{\mu\nu\alpha\beta}=-T_{\nu\mu\alpha\beta}$. An **anti-symmetric** tensor is defined in the same way as a symmetric tensor. A canonical example in physics is the electromagnetic field strength tensor/Faraday tensor $F_{\mu\nu}$. Electromagnetism is comprised of electric and magnetic fields. Often, they're treated as separate entities, but they're really two sides of the same coin. A more compact way to treat them is to put them into one tensor with components:

$$
F_{\mu\nu}=\begin{bmatrix}0 & -E_1 & -E_2 & -E_3\\ E_1 & 0 & B_3 & -B_2\\ E_2 & -B_3 & 0 & B_1 \\ E_3 & B_2 & -B_1 & 0\end{bmatrix} = -F_{\nu\mu}
$$

Note that swapping indices is the same as transposing the matrix representing the components. From these components, it's clear that $F_{\mu\nu}=-F_{\nu\mu}$. In linear algebra, this is also called a skew-symmetric matrix. As it turns out, we can take any tensor and symmetrize or anti-symmetrize it. To symmetrize a tensor, we take the sum of all permutation of the indices scaled by $\frac{1}{n!}$.

$$
T^\mu_{(\nu_1\cdots\nu_l)\rho} = \frac{1}{n!}\Big(T^\mu_{\nu_1\cdots\nu_l\rho} + \text{sum of permutations of }\nu_1\cdots\nu_l\Big)
$$

As an example, let's consider $T_{(\mu\nu\rho)\sigma}$:

$$
T_{(\mu\nu\rho)\sigma} = \frac{1}{6}\Big(T_{\mu\nu\rho\sigma}+T_{\nu\rho\mu\sigma}+T_{\rho\mu\nu\sigma}+T_{\rho\nu\mu\sigma}+T_{\nu\mu\rho\sigma}+T_{\mu\rho\nu\sigma}\Big)
$$

To anti-symmetrize a tensor, we take the alternating sum of all permutations of the indices scaled by the same factor. By alternating, we mean even exchanges of indices have a factor of $+1$ while odd number of index exchanges have a factor of $-1$.

$$
T^\mu_{[\nu_1\cdots\nu_l]\rho} = \frac{1}{n!}\Big(T^\mu_{\nu_1\cdots\nu_l\rho} + \text{alternating sum of permutations of }\nu_1\cdots\nu_l\Big)
$$

As an example, let's consider $T_{[\mu\nu\rho]\sigma}$:

$$
T_{[\mu\nu\rho]\sigma} = \frac{1}{6}\Big(T_{\mu\nu\rho\sigma}-T_{\mu\rho\nu\sigma}+T_{\rho\mu\nu\sigma}-T_{\nu\mu\rho\sigma}+T_{\nu\rho\mu\sigma}-T_{\rho\nu\mu\sigma}\Big)
$$

As I've stated before, most of these tensor operations are really about keeping careful track of the indices. With that concludes the tensor operations that we'll be using, at least for now! This is a good place to stop so I'll end the new content here since the next thing we're going to do is actually construct a manifold!

# Review 

We've covered many topics so far, so I want to take a second to review what we've learned:

* Manifolds are structures that are locally Euclidean, but globally might have some more interesting structure.
* Tensors are the most fundamental geometric object, constructed from vectors and dual vectors, and are invariant to all coordinates
* Vectors in general coordinates are defined at a point $p$ in vector space called the tangent space $T_p$.
* Any abstract vector can be decomposed into components in a basis.
* Basis vectors transform inversely to how the components transform.
* For each vector space, there is an associated dual vector space where dual vectors send elements of the vector space to the reals.
* Duals can also be represented as components in a basis whose components transform in the same way as the basis vectors.
* General tensors are comprised of vectors and duals and also written as components in a basis, constructed via the tensor product of the basis vectors and basis duals.
* The metric tensor is a $(0, 2)$-tensor that is used to compute distances and angles.
* Tensor operations like raising/lowering, contracting, and symmetrize/antisymmetrizing all maintain the Einstein notation.

With those points, we'll conclude this first part on manifolds. In the next installment, we'll actually construct a manifold 😀 and re-define most of this machinery on a manifold!

