---
layout: post
title: "Manifolds - Part 2"
excerpt: "In the second part, I'll construct a manifold from scratch and redefine vectors, dual vectors, and tensors on a manifold."
comments: true
---

In the previous article, we reviewed vectors, duals, and tensors in a flat coordinate system, i.e., Euclidean space. Now that we have a good understanding of those generalizations in flat space, we can construct a manifold and re-invent the same machinery.

# Introduction

So far, we've only dealt with Euclidean spaces. However, there are plenty of spaces that are only locally Euclidean, but, globally, have a more interesting topology. This is the informal definition of a **manifold**: a space that is locally flat but globally more interesting. This has some profound connotations for how vectors, duals, and tensors are defined, as well as how we perform any kind of calculus (differentiation and integration) on this manifold. To be more precise, we can work with manifolds that don't allow for calculus on them, i.e., non-differentiable manifolds, but those are much less interesting, and, practically, we'll usually be able to perform calculus on our manifolds.

# Manifolds

I gave an intuition of manifolds, but let me define a few concrete examples that you've likely seen or heard of:

* $\R^n$: $\R^n$ is globally Euclidean as well as locally Euclidean!
* $S^n$: the $n$-sphere is a manifold ($S^1$ is a circle; $S^2$ is a sphere; $S^3$ is a glome; etc.).
* $\mathbb{T}^n$: the $n$-dimensional torus is a manifold.
* $\mathbb{G}^n$: an $n$-genus is a manifold: the $n$ denotes the number of "holes". A $0$-genus is a sphere; a $1$-genus is a torus; a $2$-genus has two holes, like the number 8; etc.
* matrix groups: the set of continuous rotations in $\R^3$ that leave the origin fixed, i.e., a **Lie group**. We'll discuss these in a different article 😉, but they're also essential to both particle physics and astrophysics.
* spacetime, as we know it: in the previous article, I stated that my personal motivation to learn about manifolds was to understand general relativity. In that framework, spacetime is a 4D manifold, 3 space-like dimensions and 1 time-like dimension.
* $S^1\times\R$: in other words, a cylinder!
* A Mobius strip

With all of these examples, what isn't a manifold? Using that same definition, anything that isn't a manifold is a space where, at some point, it locally doesn't look like a flat, $\R^n$ space. There are a few contrived examples, but also a few practical examples:

* Anything with a boundary: at the boundary, the space doesn't look Euclidean.
* Intersections of different flat spaces, e.g., a plane with a line through it: at the intersection of the line and the plane, the space doesn't look Euclidean.
* A light cone: light cones are ubiquitous in general relativity, but they aren't manifolds because the point where the past and future light cones intersect doesn't look like a flat, Euclidean space!

## Preliminaries

As for the more rigourous definition, we'll be following Wald's textbook on general relativity; even though we rarely use the full definition of a manifold, I think it's a really neat construction that emphasizes several important characteristics of a manifold, e.g., indepedence of coordinates, no global frame, and independence of embedding space. Before we do that, however, I'll review some definitions of maps and functions since they're essential constructs in manifolds and differential geometry.

Given two sets $A$ and $B$, a **map** $\phi : A\to B$ assigns, to each $a\in A$, exactly one $b\in B$. We can think of this as a "generalization" of a function. With this definition, there are several different kinds of maps that are more specific:

* **one-to-one/injective**: $\forall b\in B$, there is at most $a\in A$ mappped to it by $\phi$. A technique you might have heard of for identifying injective function is the "horizontal line test": if there is a horizontal line that intersects with the function more than once, it is not an injective function. For example, $f(x)=x^2$ fails, since, for $f(x)=4$, $x=\pm 2$. Also, there may be $b\in B$ such that $\nexists a\in A$ such that $\phi(a)=b$. In other words, there is an element in $B$ such that no element in $A$ is mapped to it.

* **onto/surjective**: $\forall b\in B$, there is at least $a\in A$ such that $\phi(a) = b$. In other words, every $b\in B$ originates from some $a\in A$, even if $a$ is the same for more than two $b\in B$. An example of such a map is $f(x)=x^3$: each element in the $x$-axis is sent to some element in the $y$-axis. On the other hand, functions like $f(x)=e^x$ and $f(x)=\log x$ are not surjective since they don't span the entire $x$-axis.

* **one-to-one correspondence/bijective**: a function that is both one-to-one and onto. In other words, each $a\in A$ is sent to exactly one $b\in B$. For example, $x^3$ is bijective since it is both injective and surjective. As a corollary of the definition, for each bijection, there exists an inverse bijiection $\phi^{-1} : B\to A$ such that $\phi^{-1}(\phi(a)) = a$. From this definition, it's pretty easy to show that the composition of bijections is also a bjiection. 

![Types of functions](/images/manifolds-part-2/types-of-functions.svg "Types of functions")

<small>The top function is one-to-one; the middle function is onto; and the bottom function is bijective.</small>

One last thing we'll need about maps is composition of maps: if $\phi: A\to B$ and $\psi : B\to C$, then $(\psi\circ\phi): A\to C, a\mapsto\psi(\phi(a))$.

## Manifold Construction

Now that we've reviewed the preliminaries, let's construct a manifold! We'll start by defining an **open ball** as the set of all points $x\in\R^n$ such that $\lVert x - y\rVert < r$ for some $y\in\R^n$ and $r\in\R$. 

![Open ball](/images/manifolds-part-2/open-ball.svg "Open ball")

<small>An open ball is a really simple construct: a set of points inside of an open circle.</small>

(If we considered a closed ball, we'd have to worry about the boundary! As it turns out, we can completely construct a manifold with open balls rather than closed balls.) With that definition, we can define an **open subset** as a union of (a potentially infinite number of) open balls.

![Open set](/images/manifolds-part-2/open-set.svg "Open set")

<small>An open set is just a (possibly infinite) collection of open balls.</small>

In fact, we can say that a subset $U\subset\R^n$ is open iff $\forall u\in U, \exists$ an open ball at $u$ such that it is inside of $U$. In other words, we can say that $U$ defines the interior of an $(n-1)$-dimensional surface. As a concrete example, an open set $U$ for $\R^2$ defines the interior of an $1$-dimensional surface, i.e., the interior of a closed loop on a plane. For $\R^3$, this would define the interior of a closed surface.

Now that we have this arbitrary set, we can naturally and immediately define a **coordinate system**/**chart** on this open set as being a subset $U\subset M$ and a one-to-one function $\phi : U\to\R^n$ that maps the open set $U$ into the flat Euclidean space $\R^n$. For convenience, instead of applying $\phi$ to individual points, we can consider the **image** of $\phi$ for a set of points. This is defined to be the set of all points $\R^n$ that $U$ gets mapped to. As an example, we can consider the unit circle parameterized by $\theta$. Then we can define a chart such that $U=\\{ \theta \| \theta\in(0,\pi) \\}$ and $\phi(\theta)=\theta$. This maps the half-circle $\theta\in(0,\pi)$ to the real line by "flattening" it. In fact, we could have actually mapped the entire circle to the real line by flattening it, but, as we'll see, this is usually not possible for more complicated manifolds.

![Coordinate chart](/images/manifolds-part-2/coordinate-chart.svg "Coordinate chart")

<small>A coordinate chart maps an arbitrary open set to an open set in a flat space.</small>

Even though we can't usually use a single chart to cover a manifold, we could use multiple charts if we impose some additional constraints. This is called a **$C^\infty$ atlas**: an indexed family of charts $\\{(U_\alpha, \phi_\alpha)\\}$ such that

1. The union of all of the sets cover the manifold: $\bigcup_\alpha U_\alpha = M$. If they didn't, then we couldn't create a chart for some part of our manifold!

2. If two charts overlap, they are smoothly sewn together. More formally, if $U_\alpha\cap U_\beta\neq\emptyset$, then $\phi_\beta\circ\phi_\alpha^{-1} : \phi_\alpha(U_\alpha\cap U_\beta)\to\phi_\beta(U_\alpha\cap U_\beta)$. This is best explained in the figure below. This condition is the crux of manifold construction: we can smoothly sew together a bunch of locally flat spaces into a structure that is only locally flat, and we've said absolutely nothing about the global structure. The reason this is called a _$C^\infty$_ atlas is because all of the maps are $C^\infty$, in other words, continuous and infinitely differentiable.

![Smooth stitching](/images/manifolds-part-2/smooth-stitching.svg "Smooth stitching")

<small>This "smooth stitching" constraint is the most important part of the manifold definition: if we're in one open set, we can "hop" to an adjacent one using this property.</small>

Now we can finally get to the definition we've been waiting for! A **$C^\infty$ $n$-dimensional manifold** is a set $M$ with a **maximal atlas**. A **maximal atlas** is an atlas that contains every possible chart for that manifold. The reason we need a _maximal_ atlas is so we don't consider different atlases to be different manifolds. For example, if we had an atlas of a circle and another atlas that starts at 45 degrees relative to the first one, without the condition of a maximal atlas, we would have thought we had two different circles!

Note that in the construction of the manifold, we never mentioned anything about the space that the manifold may be embedded in or the global structure. We simply took a bunch of flat $\R^n$ spaces and smoothly sewed them together on their overlaps. Manifolds exist completely independent of the space they are embedded in. We can take a circle, embed it in either a plane or a space and the maps into the real line would be the same. In fact, there's a famous theorem called **Whitney's embedding theorem** that states any $n$-manifold can be embedded in _at most_ $\R^{2n}$. For example, a sphere $S^2$ can be embedded in at most $\R^4$, but, it turns out we can also embed it in $\R^3$. Another example is a Klein bottle, which is a $2$-manifold, but it can only be embedded in $\R^4$.

Now let's look at a few concrete examples of constructing a manifold from an atlas. We've seen an atlas for a circle, but we only covered it with a single chart. This doesn't quite fit the manifold construction because a single chart means we have a closed set and we need an open set. Let's fix that and use two overlapping charts to cover the circle:

$$
\begin{align*}
U_1 &=\Big\{\theta | \theta\in\Big(\frac{\pi}{4}, \frac{7\pi}{4}\Big)\Big\}, \phi_1(\theta)=\theta\\
U_2 &=\Big\{\theta | \theta\in\Big(\frac{3\pi}{4}, \frac{-3\pi}{4}\Big)\Big\}, \phi_2(\theta)=\theta\\
\end{align*}
$$

These two charts cover the circle with plenty of overlap, so they are open sets. This atlas isn't maximal, of course, but showing just one atlas is proof that a structure is a manifold.

![Atlas for a circle](/images/manifolds-part-2/atlas-for-a-circle.svg "Atlas for a circle")

<small>The atlas for a circle needs to use two charts to ensure openness, even though it could technically be covered with one chart.</small>

For a slightly more complicated example, let's consider the sphere $S^2$. This is one manifold where it is impossible to have a single chart that covers the manifold. We can split the sphere into two atlases using the Mercator projection by excluding the North and South Poles. We can use the planes $x^3=\pm 1$ as the two sets of $\R^2$ to project into. (recall that $x^3$ is a coordinate, not an exponent!) We will project a ray starting from one of the poles, intersecting the sphere, and landing on one of the planes. The two charts for our atlas are $U_1=\\{\text{all points excluding the North pole}\\}$ and $U_2=\\{\text{all points excluding the South pole}\\}$ with the maps

$$
\begin{align*}
\phi_1(x^1, x^2, x^3) &= \Big(\frac{2x^1}{1-x^3}, \frac{2x^2}{1-x^3}\Big)\\
\phi_2(x^1, x^2, x^3) &= \Big(\frac{2x^1}{1+x^3}, \frac{2x^2}{1+x^3}\Big)\\
\end{align*}
$$

This atlas hits all points on the sphere twice except for the North and South poles, which are hit only once; therefore, we still have an open set, and we can see that this hits all points on the sphere.

![Mercator projection](/images/manifolds-part-2/mercator-projection.svg "Mercator projection")

<small>Take either pole, project a beam from the inside through the surface to the outside, and record where it falls on the "catching" plane. This gives us a smooth map that projects the points on the circle into a flat space.</small>

So we've shown a sphere is indeed a manifold. Moreoever, since we're mapping the atlas into $\R^2$, we've shown it is specifically a 2-dimensional manifold.

## Tensors on a Manifold

Now that we've constructed the manifold, we need to re-introduce tensors, starting with vectors in the tangent space. In flat space, we already defined vectors to exist only at a point (to get around vectors in a curved coordinate system) and the collection of them all pointing in each direction to be the tangent space $T_p M$ at that point. First off, let's construct the tangent space. Unlike in flat space, we can't simply construct it by considering all vectors pointing in every direction because we haven't defined the tangent space! Instead, we might think of "creating" vectors by looking at all possible _curves_ $\xi : \R\to M, \lambda\mapsto\xi(\lambda)$ that go through a point $p$ and their tangent vectors at $p$. That would seem to give us basically the same result, but the problem lies in the parametrization of $\xi$: it's dependent on the coordinates of the manifold! In other words, our tangent vectors would be $\frac{\d\xi^\mu}{\d\lambda}$, which depend on the coordinates $\xi^\mu$. Recall that vectors are independent of all coordinates since they are geometric objects so we can't use this definition. Also, we're cheating here since we haven't defined what "tangent to a curve" even means!

We're still pretty close though. Instead, let's flip this notion and define the set of all continuous, infinitely-differentiable functions on the manifold $\mathcal{F}=\\{\text{all } C^\infty f : M\to\R\\}$. Given any function, we can define a directional derivative operator $\frac{\d}{\d\lambda}$ that can act on a function $f$ to produce $\frac{\d f}{\d\lambda}$. Notice that this doesn't depend on the coordinates since we're using a _scalar_ function $f$, not a curve under some coordinates! Now we can take a similar approach where we look at all possible directional derivative operators of functions through $p$ and define the tangent space to be that.

![Directional derivatives of curves](/images/manifolds-part-2/directional-derivatives-on-curves.svg "Directional derivatives of curves")

<small>At a point p, consider all possible (scalar) functions through that point. We can always take the directional derivative of a parameterized curve with respect to the parameter.</small>

However, in order to make that statement, we need to show the following conditions hold:

1. The space of all directional derivative operators forms a valid vector space. After all, a tangent space is a vector space.
2. The dimensionality of this vector space is the same as the manifold, i.e., $n$. Recall an $n$-manifold has tangent spaces of dimensionality $n$. This is because we've constructed the manifold with tangent spaces of $\R^n$ so the dimensionality has to match.

To show that the space of directional derivatives is a vector space, we need to show that two of these operators can be added and scaled and the result is also a directional derivative operator. The first part of this is pretty easy:

$$
a\frac{\d}{\d\lambda} + b\frac{\d}{\d\tau}
$$

The second part is a bit trickier. A directional derivative operator must be linear and obey the Leibniz product rule. From the equation above, we can already see that the operator is linear so we just need to show the product rule holds:

$$
\begin{align*}
\Big(\frac{\d}{\d\lambda}+\frac{\d}{\d\tau}\Big)(fg) &= f\frac{\d g}{\d\lambda} + g\frac{\d f}{\d\lambda} + f\frac{\d g}{\d\tau} + g\frac{\d f}{\d\tau}\\
&= \Big(\frac{\d f}{\d\lambda}+\frac{\d f}{\d\tau}\Big)g + f\Big(\frac{\d g}{\d\lambda}+\frac{\d g}{\d\tau}\Big)\\
&= \Big(\frac{\d}{\d\lambda}+\frac{\d}{\d\tau}\Big)(f)g + f\Big(\frac{\d}{\d\lambda}+\frac{\d}{\d\tau}\Big)(g)
\end{align*}
$$

Therefore directional derivatives form a valid vector space. It sounds rather interesting that an "operator" can form a vector space, but really any kind of object can form a vector space as long as it satisfies the constraints! (Personally, I think "linear space" is maybe a better name since the properties of a vector space are really just linearity and closure.)

The last thing we have to do is show that the dimensionality of this vector space is the same as that of the manifold. In Wald's textbook on general relativity, he shows this directly, but, in Sean Carroll's book, he uses a clever identity: the dimensionality of a vector space is the same as the number of basis vectors. Therefore we just need to show that the number of basis vectors for the tangent space is the same as the dimensionality of the manifold. In other words, we need to construct a basis for the tangent space.

Let's start by assuming some arbitrary coordinates $x^\mu$. Given that, there's a natural choice for the basis of directional derivatives: the partial derivatives of the coordinates $\partial_\mu$! Let's define the directional derivatives as a linear combination of the partial derivatives of some arbitrary coordinates. Then we need to show that the set of partial derivatives form a basis and the number of elements in that set is $n$, i.e., the dimensionality of our manifold. Since we're defining the directional derivatives as partial derivatives, we need to show that any directional derivative $\frac{\d}{\d\lambda}$ can be decomposed into a linear combination of the partial derivatives $\partial_\mu$.

![Partial derivatives](/images/manifolds-part-2/partial-derivatives.svg "Partial derivatives")

<small>For a set of coordinate functions on the manifold, the partial derivatives can form a basis for the directional derivatives.</small>

Since we're dealing with operators, it's much less error-prone if we define some arbitrary function $f:M\to\R$ that the operators act on that we'll remove at the end. We'll also need a curve $\xi:\R\to M$ since $\xi$ is the function that is actually parameterized by the $\lambda$ in the directional derivative $\frac{\d}{\d\lambda}$. Since we're at a point $p$, we'll also get a chart $\phi:M\to\R^n$ with coordinates $x^\mu$ for free!

To reiterate, our goal is to show that we can write $\frac{\d}{\d\lambda}$ as a linear combination of partial derivatives $\partial_\mu$. With all of the maps and spaces, we can draw this picture.

![Directional derivative map](/images/manifolds-part-2/directional-derivatives-maps.svg "Directional derivative map")

<small>The complicated set of maps can be used to show how any directional derivative can be teased apart into scalars and partial derivatives.</small>

Conceptually, we'll be applying $\frac{\d}{\d\lambda}$ to $f$, but realistically, we need to compose with $\xi$ since $\xi$ is the thing that is parameterized by $\lambda$.

$$
\begin{align*}
\frac{\d}{\d\lambda}f&\to\frac{\d}{\d\lambda}(f\circ\xi)\\
&=\frac{\d}{\d\lambda}[(f\circ\phi^{-1})\circ(\phi\circ\xi)]\\
&=\frac{\partial}{\partial x^\mu}(f\circ\phi^{-1})\frac{\d}{\d\lambda}(\phi\circ\xi)\\
&=\frac{\d}{\d\lambda}(\phi\circ\xi)\partial_\mu(f\circ\phi^{-1})\\
&=\frac{\d x^\mu}{\d\lambda}\partial_\mu(f\circ\phi^{-1})\\
&\to\frac{\d x^\mu}{\d\lambda}\partial_\mu f\\
\end{align*}
$$

In the last step, we use the fact that $\phi$ has coordinates $x^\mu$. Now we can remove $f$ since it was arbitrary:

$$
\frac{\d}{\d\lambda}=\frac{\d x^\mu}{\d\lambda}\partial_\mu
$$

Now we've shown that we can decompose an arbitrary directional derivative $\frac{\d}{\d\lambda}$ into a scalar $\frac{\d x^\mu}{\d\lambda}$ and a vector $\partial_\mu$. Thus, the set of $n$ partial derivatives actually do form a basis for the tangent space and we have $n$ of them! It's a little strange to think that an operator is a vector! (Maybe this is less surprising if you've taken any quantum mechanics and learned that operators can be represented as matrices.) In fact, this basis is so convenient that we give it a name: the **coordinate basis** $\hat{e}_{(\mu)}\equiv\partial\_\mu$. We don't have to use this basis, but it's often easy and convenient. One important thing to note is that this basis is not orthonormal everywhere like Cartesian coordinates in a flat space. In fact, if that were the case, then we would actually have a flat space!

Given this basis, we can write out the general vector and basis transformation laws from the index notation (this isn't exactly rigourous, but it works for now):

$$
\begin{align*}
\partial_{\mu'}&=\frac{\partial x^\mu}{\partial x^{\mu'}}\partial_{\mu}\\
V^{\mu'}&=\frac{\partial x^{\mu'}}{\partial x^{\mu}}V^{\mu}\\
\end{align*}
$$

Since we're using a coordinate basis, the components will change when the basis changes, and a change of coordinates means a change of basis as well.

So far, we've constructed the tangent space using partial derivatives as the basis vectors, but what about the cotangent space $T_p^* M$? How do we construct/define the basis for this space? Analogously to how we used the partials for the basis, we can use the gradients $\d x^\mu$ as the basis for $T_p^* M$. They used to be defined $\hat{\zeta}^{(\mu)}(\hat{e}_{(\nu)})=\delta^\mu\_\nu$, but we're going to upgrade them using our calculus notation:

$$
\d x^\mu(\partial_\nu)\equiv\delta^\mu_\nu=\frac{\partial x^\mu}{\partial x^\nu}
$$

In this case, $\d x$ is not an infinitesimal, but actually a kind of object called a differential form (specifically a one-form, also known as a gradient). A **differential form** is a $(0, p)$ antisymmetric tensor; a $0$-form is a scalar or scalar function, and a $1$-form is a gradient. There's more work we have to do to discuss differential forms, so, for now, it's ok to think of these as just dual vectors. From the definition, the set of gradients also form a basis for the cotangent space. (We can go through a similar process to apply the one-forms to vectors and show this, but it looks very similar to vectors so I'm going to skip it.) Similar to vectors, we can derive the transformation laws.

$$
\begin{align*}
\d x^{\mu'}&=\frac{\partial x^{\mu'}}{\partial x^\mu}\d x^\mu\\
\omega_{\mu'}&=\frac{\partial x^\mu}{\partial x^{\mu'}}\omega_\mu
\end{align*}
$$

Now that we've re-invented vectors and duals using the language of manifolds, we're ready to construct tensors. As you might think, this construction follows straightforwardly from the construction in flat space: we take the tensor product of the basis vectors (partial derivatives) and duals (gradients).

$$
\begin{align*}
T^{\mu_1\cdots\mu_k}_{\nu_1\cdots\nu_l}&=T(\d x^{\mu_1}, \cdots, \d x^{\mu_k}, \partial_{\nu_1}, \cdots, \partial_{\nu_l})\\
T&=T^{\mu_1\cdots\mu_k}_{\nu_1\cdots\nu_l} \partial_{\mu_1}\otimes\cdots\otimes\partial_{\mu_l}\otimes\d x^{\nu_1}\otimes\cdots\otimes\d x^{\nu_k}\\
T^{\mu_1'\cdots\mu_k'}_{\nu_1'\cdots\nu_l'}&=\frac{\partial x^{\mu_1'}}{\partial x^{\mu_1}}\cdots\frac{\partial x^{\mu_k'}}{\partial x^{\mu_k}}\frac{\partial x^{\nu_1}}{\partial x^{\nu_1'}}\cdots\frac{\partial x^{\nu_l}}{\partial x^{\nu_l'}}T^{\mu_1\cdots\mu_k}_{\nu_1\cdots\nu_l}
\end{align*}
$$

Almost everything is the same as it was in a flat space, except we upgraded our basis vectors and duals to partial derivatives and gradients (this also technically works in a flat space but is a bit overkill in that context). Just as with flat space, we have the metric tensor $g_{\mu\nu}$.

The last few things I'll point out is a small nuiance with notation. Recall the polar coordinates metric

$$
\d s^2=\d r^2+r^2\d\theta^2
$$

$\d s^2$ is just a symbol, but $\d r^2$ and $\d\theta^2$ are honest basis one-forms. That being said, for this case, our use of basis one-forms is consistent with the infinitesimal philosophy for now.

I'll end on some nomenclature that is popular in other sources (as well as some foreshadowing). A metric is said to be in **canonical form** if it is written as $g_{\mu\nu}=\mathrm{diag}(-1,\cdots,-1,+1,\cdots,+1,0,\cdots,0)$ where $\mathrm{diag}$ is a diagonal matrix with the diagonal entries as the arguments to the function. At a point, it's always possible to put the metric in this form: for a point $p\in M$, there exist coordinates $x^{\hat{\mu}}$ such that $g_{\hat{\mu}\hat{\nu}}$ is canonical and $\partial_\hat{\sigma}g_{\hat{\mu}\hat{\nu}}=0$. In other words, the metric is flat and its components are constant at $p$. Coordinates that satisfy these conditions are called **Riemann Normal Coordinates**:

$$
\begin{align*}
g_{\hat{\mu}\hat{\nu}}(p)&=\delta_{\hat{\mu}\hat{\nu}}\\
\partial_\hat{\sigma}g_{\hat{\mu}\hat{\nu}}(p)&=0
\end{align*}
$$

This gives us a convenient set of coordinates to work in initially, then we can generalize using tensor notation. If we can show our equation is true in this coordinate system, then it must be true in all coordinate systems because a tensor equation is true in all coordinate systems. We'll need some extra machinery to make this claim, but it stands nonetheless.

One last bit of terminology is the **metric signature**: the number of positive and negative eigenvalues of the metric. A metric is **Euclidean/Riemannian/positive-definite** if all eigenvalues are positive. This is the signature for most mathematical manifolds. A metric is **Lorentzian/pseudo-Riemannian** if it has exactly one negative eigenvalue and the rest are positive. This is the metric used in relativity as the metric of spacetime, with the negative eigenvalue acting as the time coordinate. (Alternative, we could flip the spacetime metric to have three negative eigenvalues for the spatial components and a positive eigenvalue for the temporal component.) A metric is **indefinite** if it has a mixture of positive and negative eigenvalues. A metric is **degenerate** if it has any zero eigenvalues; note that this means an inverse metric doesn't exist. If a metric is continuous and non-degenerate, its _signature_ is the same everywhere. In other words, if we start in a Lorentzian spacetime, the metric is non-degenerate and continuous so spacetime stays Lorentzian everywhere (at least, that's what we think now). In practice, we don't usualy deal with indefinite or degenerate metrics; in fact, in special relativity, we often assume a non-degenerate metric because a degenerate one wouldn't be terribly useful in the first place!

# Conclusion

In this post, we learned how to construct a manifold from fundamental objects like sets and how to re-invent vectors, duals, and tensors on the manifold. Let's take a second to review what we've learned in this part:

* Manifolds are constructed by smoothly sewing together atlases, which are open sets with a coordinate system/chart mapping the set to the tangent space/a flat space.
* The tangent space consists of partial derivatives of the coordinates, which are used to build directional derivatives.
* The cotangent space consists of gradients (one-forms) which are defined analogously to basis covectors in flat space.
* General tensors are still the tensor product of tangent and cotangent spaces.

In the next installment, we'll discuss the most important property of a manifold: curvature 😀
