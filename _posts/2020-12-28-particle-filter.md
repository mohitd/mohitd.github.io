---
layout: post
title: "Particle Filters for Robotic State Estimation"
excerpt: "Going beyond EKFs, I'll motivate particle filters as a more advanced state estimator that can compensate for the limitations of EKFs."
comments: true
---

In the [previous post](/ekf), we discussed Extended Kalman Filters for robotic state estimation. These are widely used for estimating unknown variables of a Gaussian, nonlinear dynamical system; specifically, we saw how we can use them to estimate the state of a robot traveling in the world. Although they're popular state estimators and work really well for some systems, they are not without their faults. In this post, we're going to explore another kind of state estimator that can overcome the limitations of the EKF: the particle filter!

# Limitations of EKFs

To motivate particle filters, let's take a second to recall some assumptions we've made about Kalman Filters (KFs) and Extended Kalman Filters (EKFs). Both work on the Gaussian assumption: the noises are fundamentally Gaussian. We always draw the process and sensor noise from Gaussian distributions. While many kinds of distributions in the real world are indeed Gaussians, it's a sliding scale: while some distributions are completely Gaussian, others are very poorly approximated by Gaussians. As an example, consider the exponential distribution.

![The exponential distribution](/images/particle-filters/exponential-distribution.svg "The exponential distribution")

<small>The exponential distribution is parameterized by $\lambda$, called the decay factor, which controls the "sharpness" or "flatness" of the graph.</small>

From the plot, we can see that this distribution is incredibly non-Gaussian and would be very poorly approximated as such. (This distribution isn't just some counterexample for the sake of counterexample: for example, bacteria growth is well-modeled by an exponential distribution.) Antoher such example is a multimodal distribution, i.e., one with multiple peaks. A Gaussian approximation would effectively average the peaks with a wide variance, depending on the distance between the peaks. These distributions are also common in robot localization: we have two rooms that have some similar features so the estimate can be in either of those rooms. (With subsequent navigation, we can further narrow down where we are.)

Additionally, we know KFs are only good for linear systems, and EKFs can, to some degree, handle nonlinear systems. Recall that, for EKFs, we simply replace the KF matrices with Jacobians linearized from the nonlinear motion and sensor models at the state estimate. But this linearization introduces error since we're just approximating the nonlinear function with a linearlized version. This causes some artifacts: the linearization point can be unstable, i.e., on a peak, and the farther we move from the linearization point, the worse the error.

![A very nonlinear function](/images/particle-filters/very-nonlinear-fuction.svg "A very nonlinear function")

<small>An example of a very nonlinear function. There are many bad linearization points that could throw off the approximation.</small>

For systems that are only *somewhat* nonlinear, we can afford to move slightly farther away from the linearization point without inducing too much error, assuming the linearization point isn't a bad one. However, for *very* nonlinear systems, the error accumulates much faster.

# Particle Filters

With those two caveats in mind, what can we do to support a broader case of distributions (non-Gaussians and multimodal distributions) and functions (highly-nonlinear functions and discontinuous functions)? Let's start by writing down our EKF equations from [the previous post](/ekf):

$$
\begin{align*}
    \hat{x}_k &= f(\hat{x}_{k-1}, u_k)\\
    P_k &= F_k \hat{x}_{k-1} F^T_k + Q_k\\
    \hat{x}_k' &= \hat{x}_k + K(z_k-h(\hat{x}_k))\\
    P_k' &= P_k - KH_k P_k\\
    K &= P_k H^T_k(H_k P_k H^T_k + R_k)^{-1}
\end{align*}
$$

An EKF is an example of a **parametric** state estimator (sometimes called **model-based**): we're directly parameterizing our result with a mean of $\hat{x}$ and covariance of $P_k$, and the points is to try to estimate those two *parameters*. If we want to represent distributions that aren't Gaussians or just arbitrary distributions, then we need to do away with directly estimating these parameters. Those kinds of state estimators are called **non-parametric** or **model-free**.

So how can we move from a parametric to non-parametric state estimator? The key insight is, for any distribution, instead of representing it by its parameters, e.g., mean, covariance, or decay factor, *we represent it by a collection of samples of the distribution*.

![A Gaussian represented by its samples](/images/particle-filters/sampled-gaussian.svg "A Gaussian represented by its samples")

<small>The blue plot is the true, underlying Gaussian distribution. The orange plot represents 100 samples taken from the underlying Gaussian. We can see that the resulting sampling distribution, even with just 100 samples, is giving us a good estimate of the true distribution.</small>

Let's build the intuition for how this works with concrete, known distributions. Suppose we're handed a collection of samples, called **particles**, generated from a secret distribution. Although we don't know the distribution itself, we can still try to compute some distribution-independent properties such as the mean and covariance. If we're later told that these samples were indeed taken from a Gaussian distribution, then we can compute the mean and covariance of the samples, and we're done characterizing the distribution! If we're told it's a Beta distribution, for example, although it's more work, we could solve for the parameters of the Beta distribution given the set of samples. (This also exemplifies how we can convert our non-parametric model to a parametric one for visualization or logging or other purposes.)

One other important thing to notice is how changes in the particles correspond to changes in the distribution. For example, suppose we take the same set of generated particles and translate each one by the same amount. If we're told the secret distribution is Gaussian and we recompute the mean and covariance with the new points, we'll find that the mean will be approximately translated by that same amount! The transformations of these particles correspond to transformations in the distribution. This further emphasizes the fact that the particles represent the distribution: changes in one correspond to changes in the other.

![Transformations of Gaussian samples](/images/particle-filters/transformed-gaussian.svg "Transformations of Gaussian samples")

<small>The blue plot is the underlying Gaussian before any transformation and the green plot is the underlying Gaussian after offsetting the mean by 5. The orange plot represents 100 samples taken from the pre-transformed Gaussian, and the red plot takes each sample from the pre-transformed Gaussian and translates it by 5 (notice that it's the exact same histogram translated by 5). The resulting sampling distribution after transforming the particles is also translated by 5 and looks similar to the post-transformed underlying distribution.</small>

Now that we have an intuition for this correspondence, I'm going to avoid speaking about any particular distributions, e.g., Gaussian, exponential, beta, the particles represent since the distributions tends to be arbitrary. Remember, that was the point: we want a way to represent a distribution in a non-parametric way so we'll be working exclusively with the particles and not any parameters of any particular distribution.

Now that we've established particles as the non-parametric way of modeling distributions, let's be slightly more specific about particle filters for localization (as the title of this post seemed to hint at). Let's assume a robot on a plane with a position and velocity so our state looks like the following.

$$
x_k = \begin{aligned}\begin{bmatrix}
	p_k \\
	v_k \\
\end{bmatrix}\end{aligned}
$$

If we were using EKFs, we'd have a mean vector and covariance matrix representing the distribution of all of the places our robot could be, but we've committed ourselves to a non-parametric approach so we can't use that! Instead, we can consider samples of this distribution to be our particles so each particle gets its own $\begin{bmatrix} p_k & v_k \end{bmatrix}^T$. In other words, each particle represents *an estimate of the true state of the robot*, i.e., it's position and velocity. We'll denote the state of a particular particle $i$ at a timestep $k$ as $x_k^{(i)}$ where I've kept the particle index $i$ in parentheses to avoid confusing it with an exponent.

To initialize our filter, we need to randomly initialize a bunch of the particles with random states. How many is a "bunch"? There's no one good answer; it's a tuning parameter that has to be empirically determined. We'll use $N$ as the placeholder for the number of particles. If we're stationary at the origin, then perhaps a good way to initialize each particle is to randomly sample from a uniform distribution centered around $0$ for both the position and velocity.

$$
\Big\{x^{(i)}_k\sim\mathcal{U}(-\varepsilon, +\varepsilon)\Big\}_{i=1,...,N}
$$

where $\mathcal{U}(a, b)$ is the uniform distribution for the interval $[a, b]$ and $\varepsilon$ is some small, positive number, i.e., $\varepsilon > 0$ and $\|\varepsilon\| \ll 1$. (Of course, we could have sampled from a Gaussian, but I decided to use the uniform distribution to show that we can use any distribution; also, the uniform distribution isn't a bad choice for this kind of initialization.)

With this, we've initialized our particle filter! Now, like any filtering approach, we need an motion and sensor model (see the [previous post](/ekf) for the motivation for these). Furthermore, just like with the filter initialization, we'll need to come up with non-parametric ways to apply these models.

## Predict

Our prediction model $f$ time-evolves our state from time $k$ to $k+1$. With EKFs, we had a nonlinear function transforming our mean and the Jacobian of that function transforming the covariance. Since we're working without an explicit mean and covariance, we're only allowed to work with the particles. The hint in figuring out the right way to implement a non-parametric motion model is something we've already discussed! Recall we noticed that transformations of the particles correspond to transformations in the underlying distribution. So, for the most part, all we need to do is apply a motion model $f$ to each particle independently. The additional nuiance is that we need to encode the uncertainty of the motion model $q_k$ directly in this update because we no longer have an explicit covariance matrix to represent that uncertainty. (You can pick $q_k$ to be sampled from your favorite distribution, Gaussian or otherwise: $q_k\sim\mathcal{N}(0, \sigma_q)$).

![Applying motion model to particles](/images/particle-filters/motion-model.svg "Applying motion model to particles")

<small>The blue particles represent the previous state and the orange particles represent the particles after the motion model has been applied to the previous state, with some added noise.</small>

After applying this to each particle, we get a new particle distribution.

$$
\Big\{x^{(i)}_{k+1} = f(x^{(i)}_k, u_k) + q_k\Big\}_{i=1,...N}
$$

(where $u_k$ is an optional control input. In our case, I've chosen to ignore it for simplicity since we already encode velocity in our state. If we wanted to account for second-order effects like acceleration/angular acceleration, then we could include it.)

The update function $f$ can actually be taken right from the EKF; remember to add in the noise at the end too. And that's the entirety of the motion model: we simply apply this function to each particle to get its new state estimate.

## Digression: Bayes Nets and Hidden Markov Models

The non-parametric motion model followed rather naturally from the parametric EKF motion model. However, the sensor model is a bit tricker to motivate directly from the EKF. Instead, I want to digress for a while to discuss the generalization of EKFs and particle filters: the **Bayes Net (BN)**. This might seem unrelated at first glance, but EKFs and particle filters share the BN as a common ancestor. In fact, EKFs are just a special case of the BN: one where all functions are linearized and all errors are Gaussian. The particle filter, as we've motivated before, doesn't have these requirements but is still a realization of an BN. From this generalization, we can easily motivate the sensor model.

Specifically, the BN structure that we're interested in is called a **Hidden Markov Model (HMM)** and looks like this.

![Hidden Markov Model](/images/particle-filters/hmm.svg "Hidden Markov Model")

<small>The HMM is a kind of Bayes Net where the variables, i.e., the blue rounded rectangles, are the unknown variables while the measurements, i.e., the green ellipses, are the known. The goal is to solve for the unknown state given just the very first prior distribution $x_0$ and the measurements $z_1, ..., z_k$.</small>

This structure makes sense because we're interested in figuring out the state of our robot given the previous state and the sensor measurements. The "catch" is that the previous state is a random variable! We don't know what its value is, but we do know the sensor measurements. So the goal is to solve for all of the random variables given the sensor measurements. But we can simplify this by noting that we only care about solving for the most recent state: if we're at $x_k$, we don't want to solve for $x_{k-4}$. It'd be simpler if we had *folded in* the previous state and sensor measurements so that we keep a sort of "running tally" state estimate. When we're adding a new state $x_k$, we want to *fold in* the information of the previous state $x_{k-1}$ into $x_k$. Similarly, when we get a new sensor measurement $z_k$ in $x_k$, we want to *fold in* $z_k$ into $x_k$ to refine it. We'll discuss precisely and mathematically what I mean by "*fold in*" in just a minute, but this process of "get a new state estimate by folding in the previous one; then fold in the sensor measurement; repeat" is called **recursive Bayes estimation**; we maintain only one "running" state, not the entire history.

One thing we notice about this structure is that there seems to be two kinds of arrows: one connecting states to states and one connecting states to sensor measurements. These are actually our motion and sensor models! Recall that our motion model tells us how to get a new state by time-evolving our previous state; this is exactly the dependence that arrow is capturing. In fact, we can use probability notation to generically discuss both the motion and sensor models. The motion model is written as $p(x_k\|x_{k-1})$: given the distribution over $x_{k-1}$ (explicltly mean and covariance in the case of KFs/EKFs), the motion model tells us how to get the distribution over $x_k$. This is exactly what the notation is telling us!

Interestingly, we could have structured the problem differently by requiring a particular state $x_k$ be conditioned on by the *past two* states $x_{k-1}$ and $x_{k-2}$ or the past $m$ states. While this is certainly something we can do (and is done in reality, but that's a different kind of state estimator for a different time 😉️), this makes the problem more difficult and doesn't align with how our EKF works. This assumption that any particular state $x_k$ is only dependent on the previous state $x_{k-1}$ is so important and widely used that it has a name: the **Markov assumption**. 

For our sensor model, recall that we thought of it as "mapping our state space into the observation space"; again, this is exactly what the other arrow is capturing. In probabilistic notation, the sensor model is written as $p(z_k\|x_k)$: given the distribution over $x_k$, the sensor model tells us how to get the distribution over $z_k$. This also sets up another assumption: a sensor measurement $z_k$ is only dependent on the corresponding state $x_k$ we made that measurement in.

Hopefully, by now, I've convinced you that this HMM structure makes sense for the problem at hand, and, at this stage, we have all of the tools and insight we need to try to solve the HMM for the most recent state.

But after all of that motivation, what's the quantity we're actually trying to solve for? State estimation in general is trying to solve for the latest state given the previous states and sensor measurements. From that intuition, we might think that $p(x_k\|x_1,x_2,...,x_{k-1}, z_1, z_2, ..., z_k)$ is the distribution to solve for, but that's not quite right because the unknown previous states are on the right-side of the "given" symbol even though they're unknown. So the distribution we're actually after is $p(x_k\|z_1, z_2, ..., z_k)$ and we can recursively "fold in" only the previous state (as per our Markov assumption). As a shorthand, we can write $p(x_k\|z_{1:k})$. Finally, this is the quantity in question we've been building up to solve!

Since recursive Bayes estimation is recursive, we can assume we're given $p(x_{k-1}\|z_{1:k-1})$. We also need a prior $p(x_0)$ for initialization which gives us the initial state distribution. Our task is to compute the latest state estimate $p(x_k\|z_{1:k})$ given the previous one $p(x_{k-1}\|z_{1:k-1})$, the motion model $p(x_k\|x_{k-1})$, and the sensor model $p(z_k\|x_k)$.

The first thing we need to do is to apply the motion model. This results in transforming the previous state $p(x_{k-1}\|z_{1:k-1})$ to $p(x_k\|z_{1:k-1})$.

$$
p(x_k|z_{1:k-1}) \overset{?}{=} p(x_k|x_{k-1})p(x_{k-1}|z_{1:k-1})
$$
 
But this isn't quite right because we have a free variable $x_{k-1}$ that's unaccounted for! We need some heavy-duty probability theory for this, but, as it turns out, we're not too far off from the right answer. We can directly the [Chapman-Kolmogorov Equation](https://en.wikipedia.org/wiki/Chapman%E2%80%93Kolmogorov_equation), which, in our scenario, states $p(A\|C) = \int p(A\|B)p(B\|C)\text{d}B$. Applying this, we get the right answer:

$$
p(x_k|z_{1:k-1}) =\int p(x_k|x_{k-1})p(x_{k-1}|z_{1:k-1}) \mathrm{d}x_{k-1}
$$

We got rid of that extra variable by integrating it out. Intuitively, this "folds it into" the resulting distribution. Finally, we get to the mathematical definition of "folding in": **marginalization**! To "fold in" a random variable, we **marginalize** it out by integrating over all possible values of that random variable. (Note that this operation is well-defined for random/unknown variables.) The information in $x_{k-1}$ is "folded into" the resulting distribution.

(For the case of KFs/EKFs, we can derive the predict step by plugging in Gaussians for $p(x_k\|x_{k-1})$ and $p(x_{k-1}\|z_{1:k-1})$ and solving for $p(x_k\|z_{1:k-1})$ in terms of the Gaussians for $p(x_k\|x_{k-1})$ and $p(x_{k-1}\|z_{1:k-1})$. The result can be massaged into looking like a Gaussian, and we can extract the equations for the mean and covariance from there. Remember that applying a Gaussian to another Gaussian produces a Gaussian.)

Now that we have $p(x_k\|z_{1:k-1})$, we need to apply the sensor model $p(z_k\|x_k)$ to refine the estimate into the final result of $p(x_k\|z_{1:k})$. For this, we can use Bayes Theorem to get the right answer:

$$
p(x_k|z_{1:k}) = \displaystyle\frac{p(z_k|x_k)p(x_k|z_{1:k-1})}{p(z_k|z_{1:k-1})}\propto p(z_k|x_k)p(x_k|z_{1:k-1})
$$

where

$$
p(z_k|z_{1:k-1}) = \int p(z_k|x_k)p(x_k|z_{1:k-1}) \mathrm{d}x_k
$$

is the normalization constant. (Notice the use of the Chapman-Kolmogorov Equation again.) This isn't easy to compute directly, but it rarely matters since, in practice, we usually just compute $p(z_k\|x_k)p(x_k\|z_{1:k-1})$ (which is proportional to $p(x_k\|z_{1:k})$) and manually normalize the result.

And that's it! We have the predict and update steps written in a generic way with no assumptions on nonlinearity or distributions:

$$
\begin{align}
p(x_k|z_{1:k-1}) &=\int p(x_k|x_{k-1})p(x_{k-1}|z_{1:k-1}) \mathrm{d}x_{k-1}\\
p(x_k|z_{1:k}) &= \displaystyle\frac{p(z_k|x_k)p(x_k|z_{1:k-1})}{p(z_k|z_{1:k-1})} \propto p(z_k|x_k)p(x_k|z_{1:k-1})\\
\end{align}
$$

That was a long digression, but it served the purpose to show the generalized predict and update steps so we can apply them (specifically the update step) to our particle filter because the update step doesn't follow so nicely from EKF.


## Update

Our sensor model updates the state to "fold in" information taken from sensor measurements. In the digression, we showed that the sensor update corresponds to the following probabilistic representation:

$$
p(x_k|z_{1:k}) \propto p(z_k|x_k) p(x_k|z_{1:k-1})
$$

where $p(x_k\|z_{1:k})$ is our latest state, given all of the sensor measurements, and $p(x_k\|z_{1:k-1})$ is the result of applying just the motion model to the previous state $p(x_{k-1}\|z_{1:k-1})$. So what's that extra term $p(z_k\|x_k)$? That's our sensor model! It's the distribution of our sensor measurement given the forward-predicted-by-the-motion-model state! To reiterate, remember that for EKFs, we thought of the sensor model as mapping our state space into our observation space. This is exactly what $p(z_k\|x_k)$ represents: given a state $x_k$, we want to know how likely seeing a particular sensor measurement $z_k$ is.

Notice that the above equation has "proportional to" $\propto$ instead of "equals" $=$. This is because the quantity on the right-hand-side has yet to be normalized. Specifically, we know for a fact that $p(x_k\|z_{1:k-1})$ is already normalized so we really just need to normalize our sensor model $p(z_k\|x_k)$ for the latest state $p(x_k\|z_{1:k})$ to be a valid probability distribution.

Another interpretation is that the normalized $p(z_k\|x_k)$ act as a set of **weights** that are large if the state "agrees" with the sensor measurement and small if the state "disagrees" with the sensor measurement. In the context of our particle filter, after moving each particle according to the motion model, we *weight* each of the particles by how well they agree with the sensor measurement. The particles that are in the most agreement will have a larger weight, and we normalize the weights so that the result is a valid probability distribution.

In fact, after the sensor model update, we can compute the best state estimate given the particles $x^{(i)}_k$ and their associated weights $w^{(i)}_k$ by a simple weighted sum:

$$
\bar{x}_k = \frac{1}{N}\sum_i w^{(i)}_k x^{(i)}_k
$$

To make the weights more concrete, let's look at an example where the sensor is a GPS. The GPS will tell us, with some noise covariance matrix $R$, the position of our robot. We want to construct our sensor update such that the largest values are the ones closest to the GPS reading. One way to do this is to compute the distance between our best state position and the GPS measurement and treat that as the mean of a Gaussian with the covariance being the GPS noise $R$. Then we evaluate the Gaussian pdf at the sensor measurement. This works because the Gaussian pdf is always largest at the mean, which is the "error" between our estimate position and what the GPS is telling us.

The combination of the particles and weights finally form the distribution corresponding to the most recent state given all of the sensor measurements $p(x_k\|z_{1:k})$! 

![Computing the weights from a sensor measurement](/images/particle-filters/sensor-model-weights.svg "Computing the weights from a sensor measurement")

<small>The robot is stationed at the origin and receives a positional sensor measurement (depicted as a magenta 'X') that, with noise included, actually tells us we're slightly above the origin ($y=0.1$). The left pane shows a uniform distribution of weights for all particles, i.e., each particle gets a weight of $\frac{1}{N}$. The right pane adjusts these weights by factoring in the sensor model; we see that particles closer to the measurement are weighted higher. Finally, we normalize the weights so they form a valid probability distribution.</small>

(Note that we're not assuming any particular distribution. The mean of any distribution is always well-defined)

So now we have our sensor update to apply after our motion model! We're almost finished with the full particle filter algorithm.

## Resampling

The last thing we need to discuss is resampling. Repeatedly applying the motion model and assigning weights to each particle runs into a problem as the particle filter evolves. Suppose we direct our robot to travel straight ahead for a long time. Since the particles are initially randomly distributed, it is often the case that many of them move away significantly away from the best estimated state. Naturally, these would have small weights once the sensor measurements are accounted for. What we're left with is only a few particles accurately representing the true state of the robot. We're wasting perfectly good particles for no benefit! This problem is often called **degeneracy**.

For the poorly-weighted particles, we'd like to swap them for higher-quality particles, and we can do this through the process of **resampling**. Since we know the combination of the weights and particles form a valid probability distribution, we can simply sample from this weighted distribution. Particles at the fringe with small weights will be less likely to be selected while particles near the true state of the robot will be more likely to be included in the resampled particle set. In other words, we swap out the lower-weighted particles with copies of the higher-weighted particles to get rid of bad particles while maintaining the same number of particles overall. This technique is called **sequential importance resampling (SIR)**. While it's certainly not the only or most advanced sampling technique, it's fairly popular and works well in practice.

![Resampling](/images/particle-filters/resampling.svg "Resampling")

<small>Suppose our weight distribution looked like that in the left pane. The right pane has the resampled particles; notice how the higher-weighted particles were sampled more often and the lower-weighted particles at the fringe aren't selected as part of the resampling. The new weights are reset to $\frac{1}{N}$ after resampling.</small>

Practically, an easy way to implement resampling is to create a bin for each weight whose size is proportional to the size of the weight. Then sample from a uniform distribution of $[0, \sum_i w_k^{(i)}]$ and take the index of the bin the sample falls into. A larger weight means a larger bin and the greater the chance that particle is selected as part of the resampling process.

As for how often to resample, we don't resample at each state update. Resampling too often means we lose particle diversity, i.e., many of our particles will actually be the same state. In the worse case, all of our particles would have the same estimate, effectively just being represented by one particle! This problem is called **sample impoverishment**. There are plenty of different techniques to remedy this, but a simple one is to compute a quantity called the **effective number of particles**.

$$
\hat{N}_{\text{eff}} = \displaystyle\frac{1}{\displaystyle\sum_i (w^{(i)})^2}
$$

(Note that the $2$ is actually an exponent, not a particular particle index.)

This is a curious metric, but, intuitively, we can think of it as measuring the "information content" of the particle set given their weights. For example, consider if each particle has the same, uniform weight $\frac{1}{N}$. This is also called a *high entropy* distribution because it doesn't really tell us much about which particle is most representative of the state: they're all equally representative! In that case, we'll need all of the particles we can get, and, indeed $\hat{N}_\mathrm{eff}=N$.

To the other extreme, suppose all of our weights are $0$ except for one which is $1$. We call this a *low entropy* because it gives us a very good idea of which particle is representative of the best state. In that case, our information content is really just that one particle, and, indeed, $\hat{N}_\mathrm{eff}=1$.

With this heuristic, we can pick a threshold $N_\mathrm{thresh}$ such that we resample if $\hat{N}_ \mathrm{eff} < N_\mathrm{thresh}$. A good initial value to try is $\frac{N}{2}$ or $\frac{N}{3}$; this should be tuned later on.

One last thing to say about resampling is what happens to the weights after resampling or up to the point of resampling? Initially, we start with the weights taking a uniform distribution $w_k^{(i)}=\frac{1}{N}$. Similarly, right after resampling, we reset the weights to that same uniform distribution. For the time steps that we don't resample at, we accumulate the weights rather than recomputing new ones from scratch with the following simple update rule:

$$
w^{(i)}_{k} = w^{(i)}_{k-1}p(z_k|x^{(i)}_ k)
$$

Remember to normalize afterwards! This allows us to preserve some historical information about the weights across time steps.

## The Particle Filter Algorithm

We've seen derivations of the particle filter rules over the past few sections so it's time to bring together the fruits of our labor!

The particle filter algorithm (specifically with sequential importance resampling) has the following actors:

* $N$: the number of particles
* $N_\text{thresh}$: the resampling criterion threshold
* $p(x_k\|x_{k-1})$: the motion model
* $p(z_k\|x_k)$: the sensor model.

And the algorithm is

1. Randomly initialize $N$ particles $x_0^{(i)}$ and uniformly initialize $N$ weights $w_0^{(i)} = \frac{1}{N}$ to get the prior distribution $p(x_0)$.
2. Apply the motion model $f$ to each particle independently $x_k^{(i)} = f(x_{k-1}^{(i)}, u_k) + q_k$, where $u_k$ is an optional control and $q_k$ is added noise, to get the post-motion-model distribution $p(x_k\|z_{k-1})$.
3. From the sensor model, update the weights: $w_k^{(i)}\gets w_{k-1}^{(i)} p(z_k\|x_k)$
4. Renormalize the weights: $w_k^{(i)}\gets\frac{w_k^{(i)}}{\sum_j w_k^{(j)}}$
5. Compute the resampling criterion: $\hat{N}_{\text{eff}} = \frac{1}{\sum_i (w^{(i)})^2}$
6. Resample $N$ particles if the resampling criterion falls below the threshold: $\hat{N}_ {\text{eff}} < N_{\text{thresh}}$ and reset the weights to the uniform distribution: $w_k^{(i)} = \frac{1}{N}$.
7. Compute and report the the best state estimate: $\bar{x}_k = \frac{1}{N}\sum_i w^{(i)}_k x^{(i)}_k$
8. Go to 2. until forever.

Particle filters aren't without problems however. We already discussed degeneracy and sample impoverishment. Another problem with particle filters is that it can be difficult to find the right number of particles. It's not immediately obvious; $N$ is a function of both the dimensionality of the space, e.g., a full 6DOF pose would require more particles than a just a 3DOF position and heading, as well as the complexity of the environment. A larger number of particles means more computation; particle filters are generally more computationally expensive since they require the motion model and weight computation for each particle.

# Conclusion

EKFs as state estimators can work really well but do have some limitations that disqualify them from certain kinds of systems. Specifically, they're only good for Gaussian and slighly nonlinear systems. We saw that a large part of these limitations stemmed from EKFs being parametric state estimators that required explicit updating of a mean and covariance. To get around this problem and move into the non-parametric domain, our key insight was that any arbitrary distribution can be represented by samples, or particles, of that distribution. We dubbed this non-parametric approach of state estimation a particle filter. We started by initializing $N$ randomly-generated particles with equally distributed weights, i.e., all are $\frac{1}{N}$. For our motion model, we simply apply it, with some noise, to each particle independently. Since the sensor update didn't follow so nicely, we sojourned to Hidden Markov Models for just long enough to motivate the particle filter sensor model. We interpreted the model as computing and maintaining a weight for each particle that measured sensor agreement. If we needed to resample to avoid degeneracy, we can do that. At the end of the day, we computed a state estimate by taking the weighted sum of the particles (using the weights as weights, of course). Then we simply "rinse-and-repeated" for the duration of the particle filter and that's our non-parametric state estimator!

This post ended up being much, much longer than I had originally anticipated so kudos if you've read everything 😀️! I wanted to make sure I properly motivated each equation and formulation so it read more like a story and less like an itemized list of facts. Besides particle filters, there are some more bespoke techniques for robotic state estimation and localization that we'll maybe get to next time.
