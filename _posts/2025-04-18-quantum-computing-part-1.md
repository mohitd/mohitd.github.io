---
layout: post
title: "Quantum Computing - Part 1: Basic Quantum Circuits"
excerpt: "Quantum Computing is a growing field of study that marries quantum physics with computer science which is already showing some promising results in speeding up specific kinds of computing problems. In this post, we'll begin our sojourn into the quantum realm!"
comments: true
---

Quantum Computing has started to enter the popsci news in the past few years as it matures from theoretical knowledge to practical application. For example, some years ago, major automobile manufacturer Volkswagen announced a partnership with D-Wave Quantum Inc. where it showcased small-scale proof-of-concept [Quantum Routing](https://www.vw.com/en/newsroom/future-of-mobility/quantum-computing.html) algorithm in Lisbon, Portugal to reduce the waiting times for passengers in the bus system. In more recent news, Google recently announced [Willow](https://blog.google/technology/research/google-willow-quantum-chip/), their new quantum chip with 106 superconducting qubits. Just a few months ago, Microsoft announced their [Majorana quantum chip](https://azure.microsoft.com/en-us/blog/quantum/2025/02/19/microsoft-unveils-majorana-1-the-worlds-first-quantum-processor-powered-by-topological-qubits/). I expect more major companies to start investing in and using this technology in the coming years and decades as it shows extraordinary speed-ups on algorithms core to many businesses! Now is the best time to start understanding how it all works and especially its limitations in the kinds of problems it can solve well.

Before you get concerned about the "quantum" in "quantum computing", at least for the kinds of computer science use-cases we'll be going into, we won't need a very detailed understanding of quantum mechanics like knowing how to solve the Schrödinger equation, but we will need to at least understand and accept some core concepts like *state*, *superposition*, and *measurement*. While Quantum Computing is unsurprisingly also used for modeling quantum mechanical systems like interactions between molecules in quantum chemistry, we're not going to get into those particular applications of quantum computing since they require substantial background knowledge. All this being said, if you want to really understand quantum computing, you'd benefit from learning more about quantum physics. (Even with that knowledge, there's certainly an aspect of "we don't know why but this is the empirical way the universe works! We'll even encounter some of that in this post itself!)

This is just the start in our quantum journey! We'll need to understand the basics before delving into some more practical use-cases for quantum computing. For example, the **vehicle routing problem** of identifying the most efficient routes for a fleet of vehicles to perform deliveries can be formulated as a quantum computing problem that can be solved significantly faster than classical approaches. I'm sure you've heard of **Shor's Algorithm** for factoring large prime numbers that could be used to break certain RSA encryption which is ubiquitous in cybersecurity. There's also **Grover's Algorithm** for searching through an unstructured database faster than classical methods. All of these are quantum algorithms that have potential to be much faster than their classical counterparts!

In this post, I'll walk through some basics of quantum computing, starting with some basic quantum mechanics concepts that we'll need to accept. Then I'll define a qubit by making analogues to classical bits. We'll see some ways to manipulate a single qubit before moving on to multi-qubit systems. Finally, we'll apply everything we've learned to two interesting quantum circuits that can be used to transmit quantum state using classical bits and vice-versa with transmitting classical bits using quantum state.

I'll assume you know some basic linear algebra with vectors and matrices and give a quick refresher on complex numbers but you don't have to know any quantum phyics!

# Quantum Mechanics Concepts

Quantum computing lives at the intersection of quantum mechanics and computer science; while we won't have to cover the entirety of quantum mechanics, we'll still need to accept some concepts that are the basis of quantum computing. We'll use a few historical experiments to motivate the concepts but we won't be getting too much into the underlying maths and physics. For a more rigorous treatment, any introductory quantum mechanics textbook will do (Griffiths is good one).

In the early 20th century, physicists were preoccupied with experiments that classical physics couldn't explain. One such experiment was the the Stern-Gerlach experiment. Vaporize some silver in an oven, send the beam of silver atoms through a magnetic field, and measure the deflection on a detector screen.

![The Stern-Gerlach Experiment](/images/quantum-computing-part-1/stern-gerlach.svg "The Stern-Gerlach Experiment")

<small><i>Credit: Wikipedia.</i> The Stern-Gerlach experiment had silver atoms traveling through a magnetic field into a detector screen to measure their deflection. (1) The oven vaporizing the silver, (2) the beam of silver atoms, (3) the magnetic field, (4) the expected result using classical electrodynamics, (5) the actual result.</small>

While the silver atom is neutral, the electron in the farthest shell will have a magnetic moment which behaves almost like the entire atom has a little magnet; this is an intrinsic, fundamental property of all particles called **spin**. From classical electrodynamics, if a charged object passes through a magnet, it experiences a force proportional to the "alignment" of its "north pole" with the north pole of magnet. Since the spin of the farthest out electron is a vector, we'd expect a Gaussian distribution with the mean being a straight line from the emitter, i.e., no deflection, and then some linear spread indicating some atoms that were slightly deflected only along one axis.

However this was not the observed result! Instead physicists observed two distinct peaks! Instead of the spin being a continuous distribution it was *quantized* into two values! Let's give these outcomes symbols: $\ket{\uparrow}$ for silver atoms deflected upward and $\ket{\downarrow}$ for silver atoms deflected downward. For each atom, we'll get one or the other outcome with some probability but not both and not something in between. We can describe the outcome of this system using an equation:

$$
\ket{\psi} = \alpha\ket{\uparrow} + \beta\ket{\downarrow}
$$

where $\ket{\uparrow}$ and $\ket{\downarrow}$ are the two possible outcomes, $\ket{\psi}$ represents the combination of all possible outcomes, and $\alpha^2 + \beta^2 = 1$ since we have a probability of being in one or the other outcome. (The $\ket{\cdot}$ is just a physics notation.) As it turns out, instead of these being real numbers in $\R$, in quantum mechanics, these are always generalized to complex numbers in $\C$ because there are many kinds of wave-like equations and other structures that are much more easily described using complex exponentials and complex numbers. There are some reformulations of quantum mechanics that use purely real numbers but they're less canonical and much more difficult. (As a refresher, a complex number $z\in\C$ is a number $z = a + bi$ such that $i\equiv\sqrt{-1}$.) So with $\alpha,\beta\in\C$, we should modify the constraint in the above equation to take the norm like $\abs{\alpha}^2 + \abs{\beta}^2 = 1$ where $\abs{z} \equiv \sqrt{a^2+b^2}$. QM tells us the *statistics* of a particle so *on average* $\abs{\alpha}^2$ percent of the time, the system will be in $\ket{\uparrow}$ and the other $\abs{\beta}^2$ percent of the time, the system will be in $\ket{\downarrow}$.

Beforehand, we don't know what the outcome is going to be. We say that the outcome is a **superposition** of the $\ket{\uparrow}$ and $\ket{\downarrow}$ states; this is just a fancy word for describing that the measured state could be one of many possible outcomes. After we take a **measurement**, then we get a *single* outcome of the possibilities of the possibilities. It's an open physics/meta-physics question as to why this happens and how to interpret it but the reality is that measuring a quantum mechanical system produces exactly one outcome from all possible outcomes.

# From Classical Computing to Quantum Computing: the Qubit

To motivate quantum bits, let's start with classical bits. The most fundamental unit of computing is the **bit** which takes a value of exactly and only 0 or 1. Something that we might forget in the modern era of computing is that a bit is a *logical* object but the *physical* representation of a bit depends on the kind of hardware used to represent it. In modern computing, we use transistors, specifically a metal-oxide-semiconductor field-effect transistor (MOSFET), where the logical value of 0 means the transistor isn't conducting any current while the value of 1 means that current is flowing through it. Going back over half a century, we were using magnetic tapes, disks, and other magnetic medium where we'd align a little region on the medium either "down" or "up" which represented 0 or 1 respectively. For most practical computer science, we generally don't worry about the physical representation and assume it's reliable; after all, that's a job for the electrical engineers and physicists!

Now what if we stretch our notion of a "bit" from the classic sense into the quantum sense of superposition, probability, and possibilities. A **quantum bit** or **qubit** is the most fundamental unit of computing for quantum computing that also takes the value of either $\ket{0}$ or $\ket{1}$ but is only known at most with some probability before measuring. The most general kind of qubit is in some superposition of $\ket{0}$ or $\ket{1}$ with respective **probability amplitudes** $\alpha$ and $\beta$.

$$
\ket{\psi} = \alpha\ket{0} + \beta\ket{1}\\
$$

such that $\abs{\alpha}^2 + \abs{\beta}^2 = 1$. The actual probabilities are $\abs{\alpha}^2$ and $\abs{\beta}^2$ so we call $\alpha$ and $\beta$ probability amplitudes.

The simplest examples of qubits are $\ket{\psi}=\ket{0}$ and $\ket{\psi}=\ket{1}$. Let's use IBM's quantum computing library Qiskit to represent the first state as a quantum circuit and simulate it on our classical hardware.

```python
# Remember to install the following!:
# pip3 install qiskit qiskit-aer

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

# define a quantum circuit with a single qubit
circuit = QuantumCircuit(1, 0)
# measure all qubits
circuit.measure_all()
# print an ASCII-art version of the circuit
print(circuit)

# create simulator to run the circuit against
simulator = AerSimulator()
# transpile the circuit from the software representation
# to a version that's optimized for quantum computing hardware
# (in this case, we're just using our simulator on our classical hardware)
circuit = transpile(circuit, simulator)

# simulate the circuit for 2^10 trials and get the results
result = simulator.run(circuit).result()
# fetch and print the counts of the distribution
counts = result.get_counts(circuit)
print(counts)
```

If we run this, unsurprisingly, we'll see that all trials measure the same state: 0.

```
         ░ ┌─┐
     q: ─░─┤M├
         ░ └╥┘
meas: 1/════╩═
            0 
{'0': 1024}
```

Remember that quantum physics tells us the statistics of *distributions* not of a single individual particle so we need to run this quantum circuit for a number of trials. The necessity of running a number of trials instead of just a single one will become apparent in a little while. What about the other state where $\ket{\psi}=\ket{1}$? We can initialize the qubit to $\ket{1}$ just by adding a call to `circuit.initialize` before `circuit.measure_all()`:

```python
# initialize qubit 0 to 0*|0> + 1*|1>
circuit.initialize([0, 1], 0)
```

Note we use a vector to represent the coefficients of $\ket{0}$ and $\ket{1}$! An essential representation of a quantum state $\ket{\psi}$ is as a vector in a vector space (specifically a **Hilbert space**) of some **basis states**, for example $\ket{0}$ and $\ket{1}$. We'll get more into this when we discuss quantum gates and operators. Let's run this code!

```
        ┌─────────────────┐ ░ ┌─┐
     q: ┤ Initialize(0,1) ├─░─┤M├
        └─────────────────┘ ░ └╥┘
meas: 1/═══════════════════════╩═
                               0 
{'1': 1024}
```

Also unsurprisingly, we find that the result is always 1 (and there's an `Initialize` block in the circuit).

A more interesting example is a uniform superposition of both.

$$
\ket{\psi} = \frac{1}{\sqrt{2}}\ket{0} + \frac{1}{\sqrt{2}}\ket{1}\\
$$

(Verify that the norm of the coefficients sum to 1!) This means that, if we prepare and measure this qubit, for about half the trials, the final outcome will be 0 and the other half of the time, the final outcome will be 1. 

We can simulate this using the same `circuit.initialize` function, being careful with the normalization.

```python
# initialize qubit 0 to 1/sqrt(2)*|0> + 1/sqrt(2)*|1>
circuit.initialize([1./np.sqrt(2), 1./np.sqrt(2)], 0)
```

Now our results are more interesting!

```
        ┌─────────────────────────────┐ ░ ┌─┐
     q: ┤ Initialize(0.70711,0.70711) ├─░─┤M├
        └─────────────────────────────┘ ░ └╥┘
meas: 1/═══════════════════════════════════╩═
                                           0 
{'0': 518, '1': 506}
```

Now when we measure, we have roughly even counts of 0 and 1 as expected! The counts are exactly equal since we're just simulating the quantum circuit and the overall system has noise! This is something we'll have to get accustomed to: quantum computing is noisy! But now it's clear why we need to run multiple trials: quantum computing is not deterministic so we need a statistically significant number of trials for each circuit to get meaningful results.

# Quantum Logic Gates for Single Qubits

Now that we have some familiarity with a single qubit's initialization and measurements, let's see what kinds of operations we can perform on that qubit. Just like with classical computing and what I've alluded to when using Qiskit, quantum computing also has a notion of logic gates that qubits pass through and have their state changed as part of a quantum circuit.

The most critical and common single-qubit quantum logic gates are $X$, $Y$, $Z$, and $H$. The first three are sometimes called **Pauli gates** since they correspond to the Pauli matrices $\sigma_x$, $\sigma_y$, and $\sigma_z$ in quantum physics. An alternative geometric representation is that those operators represent rotations of $\frac{\pi}{2}$ about their respective axes on a kind of unit sphere called the **Bloch sphere** where we represent each qubit as a point on the sphere such that the basis states $\ket{1}$ and $\ket{0}$ are at $z=1$ and $z=-1$ respectively. I don't find much insight in this geometric representation especially since it doesn't generalize to multiple qubits well.

We can define these operators based on how they transform a generic qubit $\ket{\psi}=\alpha\ket{0}+\beta\ket{1}$.

$$
X\ket{\psi} \equiv \alpha\ket{1} + \beta\ket{0}
$$

So the $X$ gate effectively swaps the probability amplitudes of the two states! Equivalently, we could have defined the $X$ gate based on how it transformed the *basis states themselves*! 

$$
\begin{align*}
X\ket{0} &= \ket{1} \\
X\ket{1} &= \ket{0}
\end{align*}
$$

This becomes clear if we substitute $\ket{\psi}=1\ket{0}+0\ket{1}$ and $\ket{\psi}=0\ket{0}+1\ket{1}$. We can think about the $X$ gate as being roughly like a `NOT` gate from classical computing! Remember when we were initializing the state of a qubit and we learned we could represent it as a vector of coefficients? Well if we have the "before" qubit and the "after" qubit, a *matrix* is how we can represent a linear transform between the two! In quantum mechanics, all operators can be represented as matrices since quantum mechanics is a *linear* framework. Therefore, we can represent all quantum logic gates as matrices too. Specifically, we're looking for the matrix that maps the **state vector** $\begin{bmatrix}\alpha \\\\ \beta\end{bmatrix}$ to $\begin{bmatrix}\beta \\\\ \alpha\end{bmatrix}$. With some effort, we can figure this out:

$$
X =
\begin{bmatrix}
0 & 1\\
1 & 0
\end{bmatrix}
$$

And we can verify this matrix is correct:

$$
\begin{align*}
X\ket{\psi} &=
\begin{bmatrix}
0 & 1\\
1 & 0
\end{bmatrix} \begin{bmatrix}\alpha \\\\ \beta\end{bmatrix}\\

&=\begin{bmatrix}\beta \\\\ \alpha\end{bmatrix}

\end{align*}
$$

All quantum gates/operators must **unitary**: their inverse must equal to their own conjugate transpose, i.e., $U^\dagger U=UU^\dagger=I$. This property is a generalization of real orthogonal matrices where their transpose equals their inverse, i.e., $Q^TQ=QQ^T=I$. Recall that the conjugate of a complex number $z=a+bi$ is just $\bar{z}=a-bi$ so the conjugate transpose $U^\dagger$ of a matrix $U$ entries must obey $a_{ij} = \bar{a_{ji}}$ where $a_{ij}\in\C$ is the entry in the $i$th row and $j$th column of $U$. In other words, we transpose the matrix and then take the complex conjugate of each entry.

We can verify the $X$ gate is unitary:

$$
X^\dagger X = 
\begin{bmatrix}
0 & 1\\
1 & 0
\end{bmatrix}^\dagger
\begin{bmatrix}
0 & 1\\
1 & 0
\end{bmatrix}
=
\begin{bmatrix}
0 & 1\\
1 & 0
\end{bmatrix}
\begin{bmatrix}
0 & 1\\
1 & 0
\end{bmatrix}
=
\begin{bmatrix}
1 & 0\\
0 & 1
\end{bmatrix}
= I
$$

This is the most important property of all quantum logic gates/operators because it *preserves normalization*! Every quantum state must be normalized so this property ensures that, after we apply a any quantum operators to any state, we always end up with a properly normalized state. Another consequence of unitary operators is that *all quantum gates are reversible*! This is generally not true for classical gates. Consider a classical AND gate: we can't know what the two operands were from just the result of the AND gate. We'll see a number of quantum circuits of the form "perform some operations to map the input into a different space, manipulate the state in that space, perform the inverse operations from the beginning to map the state back into the original space". 

But for now, let's build a quantum circuit using the $X$ gate.

```python
# replace the initialize call
# apply X gate to qubit 0
circuit.x(0)
```

Applying this to qubit 0 in state $\ket{0}$ yields 1 always.

```
        ┌───┐ ░ ┌─┐
     q: ┤ X ├─░─┤M├
        └───┘ ░ └╥┘
meas: 1/═════════╩═
                 0 
{'1': 2048}
```

Applying it to the $\ket{1}$ state always yields 0. This is the quantum version of the NOT gate!

Moving on, before we get to the $Y$ gate, let's first talk about the $Z$ gate. We can represent it matrix form as the following.

$$
Z = 
\begin{bmatrix}
1 & 0\\
0 & -1
\end{bmatrix}
$$

Applying the $Z$ gate to $\ket{0}$ leaves it unchanged, i.e., $Z\ket{0}=\ket{0}$, but to $\ket{1}$, this maps it to $-1\ket{1}$, i.e., $Z\ket{1}=-\ket{1}$. To a general qubit $\ket{\psi}=\alpha\ket{0}+\beta\ket{1}$, the $Z$ gate maps it to $Z\ket{\psi}=\alpha\ket{0}-\beta{\ket{1}}$. Does this affect the probabilities of observing either outcome? Nope! Recall that the likelihood of each state is a *norm* and $\abs{-\beta}=\abs{\beta}$ so the $Z$ gate doesn't change the final observation likelihoods. This extra factor is called the **phase** (specifically **relative phase**) and the $Z$ gate is sometimes called the *phase flip* gate because it flips the sign of $\ket{1}$. On the surface, phase doesn't *seem* to affect the final measurement of the qubit but, used in conjunction with other quantum gates and operators, it's essential to all complex quantum algorithms like Grover's Algorithm and the famous Shor's Algorithm to factor large prime numbers.

But for now, let's build a circuit with the $Z$ gate.

```python
circuit.z(0)
```

And then run it.

```
        ┌───┐ ░ ┌─┐
     q: ┤ Z ├─░─┤M├
        └───┘ ░ └╥┘
meas: 1/═════════╩═
                 0 
{'0': 2048}
```

As expected, this phase didn't change the measured result! Phase is one of the unique facets of quantum computing with no classical analogue that provides an entirely new dimension to quantum algorithms.

Circling back, the $Y$ gate flips the qubit and adds a complex phase of $i$ and can also be represented by a matrix.

$$
Y = 
\begin{bmatrix}
0 & -i\\
i & 0
\end{bmatrix}
$$

So $Y\ket{0}=i\ket{1}$ and $Y\ket{1}=-i\ket{0}$. Note that we can represent the $Y$ gate as $Y=iXZ$! In fact, we can represent each of the Pauli gates in terms of the others! I've found the $Y$ gate to be less useful as the $X$ and $Z$ gates but it *does* correspond to a Pauli matrix so it's worth mentioning it for completeness.

Moving on to arguably the most important single-qubit gate, the **Hadamard $H$ gate** is used to create uniform superpositions of qubits. Remember $\ket{\psi} = \frac{1}{\sqrt{2}}\ket{0} + \frac{1}{\sqrt{2}}\ket{1}$? Well we can create it using the Hadamard gate applied to $\ket{0}$. We can represent it as a matrix:

$$
H =
\frac{1}{\sqrt{2}}
\begin{bmatrix}
1 & 1\\
1 & -1
\end{bmatrix}
$$

Applying this to the basis states, $H\ket{0} = \frac{1}{\sqrt{2}}\ket{0} + \frac{1}{\sqrt{2}}\ket{1}$ and $H\ket{1} = \frac{1}{\sqrt{2}}\ket{0} - \frac{1}{\sqrt{2}}\ket{1}$. So given either basis state, the Hadamard gate can be used to create a uniform superposition (sometimes with a phase)! Notationally, some resources define $\ket{+}\equiv H\ket{0} = \frac{1}{\sqrt{2}}\ket{0} + \frac{1}{\sqrt{2}}\ket{1}$ and $\ket{-}\equiv H\ket{1} = \frac{1}{\sqrt{2}}\ket{0} - \frac{1}{\sqrt{2}}\ket{1}$ as a shorthand.

Let's build a quantum circuit that uses it!

```python
circuit.h(0)
```

And then run it.

```
        ┌───┐ ░ ┌─┐
     q: ┤ H ├─░─┤M├
        └───┘ ░ └╥┘
meas: 1/═════════╩═
                 0 
{'1': 1034, '0': 1014}
```

As expected it produces a roughly equal number of 0s and 1s! Now we have some of the building blocks to properly initialize our qubits. In practice, we don't initialize qubits arbitrarily beyond just the basis states, but we use gates to put them into whatever state we want before invoking quantum algorithms.

Before wrapping up with single-qubit gates, I *did* want to mention that there are other single-qubit gates like the rotation gate $R_\theta$ that applies a phase of $e^{i\theta}$ to $\ket{1}$ while leaving $\ket{0}$ unchanged. Technically, we can represent any single-qubit gates as a combination of rotation gates about some axis by some amount but the ones we've discussed are the most common. Of course, we can construct novel gates and, as long as we can show the gate is unitary, then it's a valid gate!

# Multi-qubit Systems

So far, we've discussed a single qubit but the real power of quantum computing comes from how multiple qubits are handled. We always represent multiple qubits by enumerating all possible outcomes.

$$
\ket{\psi} = a_{00}\ket{00} + a_{01}\ket{01} + a_{10}\ket{10} + a_{11}\ket{11}\\
$$

where $\sum_i\abs{a_i}^2 = \abs{a_0}^2 + \abs{a_1}^2 + \abs{a_2}^2 + \abs{a_3}^2 = 1$. This is particularly powerful in its ability to represent $2^n$ classical bits/outcomes using only $n$ qubits: with 2 qubits, we've represented 4 possible outcomes.

Similar to the one qubit case, we can put two qubits in a uniform superposition across all possible outcomes.

$$
\ket{\psi} = \frac{1}{2}\ket{00} + \frac{1}{2}\ket{01} + \frac{1}{2}\ket{10} + \frac{1}{2}\ket{11}\\
$$

When we measure this state, we get all possible outcomes equally on average! We can construct such a state using two Hadamard operators but, when dealing with multi-qubit systems, we have to specify which qubit we're applying the operator to. Conventionally similar to classical computing, qubit 0 is the rightmost qubit. So to build this state, we need to apply two Hadamard operators to the first and second qubits.

$$
\begin{align*}
H_1 H_0\ket{00} &= H_1\Bigg[\frac{1}{\sqrt{2}}(\ket{00} + \ket{01})\Bigg]\\
&= \frac{1}{\sqrt{2}}\Bigg(H_1\ket{00} + H_1\ket{01}\Bigg)\\
&= \frac{1}{\sqrt{2}}\Bigg[\frac{1}{\sqrt{2}}\Bigg(\ket{00} + \ket{10}\Bigg) + \frac{1}{\sqrt{2}}\Bigg(\ket{01}+\ket{11}\Bigg)\Bigg]\\
&= \frac{1}{2}\ket{00} + \frac{1}{2}\ket{01} + \frac{1}{2}\ket{10} + \frac{1}{2}\ket{11}\\
\end{align*}
$$

Note that when we're applying an operation to a single qubit of a multi-qubit system, we leave the unaffected qubits unchanged but still write them all out. Just like with single-qubit gates, we can represent this with a single $4\times 4$ matrix. To do this, we need to define a kind of product called the **tensor product** used to combine two independent qubits into a single multi-qubit system. The tensor product has a very specific mathematical definition but can think of it as a way to combinatorially combine two qubit states into a single joint state. Suppose we have two qubits.

$$
\begin{align*}
\ket{\psi_0} &= \alpha\ket{0} + \beta\ket{1}\\
\ket{\psi_1} &= \gamma\ket{0} + \delta\ket{1}\\
\end{align*}
$$

Then we can define the tensor product.

$$
\ket{\psi_1}\otimes\ket{\psi_0} = \ket{\psi_1}\ket{\psi_0} = \ket{\psi_1\psi_0} = \gamma\alpha\ket{00} + \gamma\beta\ket{01} + \delta\alpha\ket{10} + \delta\beta\ket{11}
$$

We're basically multiplying each state of $\psi_0$ with each state of $\psi_1$. In terms of the coefficients, we can do the same thing.

$$
\begin{align*}
\ket{\psi_1}\otimes\ket{\psi_0} &= \begin{bmatrix}\gamma\\\delta\end{bmatrix}\otimes\begin{bmatrix}\alpha\\\beta\end{bmatrix}\\
&= \begin{bmatrix}
\gamma\otimes\begin{bmatrix}\alpha\\\beta\end{bmatrix} \\
\delta\otimes\begin{bmatrix}\alpha\\\beta\end{bmatrix}
\end{bmatrix}\\
&= \begin{bmatrix}
\gamma\alpha\\
\gamma\beta\\
\delta\alpha\\
\delta\beta\\
\end{bmatrix}
\end{align*}
$$

The coefficients of all of the states line up with the matrix representation! This is also sometimes called the **Kronecker product**. Now we can construct the matrix that represents $H_1\otimes H_0 = H^{\otimes 2}$.

$$
\begin{align*}
H_1\otimes H_0 = H^{\otimes 2} &= 
\frac{1}{\sqrt{2}}
\begin{bmatrix}
1 & 1\\
1 & -1
\end{bmatrix}
\otimes
\frac{1}{\sqrt{2}}
\begin{bmatrix}
1 & 1\\
1 & -1
\end{bmatrix}\\
&= \frac{1}{2}
\begin{bmatrix}
\begin{bmatrix}
1 & 1\\
1 & -1
\end{bmatrix}
\otimes 1 &
\begin{bmatrix}
1 & 1\\
1 & -1
\end{bmatrix}
\otimes 1 \\
\begin{bmatrix}
1 & 1\\
1 & -1
\end{bmatrix}
\otimes 1 &
\begin{bmatrix}
1 & 1\\
1 & -1
\end{bmatrix}
\otimes -1
\end{bmatrix}\\
&=
\frac{1}{2}
\begin{bmatrix}
1 & 1 & 1 & 1\\
1 & -1 & 1 & -1\\
1 & 1 & -1 & -1\\
1 & -1 & -1 & 1
\end{bmatrix}
\end{align*}
$$

Applying this to the vector representing $\ket{00}$, we get the expected result: a uniform superposition.

$$
\frac{1}{2}
\begin{bmatrix}
1 & 1 & 1 & 1\\
1 & -1 & 1 & -1\\
1 & 1 & -1 & -1\\
1 & -1 & -1 & 1
\end{bmatrix}
\begin{bmatrix}
1\\ 0\\ 0\\ 0
\end{bmatrix}
=
\frac{1}{2}
\begin{bmatrix}
1\\ 1\\ 1\\ 1
\end{bmatrix}
$$

Now let's build the corresponding quantum circuit!

```python
# need to specify 2 quantum bits this time!
circuit = QuantumCircuit(2, 0)
# apply Hadamard to both qubits (equivalent to applying h to each one individually)
circuit.h([0, 1])
```

And then run it.

```
        ┌───┐ ░ ┌─┐   
   q_0: ┤ H ├─░─┤M├───
        ├───┤ ░ └╥┘┌─┐
   q_1: ┤ H ├─░──╫─┤M├
        └───┘ ░  ║ └╥┘
meas: 2/═════════╩══╩═
                 0  1 
{'01': 519, '00': 512, '10': 508, '11': 509}
```

As expected, this measures roughly equal counts across all possible states! 

# Quantum Logic Gates for Multi-qubit Systems

In addition to single-qubit quantum gates, there are also quantum logic gates that work with more than one qubit. The most important of which is called the **controlled NOT (CNOT)** gate that takes a **control qubit** and flips the **target qubit** if the control qubit is 1 otherwise the target qubit is left unchanged: $\text{CNOT} : \ket{x_1,x_0} = \ket{x_1\oplus x_0,x_0}$ where $\oplus$ is a classical XOR operation. I'll use $\text{CNOT}_{0,1}$ to refer to applying the CNOT gate with control qubit 0 and target qubit 1. The CNOT gate is like a quantum version of the XOR gate. The difference is that the CNOT gate requires a control qubit that's passed through unchanged. This is because of the property that quantum gates are reversible: a classical XOR gate is not reversible unless we know the value of one of the operands which is what the control qubit represents.

Let's construct the quantum circuit that uses it and see for ourselves.

```python
# CNOT gate with qubit 0 as the control qubit
# and qubit 1 as the target qubit
circuit.cx(0, 1)
```
Running this leaves the system unchanged since the control qubit is 0.

```
              ░ ┌─┐   
   q_0: ──■───░─┤M├───
        ┌─┴─┐ ░ └╥┘┌─┐
   q_1: ┤ X ├─░──╫─┤M├
        └───┘ ░  ║ └╥┘
meas: 2/═════════╩══╩═
                 0  1 
{'00': 2048}
```

But if we change the circuit so that the control qubit is 1 (add `circuit.x(0)` before the CNOT), then we always get 11. Just like with every quantum gate, we can represent it with a unitary matrix.

$$
\text{CNOT}=
\begin{bmatrix}
1 & 0 & 0 & 0\\
0 & 1 & 0 & 0\\
0 & 0 & 0 & 1\\
0 & 0 & 1 & 0
\end{bmatrix}
$$

These examples aren't particularly intersting but what if we pass the control qubit into a Hadamard gate before applying the CNOT gate? This is far more interesting since now the control qubit is in a uniform superposition of 0 and 1 so the CNOT gate *might* flip the target qubit half the time depending on the state of the control qubit. Let's figure out what would happen analytically.

$$
\begin{align*}
\text{CNOT}_{0,1}H_0\ket{00} &= \text{CNOT}_{0,1}\Bigg[\frac{1}{\sqrt{2}}\Bigg(\ket{00}+\ket{01}\Bigg)\Bigg]\\
&= \frac{1}{\sqrt{2}}\Bigg(\text{CNOT}_{0,1}\ket{00}+\text{CNOT}_{0,1}\ket{01}\Bigg)\\
&= \frac{1}{\sqrt{2}}\Big(\ket{00}+\ket{11}\Big)\\
\end{align*}
$$

Now this is a very interesting state! If we measure qubit 0 as 0, then qubit 1 will definitely be 0, and, if we measure qubit 0 as 1, then qubit 1 will definitely be 1. The qubits' final measured values are coupled! This is called **entanglement** in quantum physics and these kinds of coupled states are called **Bell states** (after John Bell), **EPR (Einstein-Podolsky-Rosen) pairs** (after Albert Einstein, Boris Podolsky, and Nathan Rosen), or just **entangled pairs**. The scary part is that no one currently knows *how* entanglement works but just that it does. In fact, it even works across arbitrary distances! If we put the two qubits on opposite sides of the galaxy, much farther than the speed of light could transmit any information, measuring one of them immediately tells us what the other one is.

Let's build a quantum circuit to show this empirically!

```python
circuit.h(0)
circuit.cx(0, 1)
```

And run it.

```
        ┌───┐      ░ ┌─┐   
   q_0: ┤ H ├──■───░─┤M├───
        └───┘┌─┴─┐ ░ └╥┘┌─┐
   q_1: ─────┤ X ├─░──╫─┤M├
             └───┘ ░  ║ └╥┘
meas: 2/══════════════╩══╩═
                      0  1 
{'11': 1015, '00': 1033}
```

As expected, with this multi-qubit system, the only two possible outcomes are 00 and 11 with roughly equal probability! Depending on the input qubits, we could get one of four possible Bell states (try to figure them out on your own!) They all share the same characteristic that knowing the result of one qubit determines the value of the other qubit. We can write the closed-form definition of a Bell state.

$$
\ket{B_{x,y}}\equiv\frac{\ket{0,y} + (-1)^x\ket{1,\bar{y}}}{\sqrt{2}}
$$

So the Bell state we created above was $\ket{B_{0,0}}$. The CNOT gate and single-qubit gates give us most of the foundation we need to construct more complex and practical quantum circuits that realize quantum algorithms. Let's see a few examples!

# Quantum Teleportation

One interesting application of the CNOT gate and entanglement is **quantum teleportation**. Suppose Alice wants to transmit some arbitrary quantum state $\ket{\psi}$ to Bob. On the surface, we might think to just copy $\ket{\psi}$ and send it to Bob, but quantum physics has a **No-cloning Theorem** that states that it is impossible to perfectly copy an unknown quantum state. This can be proven via proof-by-contradiction but, intuitively, if we could perfectly copy an unknown quantum state then we would be violating the Heisenberg Uncertainty Principle since we'd need to perfeclty know all of the properties of that unknown quantum state in order to perfectly copy it. The tangible consequence is that we can't just copy $\ket{\psi}$ and send it to Bob. 

Instead, suppose Alice and Bob had shared two halves of an entangled pair ahead of time. She can interact her arbitrary state $\ket{\psi}$ with her half of the entangled pair, measure it, and then send Bob the result over a classical communication channel. Based on the result, Bob can apply an operator to his half of the entangled pair to recover Alice's $\ket{\psi}$.

This time, let's first build the quantum circuit and then analyze it after. We'll take a slightly different approach than before to define the circuit just to showcase another way to use Qiskit.

```python
q = QuantumRegister(1, 'q')

bell_0 = QuantumRegister(1, 'B_0')
bell_1 = QuantumRegister(1, 'B_1')

c_0 = ClassicalRegister(1, 'c_0')
c_1 = ClassicalRegister(1, 'c_1')
c_2 = ClassicalRegister(1, 'c_2')

qc = QuantumCircuit(q, bell_0, bell_1, c_0, c_1, c_2)

# prep bell state
qc.h(bell_0)
qc.cx(bell_0, bell_1)
qc.barrier(label='ψ_0')

# Alice entangles her qubit with her half of the Bell state
qc.cx(q, bell_0)
qc.h(q)
qc.barrier(label='ψ_1')

# Alice measures to affect Bob's Bell state and sends him the classical qubits
qc.measure(q, c_0)
qc.measure(bell_0, c_1)
qc.barrier(label='ψ_2')

# Bob applies the right operators to his Bell state based on the
# classical qubits received from Alice
with qc.if_test((c_1, 1)):
    qc.x(bell_1)
with qc.if_test((c_0, 1)):
    qc.z(bell_1)
qc.barrier(label='ψ_3')

# Bob measures his Bell state
qc.measure(bell_1, c_2)
print(qc)
```

Running this for $\ket{\psi}=\ket{0}$, we see that the last qubit (the leftmost qubit) is always measured to be 0!

```
                  ψ_0      ┌───┐ ψ_1 ┌─┐    ψ_2                                                ψ_3    
    q: ────────────░────■──┤ H ├──░──┤M├─────░──────────────────────────────────────────────────░─────
       ┌───┐       ░  ┌─┴─┐└───┘  ░  └╥┘┌─┐  ░                                                  ░     
  B_0: ┤ H ├──■────░──┤ X ├───────░───╫─┤M├──░──────────────────────────────────────────────────░─────
       └───┘┌─┴─┐  ░  └───┘       ░   ║ └╥┘  ░  ┌────── ┌───┐ ───────┐ ┌────── ┌───┐ ───────┐   ░  ┌─┐
  B_1: ─────┤ X ├──░──────────────░───╫──╫───░──┤ If-0  ┤ X ├  End-0 ├─┤ If-0  ┤ Z ├  End-0 ├───░──┤M├
            └───┘  ░              ░   ║  ║   ░  └──╥─── └───┘ ───────┘ └──╥─── └───┘ ───────┘   ░  └╥┘
                                      ║  ║         ║                   ┌──╨──┐                      ║ 
c_0: 1/═══════════════════════════════╩══╬═════════╬═══════════════════╡ 0x1 ╞══════════════════════╬═
                                      0  ║      ┌──╨──┐                └─────┘                      ║ 
c_1: 1/══════════════════════════════════╩══════╡ 0x1 ╞═════════════════════════════════════════════╬═
                                         0      └─────┘                                             ║ 
c_2: 1/═════════════════════════════════════════════════════════════════════════════════════════════╩═
                                                                                                    0 
{'0 0 1': 525, '0 1 1': 506, '0 1 0': 479, '0 0 0': 538}
```

If we initialized $\ket{\psi}=\ket{1}$, we'd see that the last qubit is always measured to be 1!

Let's analyze this circuit for the general case where Alice has some arbitrary qubit $\ket{\psi}=\alpha\ket{0} + \beta\ket{1}$ that she wants to transmit to Bob. We first start by creating a Bell state with the two leftmost qubits $\ket{B_{0,0}}=\frac{1}{\sqrt{2}}(\ket{00} + \ket{11})$.

$$
\begin{align*}
\ket{\psi_0} &= \ket{B_{0,0}}\ket{\psi}\\
&= \frac{1}{\sqrt{2}}\Big(\ket{00} + \ket{11}\Big)\Big(\alpha\ket{0} + \beta\ket{1}\Big)\\
&= \frac{1}{\sqrt{2}}\Bigg[\Big(\ket{00} + \ket{11}\Big)\alpha\ket{0} + \Big(\ket{00} + \ket{11}\Big)\beta\ket{1}\Bigg]\\
\end{align*}
$$

Now let's apply the CNOT first with the control bit being the rightmost qubit and the target qubit being the midddle qubit.

$$
\ket{\psi'_1} = \frac{1}{\sqrt{2}}\Bigg[\Big(\ket{00} + \ket{11}\Big)\alpha\ket{0} + \Big(\ket{01} + \ket{10}\Big)\beta\ket{1}\Bigg]
$$

The $\alpha$ terms aren't affected since the control bit is 0 but the $\beta$ terms have their middle (or rightmost in their Bell state) qubit flipped. Now let's apply the Hadamard to the rightmost qubit, i.e., Alice's original quantum state.

$$
\begin{align*}
\ket{\psi_1} &= \frac{1}{\sqrt{2}}\Bigg[\Big(\ket{00} + \ket{11}\Big)\alpha\frac{1}{\sqrt{2}}(\ket{0}+\ket{1}) + \Big(\ket{01} + \ket{10}\Big)\beta\frac{1}{\sqrt{2}}(\ket{0}-\ket{1})\Bigg]\\
&= \frac{1}{2}\Bigg[\Big(\ket{00} + \ket{11}\Big)\alpha(\ket{0}+\ket{1}) + \Big(\ket{01} + \ket{10}\Big)\beta(\ket{0}-\ket{1})\Bigg]\\
\end{align*}
$$

In the last step, we pulled out the common factor of $\frac{1}{\sqrt{2}}$. Let's pull $\alpha$ and $\beta$ out to the front of their respective terms and expand out the two products.

$$
\begin{align*}
&= \frac{1}{2}\Bigg[\alpha\Big(\ket{00} + \ket{11}\Big)(\ket{0}+\ket{1}) + \beta\Big(\ket{01} + \ket{10}\Big)(\ket{0}-\ket{1})\Bigg]\\
&= \frac{1}{2}\Bigg[\alpha\Big(\ket{00}\ket{0} + \ket{11}\ket{0} + \ket{00}\ket{1} + \ket{11}\ket{1}\Big) + \beta\Big(\ket{01}\ket{0} + \ket{10}\ket{0} - \ket{01}\ket{1} - \ket{10}\ket{1}\Big)\Bigg]\\
\end{align*}
$$

Now let's regroup the qubits from $\ket{B_2 B_1}\ket{q}$ to $\ket{B_2}\ket{B_1 q}$ since Alice is going to send over the rightmost two qubits. (This is mathematically legal since the tensor product is associative.) Bob will read the two rightmost values to figure out which gates to apply to his half of the Bell state.

$$
= \frac{1}{2}\Bigg[\alpha\Bigg(\ket{0}\ket{00} + \ket{1}\ket{10} + \ket{0}\ket{01} + \ket{1}\ket{11}\Bigg) + \beta\Bigg(\ket{0}\ket{10} + \ket{1}\ket{00} - \ket{0}\ket{11} - \ket{1}\ket{01}\Bigg)\Bigg]
$$

Now let's regroup the terms where the righmost two qubits are $\ket{00}$, $\ket{01}$, $\ket{10}$, and $\ket{11}$.

$$
\begin{align*}
= \frac{1}{2}\Bigg[
&\phantom{+}\Big(\alpha\ket{0} + \beta\ket{1}\Big)\ket{00}\\
&+ \Big(\alpha\ket{0} - \beta\ket{1}\Big)\ket{01}\\
&+ \Big(\alpha\ket{1} + \beta\ket{1}\Big)\ket{10}\\
&+ \Big(\alpha\ket{1} - \beta\ket{1}\Big)\ket{11}
\Bigg]
\end{align*}
$$

Now this is interesting! If Bob receives $\ket{00}$ from Alice, he can recover the state that Alice originally sent $\ket{\psi}=\alpha\ket{0} + \beta\ket{1}$! But when Bob receives $\ket{01}$ from Alice, he needs to apply a Z gate to his half of the Bell state so that his qubit can be mapped from $\alpha\ket{0} - \beta\ket{1}$ to the original state $\alpha\ket{0} + \beta\ket{1}$. This is thanks to the corollary of the unitary property of quantum gates: they must be invertible! Similarly, if Bob receives $\ket{10}$, he needs to apply an X gate, and, if he receives a $\ket{11}$, then he needs to apply both an X and Z gate. This is exactly what the circuit does!

This is really phenominal! Alice can create an arbitrary quantum state, share an entangled qubit with a receiver, interact her arbitrary quantum state with half of the entangled pair so that it produces an effect on the other half of the entangled pair, measure and transmit the state and entangled pair to Bob, and Bob can recreate the original arbitrary quantum state! Note that we're *not* violating the No-cloning Theorem since Alice actually *measures* her arbitrary quantum state which makes it a known state.

The most important thing to note about quantum teleportation is that it *does not allow for faster-than-light communication!* We still need to transmit classical bits which are limited by the speed of light/causality. The term "speed of light" isn't quite complete since other things travel at the speed of light (namely a particle called a gluon or gravitational waves). A better term would be the **speed of causality**. No information of any kind can be transmitted faster than the speed of causality. But with this circuit, we can transmit an arbitrary quantum state over classical channels and perfectly recover it on the other side!

# Superdense Coding

The inverse of quantum teleportation is called **superdense coding** where we take some *classical* bits, encode them into a *quantum* state, and send it over a quantum communication channel to recover the classical bits. The neat part is that we only need to transmit one qubit for every 2 classical bits! That's why it's called *superdense* coding!

It's almost the inverse of quantum teleportation! Suppose Alice has 2 classical bits $d,c$ that she wants to transmit in a single qubit to Bob. Like with quantum teleportation, we'll start with both Alice and Bob sharing an entangled pair and then Alice will perform some operations on her qubit and send it to Bob. Bob now has both Alice's qubit and his half of the entangled pair to interact to recover the original two bits that Alice encoded. Alice transmitted only one qubit! Let's look at the the circuit first.

```python
qc = QuantumCircuit(2)

# Classical bits that Alice wants to encode
d, c = 0, 0

# Prep Bell state
qc.h(0)
qc.cx(0, 1)
qc.barrier()

# Alice performs some operations on her half of the entangled pair
if c == 1:
    qc.z(0)
if d == 1:
    qc.x(0)
qc.barrier()

# Bob receives Alice's qubit
# Bob interacts it with his half of the entangled pair and measures both
qc.cx(0, 1)
qc.h(0)
qc.measure_all()

print(qc)
```

Note that Alice's operations change the circuit based on which two bits she wants to encode.

```
        ┌───┐      ψ_0  ψ_1      ┌───┐ ψ_2  ░ ┌─┐   
   q_0: ┤ H ├──■────░────░────■──┤ H ├──░───░─┤M├───
        └───┘┌─┴─┐  ░    ░  ┌─┴─┐└───┘  ░   ░ └╥┘┌─┐
   q_1: ─────┤ X ├──░────░──┤ X ├───────░───░──╫─┤M├
             └───┘  ░    ░  └───┘       ░   ░  ║ └╥┘
meas: 2/═══════════════════════════════════════╩══╩═
                                               0  1 
{'00': 2048}
```

When Bob measures both his half of the entangled state as well as Alice's transmitted qubit, he gets the encoded classical bits with 100% accuracy!

Let's analyze this circuit. First, we create a Bell state $\ket{B_{0,0}}$.

$$
\ket{\psi_0} = \ket{B_{0,0}} = \frac{1}{\sqrt{2}}\Big(\ket{00} + \ket{11}\Big)
$$

Now depending on the classical bits, we either apply an X gate, Z gate, or both. Sound familiar to quantum teleportation? Let's list out the scenarios, starting with trying to encode $00$. In this case, we don't do anything and let the state pass to the next set of gates.

$$
\ket{\psi^{00}_1} = \ket{\psi_0} = \frac{1}{\sqrt{2}}\Big(\ket{00} + \ket{11}\Big)
$$

After the CNOT, we get the following state.

$$
\begin{align*}
\ket{\psi'^{00}_2} &= \frac{1}{\sqrt{2}}\Big(\ket{00} + \ket{01}\Big)\\
&= \ket{0}\frac{1}{\sqrt{2}}\Big(\ket{0} + \ket{1}\Big)\\
\end{align*}
$$

Applying a Hadamard to the rightmost qubit collapses the superposition.

$$
\ket{\psi^{00}_2} = \ket{0}\ket{0} = \ket{00}
$$

So when Bob measures, he'll always get the bits $00$! An even quicker way to see this for $0,0$ is that the quantum circuit is mirrored about $\ket{\psi_0}$ and quantum gates are invertible so this entire circuit is effectively a no-op if the input state is $\ket{00}$ so of course we get $\ket{00}$ at the very end.

Let's try this with trying to send $01$. We start with the Bell state but then we apply a Z gate to the first qubit, which flips the sign of the second term.

$$
\ket{\psi^{01}_1} = \frac{1}{\sqrt{2}}\Big(\ket{00} - \ket{11}\Big)
$$

Then we follow the same steps of applying the CNOT, regrouping the states, and applying a Hadamard.

$$
\begin{align*}
\ket{\psi'^{01}_2} &= \frac{1}{\sqrt{2}}\Big(\ket{00} - \ket{01}\Big)\\
&= \ket{0}\frac{1}{\sqrt{2}}\Big(\ket{0} - \ket{1}\Big)\\
\ket{\psi^{01}_2} &= \ket{0}\ket{1}=\ket{01}\\
\end{align*}
$$

For transmitting $10$, we need to apply an X gate to the first qubit of the Bell state, which flips the rightmost qubit for both terms.

$$
\ket{\psi^{10}_1} = \frac{1}{\sqrt{2}}\Big(\ket{01} + \ket{10}\Big)
$$

Following the same steps, we get the right answer.

$$
\begin{align*}
\ket{\psi'^{10}_2} &= \frac{1}{\sqrt{2}}\Big(\ket{11} + \ket{10}\Big)\\
&= \ket{1}\frac{1}{\sqrt{2}}\Big(\ket{1} + \ket{0}\Big)\\
\ket{\psi^{10}_2} &= \ket{1}\ket{0}=\ket{10}\\
\end{align*}
$$

Finally for transmitting $11$, we need to apply an X gate then a Z gate which will first flip the rightmost qubit and then flip the sign on the left term.

$$
\ket{\psi^{11}_1} = \frac{1}{\sqrt{2}}\Big(\ket{10} - \ket{01}\Big)
$$

Following the same steps, we get the right answer.

$$
\begin{align*}
\ket{\psi'^{11}_2} &= \frac{1}{\sqrt{2}}\Big(\ket{10} - \ket{11}\Big)\\
&= \ket{1}\frac{1}{\sqrt{2}}\Big(\ket{0} - \ket{1}\Big)\\
\ket{\psi^{11}_2} &= \ket{1}\ket{1}=\ket{11}\\
\end{align*}
$$

Now we know how to send classical information encoded into a qubit and decode it perfectly on the other end! Similar to quantum teleportation, this relies on Alice and Bob originally sharing an entangled pair before moving away from each other and Alice needs to reliably send her qubit to Bob somehow (also not faster than the speed of causality). But with this circuit, we can now also transmit arbitrary classical bits over quantum channels and perfectly decode the classical bits on the other side!


# Physical Representation of Qubits

So far, we've only been talking about qubits as mathematical entities, but I wanted to take a very brief aside to talk about how they're actually physically realized. I've already mentioned the spinning hard disks used for classical bits some decades ago and the more modern solid-state disks (SSDs) consisting of arrays of transistors that can be electronically controlled.

The qubits we've seen so far are called **logical qubits** since we can regard them as being "perfect" for our computational needs. However, **physical qubits** are the actual hardware representation of these qubits, much like a transistor is the physical representation of a classical bit. The two most prevalent kinds of qubits are **superconducting qubits** and **trapped-ion qubits**.

**Superconducting quantum computers**, used by IBM, Google, and Intel, utilize superconductors, sometimes called **quantum dots**, cooled to almost absolute zero where the logical $\ket{0}$ and $\ket{1}$ states are physically represented as the ground state $\ket{g}$ and excited state $\ket{e}$ of the superconductor. Superconducting qubits are manipulated by electrical signals on the order of nanoseconds, and they can scale up nicely since we can fabricate and connect arrays and lattices of quantum dots on the same chip. The primary challenges with superconducting qubits are that (i) we need to cool them to almost absolute zero in a dilution fridge to get the superconducting property of the superconductor and (ii) they are very susceptible to **decoherence** where the initialized qubits decay to random states after a short period of time due to noise in their environment.

**Trapped-ion quantum computers**, used by IonQ, realize qubits as ions held in free space by electromagnetic fields where the logical states map to electrical states of the ions. Many different kinds of ions can be used (IonQ uses ytterbium). These ions are manipulated using lasers of various frequencies to modify their electrical states corresponding to the quantum gates of the given quantum circuit. Trapped-ion qubits tend to have longer decoherence times since they're represented as stable ions from nature, and they can move among the lattice of traps to interact with any other arbitrary qubit; they're not just limited by the pre-fabricated lattice structure of the chip. On the other hand, gate operations tend to be slower than superconducting qubits, and it's more challenging to scale up ion-trap quantum computers since the traps allow for any qubit configuration as opposed to the fixed-lattice structure of a superconducting qubit chip.

Neither physical realization of qubits is better than the other: they just have different trade-offs between them. Perhaps people will invent another physical realization of qubits in the future!

# Conclusion

Quantum Computing is starting to be set up as "the next big thing"! Many top companies like Google, Intel, Microsoft, and IBM have started to construct their own quantum computers and showcase their progress on applying their quantum computers to solve practical business problems. We started our quantum journey with defining the qubit and comparing it to classical bits, and then discussed the various kinds of single-qubit gates. We quickly moved on to multi-qubit systems and their operations. Finally, we applied our knowledge to quantum teleportation  to transmit quantum state using classical bits and superdense coding to transmit classical bits using a quantum state.

This is just the start in our journey towards better understanding quantum algorithms and how we can move towards using quantum computing to solve practical problems in the real world today 🙂
