# 1. The Original Hopfield Network: Energy, Stability, and Fixed Points

> *Before deep learning was deep, before "attention" was a concept in machine learning, there was a simple recurrent network that converged to fixed points by minimizing an energy function.*

## The Model

In 1982, John Hopfield published a paper that revitalized neural networks at a moment when the field was stagnant. His idea was elegant: a network of binary neurons, all connected to each other, that evolved toward stable states — attractors — by minimizing an energy function.

A Hopfield Network is a fully connected recurrent network where every neuron is connected to every other neuron (but not to itself). The neurons are binary (on or off), and the network updates its state asynchronously, one neuron at a time.

## The Energy Function

Hopfield's key contribution was defining an energy (or Lyapunov) function that always decreases with every update:

$$E = -\frac{1}{2} \sum_{i,j} w_{ij} s_i s_j + \sum_i \theta_i s_i$$

Where:
- $w_{ij}$ is the weight of the connection between neurons $i$ and $j$
- $s_i$ is the state (0 or 1) of neuron $i$
- $\theta_i$ is the threshold of neuron $i$

Every state update guarantees that the energy never increases. Since the energy is bounded below, the network inevitably converges to a local minimum — an attractor.

## Attractors as Memory

Hopfield Networks became popular as **associative memories**: patterns are stored as energy minima, and the network recovers the full pattern from a partial or noisy version.

If we train the network on images of digits, each digit becomes an attractor. When presented with a noisy digit, the network converges to the nearest pattern — it "recovers" the clean image.

## Two Fundamental Limitations

1. **Limited capacity**: A Hopfield Network with $n$ neurons can store roughly $0.15n$ patterns before spurious attractors start to dominate.

2. **Tendency toward spurious minima**: The network converges to the nearest *local minimum*, which doesn't always correspond to a stored pattern. This is especially problematic when using it for combinatorial optimization, as we'll see in the next chapter.

## Why This Matters

The Hopfield update mechanism — computing a weighted sum of inputs and applying a nonlinearity — is structurally identical to the attention mechanism that defines Transformers. The difference lies in the nonlinearity (sign/sigmoid vs. softmax) and in how the connections are structured.

But we're getting ahead of the story. First, let's look at how Hopfield networks were applied to combinatorial optimization.

---

**Next: [Chapter 2 — My 1998 Thesis: Hopfield for the Shortest Path Problem](02_thesis_1998.md)**
