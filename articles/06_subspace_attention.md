# 6. Subspace Attention: From Standard Attention to Subdimensional Space

> *Once the equivalence between Hopfield Networks and Attention is established, a question arises: what other forms of attention — or Hopfield update — can we design? The AdvancedHopfieldModel from my shortest-path solver points in one direction: attention over subspaces.*

## The Scalability Problem

Standard attention has $O(n^2)$ complexity in the number of tokens. For long sequences, this is prohibitive. But there's a subtler problem: in a 5000-node graph, most edges don't exist (the graph is sparse). The full $n \times n$ adjacency matrix wastes memory and computation.

With Hopfield SPP, I addressed this with **sparse tensors**: only storing the edges that exist ($E$ instead of $n^2$), reducing memory from $O(n^2)$ to $O(E)$.

## Subspaces in Hopfield SPP

The `AdvancedHopfieldModel` in my shortest-path solver implements several techniques that have direct parallels in attention:

### Sparse Tensors

```python
# Only store existing edges instead of the full matrix
# Memory: O(E) instead of O(n²)
sparse_indices = tf.constant(edge_list, dtype=tf.int64)  # shape (E, 2)
sparse_values = tf.Variable(initial_values, shape=(E,))
```

### Attention Mechanism

The advanced model includes an attention mechanism that weighs the influence of neighboring nodes by their relevance:

```python
attention_weights = tf.nn.softmax(
    beta * tf.reduce_sum(node_features * target_features, axis=-1)
)
update = tf.reduce_sum(attention_weights * neighbor_values, axis=0)
```

### Adaptive Hyperparameters

Instead of a fixed temperature, the model dynamically adjusts:

- **Temperature**: controls the sharpness of attention/activation
- **Learning rate**: adapts based on convergence
- **Number of restarts**: increases for harder problems

### Beam Search for Extraction

The advanced model uses beam search (similar to decoding in transformers) to extract the optimal path from the activation matrix:

```python
# Beam search: keep the top-k partial paths
beam = [(source, [source], 0.0)]
for step in range(max_steps):
    candidates = []
    for node, path, score in beam:
        for next_node in adjacency[node]:
            candidates.append((next_node, path + [next_node], score + cost))
    beam = sorted(candidates, key=lambda x: x[2])[:beam_width]
```

## From Hopfield SPP to Subspace Attention

The techniques I implemented to scale Hopfield to large graphs are directly applicable to attention in transformers:

| Hopfield SPP Technique | Attention Equivalent |
|---------------------|------------------------|
| Sparse tensors | Sparse attention (Reformer, BigBird) |
| Local neighborhood attention | Sliding-window attention |
| Beam search | Autoregressive decoding |
| Adaptive hyperparameters | Temperature scaling at inference |
| Per-subgraph energy | Patch-based attention |

## Subspace Attention as a General Principle

The central idea that emerges is: **not all attention needs to operate over the full space**. We can restrict attention to relevant subspaces:

- **Token subspace**: local attention, windows, neighborhoods
- **Feature subspace**: factorized (low-rank) attention, projections
- **Temporal subspace**: attention over time windows, compressed memory

Each of these is a Hopfield Network operating in a subspace of the full space. And each inherits the convergence and stability properties of Hopfield theory.

---

**Next: [Chapter 7 — Looking Ahead: Hopfield Layers in Modern Deep Learning](07_future.md)**
