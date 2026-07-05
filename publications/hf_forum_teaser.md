---
title: "Hopfield Networks and Attention Are the Same Operation — and I Implemented It in 100 Lines of Python"
date: 2026-05-11
author: "Roberto de la Cámara"
tags: [hopfield-networks, transformers, attention, python, numpy]
platforms: [huggingface]
---

# Hopfield Networks and Attention Are the Same Operation

**TL;DR:** A modern Hopfield Network's update and a Transformer's attention mechanism produce identical results when configured with the same parameters. I've verified this with numerical tests.

## The equivalence in 30 seconds

```python
# Hopfield: V_new = softmax(β · Ξ · V) · Ξ
sims = beta * patterns @ state
weights = softmax(sims)
new_state = weights @ patterns

# Attention: Attn = softmax(Q · K^T / √d) · V
scale = sqrt(d_k)
sims = (query @ keys.T) / scale
weights = softmax(sims)
output = weights @ values

# They're identical when: Q=V, K=patterns, V=patterns, scale=1/β
```

## Test results

I implemented `hopfield_attention_bridge.py` (100 lines, pure NumPy) that demonstrates the equivalence:

| Test | Result |
|------|-----------|
| Hopfield vs Attention difference | 0.0e+00 (identical) |
| 10 random states | ✅ All verified |
| High β → low entropy | ✅ More focused attention |
| Iterative convergence | ✅ Stable energy |

## Question for the community

If attention = Hopfield update, what are the implications for:

1. **New architectures** — iterative attention with multiple energy steps
2. **Explicit memory** — Hopfield as external memory for continual learning
3. **Dynamic temperature** — β learned per token instead of fixed

Has anyone explored any of these directions? I'm especially interested in iterative attention as an alternative to stacks of chained transformers.

---

**Full article (7 chapters):** [Hashnode](URL_HERE)
**Code + interactive dashboard:** [GitHub](https://github.com/RobertoDeLaCamara/HopfieldAttention)
**Live demo:** 🤗 *coming soon*

*Built in my homelab — WSL2 + RTX 5060 Ti + NumPy for the equivalence tests, TensorFlow for the Shortest Path solver.*
