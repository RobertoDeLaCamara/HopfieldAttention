---
title: HopfieldAttention
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: 1.42.0
app_file: dashboard/app.py
pinned: false
short_description: "De Hopfield Networks a Transformers — la conexión interactiva"
tags:
  - hopfield-networks
  - transformers
  - attention
  - demo
  - streamlit
license: mit
---

# HopfieldAttention

**From Hopfield Networks (1982) to Transformers (2017): the mathematical bridge, traced across 7 articles.**

[**Interactive Demo**](https://huggingface.co/spaces/RobertoDeLaCamara/HopfieldAttention) · [Articles](articles/) · [Bridge Code](bridge/)

The attention mechanism in Transformers is, mathematically, a single-step Hopfield update with softmax as activation. This project traces that connection from first principles — starting with Hopfield's 1982 energy function, through a 1998 university thesis on shortest path problems, to Ramsauer et al. (2021) and modern LLMs.

## The Core Equivalence

| Hopfield Update | Transformer Attention |
|---|---|
| `V_new = softmax(β · Ξ · V) · Ξ` | `Attention(Q,K,V) = softmax(Q·Kᵀ/√d) · V` |

Both compute a softmax-weighted sum over a set of stored vectors. The correspondence is exact:

| Hopfield | Attention | Meaning |
|---|---|---|
| State V | Query Q | What we're processing |
| Patterns Ξ | Keys K | What we know / the memory |
| Patterns Ξ | Values V | What we retrieve |
| Temperature β | Scale 1/√d | How sharp the attention is |

## Contents

### Articles — 7-part series

| # | Title | Topic |
|---|---|---|
| 1 | [La Hopfield Original](articles/01_hopfield_original.md) | Energy function, attractors, associative memory |
| 2 | [Mi tesis de 1998](articles/02_thesis_1998.md) | Hopfield for Shortest Path Problem |
| 3 | [Siete correcciones](articles/03_seven_fixes.md) | Why classical Hopfield fails at optimization |
| 4 | [Modern Hopfield](articles/04_modern_hopfield.md) | Exponential capacity, continuous values |
| 5 | [La Conexión](articles/05_hopfield_attention.md) | **The bridge: Hopfield IS attention** |
| 6 | [Subspace Attention](articles/06_subspace_attention.md) | From standard to subspace attention |
| 7 | [Hacia el Futuro](articles/07_future.md) | Hopfield layers in modern deep learning |

### Bridge Code

`bridge/hopfield_attention_bridge.py` implements the same operation as both a Hopfield update and as attention, and verifies they are numerically identical:

```python
from bridge.hopfield_attention_bridge import HopfieldAttention
import numpy as np

layer = HopfieldAttention(dim=16, beta=2.0)
state = np.random.randn(16).astype(np.float32)
state /= np.linalg.norm(state)

result = layer.compare(state)
print(result["are_identical"])       # True
print(result["difference_norm"])     # < 1e-6
```

Run the tests:

```bash
cd bridge && pytest test_bridge.py -v
```

### Interactive Dashboard

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

Or use the [live demo on HuggingFace Spaces](https://huggingface.co/spaces/RobertoDeLaCamara/HopfieldAttention).

Adjust temperature β, number of stored patterns, and noise level — and see in real time how the Hopfield update and the attention mechanism produce identical outputs.

## Background

In 1998 I wrote a university thesis implementing a Hopfield Network for the Shortest Path Problem. In 2025, working on ML systems for 5G networks, I realized the attention mechanism I was using daily was mathematically the same operation I had studied 25 years earlier. This project documents that connection.

## License

MIT — see [LICENSE](LICENSE)
