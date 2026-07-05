---
title: "From My 1998 Thesis to Transformers: 25 Years, 7 Fixes, One Unexpected Connection"
date: 2026-05-11
author: "Roberto de la Cámara"
tags: [hopfield-networks, transformers, attention, personal-story, history-of-ml, neural-networks]
platforms: [devto]
---

# From My 1998 Thesis to Transformers: 25 Years, 7 Fixes, One Unexpected Connection

In 1998 I was in my final year of college, sitting in front of a CRT monitor, debugging a neural network that refused to converge. My thesis: applying Hopfield Networks to the shortest path problem.

The network worked sometimes. 40-60% of the time. And when it failed, I didn't know why.

Twenty-five years later, I work defining the AI-native strategy for 5G Core Networks at Ericsson. And one day, reviewing the attention mechanism in Transformers, I realized something that stopped me cold:

**A Transformer's attention and a Hopfield Network's state update are the same mathematical operation.**

Not "inspired by." Not "related to." The same.

---

## The "aha" moment

Look at these two formulas:

**Hopfield Network (modern):**
```
V_new = softmax(β · Ξ · V) · Ξ
```

**Transformer Attention:**
```
Attention(Q, K, V) = softmax(Q · K^T / √d) · V
```

They're structurally identical. In Hopfield, you search your memory (Ξ) for the pattern closest to your current state (V), and retrieve a weighted combination. In Attention, you search your keys (K) for what's most relevant to your query (Q), and retrieve a weighted combination of values (V).

The only real difference: in Transformers you learn to project Q, K, V from the same input using distinct weight matrices. In classic Hopfield, the patterns Ξ are fixed.

---

## What my thesis taught me (and what's still true today)

Debugging Hopfield Networks in 1998, I learned lessons I still apply daily:

**1. Correct constraints are everything.** My original implementation had a subtle bug: it encoded Traveling Salesman Problem constraints (a Hamiltonian cycle) instead of Shortest Path constraints (flow conservation). Fixing this took reliability from 40% to 95%. A single line of code.

**2. No shared state between queries.** Reusing the Adam optimizer across queries contaminated the momentum from previous queries. Every query needs fresh state. This is even more relevant today with models serving millions of users.

**3. Always have a fallback.** My solver now runs Dijkstra in parallel. If the Hopfield solution is within 5% of optimal, I use it. Otherwise, I fall back to Dijkstra. This gives 100% reliability. In production, you always need a plan B.

**4. Early stopping saves 40-60% of the time.** When energy stabilizes, stop. Don't keep iterating without improvement. This principle is everywhere today: early stopping in training, speculative decoding in inference.

---

## The 7 fixes (summary)

1. Flow constraints vs. TSP (40% → 95% reliability)
2. Zero offline training (30-60s saved)
3. Fresh optimizer per query
4. Dijkstra fallback (100% reliability)
5. Early stopping on stable energy
6. BFS instead of greedy argmax
7. In-memory model cache (2-3s → 50ms)

---

## The demo that visualizes it

I built an interactive dashboard where you can adjust temperature, noise, and see how the same operation is visualized as Hopfield or as Attention:

- **Low β** → diffuse attention, all patterns weigh similarly
- **High β** → focused attention, a single pattern dominates

It's exactly the same principle as softmax temperature in LLMs.

---

## Why this matters (and it's not just nostalgia)

1. The Hopfield-Attention connection provides a **solid theoretical framework** for understanding why transformers work
2. It suggests **new architectures**: iterative attention (multiple Hopfield steps), hierarchical memory, dynamic temperature
3. It closes a personal 25-year loop

The 1982 Hopfield Network, which seemed like a dead end, turned out to be the embryo of the architecture that defines modern AI. It didn't take a new idea. It took better activation functions, better hardware, and 25 years of engineering.

---

*The code, the dashboard, and all 7 full chapters are here:*
**GitHub:** [RobertoDeLaCamara/HopfieldAttention](https://github.com/RobertoDeLaCamara/HopfieldAttention)

*Homelab hardware used: RTX 5060 Ti + WSL2 (Hawkeye) + Raspberry Pi cluster for network monitoring.*

*If you want to read the full technical article (with math, code, and benchmarks): [Hashnode — From Hopfield Networks to Transformers](URL_HERE)*
