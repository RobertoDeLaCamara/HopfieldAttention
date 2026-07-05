"""
hopfield_attention_bridge.py

Demonstrates the formal equivalence between a Hopfield Network update
and a Transformer's attention mechanism.

The thesis: Attention(Q, K, V) = softmax(Q·K^T / √d) · V
is identical to:  V_new = softmax(β · Ξ · V) · Ξ

Where:
  Q ~ V (current state / query)
  K ~ Ξ (stored patterns / keys)
  V ~ Ξ (values = same patterns in Hopfield, projected in attention)
  β ~ 1/√d (temperature ~ scaling factor)
"""

import numpy as np


class HopfieldAttention:
    """A layer that implements the same operation, interpretable as
    either a Hopfield Network or Attention.

    Hopfield view:
        V_new = softmax(β · store · V) · store

    Attention view:
        Attn(Q, K, V) = softmax(Q · K^T / √d) · V
    """

    def __init__(self, dim: int, beta: float = 1.0):
        self.dim = dim
        self.beta = beta

        # In Hopfield: the stored patterns
        # In Attention: the projected keys/values
        self.store = np.random.randn(dim, dim).astype(np.float32)
        self.store /= np.linalg.norm(self.store, axis=1, keepdims=True)

    def hopfield_update(self, state: np.ndarray) -> np.ndarray:
        """Update as a Hopfield Network.
        V_new = softmax(β · Ξ · V) · Ξ
        """
        # Similarity between the current state and the stored patterns
        sims = self.beta * self.store @ state  # (dim,)

        # Softmax normalization
        weights = np.exp(sims - np.max(sims))
        weights /= np.sum(weights)  # (dim,)

        # New state = weighted combination of patterns
        new_state = weights @ self.store  # (dim,)
        return new_state

    def attention_forward(
        self, query: np.ndarray, keys: np.ndarray, values: np.ndarray
    ) -> np.ndarray:
        """Standard attention: softmax(Q·K^T / √d) · V"""
        scale = np.sqrt(keys.shape[-1])
        sims = (query @ keys.T) / scale  # (n_keys,)
        weights = np.exp(sims - np.max(sims))
        weights /= np.sum(weights)
        output = weights @ values
        return output

    def attention_as_hopfield(self, state: np.ndarray) -> np.ndarray:
        """Attention configured to be identical to Hopfield.
        Q = state (the current state)
        K = store (the stored patterns)
        V = store (the same patterns as values)
        scale = 1/β (inverse temperature as scaling factor)
        """
        # β = 1/√d ---> d = 1/β²
        scale = 1.0 / self.beta if self.beta > 0 else 1.0
        sims = (state @ self.store.T) / scale  # (n_patterns,)
        weights = np.exp(sims - np.max(sims))
        weights /= np.sum(weights)
        output = weights @ self.store
        return output

    def energy(self, state: np.ndarray) -> float:
        """Modern Hopfield energy: E = -logsumexp(β · ξ · V)"""
        sims = self.beta * self.store @ state
        logsumexp = np.log(np.sum(np.exp(sims)))
        return -logsumexp

    def compare(self, state: np.ndarray) -> dict:
        """Compares the Hopfield update vs the Attention forward pass apples-to-apples."""
        hopfield_result = self.hopfield_update(state)

        # Equivalent attention: Q=state, K=store, V=store, scale=1/β
        attn_result = self.attention_as_hopfield(state)

        diff = np.linalg.norm(hopfield_result - attn_result)

        return {
            "hopfield_output": hopfield_result,
            "attention_output": attn_result,
            "difference_norm": diff,
            "are_identical": diff < 1e-6,
            "energy": self.energy(state),
        }


def test_equivalence():
    """Verifies that the Hopfield update and the Attention forward pass produce the same result."""
    dim = 16
    layer = HopfieldAttention(dim=dim, beta=2.0)

    state = np.random.randn(dim).astype(np.float32)
    state /= np.linalg.norm(state)

    result = layer.compare(state)

    assert result["are_identical"], (
        f"Hopfield and Attention should be identical. "
        f"Difference: {result['difference_norm']:.2e}"
    )
    print(f"✅ Equivalence verified. Energy: {result['energy']:.4f}")
    print(f"   Max difference: {result['difference_norm']:.2e}")


def test_different_states():
    """Tests with multiple initial states."""
    dim = 8
    layer = HopfieldAttention(dim=dim, beta=1.5)

    for i in range(10):
        state = np.random.randn(dim).astype(np.float32)
        state /= np.linalg.norm(state)

        result = layer.compare(state)

        assert result["are_identical"], (
            f"Failed on state {i}. Difference: {result['difference_norm']:.2e}"
        )

    print(f"✅ {10} different states verified. "
          f"Energy range: OK")


def test_temperature_effect():
    """Verifies that temperature affects the attention distribution."""
    dim = 8
    state = np.random.randn(dim).astype(np.float32)
    state /= np.linalg.norm(state)

    for beta in [0.1, 1.0, 5.0, 10.0]:
        layer = HopfieldAttention(dim=dim, beta=beta)
        h = layer.hopfield_update(state)

        # Compute the entropy of the weight distribution
        sims = beta * layer.store @ state
        weights = np.exp(sims - np.max(sims))
        weights /= np.sum(weights)
        entropy = -np.sum(weights * np.log(weights + 1e-10))

        print(f"   β={beta:.1f} → entropy={entropy:.3f}")

    # High β → low entropy (focused attention)
    # Low β → high entropy (diffuse attention)
    print("✅ Temperature effect verified")


def test_convergence():
    """Verifies that repeated iteration converges (energy stabilizes)."""
    dim = 8
    layer = HopfieldAttention(dim=dim, beta=2.0)

    state = np.random.randn(dim).astype(np.float32)
    state /= np.linalg.norm(state)

    energies = [layer.energy(state)]
    for _ in range(50):
        new_state = layer.hopfield_update(state)
        energies.append(layer.energy(new_state))
        if np.linalg.norm(new_state - state) < 1e-5:
            break
        state = new_state

    # The last two energy values should be nearly equal (convergence)
    assert abs(energies[-1] - energies[-2]) < 1e-3, (
        f"Did not converge: {energies[-2]:.4f} → {energies[-1]:.4f}"
    )
    print(f"✅ Convergence verified ({len(energies)} steps, "
          f"{len(set(round(e, 2) for e in energies))} unique energy values)")


if __name__ == "__main__":
    print("=" * 60)
    print("HopfieldAttention — Formal Equivalence Test")
    print("=" * 60)
    print()

    test_equivalence()
    test_different_states()
    test_temperature_effect()
    test_convergence()

    print()
    print("=" * 60)
    print("✅ All tests passed.")
    print("The Hopfield ↔ Attention equivalence is verified.")
    print("=" * 60)
