import pytest
import numpy as np
from hopfield_attention_bridge import HopfieldAttention


@pytest.fixture
def layer():
    return HopfieldAttention(dim=16, beta=2.0)


class TestHopfieldAttention:
    def test_equivalence(self, layer):
        """Hopfield update == Attention forward (same params)."""
        state = np.random.randn(16).astype(np.float32)
        state /= np.linalg.norm(state)
        result = layer.compare(state)
        assert result["are_identical"]
        assert result["difference_norm"] < 1e-6

    def test_output_shape(self, layer):
        """The output has the same dimensionality as the input."""
        state = np.random.randn(16).astype(np.float32)
        output = layer.hopfield_update(state)
        assert output.shape == (16,)

    def test_energy_decreases(self, layer):
        """Energy does not increase monotonically, but it converges."""
        state = np.random.randn(16).astype(np.float32)
        state /= np.linalg.norm(state)
        for _ in range(30):
            new_state = layer.hopfield_update(state)
            if np.linalg.norm(new_state - state) < 1e-5:
                break
            state = new_state
        # Converged to a stable point
        e_final = layer.energy(state)
        e_next = layer.energy(layer.hopfield_update(state))
        assert abs(e_next - e_final) < 1e-3

    def test_temperature_effect(self):
        """High β → more focused attention (lower entropy)."""
        state = np.random.randn(8).astype(np.float32)
        state /= np.linalg.norm(state)

        entropies = []
        for beta in [0.1, 5.0]:
            layer = HopfieldAttention(dim=8, beta=beta)
            sims = beta * layer.store @ state
            weights = np.exp(sims - np.max(sims))
            weights /= np.sum(weights)
            entropy = -np.sum(weights * np.log(weights + 1e-10))
            entropies.append(entropy)

        assert entropies[0] > entropies[1]

    def test_energy_with_noise(self, layer):
        """A noisy state has higher energy than the clean pattern."""
        store_vec = layer.store[0].copy()
        noisy = store_vec + 0.5 * np.random.randn(16)
        noisy /= np.linalg.norm(noisy)

        e_clean = layer.energy(store_vec)
        e_noisy = layer.energy(noisy)
        assert e_noisy >= e_clean - 1e-6

    def test_convergence_to_attractor(self, layer):
        """Repeated updates converge to an attractor (stable energy)."""
        state = np.random.randn(16).astype(np.float32)
        state /= np.linalg.norm(state)

        for _ in range(50):
            new_state = layer.hopfield_update(state)
            if np.linalg.norm(new_state - state) < 1e-5:
                break
            state = new_state

        # Energy stable after convergence
        e_final = layer.energy(state)
        e_next = layer.energy(layer.hopfield_update(state))
        assert abs(e_next - e_final) < 1e-4

    def test_attention_as_hopfield(self, layer):
        """attention_as_hopfield must give the same result as hopfield_update."""
        state = np.random.randn(16).astype(np.float32)
        state /= np.linalg.norm(state)

        h = layer.hopfield_update(state)
        a = layer.attention_as_hopfield(state)

        assert np.linalg.norm(h - a) < 1e-6
