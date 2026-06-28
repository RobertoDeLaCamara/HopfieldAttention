import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="HopfieldAttention", layout="wide")
st.title("🧠 HopfieldAttention: La Conexión Hopfield ↔ Attention")

st.markdown("""
Esta demo interactiva muestra la equivalencia matemática entre una **Hopfield Network**
y el **mecanismo de atención** de los transformers.

Ambos lados realizan la misma operación:
**softmax(similitudes) × valores**
""")

# ---- Sidebar controls ----
st.sidebar.header("Parámetros")

n_patterns = st.sidebar.slider("Número de patrones (memoria)", 3, 10, 5)
n_neurons = st.sidebar.slider("Dimensión de cada patrón", 4, 20, 8)
temperature = st.sidebar.slider("Temperatura β (1/temperatura)", 0.1, 5.0, 1.0, 0.1)
noise = st.sidebar.slider("Ruido en el estado inicial", 0.0, 2.0, 0.5, 0.1)

col1, col2 = st.columns(2)

# ---- Generate data ----
rng = np.random.RandomState(42)

# Patrones almacenados (memoria Hopfield ≡ Keys/Values)
patterns = rng.randn(n_patterns, n_neurons)
patterns = patterns / np.linalg.norm(patterns, axis=1, keepdims=True)

# Estado inicial (Query noisy)
clean_state = patterns[0].copy()
noisy_state = clean_state + noise * rng.randn(n_neurons)
noisy_state = noisy_state / np.linalg.norm(noisy_state)

# ---- Hopfield update ----
# similarity = β · Ξ · V  (Hopfield)  ≡  Q · K^T / √d  (Attention)
similarities = temperature * patterns @ noisy_state  # shape (n_patterns,)
attn_weights = np.exp(similarities) / np.sum(np.exp(similarities))

# Hopfield: new_state = softmax(β · Ξ · V) · Ξ
# Attention: output = softmax(Q · K^T / √d) · V
new_state = attn_weights @ patterns  # identical operation

# Energy (Modern Hopfield: E = -logsumexp(β · ξ · V))
logsumexp = np.log(np.sum(np.exp(similarities)))
energy = -logsumexp  # Modern Hopfield energy

# ---- Left: Hopfield interpretation ----
with col1:
    st.subheader("🔵 Como Hopfield Network")
    st.markdown("""
    **Actualización:** $V_{new} = softmax(\\beta \\cdot \\Xi \\cdot V) \\cdot \\Xi$
    """)

    # Weights bar chart
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=[f"ξ<sub>{i}</sub>" for i in range(n_patterns)],
        y=attn_weights,
        marker_color="royalblue",
        showlegend=False,
    ))
    fig1.update_layout(
        title=f"Pesos de atención (softmax) — Energía: {energy:.3f}",
        xaxis_title="Patrón almacenado",
        yaxis_title="Peso",
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig1, use_container_width=True)

    # State comparison
    state_fig = go.Figure()
    state_fig.add_trace(go.Scatter(
        y=clean_state, mode="lines+markers",
        name="Patrón limpio (atractor)", line=dict(color="green", width=2),
    ))
    state_fig.add_trace(go.Scatter(
        y=noisy_state, mode="lines+markers",
        name="Estado inicial (ruidoso)", line=dict(color="gray", width=2, dash="dash"),
    ))
    state_fig.add_trace(go.Scatter(
        y=new_state, mode="lines+markers",
        name="Estado recuperado (Hopfield)", line=dict(color="royalblue", width=3),
    ))
    state_fig.update_layout(
        title="Recuperación del patrón desde estado ruidoso",
        xaxis_title="Dimensión",
        yaxis_title="Valor",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(state_fig, use_container_width=True)

# ---- Right: Attention interpretation ----
with col2:
    st.subheader("🟠 Como Transformer Attention")
    st.markdown("""
    **Atención:** $\text{softmax}\left(\\frac{Q K^T}{\sqrt{d_k}}\right) V$
    """)

    # Similarity heatmap
    # In attention: Q=noisy_state, K=patterns, V=patterns
    Q = noisy_state.reshape(1, -1)  # (1, d)
    K = patterns  # (n_patterns, d)
    sim_matrix = (temperature * Q @ K.T).numpy() if hasattr(temperature * Q @ K.T, "numpy") else temperature * (Q @ K.T)  # (1, n_patterns)
    sim_matrix = temperature * noisy_state @ patterns.T

    heat_fig = go.Figure(data=go.Heatmap(
        z=sim_matrix.reshape(1, -1),
        x=[f"K<sub>{i}</sub>" for i in range(n_patterns)],
        y=["Q (noisy state)"],
        colorscale="Viridis",
        colorbar=dict(title="β · Q·K"),
    ))
    heat_fig.update_layout(
        title="Matriz de similitud Q · K<sup>T</sup> (producto punto escalado)",
        height=150,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(heat_fig, use_container_width=True)

    # Distribution of attention
    attn_fig = go.Figure()
    attn_fig.add_trace(go.Bar(
        x=[f"V<sub>{i}</sub>" for i in range(n_patterns)],
        y=attn_weights,
        marker_color="darkorange",
        showlegend=False,
    ))
    attn_fig.update_layout(
        title="Distribución de atención softmax(Q·K<sup>T</sup>/√d)",
        xaxis_title="Value",
        yaxis_title="Peso de atención",
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(attn_fig, use_container_width=True)

    # Output = attended values
    out_fig = go.Figure()
    out_fig.add_trace(go.Scatter(
        y=new_state, mode="lines+markers",
        name="Salida = Attention(Q,K,V)",
        line=dict(color="darkorange", width=3),
    ))
    out_fig.add_trace(go.Scatter(
        y=clean_state, mode="lines+markers",
        name="V original (patrón limpio)",
        line=dict(color="green", width=2),
    ))
    out_fig.update_layout(
        title="Salida de atención (misma que actualización Hopfield)",
        xaxis_title="Dimensión",
        yaxis_title="Valor",
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(out_fig, use_container_width=True)

# ---- Bottom: Equivalence proof ----
st.divider()
st.subheader("⚡ La Equivalencia")

col_eq1, col_eq2, col_eq3 = st.columns(3)
with col_eq1:
    st.markdown("""
    **Hopfield Update**
    ```
    V_new = softmax(β · Ξ · V) · Ξ
    ```
    """)
with col_eq2:
    st.markdown("""
    **Attention**
    ```
    Attn(Q,K,V) = softmax(Q·K^T/√d) · V
    ```
    """)
with col_eq3:
    st.markdown("""
    **Misma operación**
    ```
    yᵢ = Σⱼ wⱼ · xⱼ
    w = softmax(similitudes)
    ```
    """)

# Quantitative comparison
recovery_error = np.linalg.norm(new_state - clean_state)
attention_entropy = -np.sum(attn_weights * np.log(attn_weights + 1e-10))

st.divider()
m1, m2, m3 = st.metric(
    "Error de recuperación vs patrón limpio",
    f"{recovery_error:.4f}",
    delta=None,
)

m2, m3 = st.columns(2)
with m2:
    st.metric("Entropía de atención", f"{attention_entropy:.3f}")
with m3:
    st.metric("Temperatura β", f"{temperature:.1f}")

st.caption("Cuando la entropía es baja → atención enfocada en un patrón | Cuando es alta → atención difusa entre varios patrones")
