"""
tau-chrono interactive demo.

Run: streamlit run demo.py
"""

import streamlit as st
import numpy as np

# Import tau_chrono from local package
from tau_chrono import (
    depolarizing,
    amplitude_damping,
    dephasing,
    verify_cptp,
    bayesian_compose,
    tau_parameter,
    fidelity,
    GateResult,
    CompositionResult,
)

st.set_page_config(
    page_title="tau-chrono demo",
    page_icon="⏳",
    layout="wide",
)

st.title("tau-chrono")
st.markdown("**Bayesian noise tracking for quantum circuits via Petz recovery maps**")
st.markdown("---")

# --- Sidebar ---
st.sidebar.header("Circuit Parameters")
n_gates = st.sidebar.slider("Number of gates (depth)", 1, 60, 20)
noise_type = st.sidebar.selectbox("Noise channel", ["Depolarizing", "Amplitude damping", "Dephasing"])
error_rate = st.sidebar.slider("Per-gate error rate", 0.001, 0.15, 0.05, 0.001)

st.sidebar.markdown("---")
st.sidebar.markdown("**What this shows**")
st.sidebar.markdown(
    "The naive (independent) model multiplies per-gate errors, "
    "overestimating total noise. Bayesian composition via the Petz "
    "recovery map captures noise saturation."
)

# --- Initial states ---
rho = np.array([[1, 0], [0, 0]], dtype=complex)    # |0>
sigma = np.eye(2, dtype=complex) / 2                # maximally mixed

# --- Build gate list ---
gate_channels = []
for i in range(n_gates):
    if noise_type == "Depolarizing":
        kraus = depolarizing(error_rate)
    elif noise_type == "Amplitude damping":
        kraus = amplitude_damping(error_rate)
    else:
        kraus = dephasing(error_rate)
    gate_channels.append(kraus)

# --- Compute ---
result: CompositionResult = bayesian_compose(gate_channels, sigma_0=sigma, rho=rho)

# --- Depth-by-depth accumulation ---
depths = list(range(1, n_gates + 1))
tau_naive_acc = []
tau_bayes_acc = []

for d in depths:
    sub = bayesian_compose(gate_channels[:d], sigma_0=sigma, rho=rho)
    tau_naive_acc.append(sub.tau_multiplicative_total)
    tau_bayes_acc.append(sub.tau_bayesian_total)

# --- Layout ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Naive tau", f"{result.tau_multiplicative_total:.4f}")
with col2:
    st.metric("Bayesian tau", f"{result.tau_bayesian_total:.4f}")
with col3:
    st.metric("Improvement", f"{result.improvement_percent:.1f}%")

st.markdown("---")

# --- Chart ---
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=depths, y=tau_naive_acc,
    mode='lines+markers', name='Naive (independent)',
    line=dict(color='#EF4444', width=2),
    marker=dict(size=4),
))
fig.add_trace(go.Scatter(
    x=depths, y=tau_bayes_acc,
    mode='lines+markers', name='Bayesian (Petz)',
    line=dict(color='#3B82F6', width=2),
    marker=dict(size=4),
))
fig.add_hline(y=0.5, line_dash="dash", line_color="#888",
              annotation_text="tau = 0.5 threshold")

fig.update_layout(
    title="Noise accumulation: naive vs Bayesian",
    xaxis_title="Circuit depth (gates)",
    yaxis_title="Total circuit noise (tau)",
    yaxis=dict(range=[0, 1.05]),
    template="plotly_dark",
    height=450,
    legend=dict(x=0.02, y=0.98),
)
st.plotly_chart(fig, use_container_width=True)

# --- Table ---
st.markdown("### Gate-by-gate breakdown")

table_data = {
    "Depth": depths,
    "tau_naive": [f"{v:.4f}" for v in tau_naive_acc],
    "tau_bayes": [f"{v:.4f}" for v in tau_bayes_acc],
    "Improvement": [
        f"{(tn - tb) / tn * 100:.1f}%" if tn > 0 else "0%"
        for tn, tb in zip(tau_naive_acc, tau_bayes_acc)
    ],
}
st.dataframe(table_data, use_container_width=True, hide_index=True)

# --- Composition inequality ---
st.markdown("### Composition inequality")
st.latex(r"\sqrt{\tau_{\text{total}}} \leq \sum_j \sqrt{\tau_j^{\text{eff}}}")

st.markdown(
    f"sqrt(tau_total) = **{result.composition_lhs:.4f}** &le; "
    f"sum(sqrt(tau_i)) = **{result.composition_rhs:.4f}** — "
    f"{'**HOLDS**' if result.composition_holds else '**VIOLATED**'}"
)

# --- Footer ---
st.markdown("---")
st.markdown(
    "Built on: Petz (1986) · Parzygnat & Buscemi (2023) · Junge et al. (2018) | "
    "[GitHub](https://github.com/akaiHuang/tau-chrono) · "
    "[Website](https://tau-chrono.pages.dev)"
)
