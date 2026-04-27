#!/usr/bin/env python3
"""Generate v2 figures from raw JSON data files.

Outputs to tau-chrono/results/:
  fig_v2_anomaly_gsweep.png         — F_anomaly continuity validation
  fig_v2_anomaly_decay.png          — T_anomaly = 101 ns / 500 ns DD
  fig_v2_cross_platform.png         — 4-backend F_anomaly bar chart
  fig_v2_t17_pair_shopping.png      — Hardware non-uniformity within Tuna-17
  fig_v2_chemistry_sprint.png       — 4-molecule baseline / v2 dissociation curves
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List

os.environ['MPLCONFIGDIR'] = '/tmp/mpl'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 13,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10,
    'figure.dpi': 200, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
    'font.family': 'serif', 'mathtext.fontset': 'cm',
})

# Hardware data dir (in companion repo)
DATA_DIR = Path("/Users/akaihuangm1/Desktop/github/quantum-llm/experiments/future_signal")
OUT_DIR = Path(__file__).resolve().parent

C_THEORY = "#76e09d"   # green
C_TUNA9 = "#ff8a6a"    # warm red (depolarizing)
C_GARNET = "#7c9cff"   # blue (amp damp)
C_SIRIUS = "#a98cff"   # purple (resonator)
C_EMERALD = "#5fc6c4"  # teal (amp damp)
C_TUNA17 = "#ffa44a"   # orange
C_V2 = "#ffd84a"       # gold
C_BASELINE = "#ff8a6a"
C_GRID = (0.85, 0.85, 0.92)


# ---------------------------------------------------------------------------
# Fig 1: anomaly g-sweep (continuous control)
# ---------------------------------------------------------------------------

def fig_anomaly_gsweep():
    # From awv_g_sweep_tuna9_*.json
    cands = sorted(DATA_DIR.glob("awv_g_sweep_tuna9_*.json"))
    if not cands:
        print("  [skip] no anomaly g-sweep data"); return
    data = json.load(open(cands[-1]))
    g_vals = data["metadata"]["g_values"]
    theory_pi0 = [data["theory"][f"{g}"]["pi0_observed"] for g in g_vals]
    measured_pi0 = []
    measured_err = []
    for g in g_vals:
        m = data["measured"][f"{g}"]
        measured_pi0.append(m["wv_pi0"])
        measured_err.append(m["wv_sigma"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(0, color='#cccccc', linewidth=1, linestyle='--', zorder=1)
    ax.axhline(-1, color='#cccccc', linewidth=0.6, linestyle=':', zorder=1)
    ax.fill_between([0, 0.6], -2, 0, color='#76e09d', alpha=0.07,
                    label='Negative-probability region')
    ax.plot(g_vals, theory_pi0, 'o-', color=C_THEORY, linewidth=2.4,
            markersize=8, label=r'Theory $\langle\Pi_0\rangle_w$',
            markeredgecolor='white', markeredgewidth=1.2, zorder=4)
    ax.errorbar(g_vals, measured_pi0, yerr=measured_err,
                fmt='s', color=C_TUNA9, markersize=8, capsize=4,
                markeredgecolor='white', markeredgewidth=1.2,
                label='Tuna-9 measured', zorder=5)
    ax.set_xlabel(r'Weak-coupling strength $g$')
    ax.set_ylabel(r'$\langle\Pi_0\rangle_w$  (negative probability if $<0$)')
    ax.set_title('Continuous control of anomalous weak value (Tuna-9)',
                 fontweight='bold', pad=12)
    ax.set_xlim(0, 0.55)
    ax.set_ylim(-1.05, 0.4)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='lower right', framealpha=0.95)
    fig.tight_layout()
    out = OUT_DIR / "fig_v2_anomaly_gsweep.png"
    fig.savefig(out); plt.close(fig)
    print(f"  ✓ {out.name}")


# ---------------------------------------------------------------------------
# Fig 2: anomaly decay vs past–future buffer (plain vs DD)
# ---------------------------------------------------------------------------

def fig_anomaly_decay():
    cands = sorted(DATA_DIR.glob("awv_dd_decay_tuna9_*.json"))
    if not cands:
        print("  [skip] no DD decay data"); return
    data = json.load(open(cands[-1]))
    wait_list = data["metadata"]["wait_ns_list"]
    plain = [(t, data["results"]["plain"][str(t)]) for t in wait_list
             if data["results"]["plain"][str(t)].get("status") == "ok"]
    dd = [(t, data["results"]["dd"][str(t)]) for t in wait_list
          if data["results"]["dd"][str(t)].get("status") == "ok"]

    p_t = [t for t, r in plain]
    p_v = [r["wv_pi0"] for t, r in plain]
    p_e = [r["wv_sigma"] for t, r in plain]
    d_t = [t for t, r in dd]
    d_v = [r["wv_pi0"] for t, r in dd]
    d_e = [r["wv_sigma"] for t, r in dd]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.axhline(0, color='#aaaaaa', linewidth=0.8, linestyle='--', zorder=1)
    ax.fill_between([0, 600], -1, 0, color='#76e09d', alpha=0.07)
    ax.errorbar(p_t, p_v, yerr=p_e, fmt='s-', color=C_TUNA9, linewidth=2,
                markersize=7, capsize=3, label='Plain idle')
    ax.errorbar(d_t, d_v, yerr=d_e, fmt='o-', color=C_GARNET, linewidth=2.4,
                markersize=8, capsize=3, label='X-Y-X-Y dynamical decoupling')
    ax.set_xlabel('Past-future buffer  $\\Delta t$  (ns)')
    ax.set_ylabel(r'$\langle\Pi_0\rangle_w$')
    ax.set_title('Anomaly coherence: plain vs DD-protected',
                 fontweight='bold', pad=12)
    ax.set_xlim(-20, 540)
    ax.set_ylim(-0.4, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='lower right', framealpha=0.95)
    # Annotate T_anomaly values
    ax.annotate('$T_{anom}\\approx101$ ns\n(plain)',
                xy=(101, -0.15), xytext=(170, -0.32),
                fontsize=10, color=C_TUNA9, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_TUNA9, lw=1.3))
    ax.annotate('$T_{anom}\\approx500$ ns\n(DD, 5$\\times$ extension)',
                xy=(500, 0.44), xytext=(330, 0.85),
                fontsize=10, color=C_GARNET, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_GARNET, lw=1.3))
    fig.tight_layout()
    out = OUT_DIR / "fig_v2_anomaly_decay.png"
    fig.savefig(out); plt.close(fig)
    print(f"  ✓ {out.name}")


# ---------------------------------------------------------------------------
# Fig 3: cross-platform F_anomaly bar chart
# ---------------------------------------------------------------------------

def fig_cross_platform():
    # All from single-anomaly-demo runs (g=0.30, 8192 shots)
    # F_anomaly = pointer / pointer_theory(0.30) = pointer / 0.913
    platforms = ["Tuna-9", "Garnet", "Sirius", "Emerald"]
    pi0 = [-0.316, -0.430, -0.355, -0.352]
    err = [0.031, 0.029, 0.032, 0.030]
    F = [0.814, 0.884, 0.838, 0.836]
    colors = [C_TUNA9, C_GARNET, C_SIRIUS, C_EMERALD]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(platforms))
    bars = ax1.bar(x, pi0, color=colors, yerr=err, capsize=5,
                   edgecolor='white', linewidth=0.7, alpha=0.9)
    ax1.axhline(0, color='#888', linewidth=0.8, linestyle='--')
    ax1.fill_between([-0.5, 3.5], -0.5, 0, color='#76e09d', alpha=0.07)
    ax1.set_xticks(x); ax1.set_xticklabels(platforms)
    ax1.set_ylabel(r'$\langle\Pi_0\rangle_w$')
    ax1.set_title(r'Negative weak value across 4 platforms',
                  fontweight='bold', pad=10)
    ax1.set_ylim(-0.55, 0.05)
    ax1.grid(True, alpha=0.25, axis='y')
    for i, v in enumerate(pi0):
        ax1.text(i, v - 0.04, f'{v:+.3f}', ha='center', va='top',
                 fontsize=9, fontweight='bold', color='#222')

    ax2.bar(x, F, color=colors, edgecolor='white', linewidth=0.7, alpha=0.9)
    for i, v in enumerate(F):
        ax2.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom',
                 fontsize=10, fontweight='bold')
    ax2.set_xticks(x); ax2.set_xticklabels(platforms)
    ax2.set_ylabel(r'$F_{\mathrm{anomaly}}$ (process fidelity)')
    ax2.set_title(r'$F_{\mathrm{anomaly}}$: vendor-neutral benchmark',
                  fontweight='bold', pad=10)
    ax2.set_ylim(0.7, 0.95)
    ax2.grid(True, alpha=0.25, axis='y')
    fig.tight_layout()
    out = OUT_DIR / "fig_v2_cross_platform.png"
    fig.savefig(out); plt.close(fig)
    print(f"  ✓ {out.name}")


# ---------------------------------------------------------------------------
# Fig 4: T-17 pair shopping
# ---------------------------------------------------------------------------

def fig_t17_pair_shopping():
    cands = sorted(DATA_DIR.glob("awv_t17_pairsweep_*.json"))
    if not cands:
        # fallback to hard-coded
        pairs = ["q0–q1\n(edge–edge)", "q4–q7\n(hub–hub)", "q11–q14\n(hub–edge)"]
        F = [0.629, 0.715, 0.492]
        pi0 = [-0.017, -0.156, +0.205]
        sigma = [0.039, 0.033, 0.036]
    else:
        d = json.load(open(cands[-1]))
        pair_keys = list(d.keys())
        pairs = [k.replace("_", "→").replace("A→", "edge–edge\n").replace(
                 "B→", "hub–hub\n").replace("C→", "hub–edge\n") for k in pair_keys]
        F = [d[k]["F_anomaly"] for k in pair_keys if d[k].get("status") == "ok"]
        pi0 = [d[k]["wv_pi0"] for k in pair_keys if d[k].get("status") == "ok"]
        sigma = [d[k]["wv_sigma"] for k in pair_keys if d[k].get("status") == "ok"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(pairs))
    colors = [C_TUNA9 if p > 0 else C_GARNET if abs(p) > 0.1 else '#888888'
              for p in pi0]
    ax.bar(x, pi0, yerr=sigma, color=colors, capsize=5,
           edgecolor='white', linewidth=0.7, alpha=0.9)
    ax.axhline(0, color='#888', linewidth=0.8, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(["q0–q1\nedge", "q4–q7\nhub", "q11–q14\nedge"]
                       if len(pairs) == 3 else pairs)
    ax.set_ylabel(r'$\langle\Pi_0\rangle_w$  (Tuna-17)')
    ax.set_title(r'Tuna-17 pair shopping: $\Delta F_{\mathrm{anomaly}}=0.22$ within chip',
                 fontweight='bold', pad=10)
    ax.grid(True, alpha=0.25, axis='y')
    for i, (v, e) in enumerate(zip(pi0, sigma)):
        offset = 0.04 if v < 0 else -0.05
        ax.text(i, v + offset, f'{v:+.3f}\n$F$={F[i]:.2f}',
                ha='center', va='top' if v > 0 else 'bottom',
                fontsize=9, fontweight='bold', color='#222')
    ax.set_ylim(min(pi0) - 0.15, max(pi0) + 0.15)
    fig.tight_layout()
    out = OUT_DIR / "fig_v2_t17_pair_shopping.png"
    fig.savefig(out); plt.close(fig)
    print(f"  ✓ {out.name}")


# ---------------------------------------------------------------------------
# Fig 5: chemistry sprint (4-molecule dissociation curves)
# ---------------------------------------------------------------------------

def fig_chemistry_sprint():
    # H2 from earlier sweep, plus LiH/BeH2/H2O from chemistry_nq script
    H2_DATA = [
        # R, exact, baseline, v2
        (0.50, -2.344, -2.229, -2.325),
        (0.74, -1.915, -1.854, -1.895),
        (1.00, -1.785, -1.732, -1.769),
        (1.50, -1.281, -1.251, -1.270),
        (2.00, -1.116, -1.078, -1.100),
        (2.50, -1.120, -1.071, -1.095),
        (3.00, -1.121, -1.068, -1.094),
    ]
    # LiH from h2_diss_sweep / LiH script  (using hardcoded from log)
    LIH_DATA = [
        (0.80, -8.3892, -8.3784, -8.3914),
        (1.20, -8.4062, -8.3963, -8.4105),
        (1.55, -8.4141, -8.4042, -8.4177),
        (2.00, -8.3798, -8.3696, -8.3806),
        (2.50, -8.3434, -8.3321, -8.3415),
    ]
    BEH2_DATA = [
        (0.80, -16.4920, -16.4639, -16.4912),
        (1.10, -16.5992, -16.5691, -16.5998),
        (1.33, -16.6414, -16.6115, -16.6423),
        (1.70, -16.5480, -16.5188, -16.5436),
        (2.20, -16.4307, -16.4052, -16.4237),
    ]
    H2O_DATA = [
        (0.60, -77.3045, -77.2646, -77.3022),
        (0.80, -77.4964, -77.4530, -77.4975),
        (0.96, -77.5660, -77.5232, -77.5692),
        (1.30, -77.3237, -77.2872, -77.3217),
        (1.70, -77.1104, -77.0767, -77.1008),
    ]

    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (mol, data, qubits, mean_b, mean_v2, ratio) in zip(
        axs.flat,
        [
            ("H$_2$", H2_DATA, 2, 57, 19, 3.0),
            ("LiH", LIH_DATA, 4, 10.4, 2.6, 4.0),
            ("BeH$_2$", BEH2_DATA, 6, 28.5, 2.8, 10.3),
            ("H$_2$O", H2O_DATA, 8, 39.3, 3.7, 10.7),
        ]
    ):
        R = [d[0] for d in data]
        exact = [d[1] for d in data]
        base = [d[2] for d in data]
        v2 = [d[3] for d in data]
        ax.plot(R, exact, 'o-', color=C_THEORY, lw=2.4, ms=7,
                markeredgecolor='white', markeredgewidth=1.2, label='Classical FCI')
        ax.plot(R, base, 's--', color=C_BASELINE, lw=1.8, ms=6, alpha=0.85,
                label=f'Tuna-9 baseline (mean |err|={mean_b:.0f} mHa)')
        ax.plot(R, v2, '^-', color=C_V2, lw=2.4, ms=8,
                markeredgecolor='white', markeredgewidth=1.0,
                label=f'Tuna-9 + ABR v2 (mean |err|={mean_v2:.1f} mHa, {ratio:.1f}$\\times$)')
        ax.set_xlabel('R  (Å)')
        ax.set_ylabel('Energy  (Ha)')
        ax.set_title(f'{mol}  ({qubits}-qubit ansatz)',
                     fontweight='bold', pad=8)
        ax.grid(True, alpha=0.25)
        ax.legend(loc='upper right' if mol == "H$_2$O" else 'lower right',
                  framealpha=0.95, fontsize=9)
    fig.suptitle('Chemistry vertical sprint on Tuna-9: ABR v2 across 2–8 qubits',
                 fontsize=14, fontweight='bold', y=1.005)
    fig.tight_layout()
    out = OUT_DIR / "fig_v2_chemistry_sprint.png"
    fig.savefig(out); plt.close(fig)
    print(f"  ✓ {out.name}")


# ---------------------------------------------------------------------------
# Fig 6: ABR mitigation boundary (regime map)
# ---------------------------------------------------------------------------

def fig_abr_boundary():
    fig, ax = plt.subplots(figsize=(9, 5))
    regimes = [
        ("Readout\ndominated\n(≤4 CZ)", 0, "v2 PASS\n3–10× improvement", "#76e09d"),
        ("Mixed regime\n(4–12 CZ)", 1, "v3 marginal\n+9% improvement", "#ffd84a"),
        ("Gate-dominated\n(12–20 CZ)", 2, "v2/v3 fail\nrequires ZNE/PEC", "#ffa44a"),
        ("Deep / strongly\nentangled (>20 CZ)", 3, "Hardware-limit\nlogical qubits needed", "#ff8a6a"),
    ]
    for i, (label, x, status, color) in enumerate(regimes):
        ax.barh(0, 1, left=x, height=0.6, color=color, alpha=0.7,
                edgecolor='white', linewidth=2)
        ax.text(x + 0.5, 0, status, ha='center', va='center',
                fontsize=10, fontweight='bold', color='#1a1a1a')
        ax.text(x + 0.5, -0.5, label, ha='center', va='center',
                fontsize=10, color='#222')

    ax.set_xlim(-0.05, 4.05)
    ax.set_ylim(-1.0, 1.0)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title("ABR mitigation regime map (CZ-depth dependence)",
                 fontweight='bold', pad=15)
    # Examples annotation
    ax.annotate('H₂ (1 CZ) ✓', xy=(0.5, 0.5), xytext=(0.5, 0.85),
                fontsize=9, ha='center', color='#0a3a16')
    ax.annotate('LiH (3 CZ) ✓', xy=(0.5, 0.45), xytext=(0.5, 0.75),
                fontsize=9, ha='center', color='#0a3a16')
    ax.annotate('BeH₂ (5 CZ) ✓\nH₂O (7 CZ) ✓', xy=(1.0, 0.45), xytext=(1.0, 0.75),
                fontsize=9, ha='center', color='#0a3a16')
    ax.annotate('QAOA p=2 ✗', xy=(1.5, 0.5), xytext=(1.5, 0.85),
                fontsize=9, ha='center', color='#1f2937')
    ax.annotate('QAOA p=3 ✗', xy=(2.5, 0.5), xytext=(2.5, 0.85),
                fontsize=9, ha='center', color='#1f2937')
    fig.tight_layout()
    out = OUT_DIR / "fig_v2_abr_boundary.png"
    fig.savefig(out); plt.close(fig)
    print(f"  ✓ {out.name}")


def main():
    print(f"Output dir: {OUT_DIR}")
    fig_anomaly_gsweep()
    fig_anomaly_decay()
    fig_cross_platform()
    fig_t17_pair_shopping()
    fig_chemistry_sprint()
    fig_abr_boundary()
    print("Done.")


if __name__ == "__main__":
    main()
