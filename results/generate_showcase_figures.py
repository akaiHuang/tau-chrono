#!/usr/bin/env python3
"""Generate showcase figures from real T-9 data."""

import os
import json

os.environ['MPLCONFIGDIR'] = '/tmp/mpl'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 13, 'axes.labelsize': 14, 'axes.titlesize': 15,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11,
    'figure.dpi': 200, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
    'font.family': 'serif', 'mathtext.fontset': 'cm',
})

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

# Load data
with open(os.path.join(RESULTS_DIR, 'showcase_tuna9_20260319_200118.json')) as f:
    showcase = json.load(f)

with open(os.path.join(RESULTS_DIR, 'deep_ceiling_tuna9_20260319_214908.json')) as f:
    ceiling = json.load(f)


# =====================================================================
# Figure A: Cost Savings (BV)
# =====================================================================
def fig_a_cost():
    data = showcase['experiment_a_cost']
    n_reps = [r['n_rep'] for r in data]
    naive_cost = [r['naive_cost_shots'] for r in data]
    chrono_cost = [r['chrono_cost_shots'] for r in data]
    p_correct = [r['p_correct'] for r in data]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(n_reps))
    width = 0.32

    bars_n = ax1.bar(x - width/2, [c/1000 for c in naive_cost], width,
                     color='#EF4444', alpha=0.85, label='Without $\\tau$-chrono',
                     edgecolor='white', linewidth=0.5)
    bars_c = ax1.bar(x + width/2, [c/1000 for c in chrono_cost], width,
                     color='#3B82F6', alpha=0.85, label='With $\\tau$-chrono',
                     edgecolor='white', linewidth=0.5)

    ax1.set_ylabel('QPU shots needed (thousands)')
    ax1.set_xlabel('Oracle repetitions ($n_{\\rm rep}$)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(n) for n in n_reps])

    # Annotate savings on the bar where it matters
    for i, (nc, cc) in enumerate(zip(naive_cost, chrono_cost)):
        if nc > cc:
            saving = (1 - cc/nc) * 100
            ax1.annotate(f'saves {saving:.0f}%',
                         xy=(x[i], nc/1000), xytext=(x[i], nc/1000 + 0.8),
                         ha='center', fontsize=10, color='#DC2626', fontweight='bold')

    # Secondary axis: P_correct
    ax2 = ax1.twinx()
    ax2.plot(x, p_correct, 'D-', color='#059669', linewidth=2.0,
             markersize=7, markeredgecolor='white', markeredgewidth=1,
             label='$P_{\\rm correct}$ (measured)', zorder=5)
    ax2.set_ylabel('$P_{\\rm correct}$ (measured on T-9)', color='#059669')
    ax2.tick_params(axis='y', labelcolor='#059669')
    ax2.set_ylim(0, 1.1)

    for i, pc in enumerate(p_correct):
        ax2.annotate(f'{pc:.0%}', (x[i], pc), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9, color='#059669',
                     fontweight='bold')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
               framealpha=0.95, edgecolor='#cccccc')

    # Total savings annotation
    total_n = sum(naive_cost)
    total_c = sum(chrono_cost)
    ax1.text(0.97, 0.55, f'Total savings: {(1-total_c/total_n)*100:.0f}%\n'
             f'({total_n:,} → {total_c:,} shots)',
             transform=ax1.transAxes, fontsize=10, ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#F0F9FF',
                       edgecolor='#3B82F6', alpha=0.95))

    ax1.set_title('Experiment A: $\\tau$-chrono saves QPU time\n'
                  'Bernstein-Vazirani on Tuna-9 (real hardware)',
                  fontweight='bold', pad=12)
    ax1.grid(True, axis='y', alpha=0.25, linestyle='-')
    ax1.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'fig_expA_cost_savings.png'))
    plt.close(fig)
    print('[OK] fig_expA_cost_savings.png')


# =====================================================================
# Figure B: Depth Ceiling
# =====================================================================
def fig_b_ceiling():
    data = ceiling['results']
    gates = [r['n_gates_total'] for r in data]
    actual = [r['p_correct'] for r in data]
    naive = [r['f_naive'] for r in data]
    chrono = [r['f_tauchrono'] for r in data]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    color_actual = '#059669'
    color_chrono = '#2563EB'
    color_naive = '#DC2626'

    ax.plot(gates, actual, 'o-', color=color_actual, linewidth=2.5,
            markersize=8, markeredgecolor='white', markeredgewidth=1.2,
            label='Actual fidelity (measured on T-9)', zorder=5)
    ax.plot(gates, chrono, 's--', color=color_chrono, linewidth=2.0,
            markersize=6, alpha=0.9,
            label='$\\tau$-chrono prediction', zorder=4)
    ax.plot(gates, naive, '^--', color=color_naive, linewidth=2.0,
            markersize=6, alpha=0.9,
            label='Naive prediction', zorder=3)

    # Threshold line
    ax.axhline(y=0.5, color='#374151', linestyle=':', linewidth=1.5, alpha=0.6)
    ax.text(305, 0.52, '$F = 0.5$ threshold', fontsize=10, color='#374151',
            ha='right', va='bottom')

    # Shade the "tau-chrono saves" region
    # Naive < 0.5 but actual > 0.5: between gates 30 and 50
    save_gates = [g for g, n, a in zip(gates, naive, actual) if n < 0.5 and a > 0.5]
    if save_gates:
        ax.axvspan(min(save_gates) - 5, max(save_gates) + 5,
                   alpha=0.08, color=color_chrono, zorder=0)
        mid = (min(save_gates) + max(save_gates)) / 2
        ax.annotate('$\\tau$-chrono\nsaves these\ncircuits',
                    xy=(mid, 0.7), fontsize=10, ha='center',
                    color=color_chrono, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor=color_chrono, alpha=0.9))

    # Annotate naive STOP point
    ax.annotate('Naive says\nSTOP here',
                xy=(30, naive[2]), xytext=(60, 0.15),
                fontsize=9.5, color=color_naive, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=color_naive, lw=1.2))

    # Annotate key data points
    ax.annotate(f'F={actual[3]:.2f}\n(50 gates)',
                xy=(50, actual[3]), xytext=(80, 0.80),
                fontsize=9, color=color_actual,
                arrowprops=dict(arrowstyle='->', color=color_actual, lw=1.0),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor=color_actual, alpha=0.9))

    ax.set_xlabel('Total circuit gates')
    ax.set_ylabel('Fidelity $F$ = $P(|000\\rangle)$')
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlim(0, 320)

    ax.legend(loc='upper right', framealpha=0.95, edgecolor='#cccccc')

    # Summary box
    summary = ceiling['summary']
    ax.text(0.03, 0.03,
            f"Naive max: {summary['max_layers_naive']} layers ({summary['max_layers_naive']*6} gates)\n"
            f"$\\tau$-chrono max: {summary['max_layers_actual']} layers ({summary['max_layers_actual']*6} gates)\n"
            f"Depth extension: {summary['max_layers_actual']/max(summary['max_layers_naive'],1):.1f}x\n"
            f"Circuits saved: {summary['circuits_saved']}",
            transform=ax.transAxes, fontsize=10, va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F0FDF4',
                      edgecolor='#059669', alpha=0.95))

    ax.set_title('Experiment B: $\\tau$-chrono extends usable circuit depth\n'
                 '3-qubit entangling mirror circuit on Tuna-9 (real hardware)',
                 fontweight='bold', pad=12)
    ax.grid(True, alpha=0.25, linestyle='-')
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'fig_expB_depth_ceiling.png'))
    plt.close(fig)
    print('[OK] fig_expB_depth_ceiling.png')


if __name__ == '__main__':
    fig_a_cost()
    fig_b_ceiling()
    print('Done.')
