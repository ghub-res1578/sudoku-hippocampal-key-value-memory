"""
Final consolidated comparison figure for the cross-modal heteroassociation section: classical/
covariance/pseudo-inverse Hopfield, modern Hopfield (Ramsauer 2020), SDM (Kanerva 1988), DAM
(Krotov-Hopfield 2016), our network, the Vector-HaSH-style modular scaffold (with recurrent
attractor correction), and the raw-pixel k-NN baseline -- nine methods total, all on the
identical M-sweep (sigma=0.3, M in {100,200,400,800}) and noise-sweep (M=50) conditions.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df_m = pd.read_csv('final_full_M_sweep_with_knn.csv')
df_n = pd.read_csv('full_hetero_comparison_noise_sweep.csv')

METHODS = ['outer', 'cov', 'pinv', 'modern', 'sdm', 'dam', 'ours', 'vectorhash']
LABELS = {'outer': 'Classical Hopfield', 'cov': 'Covariance-rule Hopfield',
          'pinv': 'Pseudo-inverse Hopfield', 'modern': 'Modern Hopfield (Ramsauer 2020)',
          'sdm': 'Sparse Distributed Memory (Kanerva)', 'dam': 'Dense Assoc. Memory (Krotov-Hopfield)',
          'ours': 'Our network', 'vectorhash': 'Vector-HaSH-style modular scaffold'}
COLORS = {'outer': 'tab:red', 'cov': 'tab:purple', 'pinv': 'tab:orange', 'modern': 'tab:blue',
          'sdm': 'tab:brown', 'dam': 'tab:pink', 'ours': 'tab:green', 'vectorhash': 'tab:cyan'}

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for key in METHODS:
    axes[0].plot(df_m.M, df_m[f'{key}_exact'], 'o-', color=COLORS[key], label=LABELS[key])
axes[0].plot(df_m.M, df_m.knn_exact, 's--', color='black', label='Raw-pixel k-NN')
axes[0].set_xlabel('M (number of stored images)'); axes[0].set_ylabel('Exact recovery rate')
axes[0].set_title('Exact recovery vs. M (sigma=0.3)')
axes[0].legend(fontsize=7.5); axes[0].grid(alpha=0.3)

for key in METHODS:
    axes[1].plot(df_n.sigma, df_n[f'{key}_exact'], 'o-', color=COLORS[key], label=LABELS[key])
axes[1].set_xlabel('Gaussian pixel noise sigma'); axes[1].set_ylabel('Exact recovery rate')
axes[1].set_title('Exact recovery vs. noise, fixed M=50')
axes[1].legend(fontsize=7.5); axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig('final_hetero_comparison_full.png', dpi=150)
print('Saved final_hetero_comparison_full.png')

# one-shot vs. recurrent ablation for the modular scaffold specifically
df_vh_m = pd.read_csv('vectorhash_modular_scaffold_M_sweep.csv')
fig2, ax = plt.subplots(figsize=(7, 5.2))
ax.plot(df_vh_m.M, df_vh_m.vectorhash_oneshot_exact, 's--', color='gray',
        label='One-shot (independent per-module argmax)')
ax.plot(df_vh_m.M, df_vh_m.vectorhash_exact, 'o-', color='tab:cyan',
        label='Recurrent attractor correction (beta=100)')
ax.set_xlabel('M (number of stored images)'); ax.set_ylabel('Exact recovery rate')
ax.set_title('Modular scaffold: effect of recurrent cross-module correction (sigma=0.3)')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig('vectorhash_oneshot_vs_recurrent.png', dpi=150)
print('Saved vectorhash_oneshot_vs_recurrent.png')
