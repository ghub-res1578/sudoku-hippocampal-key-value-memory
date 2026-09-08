"""
Regenerates the critical-slowing-down figure using the CORRECT, manually-windowed fit
(decay branch restricted to clue_fraction in [0.21, 0.37], excluding both the plateau and
the floor) -- the figure previously used (critical_slowing_down_extended.png) was produced
by analyze_and_plot()'s automatic decay-branch heuristic, which sweeps in floor-contaminated
points and gives a bogus, boundary-pinned p_c=0.85 with the wrong exponent sign in the plot
label (only the print statements were patched earlier, not the plotting code). This
reproduces p_c=0.395, nu=1.32+-0.08, R^2=0.978, matching the paper's text, with a cleaner
layout.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv('critical_slowing_down_extended.csv')
summary = df.groupby('clue_fraction').agg(
    mean_iter=('iterations', 'mean'), std_iter=('iterations', 'std'),
    acc=('exact_acc', 'mean'),
).reset_index()

PLATEAU_MAX = 0.21
DECAY_MIN, DECAY_MAX = 0.21, 0.37


def fit_power_law(sub, p_c_range):
    best = None
    for p_c in p_c_range:
        dist = p_c - sub.clue_fraction
        valid = dist > 0
        if valid.sum() < 4:
            continue
        logx = np.log10(dist[valid]); logy = np.log10(sub.mean_iter[valid])
        slope, intercept, r, p, se = stats.linregress(logx, logy)
        if best is None or r ** 2 > best[3]:
            best = (p_c, slope, se, r ** 2, intercept)
    return best


window = summary[(summary.clue_fraction >= DECAY_MIN) & (summary.clue_fraction <= DECAY_MAX)]
p_c, slope, se, r2, intercept = fit_power_law(window, np.arange(DECAY_MAX + 0.005, DECAY_MAX + 0.30, 0.005))
print(f'Corrected fit: p_c={p_c:.3f}, nu={slope:.3f}+-{se:.3f}, R2={r2:.4f}')

plateau = summary[summary.clue_fraction <= PLATEAU_MAX]
plateau_mean, plateau_std = plateau.mean_iter.mean(), plateau.mean_iter.std()
print(f'Plateau: {plateau_mean:.1f} +- {plateau_std:.1f}')

PLATEAU_COLOR = '#4f7396'
DECAY_COLOR = '#b8631f'
FLOOR_COLOR = '#5a8f69'
ACC_COLOR = '#2f7350'

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5), dpi=150)

# --- panel 1: iterations vs clue fraction, three regimes colored ---
ax = axes[0]
plateau_pts = summary[summary.clue_fraction <= DECAY_MIN]
decay_pts = summary[(summary.clue_fraction > DECAY_MIN) & (summary.clue_fraction <= DECAY_MAX)]
floor_pts = summary[summary.clue_fraction > DECAY_MAX]

for pts, color, label in [(plateau_pts, PLATEAU_COLOR, 'sub-critical (plateau)'),
                           (decay_pts, DECAY_COLOR, 'power-law decay'),
                           (floor_pts, FLOOR_COLOR, 'floor (solved)')]:
    ax.errorbar(pts.clue_fraction, pts.mean_iter, yerr=pts.std_iter, fmt='o-',
                color=color, capsize=3, markersize=5, linewidth=1.8, label=label)
ax.axvline(p_c, color=DECAY_COLOR, linestyle=':', linewidth=1.5, alpha=0.8)
ax.text(p_c, 8, f'  $p_c$={p_c:.3f}', color=DECAY_COLOR, fontsize=9, va='bottom')
ax.set_xlabel('Clue fraction'); ax.set_ylabel('Mean iterations to converge')
ax.set_title(f'M=500, K=400: iteration count vs. clue fraction')
ax.grid(alpha=0.25); ax.legend(fontsize=8.5, loc='upper right')
ax.set_ylim(bottom=-5)

# --- panel 2: accuracy (order parameter) ---
ax = axes[1]
ax.plot(summary.clue_fraction, summary.acc, 'o-', color=ACC_COLOR, markersize=5, linewidth=1.8)
ax.axvline(p_c, color=DECAY_COLOR, linestyle=':', linewidth=1.5, alpha=0.8)
ax.set_xlabel('Clue fraction'); ax.set_ylabel('Exact recovery accuracy')
ax.set_title('Order parameter vs. clue fraction')
ax.grid(alpha=0.25); ax.set_ylim(-0.03, 1.05)

# --- panel 3: log-log power-law fit, decay branch only ---
ax = axes[2]
dist = p_c - window.clue_fraction
valid = dist > 0
ax.plot(dist[valid], window.mean_iter[valid], 'o', color=DECAY_COLOR, markersize=7,
        label=f'decay branch ($p\\in[{DECAY_MIN},{DECAY_MAX}]$)', zorder=3)
xs = np.linspace(dist[valid].min(), dist[valid].max(), 50)
ax.plot(xs, 10 ** intercept * xs ** slope, '--', color=DECAY_COLOR, alpha=0.6, linewidth=1.5,
        label=f'fit: $\\nu$={slope:.2f}$\\pm${se:.2f}, $R^2$={r2:.3f}', zorder=2)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('$p_c - p$ (log scale)'); ax.set_ylabel('Mean iterations (log scale)')
ax.set_title('Power-law check: decay branch only')
ax.grid(alpha=0.25, which='both'); ax.legend(fontsize=8.5, loc='lower right')

fig.suptitle('')
fig.tight_layout()
fig.savefig('critical_slowing_down_clean.png', dpi=150, bbox_inches='tight')
print('Saved critical_slowing_down_clean.png')
