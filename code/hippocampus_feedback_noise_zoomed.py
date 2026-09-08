"""
Finer-grained noise sweep restricted to the 35-55% transition window (1 percentage point
steps instead of the original's 2-5pp gaps), reusing the exact same engram setup as
hippocampus_drive_readout.py's main(). Produces a two-panel figure: full 0-100% range (left,
unchanged) + zoomed 35-55% range with the finer data (right).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '.')
from hippocampus_drive_readout import (
    SEED, M, generate_unique, relation, make_projection, make_engrams, drive_vectors,
    conflict_matrix, run_feedback_comparison,
)

ZOOM_NOISE_SWEEP = [round(x, 2) for x in np.arange(0.35, 0.551, 0.01)]  # 35% to 55%, step 1pp


def main():
    rng = np.random.default_rng(SEED)
    print(f'Reconstructing the same engram setup as hippocampus_drive_readout.py (M={M})...')
    sud = generate_unique(M, rng)
    C_all = np.asarray([relation(g).ravel() for g in sud], dtype=np.float32)
    W_SH = make_projection(rng)
    H_keys = make_engrams(C_all, W_SH)
    B_values = drive_vectors(sud)
    digits_true = sud.reshape(M, 81)
    conf = conflict_matrix()

    print(f'\nZoomed noise sweep: {len(ZOOM_NOISE_SWEEP)} points from '
          f'{ZOOM_NOISE_SWEEP[0]:.0%} to {ZOOM_NOISE_SWEEP[-1]:.0%}, 1pp steps, 100 trials/point')
    zoom_df = run_feedback_comparison(H_keys, B_values, W_SH, digits_true, conf,
                                       noise_levels=ZOOM_NOISE_SWEEP, n_trials=100)
    zoom_df.to_csv('hippocampus_feedback_noise_zoomed.csv', index=False)
    print('Saved hippocampus_feedback_noise_zoomed.csv')

    full_df = pd.read_csv('hippocampus_drive_readout_feedback_sweep.csv')

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=150)

    ax = axes[0]
    ax.plot(full_df.noise_percent, full_df.digit_cell_acc_no_feedback, 'o-', color='#c0603a',
            label='without cleanup', linewidth=1.8, markersize=5)
    ax.plot(full_df.noise_percent, full_df.digit_cell_acc_with_feedback, 'o-', color='#2f7350',
            label='with cleanup', linewidth=1.8, markersize=5)
    ax.axhline(1 / 9, color='gray', linestyle=':', linewidth=1, label='chance (1/9)')
    ax.axvspan(35, 55, color='gray', alpha=0.08, zorder=0)
    ax.set_xlabel('Query noise (% of engram bits flipped)')
    ax.set_ylabel('Digit-cell retrieval accuracy')
    ax.set_title('Full range (M=200)')
    ax.grid(alpha=0.25); ax.legend(loc='center left', fontsize=9)

    ax = axes[1]
    zoom_pct = zoom_df.noise_percent
    ax.plot(zoom_pct, zoom_df.digit_cell_acc_no_feedback, 'o-', color='#c0603a',
            label='without cleanup', linewidth=1.8, markersize=5)
    ax.plot(zoom_pct, zoom_df.digit_cell_acc_with_feedback, 'o-', color='#2f7350',
            label='with cleanup', linewidth=1.8, markersize=5)
    ax.axhline(1 / 9, color='gray', linestyle=':', linewidth=1, label='chance (1/9)')
    ax.set_xlabel('Query noise (% of engram bits flipped)')
    ax.set_ylabel('Digit-cell retrieval accuracy')
    ax.set_title('Zoomed: 35-55% (1pp steps, n=100/point)')
    ax.grid(alpha=0.25); ax.legend(loc='upper right', fontsize=9)

    fig.tight_layout()
    fig.savefig('hippocampus_feedback_noise_zoomed.png', dpi=150, bbox_inches='tight')
    print('Saved hippocampus_feedback_noise_zoomed.png')


if __name__ == '__main__':
    main()
