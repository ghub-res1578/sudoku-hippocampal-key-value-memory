"""
Regenerates Figure 2 (graceful degradation under query noise, with vs. without the
conflict-mask cleanup pass) from hippocampus_drive_readout_feedback_sweep.csv -- the CSV
written by hippocampus_drive_readout.py's own run_feedback_comparison()/main().
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv('hippocampus_drive_readout_feedback_sweep.csv')

fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=150)
ax.plot(df.noise_percent, df.digit_cell_acc_no_feedback, 'o-', color='#c0603a',
        label='without conflict-mask cleanup', linewidth=1.8, markersize=5)
ax.plot(df.noise_percent, df.digit_cell_acc_with_feedback, 'o-', color='#2f7350',
        label='with conflict-mask cleanup', linewidth=1.8, markersize=5)
ax.axhline(1 / 9, color='gray', linestyle=':', linewidth=1, label='chance (1/9)')
ax.set_xlabel('Query noise (% of engram bits flipped)')
ax.set_ylabel('Digit-cell retrieval accuracy')
ax.set_title('Graceful degradation under query noise (M=200)')
ax.grid(alpha=0.25)
ax.legend(loc='center left', fontsize=9)
fig.tight_layout()
fig.savefig('hippocampus_feedback_noise_clean.png', dpi=150, bbox_inches='tight')
print('Saved hippocampus_feedback_noise_clean.png')
