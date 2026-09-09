# M, K, H per figure

H = hippocampal/engram dimension, K = active units per engram, M = number of stored items.
Values pulled directly from each figure's generating script (not from memory), in the
paper's figure order. Where a script imports H/K from `hippocampus_drive_readout.py`
without overriding them, that means the project-wide default (H=1000, K=150).

| # | Figure | Source script | M | K | H | Notes |
|---|--------|---------------|---|---|---|-------|
| 1 | `capacity_regression_vs_attention_capacity.png` | `capacity_regression_vs_attention.py` | swept: 10,20,50,100,200,300,500,700,1000,1300,1600,2000 | 150 | 1000 | M is the figure's x-axis |
| 2 | `hippocampus_feedback_noise_zoomed.png` | `hippocampus_feedback_noise_zoomed.py` | 200 | 150 | 1000 | imports M/H/K unchanged from `hippocampus_drive_readout.py` |
| 3 | `sudoku_partial_grid_completion.png` | `sudoku_partial_grid_completion.py` | 2000 | 150 | 1000 | clue fraction is the x-axis |
| 4 | `critical_slowing_down_clean.png` | `critical_slowing_down_clean_plot.py` (data from `critical_slowing_down_extended.py`) | 500 | 400 | 1000 | K=400 chosen as the sharpest crossover in an earlier M-K grid search |
| 5 | `spiking_real_sudoku_sweetspot_raster.png` | `spiking_real_sudoku_sweetspot.py` | n/a | n/a | n/a | single-grid spiking simulation — no engram/M-K-H machinery involved, N=81 cells only |
| 6 | `spiking_emergent_coincidence_full.png` | `spiking_emergent_coincidence.py` | 200 | 150 | 1000 | imports H/K unchanged from `hippocampus_drive_readout.py` |
| 7 | `spiking_drive_noise_clean.png` | `spiking_drive_noise_robustness.py` | 200 | 150 | 1000 | clue fraction fixed at 0.30 |
| 8 | `heteroassoc_vectorhash_best_examples.png` | `heteroassoc_vectorhash_best_examples.py` | 200 | 150 | 1000 | 3 datasets (MNIST/Fashion-MNIST/CIFAR-100), 1 example/dataset |
| 9 | `heteroassoc_final_consolidation.png` | `heteroassoc_final_consolidation.py` | top-left/top-right: 200 (rigor sweep) and 200–800 (capacity sweep, see below); bottom row: 500 (ridge sensitivity) | 150 | 1000 | 4-panel figure, M differs by panel — see breakdown below |
| 10 | `hopfield_matched_noise_rigorous.png` | `hopfield_matched_noise_rigorous.py` | left panel: 100,200,400,800 (M-sweep); right panel: 50 (noise-sweep) | 150 | 1000 | 2-panel figure, M differs by panel |

## Figure 9 panel breakdown (`heteroassoc_final_consolidation.png`)
- Top-left (pipeline vs. k-NN baseline): M=200
- Top-right (capacity cliff vs. M): M swept 200, 300, 400, 500, 650, 800
- Bottom-left (Gram-matrix conditioning vs. M): same M sweep as top-right
- Bottom-right (ridge sensitivity): M=500 fixed, ridge λ swept

## Figure 10 panel breakdown (`hopfield_matched_noise_rigorous.png`)
- Left (exact recovery vs. M): M swept 100, 200, 400, 800, at fixed σ=0.3
- Right (exact recovery vs. noise): M=50 fixed, σ swept 0.0–1.0

H and K are 1000/150 in every figure that uses the engram architecture at all — never
varied across figures. The only figure-specific H/K/M-independent case is Figure 4
(K=400 instead of the default 150), and Figure 5 (no engram involved).
