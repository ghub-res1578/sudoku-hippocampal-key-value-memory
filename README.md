# Combinatorial Coding, Key-Value Recall, and Emergent Constraint Satisfaction in a Hippocampal-Inspired Spiking Network

Code, data, and a full write-up (`paper/main.pdf`, `paper/main.tex`) for a hippocampal-style
key-value memory that stores solved Sudoku grids as sparse combinatorial engrams, retrieves
them through an attention-based readout, and — the paper's central result — shows that the
symbolic legality check the architecture relies on for error correction (Sudoku's
row/column/box constraint) can instead be realized as an emergent property of a spiking
network's phase-synchrony dynamics. The dynamics-based mechanism is statistically
indistinguishable from the hard-coded rule it replaces (paired Wilcoxon signed-rank test,
$p>0.05$ at every tested clue fraction) once routed through the same associative-memory
pipeline.

Read the paper first: `paper/main.pdf`. Everything below reproduces its tables and figures.

## Repository layout

```
paper/               main.tex + main.pdf + figures/ used in the paper
code/                all scripts, flat (each script writes its own .csv/.png into this
                      directory, and several scripts load .csv/.npy files also kept here --
                      keeping everything in one directory avoids relative-path issues)
requirements.txt
LICENSE
```

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Brian2 will JIT-compile C++ code the first time each network is built in a process; this is
slow (tens of seconds) once per process and cached thereafter.

## What's excluded, and why

Two things explored during this project are **deliberately not included**: a synaptic
plasticity (STDP) variant of the excitatory synapses, and a cross-modal (MNIST-to-Sudoku)
extension. The plasticity variant was tested rigorously (a paired, statistically powered
sweep) and found *not* to reliably outperform the reported non-plastic, fixed-weight
dynamics -- the fixed-weight mechanism already matches the hard-coded rule, and we judged
added mechanism without added, validated benefit to be the wrong trade. The cross-modal
extension is a separate, unrelated negative result and out of scope for this paper.

## Reproducing each result

All commands assume you're in `code/`. Each script prints a summary and writes CSVs/PNGs
that a corresponding `_plot`-style call at the bottom of the script (or the script itself)
turns into the paper's figures. Runtime notes below are on an 8-core desktop CPU.

### Section 3.1 -- Combinatorial code: capacity and robustness (Table 1)
```bash
python3 feedback_vs_M_sweep.py        # M = 50 .. 2000, noise-collapse threshold
python3 feedback_iterate_highM.py     # M up to 10,000 (needs feedback_iterate_highM_pool.npy,
                                       #   included; regenerate via generate_unique() if needed)
python3 feedback_vs_K_sweep.py        # engram size K = 10 .. 500
python3 feedback_K_finegrain.py       # finer K grid around the interesting region
python3 feedback_vs_density_sweep.py  # projection density 0.002 .. 1.0
```
Each writes a `*_collapse_thresholds.csv` -- the numbers in Table 1.
Runtime: a few minutes each; `feedback_iterate_highM.py` is the slowest (~10 min).

### Section 3.2 -- Key-value recall: attention vs. regression (Table 2, Fig. 1)
```bash
python3 capacity_regression_vs_attention.py
```
Writes `capacity_regression_vs_attention.csv` (raw) and
`capacity_regression_vs_attention_capacity_estimates.csv` (Table 2), plus the capacity-curve
PNG used in Figure 1. Runtime: ~15-20 min (sweeps $M\in\{10,\dots,2000\}$).

### Section 3.3 -- Partial-grid pattern completion (Table 3, Fig. 2)
```bash
python3 sudoku_partial_grid_completion.py
```
Writes `sudoku_partial_grid_completion.csv` and the accuracy-vs-clue-fraction figure.
`sudoku_partial_grid_MK_sweep.py`, `sudoku_partial_grid_MKH_sweep.py`, and
`sudoku_partial_grid_MK_iterations_sweep.py` reproduce the broader $M$-$K$-$H$ characterization
referenced in the text (not required for the headline table). Runtime: ~5-10 min.

### Section 3.4 -- Relaxation dynamics near the completion threshold (Fig. 3)
```bash
python3 critical_slowing_down_test.py          # original, narrower sweep
python3 critical_slowing_down_extended.py      # corrected, properly bracketed p_c (Fig. 3)
python3 critical_slowing_down_universality.py  # non-universality check across (M,K)
```
`critical_slowing_down_extended.py` is the source of Figure 3 and the $p_c=0.395$,
$\nu=1.32\pm0.08$ numbers. `critical_slowing_down_universality.py` produces the four
additional $(M,K)$ fits discussed as an honest negative/mixed finding. Runtime: 15-30 min
each (200 trials/point for the flagship configuration).

### Section 3.5 -- Phase synchrony as an emergent conflict mask (Table 4, Fig. 4)
```bash
python3 spiking_multigrid_replication_fixedweight.py   # Table 4 (8-grid replication)
python3 spiking_real_sudoku_sweetspot.py               # single-grid version, sanity check
python3 spiking_sync_desync_test.py                    # causal random-assignment test,
                                                        #   isolates the connection-type effect
python3 spiking_steady_state_phase_test.py             # anti-phase-locking test (negative
                                                        #   result) + coincidence-window
                                                        #   classifier ceiling (63-65%)
python3 spiking_raster_comparison.py                   # Figure 4-style raster panels
```
Each spiking trial simulates 3000 ms (30 oscillation cycles) and takes roughly 6-10 seconds;
the multigrid replication (8 grids) takes about a minute, the full causal/anti-phase sweeps
several minutes each.

### Section 3.6-3.7 -- Emergent coincidence matrix closes the loop (Tables 5-6, Fig. 5)
```bash
python3 spiking_emergent_coincidence.py           # Table 5 + Figure 5 (11 clue fractions x
                                                   #   25 trials -- ~40-60 min)
python3 spiking_paired_significance_sweep.py      # Table 6: paired Wilcoxon test, dynamics
                                                   #   vs. hard-coded rule (7 clue fractions x
                                                   #   20 trials, ~35-45 min)
python3 spiking_low_clue_diagnostic_fixedweight.py  # Section 3.7 diagnostic (why low clue
                                                     #   fractions are hard)
```
These are the longest-running scripts in the repository (each launches hundreds of
independent 3000 ms spiking simulations). Precomputed output CSVs are included in `code/` so
you can inspect and re-plot results without re-running the simulations.

## Core modules

- `hippocampus_drive_readout.py` -- coincidence matrix, random projection, $k$-WTA engram,
  attention and ridge-regression readouts, conflict-mask cleanup. Imported by nearly every
  other script.
- `separated_edit_no_plasticity.py` -- the Brian2 spiking network: LIF neuron equations,
  conflict-mask (inhibitory) and complement-mask (excitatory) wiring, shared oscillatory
  drive. Imported by every `spiking_*.py` script.

## Statistics

Table 6's significance test is a two-sided Wilcoxon signed-rank test
(`scipy.stats.wilcoxon`) on paired per-trial accuracy differences -- identical target grid
and known-cell mask used across compared methods within each trial.

## License

MIT (see `LICENSE`).
