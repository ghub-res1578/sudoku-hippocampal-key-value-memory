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

Section 4 extends the engram layer to a second modality (natural images, heteroassociated to
Sudoku engrams via the same bidirectional pseudo-inverse rule used by Vector-HaSH) and
benchmarks it against classical Hopfield networks on identical data, with a single shared
noise model across all four methods. The result is double-edged and reported as such: a
paired nearest-neighbor baseline beats the pipeline at every stored-set size tested, and the
gap widens (not closes) as that size grows, from +3.5pp at M=200 to +96.0pp at M=2000 -- an
honest negative result, not just an untested caveat. But on the specific property Vector-HaSH
is built to provide (graceful, not catastrophic, degradation under scale), our architecture
succeeds where classical and even pseudo-inverse-trained Hopfield networks -- the latter using
the identical learning rule, and statistically tied with ours in noise tolerance at fixed
storage size -- collapse to zero recovery an order of magnitude sooner in storage capacity.

Two further results (Sections 3.2 and 3.4) connect the architecture directly to specific prior
work rather than by analogy: we verify the engram encoder behaves as a similarity-preserving
hash, the property the fly olfactory mushroom-body circuit is known to provide (Spearman
$\rho=0.974$ between input and engram similarity across a full corruption sweep), and we test
the recent unification of correlation-matrix memory, sparse distributed memory, dense
associative memory, and Transformer attention as one computation differing only in its
"separation operator" (Gershman, Fiete \& Irie, 2025) by swapping that operator in our own
readout -- finding that noise robustness tracks whether an operator sharpens by an absolute
similarity gap or a similarity ratio, not merely how aggressively it discriminates.

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

A synaptic plasticity (STDP) variant of the excitatory synapses, explored during this project,
is **deliberately not included**. It was tested rigorously (a paired, statistically powered
sweep) and found *not* to reliably outperform the reported non-plastic, fixed-weight
dynamics -- the fixed-weight mechanism already matches the hard-coded rule, and we judged
added mechanism without added, validated benefit to be the wrong trade.

One earlier cross-modal encoder design is not included for a genuine failure mode: a
fixed-random-projection + softmax-attention lookup over stored image exemplars
(mathematically continuous but effectively winner-take-most given the similarity scale,
producing bimodal, non-monotonic behavior).

A second design -- mapping many exemplars to a single shared per-class target via ridge
regression -- was originally excluded here as "generalizes poorly, a cliff-like collapse
under noise," based on `heteroassoc_W_IH_overlap_vs_noise.py`. That claim has since been
corrected (`heteroassoc_W_IH_ridge_corrected.py`, `paper/main.tex` Section 4.1): the cited
test used the same too-small default ridge diagnosed and fixed in Section 4.5, and with
*only* that value corrected (nothing else changed), the identical setup degrades gracefully
instead of cliffing (overlap $0.846\to0.725$ between $\sigma=0$ and $\sigma=0.10$, vs. the
originally-reported $0.835\to0.330$) and supports genuine 9-way classification (78.8% at zero
noise, exhaustive over all 900 held-out test images; an earlier 100-trial sample overestimated
this at 86%). Section 4.2 examines this classifier's confusion structure and seed robustness
in detail, including a comparison against standard $k$-NN and logistic-regression baselines'
confusion matrices on the identical split. It is still not used for the main
image-to-Sudoku-grid pipeline, but for a real,
structural reason rather than the originally-claimed one: a shared target can only support
as many distinguishable outputs as there are distinct targets, while recovering a *specific*
one of $M$ stored images needs $M$ mutually distinguishable targets -- exactly what
one-memory-per-pair (the reported architecture) provides and a shared per-class target
structurally cannot, regardless of regularization. Unlike the softmax-lookup dead end above,
both the original and corrected scripts for this one are kept in `code/` -- not because the
design is used anywhere in the main pipeline, but so the correction itself is fully
reproducible: run both and see the before/after directly, rather than take the correction on
faith.

## Reproducing each result

All commands assume you're in `code/`. Each script prints a summary and writes CSVs/PNGs
that a corresponding `_plot`-style call at the bottom of the script (or the script itself)
turns into the paper's figures. Runtime notes below are on an 8-core desktop CPU.

### Section 3.1 -- Combinatorial code: capacity and robustness (Table 2)
```bash
python3 feedback_vs_M_sweep.py        # M = 50 .. 2000, noise-collapse threshold
python3 feedback_iterate_highM.py     # M up to 10,000 (needs feedback_iterate_highM_pool.npy,
                                       #   included; regenerate via generate_unique() if needed)
python3 feedback_vs_K_sweep.py        # engram size K = 10 .. 500
python3 feedback_K_finegrain.py       # finer K grid around the interesting region
python3 feedback_vs_density_sweep.py  # projection density 0.002 .. 1.0
```
Each writes a `*_collapse_thresholds.csv` -- the numbers in Table 2.
Runtime: a few minutes each; `feedback_iterate_highM.py` is the slowest (~10 min).

### Section 3.2 -- The engram code as a similarity-preserving hash (Fig. 1)
```bash
python3 hash_similarity_preservation.py
```
Corrupts each of 30 sampled grids' true coincidence matrix at 13 levels (0-100%), and measures
Jaccard similarity of the corrupted vs. true same-digit pair sets against engram overlap of the
resulting codes. Writes `hash_similarity_preservation.csv` and Figure 1 (Spearman
$\rho=0.974$ between the two, pooled across $n=390$ corrupted instances). Runtime: under a
minute.

### Section 3.3 -- Key-value recall: attention vs. regression (Table 3, Fig. 2)
```bash
python3 capacity_regression_vs_attention.py
```
Writes `capacity_regression_vs_attention.csv` (raw) and
`capacity_regression_vs_attention_capacity_estimates.csv` (Table 3), plus the capacity-curve
PNG used in Figure 2. Runtime: ~15-20 min (sweeps $M\in\{10,\dots,2000\}$).

### Section 3.4 -- Separation operators: gap-based vs. ratio-based sharpening (Fig. 3)
```bash
python3 separation_operator_ablation.py
```
Holds the encoder, the $M=200$ stored engrams, and every corrupted query fixed (paired design)
while swapping the readout's separation operator across six variants -- identity
(correlation-matrix memory), softmax at two temperatures, a rectified-power nonlinearity
(dense associative memory), a hard top-5 threshold (sparse distributed memory), and argmax
(winner-take-all) -- re-running the query-noise sweep without the conflict-mask cleanup pass.
Writes `separation_operator_ablation.csv` and Figure 3. Runtime: a few minutes.
```bash
python3 separation_operator_competitor_ratio.py
```
Reproduces the mechanistic numbers in the text (not a figure/table): the best-competing
wrong memory's zero-noise similarity is $3.39\times$ tighter than the true match's, far
below the naive $6.67\times$ ($=H/K$) predicted from the mean chance-overlap floor alone
-- an extreme-value effect of having $M-1=199$ competitors -- and this ratio shrinks to
$1.90$ at $20\%$ query noise and $1.46$ at $30\%$, tracking the point where the
rectified-polynomial operator collapses in Figure 3. Runtime: under a minute.

### Section 3.5 -- Graceful degradation under query noise (Fig. 4)
```bash
python3 hippocampus_drive_readout.py        # writes hippocampus_drive_readout_feedback_sweep.csv
                                             #   (full 0-100% sweep, coarse steps)
python3 hippocampus_feedback_noise_zoomed.py  # reconstructs the same engram, re-sweeps just
                                               #   35-55% at 1pp resolution, and writes the
                                               #   two-panel Figure 4 (needs the CSV above)
```
Full 0-100% query-noise sweep, with and without the conflict-mask cleanup pass, at $M=200$;
the zoomed script resolves the 35-55% transition window at 1-percentage-point resolution and
is what the rescue-peak number ($42\%$ noise, $+16.8$pp) in the text comes from. Runtime: a
few minutes for each (`hippocampus_drive_readout.py` also runs the capacity/robustness spot
checks used elsewhere in the paper). `hippocampus_feedback_noise_plot.py` regenerates the
old single-panel version of Figure 4's left panel alone, if useful for other purposes.

### Section 3.6 -- Partial-grid pattern completion (Table 4, Fig. 5)
```bash
python3 sudoku_partial_grid_completion.py
```
Writes `sudoku_partial_grid_completion.csv` and the accuracy-vs-clue-fraction figure.
`sudoku_partial_grid_MK_sweep.py`, `sudoku_partial_grid_MKH_sweep.py`, and
`sudoku_partial_grid_MK_iterations_sweep.py` reproduce the broader $M$-$K$-$H$ characterization
referenced in the text (not required for the headline table). Runtime: ~5-10 min.

### Section 3.7 -- Relaxation dynamics near the completion threshold (Fig. 6)
```bash
python3 critical_slowing_down_test.py          # original, narrower sweep
python3 critical_slowing_down_extended.py      # extended sweep -- generates the RAW DATA
                                                #   (critical_slowing_down_extended.csv) used
                                                #   for Figure 6, but its own built-in
                                                #   analyze_and_plot() should NOT be used to
                                                #   make the figure -- see note below
python3 critical_slowing_down_clean_plot.py    # regenerates Figure 6 correctly from that CSV
python3 critical_slowing_down_universality.py  # non-universality check across (M,K)
```
**Known issue, worth understanding before re-running anything here:**
`critical_slowing_down_extended.py`'s own `analyze_and_plot()` picks its power-law decay
branch with an automatic heuristic (`mean_iter < plateau_mean - plateau_std`) that turns out
to sweep in points that have already hit the floor, biasing the fit to a bogus,
boundary-pinned $p_c\approx0.85$ -- and separately, its plot-label string still negates the
fitted slope (a sign bug that was only patched in the script's console `print` output, not in
the plotting code). **Do not use the PNG that script produces directly.** The correct,
paper-reported fit ($p_c=0.395$, $\nu=1.32\pm0.08$, $R^2=0.978$) restricts the decay branch by
hand to clue fraction $\in[0.21,0.37]$ (excluding both the plateau and the floor) and is what
`critical_slowing_down_clean_plot.py` does, reading the same
`critical_slowing_down_extended.csv` the first script writes. Run the two in that order.
`critical_slowing_down_universality.py` produces the four additional $(M,K)$ fits discussed
as an honest negative/mixed finding. Runtime: 15-30 min each (200 trials/point for the
flagship configuration); the clean-plot script itself is instant (it only re-fits the
already-generated CSV).

### Section 3.8 -- Phase synchrony as an emergent conflict mask (Table 5, Fig. 7)
```bash
python3 spiking_multigrid_replication_fixedweight.py   # Table 5 (8-grid replication)
python3 spiking_real_sudoku_sweetspot.py               # single-grid version, sanity check
python3 spiking_sync_desync_test.py                    # causal random-assignment test,
                                                        #   isolates the connection-type effect
python3 spiking_steady_state_phase_test.py             # anti-phase-locking test (negative
                                                        #   result) + coincidence-window
                                                        #   classifier ceiling (63-65%)
python3 spiking_raster_comparison.py                   # Figure 7-style raster panels
```
Each spiking trial simulates 3000 ms (30 oscillation cycles) and takes roughly 6-10 seconds;
the multigrid replication (8 grids) takes about a minute, the full causal/anti-phase sweeps
several minutes each.

### Section 3.9 -- Emergent coincidence matrix closes the loop (Tables 6-7, Fig. 8)
```bash
python3 spiking_emergent_coincidence.py           # Table 6 + Figure 8 (11 clue fractions x
                                                   #   25 trials -- ~40-60 min)
python3 spiking_paired_significance_sweep.py      # Table 7: paired Wilcoxon test, dynamics
                                                   #   vs. hard-coded rule (7 clue fractions x
                                                   #   20 trials, ~35-45 min)
```
These are the longest-running scripts in the repository (each launches hundreds of
independent 3000 ms spiking simulations). Precomputed output CSVs are included in `code/` so
you can inspect and re-plot results without re-running the simulations.

### Section 3.10 -- Graceful degradation under drive-current noise (Fig. 9)
```bash
python3 spiking_drive_noise_robustness.py   # ~15-20 min (7 noise levels x 20 trials); also
                                             #   regenerates Figure 9 (calls plot() at the end)
```
Same clue fraction (0.30) as the Section 3.9 headline results, but corrupts every known
cell's drive current with Gaussian jitter instead of leaving cells unknown -- a continuous,
rather than categorical, robustness test specific to the spiking mechanism.

### Section 3.11 -- Why low clue fractions are hard (diagnostic)
```bash
python3 spiking_low_clue_diagnostic_fixedweight.py
```
Measures emergent-$\hat C$ activity, engram overlap, and attention mass/rank on the true
target vs.\ an oracle, at three clue fractions. Runtime: a few minutes.

### Section 4 -- Cross-modal heteroassociation (Tables 8-17, Figs. 10-18; Tables 18-19/Fig. 19 in the Appendix)

Requires `tensorflow` (used only to load MNIST/Fashion-MNIST/CIFAR-100 via
`tf.keras.datasets`; downloads to `~/.keras/datasets/` on first use, then cached).

```bash
python3 heteroassoc_vectorhash_pipeline.py       # the core M=200 MNIST pipeline + sanity
                                                  #   checks (defines the reusable functions
                                                  #   train_pseudo_inverse_dual, k_wta,
                                                  #   iterate_sudoku_attractor, etc.)
python3 heteroassoc_vectorhash_best_examples.py  # Figure 10: largest-rescue-gap examples,
                                                  #   3 datasets
python3 heteroassoc_W_IH_ridge_corrected.py      # Section 4.1/4.2's single-W_IH, 9-class
                                                  #   digit classifier (ridge=1.0); prints the
                                                  #   78.8%-at-zero-noise headline number
python3 heteroassoc_digit_classifier_seed_sweep.py       # Table 8, Figure 11: accuracy and
                                                  #   attractor-class concentration across 6
                                                  #   independent (W_SH, class-target) draws
python3 heteroassoc_digit_classifier_vs_baseline_confusion.py  # Table 9, Figure 12: k-NN
                                                  #   (k=5) and logistic-regression baselines
                                                  #   on the identical split, and the
                                                  #   off-diagonal error-pattern correlation
                                                  #   test (r=0.71/0.68 vs. ours, 0.55 between
                                                  #   the two baselines)
python3 heteroassoc_digit_classifier_best_worst.py       # Appendix Figure 19: one best and
                                                  #   one worst held-out classification per
                                                  #   digit class, zero noise
python3 heteroassoc_digit_classifier_random_scaffold_ablation.py  # Section 4.2 caveat: swaps
                                                  #   the 9 Sudoku-derived class targets for 9
                                                  #   random K-sparse vectors (no Sudoku
                                                  #   structure) -- accuracy is unchanged
                                                  #   (79.0% vs. 78.8%, p=0.91), so the
                                                  #   scaffold contributes nothing here; the
                                                  #   classifier has no cleanup/attractor pass
                                                  #   either, unlike Section 4.1's
                                                  #   reconstruction pipeline
python3 heteroassoc_vectorhash_generalized.py    # Table 10 precursor: dataset/M generalization
python3 heteroassoc_final_consolidation.py       # Table 10 (rigorous, N=200/point, Wilson CI) +
                                                  #   Section 4.4's k-NN baseline/McNemar test +
                                                  #   Section 4.5's capacity/conditioning/ridge
                                                  #   diagnostics -- Figure 13
python3 heteroassoc_knn_vs_M.py                  # Table 11: does the k-NN gap close as M grows?
                                                  #   (no -- it widens from +3.5pp at M=200 to
                                                  #   +96.0pp at M=2000), paired McNemar per M
python3 heteroassoc_ridge_scaling_with_M.py      # does a single ridge value keep working as M
                                                  #   grows? (M up to 2000)
python3 heteroassoc_ridge_scaling_extended.py    # Table 12: extends the above to M=2500-4000 --
                                                  #   lambda~10 stays near-optimal throughout,
                                                  #   does not need to keep growing
python3 heteroassoc_H_K_sweep.py                 # does bigger H or K help noise robustness?
python3 heteroassoc_H_extends_capacity.py        # does bigger H extend the M-scaling ceiling?
                                                  #   (isolates the image-side vs. engram-side
                                                  #   bottleneck via direct Gram-matrix rank)
python3 heteroassoc_cifar_capacity.py            # same capacity diagnostic, CIFAR-100 grayscale
python3 hopfield_classic_cliff_reference.py      # sanity check: reproduces the textbook
                                                  #   random-pattern Hopfield cliff (M/N~0.138)
python3 hopfield_vs_our_network_comparison.py    # original bit-flip-vs-Gaussian comparison --
                                                  #   superseded as the paper's headline number by
                                                  #   the matched-noise version below, but still
                                                  #   the source of the M=100/150/200 pseudo-
                                                  #   inverse basin-shrinkage illustration in the
                                                  #   Section 4.6 text
python3 hopfield_pinv_basin_shrinkage.py         # standalone reproduction of that same
                                                  #   illustration (100%/25%/0% exact)
python3 hopfield_matched_noise_rigorous.py       # Table 13, Figure 14: the headline four-way
                                                  #   comparison, redone with N=200/point, Wilson
                                                  #   CIs, and a SINGLE shared noise model (every
                                                  #   method sees the same Gaussian-corrupted
                                                  #   image; the three Hopfield variants get it
                                                  #   bipolarized) -- this is what the paper
                                                  #   reports, not the script above
python3 hopfield_modern_sdm_dam_comparison.py    # Section 4.7 -- Table 14 (construction),
                                                  #   Figure 15 (M-sweep + M=50 noise-sweep):
                                                  #   extends the comparison to Modern Hopfield
                                                  #   (Ramsauer 2020), sparse distributed memory
                                                  #   (Kanerva), and dense associative memory
                                                  #   (Krotov-Hopfield), each built as a complete
                                                  #   standalone system, not a swapped operator
                                                  #   inside our own encoder (contrast Section
                                                  #   3.4). Also re-validates beta/n from scratch
                                                  #   on this data (naive carryovers from
                                                  #   elsewhere in this paper fail near-totally
                                                  #   even at zero noise)
python3 hopfield_comparison_followup_H_and_highM_noise.py  # Table 15, Figure 17 (does bigger H
                                                  #   close the gap? M=800, sigma=0.3, H swept
                                                  #   1000-3000) and Figure 16 (noise-sweep at
                                                  #   M=800, alongside the existing M=50 sweep,
                                                  #   to show the gap is a genuine high-M
                                                  #   collapse in noise tolerance, not a single
                                                  #   data point)
python3 vectorhash_modular_scaffold_comparison.py  # Section 4.8 -- Table 16 (one-shot vs.
                                                  #   recurrent cleanup): builds Vector-HaSH's
                                                  #   actual modular addressing mechanism (10
                                                  #   coprime-period modules, CRT-assigned
                                                  #   codes) rather than only its
                                                  #   heteroassociation rule, first with a naive
                                                  #   independent per-module argmax cleanup
                                                  #   (collapses to 7.5% by M=800), then with a
                                                  #   genuine recurrent Hopfield-style relaxation
                                                  #   over the space of known valid codes
                                                  #   (recovers to 81.0% at M=800) -- same
                                                  #   M-sweep/noise-sweep grid as
                                                  #   hopfield_modern_sdm_dam_comparison.py
python3 knn_at_sigma03_matched.py                # k-NN baseline reconciled to the exact
                                                  #   sigma=0.3, M in {100,200,400,800}
                                                  #   condition used throughout Section 4.7-4.8
                                                  #   (100% at every M tested)
python3 vectorhash_modular_scaffold_gram_analysis.py  # Section 4.8's condition-number/
                                                  #   effective-rank numbers (3.0e3->2.8e12,
                                                  #   rank 387 at M=800) -- these existed only
                                                  #   as an unsaved ad hoc computation when
                                                  #   first reported; this script makes them
                                                  #   reproducible
python3 final_hetero_comparison_plot.py          # Figure 18: reads the CSVs from the two
                                                  #   scripts above and plots the one-shot-vs-
                                                  #   recurrent ablation (run the two scripts
                                                  #   first)
```
Table 17 (storage/time complexity) is an analytical comparison, not a simulation -- no script
to run for it; see Section 4.8 of the paper for the derivation.
Runtime: most of these build an M-item store from scratch per configuration and are a few
seconds to low minutes each; `heteroassoc_final_consolidation.py`, `heteroassoc_knn_vs_M.py`,
and `hopfield_matched_noise_rigorous.py` sweep several $M$/noise values at $N{=}200$/point and
take several minutes; `heteroassoc_ridge_scaling_extended.py` builds stores up to $M{=}4000$
and can take 10+ minutes. `heteroassoc_digit_classifier_seed_sweep.py` retrains the classifier
from scratch across 6 seeds (a couple of minutes); `heteroassoc_digit_classifier_vs_baseline_confusion.py`
additionally requires `scikit-learn` (for the $k$-NN and logistic-regression baselines) and is
a few minutes, mostly spent fitting logistic regression on $4500\times784$ raw pixels.
`hopfield_modern_sdm_dam_comparison.py` and `hopfield_comparison_followup_H_and_highM_noise.py`
each build several $M$-item stores at $N{=}200$/point and take a few minutes total.
`vectorhash_modular_scaffold_comparison.py` is a few minutes (no iterative relaxation needed for
the one-shot pass; the recurrent decode adds a Hopfield-style fixed-point iteration per trial);
`knn_at_sigma03_matched.py` is dominated by MNIST loading, a couple of minutes total;
`final_hetero_comparison_plot.py` only reads existing CSVs and is instant. Note: when
running these from a network-mounted working directory, redirect stdout/stderr to a local path
(e.g. `python3 script.py > /tmp/log.txt 2>&1`) rather than one on the mount -- intermittent
mount I/O, not the computation itself, caused sporadic silent process termination during
development (confirmed by rerunning the identical computation with output redirected locally,
which completed in seconds).

## Core modules

- `hippocampus_drive_readout.py` -- coincidence matrix, random projection, $k$-WTA engram,
  attention and ridge-regression readouts, conflict-mask cleanup. Imported by nearly every
  other script.
- `separated_edit_no_plasticity.py` -- the Brian2 spiking network: LIF neuron equations,
  conflict-mask (inhibitory) and complement-mask (excitatory) wiring, shared oscillatory
  drive. Imported by every `spiking_*.py` script.

## Statistics

Table 7's significance test is a two-sided Wilcoxon signed-rank test
(`scipy.stats.wilcoxon`) on paired per-trial accuracy differences -- identical target grid
and known-cell mask used across compared methods within each trial.

## License

MIT (see `LICENSE`).
