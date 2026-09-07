from brian2 import *
import numpy as np
import os
import time
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ------------------------- Parameters -------------------------
DEFAULT_PARAMS = {
    'N': 81,
    'runtime_ms': 10000,    # simulation time in ms
    'dt':0.1 * ms,
    'Vth_val': 1.0,
    'Vreset_val': 0.0,
    'refractory': 0*ms,
    'p_EE': 1.0,
    'w_EE': 0.0,    # excitatory hop magnitude (unitless)
    'w_EI': 1.0,      # E->I relay
    'w_IE': 0.02,     # initial inhibitory hop magnitude (I->E) #0.005
    'w_II': 0.0,   # I->I inhibitory hop
    # drive settings:
    'drive_duration_ms': 10000,   # how long external drive is ON (ms)
    # amplitudes per Sudoku cluster (1..9). Length must be 9.
    'cluster_drive_amplitudes':  np.linspace(0.58, 0.67, 9),   #np.linspace(0.58, 0.67, 9) #np.linspace(1.08, 1.17, 9)
    'baseline_Id': 1.5,       # baseline I_d after drive is turned off#1.1
    'I_d_init': 0,          # initial I_d value (before network_operation sets drive)
    # plasticity params (iSTDP)
    'tau_trace_ms': 5.0,
    'eta': 0.0, #0.001
    'alpha': 1.0,
    'w_min': 0.005, #0.005
    'w_max': 0.005, #0.01
    # snapshot times (ms)
    'ei_snapshot_times_ms': [3000.0, 400.0, 1000.0, 1500],
    # initial potential perturbation range (added)
    'init_perturb_low': -0.1,
    'init_perturb_high': 0.1,
}

# ------------------------- Helper: Sudoku masks -------------------------
def build_conflict_mask(N=81):
    assert N == 81
    mask = np.zeros((N, N), dtype=int)
    for i in range(9):
        for j in range(9):
            src = i * 9 + j
            # row conflicts (same column j, different row)
            for r in range(9):
                if r != i:
                    mask[src, r * 9 + j] = 1
            # column conflicts (same row i, different column)
            for c in range(9):
                if c != j:
                    mask[src, i * 9 + c] = 1
            # subgrid conflicts
            base_r = (i // 3) * 3
            base_c = (j // 3) * 3
            for rr in range(base_r, base_r+3):
                for cc in range(base_c, base_c+3):
                    if (rr, cc) != (i, j):
                        mask[src, rr * 9 + cc] = 1
    return mask

# ------------------------- Network builder -------------------------
def build_network(params=DEFAULT_PARAMS, init_v=None, flat_orig=None):
    start_scope()
    N = params['N']
    # oscillation frequency/constants left here for clarity (used in eqs)
    # NOTE: we gate oscillations with per-neuron var 'osc_on'
    # Add I_d as drive and osc_on to enable/disable oscillation term.
    eqs = f'''
    dv/dt = (I_d + I - v) / (40*ms) : 1
    I = osc_on * (0.5 - 0.2*cos(2*pi*10*Hz*t)) : 1   # <<-- CHANGED: gated oscillation
    I_d : 1
    osc_on : 1                                       # <<-- CHANGED: 1 when oscillations enabled, 0 when disabled
    '''

    E = NeuronGroup(N, eqs, threshold=f'v >= {params["Vth_val"]}', reset=f'v = {params["Vreset_val"]}', refractory=params['refractory'], method='rk4', name='E')
    I = NeuronGroup(N, 'dv/dt = (0) / ( 40*ms) : 1', threshold=f'v > 0', reset=f'v = {params["Vreset_val"]}', refractory=params['refractory'], method='rk4', name='I')
    
    # initialize membrane potentials
    low = params.get('init_perturb_low', 0)
    high = params.get('init_perturb_high', 0)
    if init_v is None:
        base = np.random.rand(N) * 0.1
        perturb = np.random.uniform(low, high, size=N)
        E.v = base + perturb
    else:
        init_arr = np.asarray(init_v, dtype=float)
        if init_arr.size != N:
            raise ValueError('init_v must have length N')
        perturb = np.random.uniform(low, high, size=N)
        E.v = init_arr + perturb

    # initialize I_d to a neutral/baseline value; network_operation will set drive during run
    E.I_d = params.get('I_d_init', 0.0)
    # initialize oscillation gate: default ON (1) at start
    E.osc_on = 1.0    # <<-- CHANGED: start with oscillations enabled during drive
    I.v = 0.0

    # build masks
    conflict_mask = build_conflict_mask(N)
    complement_mask = (1 - conflict_mask) - np.eye(N, dtype=int)

    # E -> E (random) with synaptic variable w
    S_EE = Synapses(E, E, model='w:1', on_pre='v_post += w', name='S_EE')
    ii_src, ii_tgt = np.where(complement_mask == 1)
    if len(ii_src) > 0:
        S_EE.connect(i=ii_src, j=ii_tgt, p=params['p_EE'])
    S_EE.w = params['w_EE']

    # E -> I (relay using conflict_mask - here we keep simple 1:1 relay)
    S_EI = Synapses(E, I, model='w:1', on_pre='v_post += w', name='S_EI')
    S_EI.connect(i=range(N), j=range(N))
    S_EI.w = params['w_EI']

    # I -> E (inhibitory according to conflict mask) with iSTDP plasticity
    tau_trace_ms = params['tau_trace_ms']   # decay time constant for pre/post traces
    eta = params['eta']           # learning rate for STDP
    alpha = params['alpha']       # depression scaling
    w_min = params['w_min']
    w_max = params['w_max']

    S_IE = Synapses(
        I, E,
        model=f'''
        w : 1
        eta : 1
        alpha : 1
        dxpre/dt = -xpre/({tau_trace_ms}*ms) : 1 (clock-driven)
        dxpost/dt = -xpost/({tau_trace_ms}*ms) : 1 (clock-driven)
        ''',
        on_pre=f'''
        v_post -= w
        w = clip(w + eta * xpost, {w_min}, {w_max})
        xpre += 1.0
        ''',
        on_post=f'''
        w = clip(w - eta * alpha * xpre, {w_min}, {w_max})
        xpost += 1.0
        ''',
        name='S_IE_plastic'
    )

    # connect according to conflict mask deterministically
    src_idx, tgt_idx = conflict_mask.nonzero()
    if len(src_idx) > 0:
        S_IE.connect(i=src_idx, j=tgt_idx)

    #S_IE.connect(condition='i!=j', p=1.0)

    # initialize synapse variables
    S_IE.w = params['w_IE']
    S_IE.xpre = 0.0
    S_IE.xpost = 0.0
    S_IE.eta = eta
    S_IE.alpha = alpha

    # I -> I complementary connectivity
    S_II = Synapses(I, I, model='w:1', on_pre='v_post -= w', name='S_II')
    ii_src, ii_tgt = np.where(complement_mask == 1)
    if len(ii_src) > 0:
        S_II.connect(i=ii_src, j=ii_tgt, p = 0.0)
    S_II.w = params['w_II']
    
    # monitors
    spkE = SpikeMonitor(E, name='spkE')
    spkI = SpikeMonitor(I, name='spkI')
    rateE = PopulationRateMonitor(E, name='rateE')
    rateI = PopulationRateMonitor(I, name='rateI')
    stateE = StateMonitor(E, 'v', record=range(min(20, N)), name='stateE')
    


    # sample synapse weight monitor (record a subset to keep memory small)
    sample_n = min(200, len(S_IE))  # record up to 200 sample synapses
    syn_w_mon = StateMonitor(S_IE, 'w', record=range(sample_n), dt=10*ms, name='syn_w_mon')

    net = Network(collect())
    comps = {
        'E': E, 'I': I, 'S_EE': S_EE, 'S_EI': S_EI, 'S_IE': S_IE, 'S_II': S_II,
        'spkE': spkE, 'spkI': spkI, 'rateE': rateE, 'rateI': rateI, 'stateE': stateE,
        'syn_w_mon': syn_w_mon
    }
    comps['conflict_mask'] = conflict_mask
    return net, comps

# ------------------------- Raster plotting helper (unchanged) -------------------------
def plot_raster_from_h5(h5file, outname='raster_plot.png', start_ms=None, end_ms=None,
                        sudoku_mapping=None, sort_method='first'):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import h5py

    with h5py.File(h5file, 'r') as f:
        if 'spikes' not in f or 'E' not in f['spikes']:
            raise ValueError("HDF5 missing 'spikes/E' group")
        grp = f['spikes']['E']
        n = len(grp.keys())
        spike_times = []
        for i in range(n):
            data = grp[f'neuron_{i}'][:]  # assumed in ms
            spike_times.append(np.array(data, dtype=float))

    if n != 81:
        raise ValueError(f'Expected 81 neurons, found {n} in {h5file}')

    if sudoku_mapping is None:
        sudoku_mapping = [(i % 9) + 1 for i in range(81)]
    sudoku_mapping = np.asarray(sudoku_mapping).astype(int)
    if sudoku_mapping.shape[0] != 81:
        raise ValueError('sudoku_mapping must have length 81')

    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
    neuron_colors = [mcolors.to_rgb(palette[(d-1) % 9]) for d in sudoku_mapping]

    times_in_window = []
    rep_times = np.empty(n, dtype=float)
    rep_times.fill(np.inf)

    for i in range(n):
        times = spike_times[i]
        if start_ms is not None:
            times = times[times >= start_ms]
        if end_ms is not None:
            times = times[times <= end_ms]
        times_in_window.append(times)
        if times.size > 0:
            if sort_method == 'mean':
                rep_times[i] = np.mean(times)
            elif sort_method == 'first':
                rep_times[i] = times[0]
            else:
                raise ValueError("sort_method must be 'mean' or 'first'")

    order = np.argsort(rep_times)
    raster_data = [times_in_window[idx] for idx in order]
    raster_colors = [neuron_colors[idx] for idx in order]
    raster_data_ms = [np.asarray(arr) for arr in raster_data]

    plt.figure(figsize=(12, 6))
    plt.eventplot(raster_data_ms,
                  lineoffsets=np.arange(len(raster_data_ms)),
                  colors=raster_colors,
                  linelengths=0.8)
    plt.xlabel('Time (ms)')
    plt.ylabel('Neurons (sorted by spike time)')
    title_window = 'full' if (start_ms is None and end_ms is None) else f'{start_ms}-{end_ms} ms'
    plt.title(f'Raster (sorted by {sort_method}; window={title_window})')
    plt.tight_layout()
    plt.savefig(outname, dpi=300)
    plt.close()
    print(f'Raster saved to {outname}')

# ------------------------- Runner -------------------------
def run_simulation(seed=0, params=DEFAULT_PARAMS, save_path='sim_output', init_from_sudoku=True):
    np.random.seed(seed)
    defaultclock.dt = params['dt']
    runtime = params['runtime_ms'] * ms

    # optional initialization from Sudoku solution (values scaled to 0..1)
    init_v = None
    flat_orig = None
    if init_from_sudoku:
        sudoku_grid = [
            [6, 7, 4, 5, 1, 8, 2, 9, 3],
            [3, 5, 9, 6, 2, 7, 1, 8, 4],
            [2, 1, 8, 4, 3, 9, 5, 7, 6],
            [4, 6, 1, 7, 9, 5, 8, 3, 2],
            [9, 8, 7, 2, 4, 3, 6, 5, 1],
            [5, 2, 3, 1, 8, 6, 7, 4, 9],
            [1, 9, 2, 8, 5, 4, 3, 6, 7],
            [8, 3, 6, 9, 7, 1, 4, 2, 5],
            [7, 4, 5, 3, 6, 2, 9, 1, 8]
        ]
        flat_orig = np.array([val for row in sudoku_grid for val in row])
        init_v = flat_orig/10.0

    net, comps = build_network(params=params, init_v=init_v, flat_orig=flat_orig)

    E = comps['E']
    I = comps['I']
    S_IE = comps['S_IE']
    spkI = comps['spkI']
    spkE = comps['spkE']

    # Build drive vector per neuron based on cluster (flat_orig gives values 1..9)
    if flat_orig is not None:
        cluster_ids = np.array(flat_orig, dtype=int)  # values 1..9
    else:
        # fallback: random cluster assignment
        cluster_ids = np.random.randint(1,10, size=params['N'])

    # cluster amplitudes
    amps = np.array(params['cluster_drive_amplitudes'], dtype=float)
    if amps.shape[0] != 9:
        raise ValueError('cluster_drive_amplitudes must be length 9')
    drive_vector = amps[cluster_ids - 1]  # index 0..8

    drive_duration = params['drive_duration_ms'] * ms
    baseline_Id = params.get('baseline_Id', 0.0)

    # helper function to build full (N_inh x N_e) matrix from synapse arrays
    def build_EI_matrix_from_synapses(w_flat, pres_arr, post_arr, N):
        mat = np.zeros((N, N), dtype=float)  # rows = presyn inhibitory index, cols = posts excitatory index
        for kk in range(len(w_flat)):
            i_pres = pres_arr[kk]
            j_post = post_arr[kk]
            mat[i_pres, j_post] = w_flat[kk]
        return mat

    # prepare pres/posts arrays for snapshots & saving
    w_arr = S_IE.w[:].copy()
    pres = np.array(S_IE.i[:], dtype=int)
    posts = np.array(S_IE.j[:], dtype=int)

    # ---------------- prepare snapshot machinery ----------------
    snapshot_times = list(params.get('ei_snapshot_times_ms', [0.0, 200.0, 1000.0]))
    # dictionary to hold captured EI matrices {time_ms: matrix}
    EI_snapshots = {}

    # helper to save/plot a snapshot
    def save_plot_EI(mat, t_ms, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        vmin = np.min(mat)
        vmax = np.max(mat)
        plt.figure(figsize=(6,5))
        im = plt.imshow(mat, aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis', origin='lower')
        plt.colorbar(im, label='w (I->E)')
        plt.xlabel('E neuron index (post)')
        plt.ylabel('I neuron index (pre)')
        plt.title(f'I->E weight matrix at t={int(t_ms)} ms')
        fname = os.path.join(out_dir, f'EI_snapshot_{int(t_ms)}ms.png')
        plt.tight_layout()
        plt.savefig(fname, dpi=300)
        plt.close()
        print(f'Saved EI snapshot plot: {fname}')

    # capture initial (t=0) EI matrix BEFORE running the network (this reflects initial S_IE.w)
    initial_w = S_IE.w[:].copy()
    EI_snapshots[0.0] = build_EI_matrix_from_synapses(initial_w, pres, posts, params['N'])

    # ---------------- network_operation to apply drive & control oscillations ----------------
    @network_operation(dt=0.1*ms)
    def apply_drive_and_osc_control():
        if defaultclock.t < drive_duration:
            # during drive: provide per-neuron drive and enable oscillations
            E.I_d = drive_vector
            E.osc_on = 1.0           # <<-- CHANGED: enable oscillations while drive is on
        else:
            # after drive: set baseline drive and disable oscillations
            E.I_d = baseline_Id
            E.osc_on = 0.0           # <<-- CHANGED: disable oscillations when drive is off

    net.add(apply_drive_and_osc_control)
    
    # ---------------- snapshot network_operation (to capture early times) ----------------
    @network_operation(dt=1*ms)
    def snapshot_op():
        nonlocal EI_snapshots
        curr_t_ms = float(defaultclock.t/ms)
        # check snapshot list
        for target_t in snapshot_times:
            if target_t in EI_snapshots:
                continue
            if curr_t_ms >= target_t - 1e-6:
                # grab current synapse weights
                w_curr = S_IE.w[:].copy()
                pres_arr = np.array(S_IE.i[:], dtype=int)
                post_arr = np.array(S_IE.j[:], dtype=int)
                EI_snapshots[target_t] = build_EI_matrix_from_synapses(w_curr, pres_arr, post_arr, params['N'])
                print(f'[snapshot_op] captured EI at t={curr_t_ms:.1f} ms for target {target_t} ms')

    net.add(snapshot_op)

    # run
    os.makedirs(save_path, exist_ok=True)
    # save initial snapshot plot now (t=0)
    save_plot_EI(EI_snapshots[0.0], 0.0, save_path)

    t0 = time.time()
    net.run(runtime)
    t1 = time.time()

    # after the run, ensure all requested snapshots are present; if not, save final weights as fallback
    final_w = S_IE.w[:].copy()
    pres_arr = np.array(S_IE.i[:], dtype=int)
    post_arr = np.array(S_IE.j[:], dtype=int)
    final_mat = build_EI_matrix_from_synapses(final_w, pres_arr, post_arr, params['N'])

    for target_t in snapshot_times:
        if target_t not in EI_snapshots:
            EI_snapshots[target_t] = final_mat.copy()
            print(f'[post-run] Requested snapshot at {target_t} ms not reached; saved final matrix ({params["runtime_ms"]} ms) instead.')

    # save plots and write matrices to HDF5
    h5file = os.path.join(save_path, f'dcon_0.58_0.67_plaon_oson_pert0.1_baseline1.1_{seed}_wEE{params["w_EE"]}_wIE{params["w_IE"]}_wmin{params["w_min"]}_wmax{params["w_max"]}.h5')
    with h5py.File(h5file, 'w') as f:
        f.attrs['seed'] = seed
        f.attrs['runtime_ms'] = params['runtime_ms']
        f.attrs['w_EE'] = params['w_EE']
        f.attrs['w_IE_init'] = params['w_IE']

        grpE = f.create_group('spikes/E')
        for neuron_id in range(params['N']):
            times = spkE.t[spkE.i == neuron_id] / ms
            grpE.create_dataset(f'neuron_{neuron_id}', data=np.array(times, dtype=np.float64))
        grpI = f.create_group('spikes/I')
        for neuron_id in range(params['N']):
            times = spkI.t[spkI.i == neuron_id] / ms
            grpI.create_dataset(f'neuron_{neuron_id}', data=np.array(times, dtype=np.float64))

        f.create_dataset('final_v_E', data=np.array(E.v[:], dtype=np.float64))
        f.create_dataset('final_v_I', data=np.array(I.v[:], dtype=np.float64))

        # final I->E weights
        try:
            w_final = np.array(S_IE.w[:], dtype=np.float64)
            f.create_dataset('final_S_IE_w', data=w_final)
        except Exception as e:
            print('Failed to save final S_IE weights:', e)

        # store EI snapshots explicitly as full matrices under /snapshots/EI_txxx
        grp_snap = f.create_group('snapshots')
        for t_ms, mat in EI_snapshots.items():
            dsname = f'EI_{int(t_ms)}ms'
            grp_snap.create_dataset(dsname, data=mat)
        # optionally sample synaptic weight history if the monitor exists
        try:
            syn_w_mon = comps.get('syn_w_mon', None)
            if syn_w_mon is not None:
                grp_w = f.create_group('synapse_monitor/S_IE_w')
                grp_w.create_dataset('t', data=np.array(syn_w_mon.t/ms, dtype=np.float64))
                grp_w.create_dataset('w', data=np.array(syn_w_mon.w, dtype=np.float64))
        except Exception as e:
            print('Failed to save syn_w_mon:', e)

    # save snapshot plots to disk (and also re-save initial if overwritten)
    for t_ms, mat in EI_snapshots.items():
        save_plot_EI(mat, t_ms, save_path)

    print(f'Simulation done in {t1-t0:.2f} s. Output saved to {h5file}')
    return h5file

# ------------------------- If run as script -------------------------
if __name__ == '__main__':
    out = run_simulation(seed= 18270, params=DEFAULT_PARAMS, save_path='sim_output', init_from_sudoku=True)
    # plot the last chunk of spikes
    print('Output:', out)
