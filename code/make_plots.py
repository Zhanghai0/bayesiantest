"""
make_plots.py
=============
Run AFTER simulation.py completes. Generates:
  Fig 1: fig1_regret_curves.pdf       (panel per setting; up to 4)
  Fig 2: fig2_setting_c_per_task.pdf  (per-task regret in Setting C, with
                                        paired Wilcoxon test on task 0)
  Fig 3: fig3_sensitivity.pdf         (hyperparameter sensitivity)
  Table 1: tables/table1_results.tex  (LaTeX results, includes Setting D + LinUCB)
  Table 2: tables/table2_sensitivity.tex
  Table 3: tables/table3_per_task_test.tex (paired test for Setting C task 0)

Changes vs. previous version:
  * Imports sampler classes from `samplers.py` — no more duplicated
    implementations across simulation.py and make_plots.py.
  * Loads optional 'linucb' field from npz files; falls back gracefully
    when running on old result files.
  * Setting D included automatically if its npz file is present.
  * Per-task analysis now reports a paired Wilcoxon signed-rank test
    on task-0 final regret (HierTS - IndepTS), giving a defensible
    statistical statement rather than relying on overlapping CIs.
  * Sensitivity sweep increased to S=15 trials.
"""

import argparse, os
import numpy as np
from scipy.special import ndtr
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'axes.labelsize': 11, 'axes.titlesize': 12,
    'legend.fontsize': 9, 'xtick.labelsize': 10,
    'ytick.labelsize': 10, 'figure.dpi': 150,
})

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from samplers import (HierProbitBandit, IndepProbitBandit, LinUCBBandit)

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', type=str, default='results')
parser.add_argument('--n_sens', type=int, default=15,
                    help='Trials for sensitivity sweep (default 15)')
args = parser.parse_args()

os.makedirs(os.path.join(args.outdir, 'figures'), exist_ok=True)
os.makedirs(os.path.join(args.outdir, 'tables'),  exist_ok=True)

# ─────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────
METHODS  = ['HierTS', 'IndepTS', 'PoolTS', 'LinUCB']
MLABELS  = {'HierTS' : 'HierTS (proposed)',
            'IndepTS': 'IndepTS',
            'PoolTS' : 'PoolTS',
            'LinUCB' : 'LinUCB'}
COLORS   = {'HierTS': '#1f77b4', 'IndepTS': '#d62728',
            'PoolTS': '#2ca02c', 'LinUCB' : '#9467bd'}
LSTYLES  = {'HierTS': '-', 'IndepTS': '--', 'PoolTS': '-.', 'LinUCB': ':'}
SLABELS  = {
    'A': 'Setting A\n(High Heterogeneity)',
    'B': 'Setting B\n(Low Heterogeneity)',
    'C': 'Setting C\n(Data-Poor Task)',
    'D': 'Setting D\n(Correlated, $\\rho_{\\text{true}}=0.5$)',
}
DESC = {'A': 'High heterogeneity ($\\sigma=1.5$)',
        'B': 'Low heterogeneity ($\\sigma=0.3$)',
        'C': 'Data-poor task ($p_0=0.05$)',
        'D': 'Correlated deviations ($\\rho_{\\text{true}}=0.5$)'}

# ─────────────────────────────────────────────
# Load main results
# ─────────────────────────────────────────────
results, T = {}, None
for s in ['A', 'B', 'C', 'D']:
    fpath = os.path.join(args.outdir, f'setting_{s}_regret.npz')
    if not os.path.exists(fpath):
        print(f"NOTE: {fpath} not found — skipping Setting {s}")
        continue
    d = np.load(fpath)
    res = {'HierTS' : d['hier'],
           'IndepTS': d['indep'],
           'PoolTS' : d['pooled']}
    if 'linucb' in d.files:
        res['LinUCB'] = d['linucb']
    results[s] = res
    T = d['hier'].shape[1]
    n_methods = len(res)
    print(f"Loaded Setting {s}: {d['hier'].shape[0]} trials, "
          f"T={T}, methods={list(res.keys())}")

if not results:
    print("No results found. Run simulation.py first.")
    raise SystemExit(1)

available = list(results.keys())
time_axis = np.arange(1, T + 1)

# ─────────────────────────────────────────────
# Figure 1: regret curves (one panel per available setting)
# ─────────────────────────────────────────────
n_set = len(available)
fig, axes = plt.subplots(1, n_set, figsize=(5.2 * n_set, 4.5))
if n_set == 1: axes = [axes]
for ax, s in zip(axes, available):
    for m in METHODS:
        if m not in results[s]: continue
        arr  = results[s][m]
        mean = arr.mean(0)
        lo, hi = np.percentile(arr, [10, 90], 0)
        ax.plot(time_axis, mean, label=MLABELS[m],
                color=COLORS[m], lw=2, linestyle=LSTYLES[m])
        ax.fill_between(time_axis, lo, hi, color=COLORS[m], alpha=0.13)
    ax.set_title(SLABELS[s].replace('\n', ' '), pad=8)
    ax.set_xlabel('Round $t$'); ax.set_ylabel('Cumulative Regret $R_t$')
    ax.legend(loc='upper left'); ax.grid(True, alpha=0.25, linestyle=':')
    ax.set_xlim(1, T)
fig.suptitle('Cumulative Regret: Mean $\\pm$ 80\\% interval',
             fontsize=12, y=1.02)
plt.tight_layout()
for ext in ['pdf', 'png']:
    fig.savefig(os.path.join(args.outdir, 'figures',
                             f'fig1_regret_curves.{ext}'),
                bbox_inches='tight')
print("Saved fig1_regret_curves.pdf")
plt.close()

# ─────────────────────────────────────────────
# Figure 2 & paired test: per-task regret (Setting C only, if available)
# ─────────────────────────────────────────────
N_TASKS = 4; K_ARMS = 3; P_FEAT = 4

def make_betas_AB_C(rng, sigma=1.5):
    """Mirrors the DGP for settings A/B/C in simulation.py."""
    b0 = rng.normal(0, 1, (K_ARMS, P_FEAT))
    return b0[:, None, :] + rng.normal(0, sigma, (K_ARMS, N_TASKS, P_FEAT))

def ctx(rng):
    x = np.empty(P_FEAT); x[0] = 1.0; x[1:] = rng.normal(0, 1, P_FEAT - 1)
    return x

if 'C' in results:
    print("\nRunning per-task regret analysis for Setting C...")
    T_PT = 400; N_PT = 20; N_GIBBS_PT = 15
    TPROB_C = np.array([0.05, 0.35, 0.30, 0.30])

    pt_hier  = np.zeros((N_PT, N_TASKS, T_PT))
    pt_indep = np.zeros((N_PT, N_TASKS, T_PT))

    for trial in range(N_PT):
        rng_t = np.random.default_rng(trial + 100)
        bt = make_betas_AB_C(rng_t)
        h = HierProbitBandit (rng_t, K_ARMS, N_TASKS, P_FEAT)
        i = IndepProbitBandit(rng_t, K_ARMS, N_TASKS, P_FEAT)
        inst_h = np.zeros((N_TASKS, T_PT)); inst_i = np.zeros((N_TASKS, T_PT))
        for t in range(T_PT):
            x = ctx(rng_t); j = rng_t.choice(N_TASKS, p=TPROB_C)
            opt = max(ndtr(x @ bt[k, j]) for k in range(K_ARMS))

            a = h.select_arm(j, x); ph = ndtr(x @ bt[a, j])
            h.observe(a, j, x, int(rng_t.uniform() < ph)); h.update(N_GIBBS_PT)
            inst_h[j, t] = opt - ph

            a = i.select_arm(j, x); pi_ = ndtr(x @ bt[a, j])
            i.observe(a, j, x, int(rng_t.uniform() < pi_)); i.update(N_GIBBS_PT)
            inst_i[j, t] = opt - pi_

        pt_hier[trial]  = np.cumsum(inst_h, axis=1)
        pt_indep[trial] = np.cumsum(inst_i, axis=1)
        print(f"  Per-task trial {trial+1}/{N_PT} done")

    # ---- Paired Wilcoxon signed-rank test on task-0 final regret ----
    final_hier_t0  = pt_hier [:, 0, -1]
    final_indep_t0 = pt_indep[:, 0, -1]
    diff = final_hier_t0 - final_indep_t0
    try:
        w_stat, w_p = wilcoxon(diff, alternative='less')   # H1: Hier < Indep
        wstr = f"W={w_stat:.1f}, p={w_p:.4f}"
    except ValueError:
        wstr = "(test undefined: zero differences)"
    print(f"\nPaired Wilcoxon on task-0 final regret (HierTS < IndepTS): {wstr}")
    print(f"  mean diff = {diff.mean():.2f}, "
          f"median diff = {np.median(diff):.2f}")

    # Save Table 3
    with open(os.path.join(args.outdir, 'tables',
                           'table3_per_task_test.tex'), 'w') as f:
        f.write(r"""\begin{table}[ht]
\centering
\caption{Setting C, task 0 (rare): paired test of final cumulative regret
($T=400$, $S=%d$ trials). Difference $D_s = R_T^{\text{HierTS}} - R_T^{\text{IndepTS}}$;
negative values favour HierTS. The Wilcoxon signed-rank test uses the one-sided
alternative $H_1: \text{HierTS regret} < \text{IndepTS regret}$.}
\label{tab:per_task_test}
\begin{tabular}{lc}
\toprule
Statistic & Value \\
\midrule
Mean of $D_s$ & %.2f \\
Median of $D_s$ & %.2f \\
Wilcoxon $W$ & %.1f \\
One-sided $p$-value & %.4f \\
\bottomrule
\end{tabular}
\end{table}
""" % (N_PT, diff.mean(), np.median(diff), w_stat, w_p))
    print("Saved table3_per_task_test.tex")

    # ---- Figure 2 ----
    fig3, axes3 = plt.subplots(1, 2, figsize=(11, 4.5))
    tax = np.arange(1, T_PT + 1)
    for label, arr, color, ls in [
            ('HierTS (proposed)', pt_hier,  COLORS['HierTS'],  '-'),
            ('IndepTS',           pt_indep, COLORS['IndepTS'], '--')]:
        m  = arr[:, 0, :].mean(0)
        lo, hi = np.percentile(arr[:, 0, :], [10, 90], 0)
        axes3[0].plot(tax, m, label=label, color=color, lw=2, linestyle=ls)
        axes3[0].fill_between(tax, lo, hi, color=color, alpha=0.15)
    axes3[0].set_title(f'Task 0 (Rare, $p_0=0.05$)\n'
                       f'Wilcoxon: {wstr}', fontsize=11)
    axes3[0].set_xlabel('Round $t$'); axes3[0].set_ylabel('Cumulative Regret')
    axes3[0].legend(); axes3[0].grid(True, alpha=0.25, linestyle=':')

    for label, arr, color, ls in [
            ('HierTS (proposed)', pt_hier,  COLORS['HierTS'],  '-'),
            ('IndepTS',           pt_indep, COLORS['IndepTS'], '--')]:
        avg = arr[:, 1:, :].mean(1)
        m   = avg.mean(0)
        lo, hi = np.percentile(avg, [10, 90], 0)
        axes3[1].plot(tax, m, label=label, color=color, lw=2, linestyle=ls)
        axes3[1].fill_between(tax, lo, hi, color=color, alpha=0.15)
    axes3[1].set_title('Tasks 1–3 (Common, $p_j \\approx 0.32$)')
    axes3[1].set_xlabel('Round $t$'); axes3[1].set_ylabel('Cumulative Regret')
    axes3[1].legend(); axes3[1].grid(True, alpha=0.25, linestyle=':')
    fig3.suptitle('Setting C: Per-Task Cumulative Regret (HierTS vs IndepTS)',
                  fontsize=12, y=1.01)
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig3.savefig(os.path.join(args.outdir, 'figures',
                                  f'fig2_setting_c_per_task.{ext}'),
                     bbox_inches='tight')
    print("Saved fig2_setting_c_per_task.pdf")
    plt.close()

# ─────────────────────────────────────────────
# Figure 3: sensitivity sweep on Setting A
# ─────────────────────────────────────────────
print("\nRunning hyperparameter sensitivity (Setting A)...")
SENS_CONFIGS = {
    'lam':   [0.1, 0.5, 1.0, 2.0, 5.0],
    'a_sig': [0.5, 1.0, 2.0, 5.0],
}
DEFAULT = {'lam': 1.0, 'a_sig': 1.0, 'b_sig': 1.0}
N_SENS = args.n_sens; T_SENS = 200; N_GS = 10

def run_sens_trial(lam, a_sig, b_sig, rng):
    bt = make_betas_AB_C(rng, sigma=1.5)
    h = HierProbitBandit(rng, K_ARMS, N_TASKS, P_FEAT,
                         lam=lam, a_sig=a_sig, b_sig=b_sig)
    r = np.zeros(T_SENS)
    for t in range(T_SENS):
        x = ctx(rng); j = rng.choice(N_TASKS)
        opt = max(ndtr(x @ bt[k, j]) for k in range(K_ARMS))
        a = h.select_arm(j, x); ph = ndtr(x @ bt[a, j])
        h.observe(a, j, x, int(rng.uniform() < ph))
        h.update(N_GS)
        r[t] = opt - ph
    return np.cumsum(r)[-1]

sens_results = {}
for param, vals in SENS_CONFIGS.items():
    sens_results[param] = {'vals': vals, 'means': [], 'stds': []}
    for v in vals:
        cfg = dict(DEFAULT); cfg[param] = v
        if param == 'a_sig': cfg['b_sig'] = v
        finals = [run_sens_trial(cfg['lam'], cfg['a_sig'], cfg['b_sig'],
                                 np.random.default_rng(i + 200))
                  for i in range(N_SENS)]
        sens_results[param]['means'].append(np.mean(finals))
        sens_results[param]['stds'].append(np.std(finals))
        print(f"  {param}={v:.1f}: mean regret={np.mean(finals):.2f}")

fig4, axes4 = plt.subplots(1, len(SENS_CONFIGS),
                           figsize=(5.5 * len(SENS_CONFIGS), 4.5))
if len(SENS_CONFIGS) == 1: axes4 = [axes4]
PARAM_LABELS = {'lam':   r'$\lambda$ (global mean precision)',
                'a_sig': r'$a_\sigma = b_\sigma$ (IG shape/rate)'}
for ax, (param, res) in zip(axes4, sens_results.items()):
    vals  = np.array(res['vals'])
    means = np.array(res['means'])
    stds  = np.array(res['stds'])
    ax.errorbar(vals, means, yerr=stds, fmt='o-', color=COLORS['HierTS'],
                lw=2, capsize=5, markersize=7)
    def_val = DEFAULT[param]
    if def_val in list(vals):
        ax.axvline(def_val, color='gray', lw=1.5, linestyle='--', alpha=0.7,
                   label=f'Default ({def_val})')
    ax.set_xlabel(PARAM_LABELS.get(param, param))
    ax.set_ylabel(f'Final Regret at $T={T_SENS}$')
    ax.set_title(f'Sensitivity to {PARAM_LABELS.get(param, param)}')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25, linestyle=':')
    ax.set_xscale('log')
fig4.suptitle(f'HierTS Hyperparameter Sensitivity (Setting A, '
              f'{N_SENS} trials)', fontsize=12, y=1.01)
plt.tight_layout()
for ext in ['pdf', 'png']:
    fig4.savefig(os.path.join(args.outdir, 'figures',
                              f'fig3_sensitivity.{ext}'),
                 bbox_inches='tight')
print("Saved fig3_sensitivity.pdf")
plt.close()

# ─────────────────────────────────────────────
# Table 1: Main results
# ─────────────────────────────────────────────
methods_present = [m for m in METHODS
                   if any(m in results[s] for s in available)]
n_cols = len(methods_present)
header = ' & '.join([MLABELS[m] for m in methods_present])

lines = [r'\begin{table}[ht]', r'\centering',
         r'\caption{Mean cumulative regret at $T='+str(T)+r'$ ($\pm$ std) across '+
         str(results[available[0]]['HierTS'].shape[0]) +
         r' trials. Bold: lowest regret per setting.}',
         r'\label{tab:results}',
         r'\begin{tabular}{ll' + 'c' * n_cols + r'}', r'\toprule',
         r'Setting & Description & ' + header + r' \\', r'\midrule']
for s in available:
    vals = {m: (results[s][m][:, -1].mean(), results[s][m][:, -1].std())
            for m in methods_present if m in results[s]}
    if not vals: continue
    best = min(vals.keys(), key=lambda m: vals[m][0])
    cells = []
    for m in methods_present:
        if m not in vals:
            cells.append('—')
            continue
        mu, sd = vals[m]
        cell = f'${mu:.1f}\\pm{sd:.1f}$'
        if m == best: cell = r'\textbf{' + cell + r'}'
        cells.append(cell)
    lines.append(f'  {s} & {DESC[s]} & ' + ' & '.join(cells) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
table1 = '\n'.join(lines)
with open(os.path.join(args.outdir, 'tables', 'table1_results.tex'), 'w') as f:
    f.write(table1)
print("\nSaved table1_results.tex")
print(table1)

# ─────────────────────────────────────────────
# Table 2: Sensitivity
# ─────────────────────────────────────────────
lines2 = [r'\begin{table}[ht]', r'\centering',
          r'\caption{Hyperparameter sensitivity of HierTS (Setting A, $T='+str(T_SENS)+r'$, mean $\pm$ std across '+str(N_SENS)+r' trials). '
          r'Default values in bold.}',
          r'\label{tab:sensitivity}',
          r'\begin{tabular}{lcc}', r'\toprule',
          r'Parameter & Value & Final Regret \\', r'\midrule']
for param, res in sens_results.items():
    def_val = DEFAULT[param]
    for v, mu, sd in zip(res['vals'], res['means'], res['stds']):
        cell = f'${mu:.1f}\\pm{sd:.1f}$'
        vstr = f'{v:.1f}'
        if v == def_val:
            vstr = r'\textbf{' + vstr + r'}'
            cell = r'\textbf{' + cell + r'}'
        lines2.append(f'  {PARAM_LABELS.get(param,param)} & {vstr} & {cell} \\\\')
    lines2.append(r'\midrule')
lines2[-1] = r'\bottomrule'
lines2 += [r'\end{tabular}', r'\end{table}']
with open(os.path.join(args.outdir, 'tables', 'table2_sensitivity.tex'), 'w') as f:
    f.write('\n'.join(lines2))
print("\nSaved table2_sensitivity.tex")

print(f"\n{'='*60}")
print(f"{'Setting':<10} {'Method':<22} {'Mean':>8} {'Std':>7} {'Median':>8}")
print("-" * 60)
for s in available:
    for m in METHODS:
        if m not in results[s]: continue
        arr = results[s][m][:, -1]
        print(f"  {s:<8} {MLABELS[m]:<22} {arr.mean():>8.2f} {arr.std():>7.2f} {np.median(arr):>8.2f}")
    print()
print("\nAll outputs saved to:", args.outdir)
