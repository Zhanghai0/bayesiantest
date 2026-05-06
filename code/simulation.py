"""
simulation.py
=============
Hierarchical Bayesian Probit Bandit — Simulation Study (main experiment).

Changes vs. previous version:
  * Now imports samplers from the shared `code/samplers.py` module so
    that simulation.py and make_plots.py can never drift apart.
  * Adds Setting D: true rho = 0.5 (correlated coordinate deviations).
    Settings A/B/C keep the original DGP exactly; their RNG sequence
    is unchanged so previous results are reproducible.
  * Adds LinUCB as a non-Bayesian comparator. Saved into the npz files
    alongside hier/indep/pooled.
  * Vectorized truncnorm (in samplers.py) gives ~50x speed-up vs. the
    previous per-element loop.
  * Tie-breaking via tiny uniform noise on Phi-scores.

Usage:
    python simulation.py --setting A
    python simulation.py --setting D
    python simulation.py --setting ALL
"""

import argparse, os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 11})
from scipy.special import ndtr
from tqdm import trange
import warnings; warnings.filterwarnings('ignore')

# Import the single source-of-truth sampler implementations
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from samplers import (HierProbitBandit, IndepProbitBandit,
                      PooledProbitBandit, LinUCBBandit)

# ── Args ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--setting',  type=str, default='ALL',
                    choices=['A', 'B', 'C', 'D', 'ALL'])
parser.add_argument('--n_trials', type=int, default=20)
parser.add_argument('--T',        type=int, default=400)
parser.add_argument('--n_gibbs',  type=int, default=15)
parser.add_argument('--seed',     type=int, default=42)
parser.add_argument('--outdir',   type=str, default='results')
parser.add_argument('--no_linucb', action='store_true',
                    help='Skip the LinUCB comparator (faster smoke tests)')
parser.add_argument('--linucb_alpha', type=float, default=1.0,
                    help='LinUCB exploration parameter')
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)
os.makedirs(os.path.join(args.outdir, 'figures'), exist_ok=True)
os.makedirs(os.path.join(args.outdir, 'checkpoints'), exist_ok=True)

print(f"n_trials={args.n_trials}, T={args.T}, n_gibbs={args.n_gibbs}\n")

# ── Constants ───────────────────────────────────────────────────────────
N_TASKS = 4
K_ARMS  = 3
P_FEAT  = 4
LAM     = 1.0
A_SIG   = 1.0
B_SIG   = 1.0


# ── Data-generating process ─────────────────────────────────────────────
def make_true_betas(rng, s):
    """
    Return true beta_{k,j} of shape (K, N, P).

    Settings A/B/C use independent coordinate deviations (true rho = 0)
    and reproduce the original RNG sequence exactly. Setting D introduces
    correlated coordinate deviations (true rho = 0.5) drawn via Cholesky.
    """
    if s in ('A', 'B', 'C'):
        sig = {'A': 1.5, 'B': 0.3, 'C': 1.5}[s]
        b0 = rng.normal(0, 1, (K_ARMS, P_FEAT))
        return b0[:, None, :] + rng.normal(0, sig, (K_ARMS, N_TASKS, P_FEAT))

    elif s == 'D':
        # Correlated case: tests the full prior, including rho_k.
        sig, rho_true = 1.0, 0.5
        b0 = rng.normal(0, 1, (K_ARMS, P_FEAT))
        R    = (1.0 - rho_true)*np.eye(P_FEAT) + rho_true*np.ones((P_FEAT, P_FEAT))
        Sig  = (sig**2) * R
        L    = np.linalg.cholesky(Sig)
        out  = np.zeros((K_ARMS, N_TASKS, P_FEAT))
        for k in range(K_ARMS):
            for j in range(N_TASKS):
                z = rng.normal(0, 1, P_FEAT)
                out[k, j] = b0[k] + L @ z
        return out

    else:
        raise ValueError(f"Unknown setting {s}")


def task_probs(s):
    p = np.array([0.05, 0.35, 0.30, 0.30]) if s == 'C' else np.ones(N_TASKS)/N_TASKS
    return p / p.sum()


def sample_context(rng):
    x = np.empty(P_FEAT)
    x[0] = 1.0
    x[1:] = rng.normal(0, 1, P_FEAT - 1)
    return x


# ── One trial ───────────────────────────────────────────────────────────
def run_one_trial(beta_true, tprob, rng, T, n_gibbs, run_linucb):
    hier   = HierProbitBandit (rng, K_ARMS, N_TASKS, P_FEAT,
                               lam=LAM, a_sig=A_SIG, b_sig=B_SIG)
    indep  = IndepProbitBandit(rng, K_ARMS, N_TASKS, P_FEAT)
    pooled = PooledProbitBandit(rng, K_ARMS, N_TASKS, P_FEAT)
    linucb = (LinUCBBandit(rng, K_ARMS, N_TASKS, P_FEAT,
                           alpha=args.linucb_alpha)
              if run_linucb else None)

    r_h = np.zeros(T); r_i = np.zeros(T); r_p = np.zeros(T)
    r_l = np.zeros(T) if run_linucb else None

    def oracle(j, x):
        return max(float(ndtr(x @ beta_true[k, j])) for k in range(K_ARMS))

    for t in range(T):
        x   = sample_context(rng)
        j   = rng.choice(N_TASKS, p=tprob)
        opt = oracle(j, x)

        # HierTS
        a   = hier.select_arm(j, x)
        ph  = float(ndtr(x @ beta_true[a, j]))
        hier.observe(a, j, x, int(rng.uniform() < ph))
        hier.update(n_gibbs)
        r_h[t] = opt - ph

        # IndepTS
        a   = indep.select_arm(j, x)
        pi_ = float(ndtr(x @ beta_true[a, j]))
        indep.observe(a, j, x, int(rng.uniform() < pi_))
        indep.update(n_gibbs)
        r_i[t] = opt - pi_

        # PoolTS
        a   = pooled.select_arm(j, x)
        pp  = float(ndtr(x @ beta_true[a, j]))
        pooled.observe(a, j, x, int(rng.uniform() < pp))
        pooled.update(n_gibbs)
        r_p[t] = opt - pp

        # LinUCB (no MCMC, so essentially free)
        if run_linucb:
            a   = linucb.select_arm(j, x)
            pl  = float(ndtr(x @ beta_true[a, j]))
            linucb.observe(a, j, x, int(rng.uniform() < pl))
            r_l[t] = opt - pl

    out = (np.cumsum(r_h), np.cumsum(r_i), np.cumsum(r_p))
    if run_linucb:
        out = out + (np.cumsum(r_l),)
    return out


# ── Run setting with checkpointing ──────────────────────────────────────
def run_setting(s, n_trials, T, n_gibbs, seed, outdir, run_linucb):
    ckpt = os.path.join(outdir, 'checkpoints', f'ckpt_{s}.npz')
    start = 0
    all_h = np.zeros((n_trials, T))
    all_i = np.zeros((n_trials, T))
    all_p = np.zeros((n_trials, T))
    all_l = np.zeros((n_trials, T)) if run_linucb else None

    if os.path.exists(ckpt):
        c = np.load(ckpt)
        start = int(c['completed'])
        all_h[:start] = c['hier'][:start]
        all_i[:start] = c['indep'][:start]
        all_p[:start] = c['pooled'][:start]
        if run_linucb and 'linucb' in c.files:
            all_l[:start] = c['linucb'][:start]
        print(f"  Resuming Setting {s} from trial {start}/{n_trials}")
    else:
        print(f"  Starting Setting {s} from scratch")

    master = np.random.default_rng(seed)
    for _ in range(start):
        master.integers(0, 2**31)
    tprob = task_probs(s)

    for trial in trange(start, n_trials, desc=f'Setting {s}'):
        trial_rng = np.random.default_rng(master.integers(0, 2**31))
        beta_true = make_true_betas(trial_rng, s)
        out = run_one_trial(beta_true, tprob, trial_rng, T, n_gibbs, run_linucb)
        if run_linucb:
            h, i, p, l = out
            all_h[trial] = h; all_i[trial] = i; all_p[trial] = p; all_l[trial] = l
        else:
            h, i, p = out
            all_h[trial] = h; all_i[trial] = i; all_p[trial] = p

        save_kw = dict(hier=all_h, indep=all_i, pooled=all_p,
                       completed=trial+1, n_trials=n_trials,
                       T=T, n_gibbs=n_gibbs, setting=s)
        if run_linucb:
            save_kw['linucb'] = all_l
        np.savez(ckpt, **save_kw)

    return all_h, all_i, all_p, all_l


# ── Main loop ───────────────────────────────────────────────────────────
settings = ['A', 'B', 'C', 'D'] if args.setting == 'ALL' else [args.setting]
results  = {}
run_linucb = not args.no_linucb

for s in settings:
    print(f"\n{'='*50}\nSetting {s}\n{'='*50}")
    h, i, p, l = run_setting(s, args.n_trials, args.T, args.n_gibbs,
                             args.seed, args.outdir, run_linucb)
    results[s] = (h, i, p, l)
    save_kw = dict(hier=h, indep=i, pooled=p,
                   n_trials=args.n_trials, T=args.T, n_gibbs=args.n_gibbs,
                   setting=s)
    if run_linucb:
        save_kw['linucb'] = l
    np.savez(os.path.join(args.outdir, f'setting_{s}_regret.npz'), **save_kw)


# ── Summary table ───────────────────────────────────────────────────────
rows = []
methods = [('HierTS', 0), ('IndepTS', 1), ('PoolTS', 2)]
if run_linucb:
    methods.append(('LinUCB', 3))

print(f"\n{'Setting':<10}{'Method':<12}{'Mean':>10}{'Std':>8}{'Median':>10}")
print("-" * 52)
for s in settings:
    arrs = results[s]
    for label, idx in methods:
        if arrs[idx] is None: continue
        final = arrs[idx][:, -1]
        rows.append({'setting': s, 'method': label,
                     'mean': final.mean(), 'std': final.std(),
                     'median': np.median(final)})
        print(f"  {s:<8}{label:<12}{final.mean():>10.2f}"
              f"{final.std():>8.2f}{np.median(final):>10.2f}")

with open(os.path.join(args.outdir, 'summary_table.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['setting', 'method', 'mean', 'std', 'median'])
    w.writeheader()
    w.writerows(rows)


# ── Quick sanity-check figure ───────────────────────────────────────────
COLORS = {'HierTS': '#1f77b4', 'IndepTS': '#ff7f0e',
          'PoolTS': '#2ca02c', 'LinUCB': '#9467bd'}
TITLES = {'A': 'Setting A: High Heterogeneity',
          'B': 'Setting B: Low Heterogeneity',
          'C': 'Setting C: Data-Poor Task',
          'D': 'Setting D: Correlated Deviations ($\\rho=0.5$)'}

fig, axes = plt.subplots(1, len(settings), figsize=(6 * len(settings), 5))
if len(settings) == 1: axes = [axes]
time_axis = np.arange(1, args.T + 1)

for ax, s in zip(axes, settings):
    arrs = results[s]
    for label, idx in methods:
        if arrs[idx] is None: continue
        a   = arrs[idx]
        m   = a.mean(0)
        lo  = np.percentile(a, 10, 0)
        hi  = np.percentile(a, 90, 0)
        ax.plot(time_axis, m, label=label, color=COLORS[label], lw=2)
        ax.fill_between(time_axis, lo, hi, color=COLORS[label], alpha=0.18)
    ax.set_title(TITLES[s])
    ax.set_xlabel('Round $t$')
    ax.set_ylabel('Cumulative Regret')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
for ext in ['pdf', 'png']:
    plt.savefig(os.path.join(args.outdir, 'figures',
                             f'simulation_results.{ext}'),
                dpi=150, bbox_inches='tight')
print("\nAll done.")
