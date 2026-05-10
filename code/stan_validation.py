"""
stan_validation.py
==================
Sampler Validation and MCMC Convergence Diagnostics.

PART 1 (PRIMARY):  Multi-chain Gibbs convergence diagnostics
    - 4 independent Gibbs chains with dispersed initial values
    - Split-R-hat, ESS (Geyer's IMSE), MCSE for every monitored parameter
    - Trace plots, autocorrelation, MH acceptance rate

PART 2 (SECONDARY): Independent cross-check against Stan NUTS
    - Stan model uses a NON-CENTERED PARAMETERIZATION (NCP) to alleviate
      the funnel geometry that previously caused ~12% divergent
      transitions. With NCP, divergences are typically reduced to <1%.
    - Compare marginal posteriors via overlaid KDEs and KS tests.

Outputs:
    fig5_traceplots.pdf            (multi-chain trace plots)
    fig5b_acf.pdf                  (autocorrelation per chain)
    fig4_posterior_validation.pdf  (Gibbs vs Stan KDE)
    mcmc_diagnostics.csv           (R-hat / ESS / MCSE for all params)

Changes vs. previous version:
    * Stan model file now uses NCP (see hierarchical_probit_bandit.stan).
    * Honest R-hat reporting: the R-hat range printed in the console and
      saved to disk now covers ALL monitored parameters (beta blocks,
      sigma2, rho), not just a beta sub-selection.
    * Imports the same Gibbs sampler implementation used by the bandit
      loop (samplers.HierProbitBandit), so the validation tests THE
      sampler, not a near-copy.
"""

import numpy as np
import os, sys
from scipy.stats import ks_2samp, gaussian_kde
from scipy.special import ndtr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 11})
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

# Use the same sampler as the bandit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from samplers import HierProbitBandit


try:
    import arviz as az
    HAVE_ARVIZ = True
except ImportError:
    HAVE_ARVIZ = False
    print("Note: arviz not installed — using built-in fallbacks for R-hat/ESS.\n")

# Stan backend
try:
    import cmdstanpy
    STAN_BACKEND = 'cmdstanpy'
except ImportError:
    STAN_BACKEND = None
    print("WARNING: cmdstanpy not found. PART 2 (Stan cross-check) will be skipped.\n")



RNG_SEED = 2025
T_VAL    = 200
K_ARMS   = 3
N_TASKS  = 4
P_FEAT   = 3
LAM      = 1.0
A_SIG    = 1.0
B_SIG    = 1.0

N_CHAINS      = 4
N_GIBBS_TOTAL = 3000
N_BURNIN      = 1500
N_KEEP        = N_GIBBS_TOTAL - N_BURNIN

STAN_CHAINS        = 4
STAN_ITER_WARMUP   = 500
STAN_ITER_SAMPLING = 500



# Generate fixed dataset (true rho = 0.3)
rng_dgp = np.random.default_rng(RNG_SEED)
sigma2_true = 1.0
rho_true    = 0.3
beta0_true  = rng_dgp.normal(0, 1, size=(K_ARMS, P_FEAT))

R_true     = (1 - rho_true)*np.eye(P_FEAT) + rho_true*np.ones((P_FEAT, P_FEAT))
Sigma_true = sigma2_true * R_true
L_true     = np.linalg.cholesky(Sigma_true)

beta_true = np.zeros((K_ARMS, N_TASKS, P_FEAT))
for k in range(K_ARMS):
    for j in range(N_TASKS):
        z = rng_dgp.normal(0, 1, P_FEAT)
        beta_true[k, j] = beta0_true[k] + L_true @ z

task_prob = np.ones(N_TASKS) / N_TASKS
X_obs    = np.empty((T_VAL, P_FEAT))
arm_obs  = np.empty(T_VAL, dtype=int)
task_obs = np.empty(T_VAL, dtype=int)
Y_obs    = np.empty(T_VAL, dtype=int)

for t in range(T_VAL):
    x = np.r_[1.0, rng_dgp.normal(0, 1, P_FEAT - 1)]
    j = rng_dgp.choice(N_TASKS, p=task_prob)
    k = rng_dgp.choice(K_ARMS)
    p_t = ndtr(x @ beta_true[k, j])
    y = int(rng_dgp.uniform() < p_t)
    X_obs[t] = x; arm_obs[t] = k; task_obs[t] = j; Y_obs[t] = y

print(f"Fixed dataset: T={T_VAL}, K={K_ARMS}, N={N_TASKS}, P={P_FEAT}")
print(f"True rho={rho_true}, sigma2={sigma2_true}, mean reward={Y_obs.mean():.3f}\n")



# Adapter: run the bandit Gibbs sampler in batch mode
def run_gibbs_chain(X, arm, task, Y, n_iter, n_burnin, rng, init_mode='prior'):
    """
    Run one Gibbs chain on a fixed dataset. Reuses HierProbitBandit
    internal Gibbs steps but populates the data containers up-front.
    Returns dict of posterior draws and MH acceptance rate for rho.
    """
    T_, K_, N_, P_ = len(Y), K_ARMS, N_TASKS, P_FEAT

    # Construct sampler
    h = HierProbitBandit(rng, K_, N_, P_, lam=LAM, a_sig=A_SIG, b_sig=B_SIG,
                         tie_break_eps=0.0)

    # Dispersed initialization for R-hat
    if init_mode == 'prior':
        h.beta0[:] = rng.normal(0, 1.0/np.sqrt(LAM), size=(K_, P_))
        h.sig2[:]  = 1.0/rng.gamma(A_SIG, 1.0/B_SIG, size=K_)
        h.rho[:]   = rng.uniform(-1.0/(P_-1), 1.0, size=K_)
    elif init_mode == 'overdispersed':
        h.beta0[:] = rng.normal(0, 3.0, size=(K_, P_))
        h.sig2[:]  = np.full(K_, 5.0) * rng.uniform(0.5, 2.0, size=K_)
        h.rho[:]   = rng.uniform(-1.0/(P_-1) + 0.05, 0.95, size=K_)
    elif init_mode == 'underdispersed':
        h.beta0[:] = rng.normal(0, 0.1, size=(K_, P_))
        h.sig2[:]  = np.full(K_, 0.1)
        h.rho[:]   = np.full(K_, 0.0)
    else:  # 'tail'
        h.beta0[:] = rng.normal(2.0, 0.5, size=(K_, P_))
        h.sig2[:]  = np.full(K_, 0.3)
        h.rho[:]   = np.full(K_, 0.7)

    # Initialize beta from prior
    from samplers import cs_full
    for k in range(K_):
        for j in range(N_):
            h.beta[k, j] = rng.multivariate_normal(
                h.beta0[k], cs_full(h.sig2[k], h.rho[k], P_))

    # Pre-load all observations
    for t in range(T_):
        h.observe(int(arm[t]), int(task[t]), X[t], int(Y[t]))

    s_beta  = np.zeros((n_iter - n_burnin, K_, N_, P_))
    s_beta0 = np.zeros((n_iter - n_burnin, K_, P_))
    s_sig2  = np.zeros((n_iter - n_burnin, K_))
    s_rho   = np.zeros((n_iter - n_burnin, K_))

    # Track MH acceptance: monkey-patch by counting rho changes
    n_proposed = 0
    n_accepted = 0
    store = 0
    for it in range(n_iter):
        rho_pre = h.rho.copy()
        h.gibbs_step()
        # MH proposed once per arm in Step 5
        n_proposed += K_
        n_accepted += int(np.sum(h.rho != rho_pre))
        if it >= n_burnin:
            s_beta[store]  = h.beta
            s_beta0[store] = h.beta0
            s_sig2[store]  = h.sig2
            s_rho[store]   = h.rho
            store += 1

    return {
        'beta':  s_beta,
        'beta0': s_beta0,
        'sig2':  s_sig2,
        'rho':   s_rho,
        'mh_accept_rate': n_accepted / max(1, n_proposed),
    }



# Built-in R-hat / ESS / MCSE
def compute_rhat(chains):
    chains = np.asarray(chains, dtype=float)
    M, N = chains.shape
    if N < 4: return np.nan
    half = N // 2
    splits = np.concatenate([chains[:, :half], chains[:, half:2*half]], axis=0)
    M2, N2 = splits.shape
    chain_means = splits.mean(axis=1)
    chain_vars  = splits.var(axis=1, ddof=1)
    B = N2 * np.var(chain_means, ddof=1)
    W = chain_vars.mean()
    if W <= 0: return np.nan
    var_hat = (N2 - 1)/N2 * W + B/N2
    return float(np.sqrt(var_hat / W))

def compute_ess(chains):
    chains = np.asarray(chains, dtype=float)
    M, N = chains.shape
    def acf_chain(x):
        x = x - x.mean()
        result = np.correlate(x, x, mode='full')[N-1:]
        if result[0] == 0: return np.zeros_like(result)
        return result / result[0]
    rho_avg = np.zeros(N)
    for m in range(M): rho_avg += acf_chain(chains[m])
    rho_avg /= M
    sum_pairs = []
    for t in range(0, N-1, 2):
        s = rho_avg[t] + rho_avg[t+1]
        if s < 0: break
        sum_pairs.append(s)
    if not sum_pairs: return float(M*N)
    for i in range(1, len(sum_pairs)):
        if sum_pairs[i] > sum_pairs[i-1]:
            sum_pairs[i] = sum_pairs[i-1]
    tau = -1 + 2*sum(sum_pairs)
    return float(M*N / max(tau, 1.0))

def compute_mcse(chains, ess=None):
    if ess is None: ess = compute_ess(chains)
    return float(np.asarray(chains).std(ddof=1) / np.sqrt(max(ess, 1.0)))

def diagnostics_table(samples_per_param, var_names):
    rows = []
    for name in var_names:
        arr = samples_per_param[name]
        if arr.ndim == 2:
            ess = compute_ess(arr)
            rows.append((name, arr.mean(), arr.std(ddof=1),
                         compute_mcse(arr, ess), ess, compute_rhat(arr)))
        else:
            shape = arr.shape[2:]
            for idx in np.ndindex(*shape):
                slc = arr[(slice(None), slice(None)) + idx]
                ess = compute_ess(slc)
                rows.append((f"{name}[{','.join(map(str, idx))}]",
                             slc.mean(), slc.std(ddof=1),
                             compute_mcse(slc, ess), ess, compute_rhat(slc)))
    return rows

def print_diag_table(rows):
    print(f"\n{'Parameter':<25} {'mean':>10} {'sd':>10} {'mcse':>10}"
          f" {'ess':>10} {'R-hat':>10}")
    print("-"*78)
    for name, mean, sd, mcse, ess, rhat in rows:
        rmark = "  WARN" if (np.isfinite(rhat) and rhat > 1.05) else ""
        emark = "  LOW"  if (np.isfinite(ess)  and ess  < 200)  else ""
        print(f"{name:<25} {mean:>10.3f} {sd:>10.3f} {mcse:>10.4f}"
              f" {ess:>10.1f} {rhat:>10.3f}{rmark}{emark}")



# 4 Gibbs chains
print("="*70)
print("PART 1: Multi-chain Gibbs convergence diagnostics")
print("="*70)
print(f"Running {N_CHAINS} chains x {N_GIBBS_TOTAL} iterations "
      f"({N_BURNIN} burn-in, {N_KEEP} retained)\n")

INIT_MODES = ['prior', 'overdispersed', 'underdispersed', 'tail'][:N_CHAINS]

chain_outputs = []
for c in range(N_CHAINS):
    print(f"  Chain {c+1}/{N_CHAINS} (init={INIT_MODES[c]})...", end=' ', flush=True)
    rng_c = np.random.default_rng(RNG_SEED + 100*(c+1))
    out = run_gibbs_chain(X_obs, arm_obs, task_obs, Y_obs,
                          n_iter=N_GIBBS_TOTAL, n_burnin=N_BURNIN,
                          rng=rng_c, init_mode=INIT_MODES[c])
    chain_outputs.append(out)
    print(f"done. MH accept rate = {out['mh_accept_rate']:.3f}")

def stack_chains(key):
    return np.stack([c[key] for c in chain_outputs], axis=0)

stacked = {
    'beta':  stack_chains('beta'),
    'beta0': stack_chains('beta0'),
    'sig2':  stack_chains('sig2'),
    'rho':   stack_chains('rho'),
}

acc_rates = np.array([c['mh_accept_rate'] for c in chain_outputs])
print(f"\nMH acceptance rate (rho): mean={acc_rates.mean():.3f}, "
      f"per-chain={[f'{a:.3f}' for a in acc_rates]}")
print("  (target band: 0.20 - 0.50)\n")

print("-"*70)
print("Convergence diagnostics for monitored parameters")
print("WARN = R-hat > 1.05;  LOW = ESS < 200")
print("-"*70)

diag_inputs = {
    'beta_k0_j0': stacked['beta'][:,:,0,0,:],
    'beta_k0_j1': stacked['beta'][:,:,0,1,:],
    'beta0_k0':   stacked['beta0'][:,:,0,:],
    'sigma2':     stacked['sig2'],
    'rho':        stacked['rho'],
}
rows = diagnostics_table(diag_inputs, list(diag_inputs.keys()))
print_diag_table(rows)


all_rhats = np.array([r[5] for r in rows if np.isfinite(r[5])])
all_esss  = np.array([r[4] for r in rows if np.isfinite(r[4])])
print(f"\nOverall (across all monitored params): "
      f"R-hat range = {all_rhats.min():.3f}–{all_rhats.max():.3f}; "
      f"ESS range = {all_esss.min():.0f}–{all_esss.max():.0f}")

# Save diagnostics
import csv as _csv
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'tables')
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, 'mcmc_diagnostics.csv'), 'w', newline='') as f:
    w = _csv.writer(f)
    w.writerow(['parameter', 'mean', 'sd', 'mcse', 'ess', 'rhat'])
    for r in rows: w.writerow(r)
print(f"\nDiagnostics saved to {out_dir}/mcmc_diagnostics.csv")



# Trace plots
CHAIN_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'figures')
os.makedirs(fig_dir, exist_ok=True)

fig_trace, axes_trace = plt.subplots(2, 2, figsize=(13, 7))
axes_trace = axes_trace.flatten()
trace_params = [
    (stacked['beta'][:,:,0,0,0], r'$\beta_{k=0,j=0}[0]$'),
    (stacked['beta'][:,:,0,0,1], r'$\beta_{k=0,j=0}[1]$'),
    (stacked['sig2'][:,:,0],     r'$\sigma^2_{k=0}$'),
    (stacked['rho'][:,:,0],      r'$\rho_{k=0}$'),
]
for ax, (samples_MN, label) in zip(axes_trace, trace_params):
    M, N = samples_MN.shape
    for c in range(M):
        ax.plot(samples_MN[c], lw=0.5, alpha=0.7,
                color=CHAIN_COLORS[c % len(CHAIN_COLORS)],
                label=f'Chain {c+1}')
    ax.axhline(samples_MN.mean(), color='black', lw=1.2, linestyle='--', alpha=0.7)
    rhat = compute_rhat(samples_MN); ess = compute_ess(samples_MN)
    ax.set_title(f'Trace: {label}  (R-hat={rhat:.3f}, ESS={ess:.0f})')
    ax.set_xlabel('Iteration (post burn-in)'); ax.set_ylabel('Value')
    ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig5_traceplots.pdf'),
            dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(fig_dir, 'fig5_traceplots.png'),
            dpi=150, bbox_inches='tight')
print("Saved fig5_traceplots.pdf")
plt.close()

# ACF
def autocorr(x, max_lag=100):
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x); out = np.zeros(max_lag + 1)
    var = (x*x).sum()
    if var == 0: return out
    for lag in range(max_lag + 1):
        out[lag] = 1.0 if lag == 0 else (x[:-lag]*x[lag:]).sum()/var
    return out

fig_acf, axes_acf = plt.subplots(2, 2, figsize=(13, 7))
axes_acf = axes_acf.flatten(); max_lag = 100
for ax, (samples_MN, label) in zip(axes_acf, trace_params):
    M, N = samples_MN.shape
    for c in range(M):
        acf = autocorr(samples_MN[c], max_lag=max_lag)
        ax.plot(np.arange(max_lag+1), acf, lw=1.2, alpha=0.8,
                color=CHAIN_COLORS[c % len(CHAIN_COLORS)],
                label=f'Chain {c+1}')
    ax.axhline(0,    color='black', lw=0.5)
    ax.axhline(0.05, color='gray',  lw=0.5, linestyle=':')
    ax.axhline(-0.05,color='gray',  lw=0.5, linestyle=':')
    ax.set_title(f'Autocorrelation: {label}')
    ax.set_xlabel('Lag'); ax.set_ylabel('ACF')
    ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.2, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig5b_acf.pdf'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(fig_dir, 'fig5b_acf.png'), dpi=150, bbox_inches='tight')
print("Saved fig5b_acf.pdf\n")
plt.close()



# Stan cross-check (NCP)
print("="*70)
print("PART 2: Independent cross-check against Stan NUTS (NCP model)")
print("="*70)
print("The Stan model uses a non-centered parameterization to alleviate")
print("the funnel geometry that previously caused divergent transitions.\n")

stan_samples = None
n_divergent  = None

if STAN_BACKEND == 'cmdstanpy':
    print("Compiling and running Stan model "
          "(4 chains, 500 warmup + 500 sampling)...")

    stan_data = {
        'T':    T_VAL,
        'K':    K_ARMS,
        'N':    N_TASKS,
        'P':    P_FEAT,
        'X':    X_obs.tolist(),
        'arm':  (arm_obs + 1).tolist(),
        'task': (task_obs + 1).tolist(),
        'Y':    Y_obs.tolist(),
        'lam':  LAM,
        'a_sig': A_SIG,
        'b_sig': B_SIG,
    }
    try:
        stan_path = Path(__file__).resolve().with_name("hierarchical_probit_bandit.stan")
        model = cmdstanpy.CmdStanModel(stan_file=str(stan_path))
        fit = model.sample(
            data=stan_data,
            chains=STAN_CHAINS,
            iter_warmup=STAN_ITER_WARMUP,
            iter_sampling=STAN_ITER_SAMPLING,
            seed=RNG_SEED,
            show_progress=True,
        )
        print("\n" + fit.diagnose())

        try:
            sampler_diag = fit.method_variables()
            divergent = sampler_diag.get('divergent__', None)
            if divergent is not None:
                n_divergent = int(np.sum(divergent))
                total = STAN_CHAINS * STAN_ITER_SAMPLING
                print(f"Stan divergent transitions: {n_divergent} / {total} "
                      f"= {100*n_divergent/total:.1f}%")
        except Exception as e:
            print(f"  (could not extract divergence count: {e})")

        stan_samples = {
            'beta':   fit.stan_variable('beta'),
            'beta0':  fit.stan_variable('beta0'),
            'sigma2': fit.stan_variable('sigma2'),
            'rho':    fit.stan_variable('rho'),
        }
    except Exception as e:
        print(f"Stan failed: {e}")
        STAN_BACKEND = None
else:
    print("Stan backend not available - skipping cross-check.")



# Posterior comparison (pooled Gibbs vs Stan)
gibbs_pool = {
    'beta':  stacked['beta'].reshape(-1, K_ARMS, N_TASKS, P_FEAT),
    'beta0': stacked['beta0'].reshape(-1, K_ARMS, P_FEAT),
    'sig2':  stacked['sig2'].reshape(-1, K_ARMS),
    'rho':   stacked['rho'].reshape(-1, K_ARMS),
}

print("\n" + "="*70)
print("Marginal posterior comparison: pooled Gibbs vs Stan")
print("="*70)

def cmp(name, g, s):
    out = {'name': name, 'g_mean': g.mean(), 'g_std': g.std(ddof=1)}
    if s is not None:
        ks_stat, ks_p = ks_2samp(g, s)
        out.update({'s_mean': s.mean(), 's_std': s.std(ddof=1),
                    'ks': ks_stat, 'p': ks_p})
    return out

comparisons = []
k_focus, j_focus = 0, 0
for feat in range(P_FEAT):
    g = gibbs_pool['beta'][:, k_focus, j_focus, feat]
    s = stan_samples['beta'][:, k_focus, j_focus, feat] if stan_samples else None
    comparisons.append((g, s, cmp(f"beta[{k_focus},{j_focus}][{feat}]", g, s)))
g_s2  = gibbs_pool['sig2'][:, k_focus]
s_s2  = stan_samples['sigma2'][:, k_focus] if stan_samples else None
comparisons.append((g_s2, s_s2, cmp(f"sigma2[{k_focus}]", g_s2, s_s2)))
g_rho = gibbs_pool['rho'][:, k_focus]
s_rho = stan_samples['rho'][:, k_focus] if stan_samples else None
comparisons.append((g_rho, s_rho, cmp(f"rho[{k_focus}]", g_rho, s_rho)))

print(f"{'Parameter':<22} {'Gibbs mean':>12} {'Gibbs sd':>10}"
      f" {'Stan mean':>12} {'Stan sd':>10} {'KS':>8} {'p':>8}")
print("-"*88)
for (g, s, r) in comparisons:
    if s is not None:
        print(f"{r['name']:<22} {r['g_mean']:>12.3f} {r['g_std']:>10.3f}"
              f" {r['s_mean']:>12.3f} {r['s_std']:>10.3f}"
              f" {r['ks']:>8.3f} {r['p']:>8.3f}")
    else:
        print(f"{r['name']:<22} {r['g_mean']:>12.3f} {r['g_std']:>10.3f}"
              f" {'(N/A)':>12} {'':>10} {'':>8} {'':>8}")



# Density plot
def kde_plot(ax, samples, label, color, ls='-'):
    kde = gaussian_kde(samples)
    lo, hi = np.percentile(samples, [0.5, 99.5])
    pad = 0.3*(hi - lo)
    xs = np.linspace(lo - pad, hi + pad, 300)
    ax.plot(xs, kde(xs), label=label, color=color, lw=2, linestyle=ls)
    ax.axvline(samples.mean(), color=color, lw=1, linestyle=':', alpha=0.6)

n_panels = P_FEAT + 2
n_cols   = min(n_panels, 3)
n_rows   = (n_panels + n_cols - 1) // n_cols
fig_kde, axes_kde = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
axes_kde = np.array(axes_kde).flatten()

titles = ([f"$\\beta_{{k=0,j=0}}[{f}]$" for f in range(P_FEAT)]
          + [r"$\sigma^2_{k=0}$", r"$\rho_{k=0}$"])
plot_data = [(g, s) for (g, s, _) in comparisons]

for i, (ax, (g, s), title) in enumerate(zip(axes_kde, plot_data, titles)):
    if i < P_FEAT:
        true_val = beta_true[k_focus, j_focus, i]
    elif i == P_FEAT:
        true_val = sigma2_true
    else:
        true_val = rho_true
    ax.axvline(true_val, color='black', lw=1.4, linestyle=':',
               label='True', alpha=0.75)
    kde_plot(ax, g, 'Gibbs (pooled)', '#1f77b4')
    if s is not None:
        kde_plot(ax, s, 'Stan (NUTS, NCP)', '#d62728', ls='--')
    ax.set_title(title)
    ax.set_xlabel('Parameter value'); ax.set_ylabel('Density')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

for ax in axes_kde[n_panels:]: ax.set_visible(False)

div_note = (f"  (Stan: {n_divergent} divergent / "
            f"{STAN_CHAINS*STAN_ITER_SAMPLING})") if n_divergent is not None else ""
fig_kde.suptitle(
    f'Sampler Cross-Check: Pooled Gibbs ({N_CHAINS} chains) vs Stan NUTS (NCP)\n'
    f'(T={T_VAL} obs, dotted black = true value){div_note}',
    fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig4_posterior_validation.pdf'),
            dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(fig_dir, 'fig4_posterior_validation.png'),
            dpi=150, bbox_inches='tight')
print("\nSaved fig4_posterior_validation.pdf")
plt.close()



# Final summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
n_bad_rhat = sum(1 for r in rows if np.isfinite(r[5]) and r[5] > 1.05)
n_low_ess  = sum(1 for r in rows if np.isfinite(r[4]) and r[4] < 200)
print(f"  Monitored parameters:          {len(rows)}")
print(f"  Parameters with R-hat > 1.05:  {n_bad_rhat}")
print(f"  Parameters with ESS  < 200:    {n_low_ess}")
print(f"  R-hat range (all params):      {all_rhats.min():.3f}–{all_rhats.max():.3f}")
print(f"  ESS  range (all params):       {all_esss.min():.0f}–{all_esss.max():.0f}")
print(f"  Mean MH acceptance rate:       {acc_rates.mean():.3f}")
print(f"  Stan divergent transitions:    "
      + (f"{n_divergent}" if n_divergent is not None else "(Stan not run)"))
print("\nDone.")
