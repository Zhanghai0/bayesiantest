"""
Setting D' control experiment.

Runs HierTS and IndepTS on a control matched to Setting D in scale
(sigma=1.0) but with rho_true=0.0. Results are written to
results/setting_d_prime.csv.

Run from the repository root:
    python code/setting_d_prime.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from samplers import HierProbitBandit, IndepProbitBandit


N_TASKS = 4
K_ARMS = 3
P_FEAT = 4
LAM = 1.0
A_SIG = 1.0
B_SIG = 1.0


def compound_symmetry_cov(p: int, sigma: float, rho: float) -> np.ndarray:
    R = (1.0 - rho) * np.eye(p) + rho * np.ones((p, p))
    return (sigma**2) * R


def make_correlated_betas(
    rng: np.random.Generator,
    sigma: float = 1.0,
    rho_true: float = 0.0,
) -> np.ndarray:
    """Generate beta_{k,j} with compound-symmetry task deviations."""
    beta0 = rng.normal(0, 1, (K_ARMS, P_FEAT))
    chol = np.linalg.cholesky(compound_symmetry_cov(P_FEAT, sigma, rho_true))
    beta = np.zeros((K_ARMS, N_TASKS, P_FEAT))

    for k in range(K_ARMS):
        for j in range(N_TASKS):
            beta[k, j] = beta0[k] + chol @ rng.normal(0, 1, P_FEAT)

    return beta


def sample_context(rng: np.random.Generator) -> np.ndarray:
    x = np.empty(P_FEAT)
    x[0] = 1.0
    x[1:] = rng.normal(0, 1, P_FEAT - 1)
    return x


def run_episode(
    bandit,
    beta_true: np.ndarray,
    env_rng: np.random.Generator,
    T: int,
    n_gibbs: int,
) -> np.ndarray:
    regrets = np.zeros(T)
    task_prob = np.ones(N_TASKS) / N_TASKS

    for t in range(T):
        x = sample_context(env_rng)
        j = int(env_rng.choice(N_TASKS, p=task_prob))
        arm_probs = np.array([ndtr(float(x @ beta_true[k, j])) for k in range(K_ARMS)])
        optimal = float(arm_probs.max())

        a = bandit.select_arm(j, x)
        chosen_prob = float(arm_probs[a])
        y = int(env_rng.uniform() < chosen_prob)

        bandit.observe(a, j, x, y)
        bandit.update(n_gibbs)
        regrets[t] = optimal - chosen_prob

    return np.cumsum(regrets)


def run_one_trial(method: str, trial_seed: int, T: int, n_gibbs: int) -> float:
    truth_rng = np.random.default_rng(trial_seed)
    env_rng = np.random.default_rng(trial_seed + 20_000)
    bandit_rng = np.random.default_rng(trial_seed + 10_000)

    beta_true = make_correlated_betas(truth_rng, sigma=1.0, rho_true=0.0)

    if method == "HierTS":
        bandit = HierProbitBandit(
            bandit_rng,
            K_ARMS,
            N_TASKS,
            P_FEAT,
            lam=LAM,
            a_sig=A_SIG,
            b_sig=B_SIG,
        )
    elif method == "IndepTS":
        bandit = IndepProbitBandit(bandit_rng, K_ARMS, N_TASKS, P_FEAT)
    else:
        raise ValueError(f"Unknown method: {method}")

    regret_curve = run_episode(bandit, beta_true, env_rng, T=T, n_gibbs=n_gibbs)
    return float(regret_curve[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Setting D' control.")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--T", type=int, default=400)
    parser.add_argument("--n_gibbs", type=int, default=15)
    parser.add_argument("--out_path", type=str, default="results/setting_d_prime.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    finals_hier = np.zeros(args.n_trials)
    finals_indep = np.zeros(args.n_trials)
    t_start = time.time()

    print(
        "Setting D' control: sigma=1.0, rho_true=0.0, "
        f"S={args.n_trials}, T={args.T}, M={args.n_gibbs}"
    )
    print("-" * 70)

    for s in range(args.n_trials):
        trial_seed = 5000 + s
        finals_hier[s] = run_one_trial("HierTS", trial_seed, args.T, args.n_gibbs)
        finals_indep[s] = run_one_trial("IndepTS", trial_seed, args.T, args.n_gibbs)
        print(
            f"  trial {s + 1:>2d}/{args.n_trials}  "
            f"HierTS={finals_hier[s]:6.2f}  "
            f"IndepTS={finals_indep[s]:6.2f}  "
            f"diff={finals_hier[s] - finals_indep[s]:+6.2f}"
        )

    elapsed = time.time() - t_start
    mean_h, std_h = float(finals_hier.mean()), float(finals_hier.std(ddof=1))
    mean_i, std_i = float(finals_indep.mean()), float(finals_indep.std(ddof=1))
    diffs = finals_hier - finals_indep
    gap = float(diffs.mean())

    try:
        w_stat, p_val = wilcoxon(diffs, alternative="less")
    except ValueError:
        w_stat, p_val = float("nan"), float("nan")

    print("-" * 70)
    print(f"HierTS:  {mean_h:.2f} +/- {std_h:.2f}")
    print(f"IndepTS: {mean_i:.2f} +/- {std_i:.2f}")
    print(f"Mean paired gap (HierTS - IndepTS): {gap:+.2f}")
    print(f"Wilcoxon W = {w_stat:.1f}, one-sided p = {p_val:.4f}")
    print(f"Total elapsed: {elapsed:.1f}s")

    with open(args.out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "HierTS_R_T", "IndepTS_R_T", "diff"])
        for s in range(args.n_trials):
            writer.writerow([s, finals_hier[s], finals_indep[s], diffs[s]])
        writer.writerow([])
        writer.writerow(["mean", mean_h, mean_i, gap])
        writer.writerow(["std", std_h, std_i, float(diffs.std(ddof=1))])
        writer.writerow(["wilcoxon_W", w_stat])
        writer.writerow(["wilcoxon_p_one_sided", p_val])

    print("\nLaTeX-ready table row:")
    print(
        f"D$'$ & $0.0$ & ${mean_h:.1f} \\pm {std_h:.1f}$ "
        f"& ${mean_i:.1f} \\pm {std_i:.1f}$ & ${gap:+.1f}$ \\\\"
    )


if __name__ == "__main__":
    main()
