"""
M-ablation for Setting A.

Runs HierTS with Gibbs sweeps M in {15, 50, 100} on the Setting A data
generating process and writes final-regret summaries to
results/m_ablation.csv.

Run from the repository root:
    python code/m_ablation.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from samplers import HierProbitBandit


N_TASKS = 4
K_ARMS = 3
P_FEAT = 4
LAM = 1.0
A_SIG = 1.0
B_SIG = 1.0
SETTING_A_SIGMA = 1.5


def make_setting_a_betas(rng: np.random.Generator) -> np.ndarray:
    """Generate Setting A true beta_{k,j}: high heterogeneity, rho_true=0."""
    beta0 = rng.normal(0, 1, (K_ARMS, P_FEAT))
    deviations = rng.normal(0, SETTING_A_SIGMA, (K_ARMS, N_TASKS, P_FEAT))
    return beta0[:, None, :] + deviations


def sample_context(rng: np.random.Generator) -> np.ndarray:
    x = np.empty(P_FEAT)
    x[0] = 1.0
    x[1:] = rng.normal(0, 1, P_FEAT - 1)
    return x


def run_episode(
    bandit: HierProbitBandit,
    beta_true: np.ndarray,
    env_rng: np.random.Generator,
    T: int,
    n_gibbs: int,
) -> np.ndarray:
    """Run one Setting A episode and return cumulative regret."""
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


def run_one_trial(M: int, trial_seed: int, T: int) -> float:
    truth_rng = np.random.default_rng(trial_seed)
    env_rng = np.random.default_rng(trial_seed + 20_000)
    bandit_rng = np.random.default_rng(trial_seed + 10_000)

    beta_true = make_setting_a_betas(truth_rng)
    bandit = HierProbitBandit(
        bandit_rng,
        K_ARMS,
        N_TASKS,
        P_FEAT,
        lam=LAM,
        a_sig=A_SIG,
        b_sig=B_SIG,
    )

    regret_curve = run_episode(bandit, beta_true, env_rng, T=T, n_gibbs=M)
    return float(regret_curve[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Setting A M-ablation.")
    parser.add_argument("--M_values", type=int, nargs="+", default=[15, 50, 100])
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--T", type=int, default=400)
    parser.add_argument("--out_path", type=str, default="results/m_ablation.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    rows = []
    print(f"M-ablation: M in {args.M_values}, S={args.n_trials}, T={args.T}")
    print("-" * 64)

    for M in args.M_values:
        finals = np.zeros(args.n_trials)
        t_start = time.time()
        for s in range(args.n_trials):
            trial_seed = 1000 + s
            finals[s] = run_one_trial(M=M, trial_seed=trial_seed, T=args.T)
            print(
                f"  M={M:>3d}  trial {s + 1:>2d}/{args.n_trials}  "
                f"R_T={finals[s]:6.2f}"
            )

        elapsed = time.time() - t_start
        mean = float(finals.mean())
        std = float(finals.std(ddof=1))
        print(f"M={M}: mean={mean:.2f} std={std:.2f} ({elapsed:.1f}s)")
        print("-" * 64)
        rows.append(
            {
                "M": M,
                "mean_final_regret": mean,
                "std_final_regret": std,
                "n_trials": args.n_trials,
                "T": args.T,
                "elapsed_sec": elapsed,
                "raw": ";".join(f"{x:.4f}" for x in finals),
            }
        )

    with open(args.out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nLaTeX-ready table:")
    print(r"\begin{tabular}{lc}")
    print(r"\toprule")
    print(r"$M$ (sweeps per round) & Final Regret at $T=400$ \\")
    print(r"\midrule")
    for row in rows:
        print(
            f"${row['M']}$ & "
            f"${row['mean_final_regret']:.1f} \\pm {row['std_final_regret']:.1f}$ \\\\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
