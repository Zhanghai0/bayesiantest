# Hierarchical Bayesian Probit Bandit — PHP2530

A hierarchical Bayesian probit bandit with task-adaptive partial pooling,
Albert–Chib data augmentation, and Thompson sampling. Compared against
IndepTS, PoolTS, and a frequentist LinUCB baseline.

This repository accompanies the PHP2530 final paper *"A Hierarchical
Bayesian Probit Bandit for Task-Adaptive Tool Selection"* (Zhang, 2026)
and reproduces every figure and table in that paper.

## Repository layout

```
code/
    samplers.py                    — single source-of-truth sampler module
                                     (HierProbitBandit, IndepProbitBandit,
                                      PooledProbitBandit, LinUCBBandit)
    simulation.py                  — main bandit simulation (Settings A/B/C/D)
    make_plots.py                  — figures + LaTeX tables (Tables 1, 4)
    setting_d_prime.py             — Setting D' correlation control (Table 3)
    m_ablation.py                  — Gibbs sweeps M ∈ {15, 50, 100} (Table 5)
    stan_validation.py             — multi-chain diagnostics + Stan NCP
                                     cross-check (Table 6, Figure 3)
    hierarchical_probit_bandit.stan
                                   — Stan model (non-centered parameterization)

slurm/
    run_simulation.sh              — Settings A/B/C/D, 20 trials, T=400
    run_setting_d_prime.sh         — D' control
    run_m_ablation.sh              — M ∈ {15, 50, 100} ablation
    run_stan_validation.sh         — Gibbs convergence + Stan NCP

results/                           — output directory (created at runtime)
    setting_{A,B,C,D}_regret.npz   — main simulation arrays
    setting_d_prime.csv            — D' paired regret + Wilcoxon test
    m_ablation.csv                 — M-ablation summaries
    summary_table.csv              — flat summary across all settings
    figures/                       — fig1_regret_curves.pdf, fig2_*, fig4_*
    tables/                        — table1_results.tex, table2_sensitivity.tex,
                                     table3_per_task_test.tex,
                                     mcmc_diagnostics.csv
```

## How the experiments map to the paper

| Paper artifact         | Script                                          | Output file(s)                                                                                |
| ---------------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Table 1, Figure 1      | `code/simulation.py` + `code/make_plots.py`     | `results/tables/table1_results.tex`, `results/figures/fig1_regret_curves.pdf`                 |
| Table 2, Figure 2      | `code/make_plots.py` (Setting C per-task block) | `results/tables/table3_per_task_test.tex`, `results/figures/fig2_setting_c_per_task.pdf`      |
| Table 3 (D' control)   | `code/setting_d_prime.py`                       | `results/setting_d_prime.csv`                                                                 |
| Table 4 (sensitivity)  | `code/make_plots.py` (sensitivity block)        | `results/tables/table2_sensitivity.tex`, `results/figures/fig3_sensitivity.pdf`               |
| Table 5 (M-ablation)   | `code/m_ablation.py`                            | `results/m_ablation.csv`                                                                      |
| Table 6, Figure 3      | `code/stan_validation.py`                       | `results/tables/mcmc_diagnostics.csv`, `results/figures/fig4_posterior_validation.pdf`        |

## Quick start on Oscar

```bash
git clone https://github.com/Zhanghai0/hier-probit-bandit.git
cd hier-probit-bandit
python -m venv bandit_env
source bandit_env/bin/activate
pip install numpy scipy matplotlib tqdm
# Optional (Appendix A Part 2 only):
pip install cmdstanpy arviz
mkdir -p logs results/figures results/tables results/checkpoints
```

> **Note on cluster paths.** The SLURM scripts are configured for the
> author's Oscar setup. Before submitting, edit the `cd` line at the top
> of each `slurm/run_*.sh` to point to your local clone, or replace it
> with `cd "$SLURM_SUBMIT_DIR"` and submit from the repo root.

### Reproduce the full paper

```bash
# 1. Main simulation (Table 1, Figure 1) — Settings A/B/C/D, 20 trials, T=400
sbatch slurm/run_simulation.sh

# 2. After (1) finishes: figures + Tables 1/2/4 and Figures 1/2/sensitivity
python code/make_plots.py --outdir results

# 3. Setting D' control (Table 3)
sbatch slurm/run_setting_d_prime.sh

# 4. M-ablation (Table 5)
sbatch slurm/run_m_ablation.sh

# 5. Sampler validation (Table 6, Figure 3) — uses Stan NCP
sbatch slurm/run_stan_validation.sh
```

Steps 1, 3, 4, 5 are independent and can run in parallel.

### Reproduce a single experiment locally

```bash
# Setting A only, 5 trials (smoke test, ~2 minutes)
python code/simulation.py --setting A --n_trials 5 --T 400 --n_gibbs 15

# Setting D' control (~5 minutes for 20 trials)
python code/setting_d_prime.py --n_trials 20 --T 400 --n_gibbs 15

# M-ablation, single M value
python code/m_ablation.py --M_values 15 --n_trials 5 --T 400
```

## Resuming interrupted runs

`code/simulation.py` writes per-setting checkpoints to
`results/checkpoints/`. Re-running the same `sbatch slurm/run_simulation.sh`
resumes from the last completed trial.

## Headline results

* **HierTS dominates IndepTS** in every setting by 4–9 cumulative regret
  units, confirming partial pooling is uniformly beneficial.
* **HierTS avoids the catastrophic failures of PoolTS** on heterogeneous
  settings (A/C/D) while adapting to Setting B where pooling is appropriate.
* **The Setting D' control** (`ρ_true = 0`, scale-matched to Setting D)
  shows the HierTS–IndepTS gap is *larger* under independent deviations
  (−5.6, paired Wilcoxon `p = 0.002`) than under correlated deviations
  (−3.7). The partial-pooling advantage is therefore driven by shrinkage
  to the shared mean β_k0, **not** by the correlation parameter ρ_k —
  an honest negative result for the correlation modeling.
* **LinUCB is competitive** at T = 400: it wins three of four primary
  settings on raw regret. HierTS beats LinUCB only in Setting B, where
  shrinkage to a shared mean is the right inductive bias.

## What's new (relative to the proposal)

* **Setting D**: true `ρ = 0.5` — directly exercises the compound-symmetry
  prior's correlation parameter, which Settings A/B/C left untested.
* **Setting D' control**: `ρ_true = 0`, σ matched to D — disentangles the
  contributions of the shared mean and the correlation parameter (Table 3).
* **M-ablation**: confirms that M = 15 warm-started Gibbs sweeps per round
  suffice for the bandit's Thompson-sampling posterior approximation
  (Table 5).
* **LinUCB comparator**: a non-Bayesian baseline (Li et al. 2010) is now
  saved alongside the three Bayesian methods.
* **Shared sampler module** (`samplers.py`): both `simulation.py` and
  `make_plots.py` import the same implementation, eliminating drift.
* **Vectorized truncated normals**: ~280× faster Z-sampling on T=400 chains.
* **Stan NCP**: the Stan model uses a non-centered parameterization to
  alleviate the funnel-geometry divergent transitions.
* **Paired Wilcoxon test** on Setting C task 0 final regret, replacing
  "intervals overlap so it's suggestive" with a defensible p-value.

## Citation

If you use this code, please cite:

> Zhang, H. (2026). *A Hierarchical Bayesian Probit Bandit for
> Task-Adaptive Tool Selection.* PHP2530 final paper, Brown University.