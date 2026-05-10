# Hierarchical Bayesian Probit Bandit

NumPy/SciPy implementation; no GPU required. Stan cross-check is optional and
needs `cmdstanpy`.

## Headline result

HierTS attains lower mean final regret than IndepTS in all four primary
simulation settings (averaged over 20 trials) and avoids the catastrophic
failures of PoolTS. LinUCB is competitive at this modest horizon. A control
setting (D′) shows the partial-pooling advantage comes from shrinkage toward
the shared mean β_{k0} rather than from learning ρ_k.

## Quick start

```bash
git clone https://github.com/Zhanghai0/hier-probit-bandit.git
cd hier-probit-bandit
python -m venv bandit_env
source bandit_env/bin/activate
pip install -r requirements.txt
```

## Reproduce

```bash
sbatch slurm/run_simulation.sh         # main settings A/B/C/D
sbatch slurm/run_setting_d_prime.sh    # correlation control
sbatch slurm/run_m_ablation.sh         # Gibbs-sweep ablation
sbatch slurm/run_stan_validation.sh    # MCMC diagnostics + Stan cross-check
python  code/make_plots.py --outdir results
```

Saved seeds give deterministic results. Main simulation runs in a few hours on
1 CPU core; the M-ablation at M=100 is the slowest piece (~80 min).

## Paper artifact map

| Paper element        | Source                                      |
|----------------------|---------------------------------------------|
| Table 1, Figure 1    | `results/setting_{A,B,C,D}_regret.npz` → `make_plots.py` |
| Table 2, Figure 2    | per-task rerun in `make_plots.py`           |
| Table 3 (Setting D′) | `code/setting_d_prime.py`                   |
| Table 4              | sensitivity sweep in `make_plots.py`        |
| Table 5 (M-ablation) | `code/m_ablation.py`                        |
| Table 6              | `code/stan_validation.py`                   |
| Figure 3             | `code/stan_validation.py` (Stan vs Gibbs)   |

Supplementary figures not in the paper (`results/figures/`):
`fig3_sensitivity.pdf` (Table 4 as a plot), `fig5_traceplots.pdf`,
`fig5b_acf.pdf` (4-chain trace and autocorrelation diagnostics).

## Files

- `code/samplers.py` — HierTS, IndepTS, PoolTS, LinUCB
- `code/simulation.py` — main experiment driver
- `code/make_plots.py` — figures and LaTeX tables
- `code/setting_d_prime.py` — D′ correlation control
- `code/m_ablation.py` — Gibbs-sweep ablation
- `code/stan_validation.py` — MCMC diagnostics and Stan NCP cross-check
- `code/hierarchical_probit_bandit.stan` — non-centered Stan model
- `slurm/` — SLURM batch scripts (CPU jobs, submit from repo root)
- `results/` — saved outputs used in the paper

Generated logs, virtual environments, and checkpoints are ignored by
`.gitignore`.