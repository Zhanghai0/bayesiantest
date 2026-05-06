# Hierarchical Bayesian Probit Bandit — PHP2530

A hierarchical Bayesian probit bandit with task-adaptive partial pooling,
Albert–Chib data augmentation, and Thompson sampling. Compared against
IndepTS, PoolTS, and a frequentist LinUCB baseline.

## Repository layout

```
code/
    samplers.py                 — single source-of-truth sampler module
                                  (HierProbitBandit, IndepProbitBandit,
                                   PooledProbitBandit, LinUCBBandit)
    simulation.py               — main bandit simulation (Settings A/B/C/D)
    make_plots.py               — figures + LaTeX tables
    stan_validation.py          — multi-chain diagnostics + Stan NCP cross-check
    hierarchical_probit_bandit.stan
                                — Stan model (non-centered parameterization)
slurm/
    run_simulation.sh
    run_stan_validation.sh
results/                        — output directory
```

## Quick start on Oscar

```bash
git clone https://github.com/Zhanghai0/hier-probit-bandit.git
cd hier-probit-bandit
source bandit_env/bin/activate
mkdir -p logs results/figures results/tables results/checkpoints

# Main simulation (4 settings: A/B/C/D, 20 trials each, T=400)
sbatch slurm/run_simulation.sh

# After jobs finish: figures + LaTeX tables
python code/make_plots.py --outdir results

# Stan sampler validation (optional; uses NCP)
sbatch slurm/run_stan_validation.sh
```

## What's new

* **Setting D**: true `rho = 0.5` — directly exercises the compound-symmetry
  prior's correlation parameter, which Settings A/B/C left untested.
* **LinUCB comparator**: a non-Bayesian baseline (Li et al. 2010) is now
  saved alongside the three Bayesian methods.
* **Shared sampler module** (`samplers.py`): both `simulation.py` and
  `make_plots.py` import the same implementation. Earlier, two near-copies
  could drift apart silently.
* **Vectorized truncated normals**: ~280× faster Z-sampling on T=400 chains.
* **Stan NCP**: the Stan model now uses a non-centered parameterization to
  alleviate the funnel-geometry divergent transitions.
* **Paired Wilcoxon test** on Setting C task 0 final regret, replacing
  "intervals overlap so it's suggestive" with a defensible $p$-value.

## Resuming

Re-run the same `sbatch` command. Checkpoints in `results/checkpoints/`
are loaded automatically.
