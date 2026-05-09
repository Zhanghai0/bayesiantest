# Hierarchical Bayesian Probit Bandit 

The main implementation is NumPy/SciPy based and does not require a GPU.

## Quick start

```bash
git clone https://github.com/Zhanghai0/hier-probit-bandit.git
cd hier-probit-bandit
python -m venv bandit_env
source bandit_env/bin/activate
pip install -r requirements.txt
```

## Reproduce results

```bash
sbatch slurm/run_simulation.sh
python code/make_plots.py --outdir results
sbatch slurm/run_setting_d_prime.sh
sbatch slurm/run_m_ablation.sh
sbatch slurm/run_stan_validation.sh
```

The SLURM scripts are CPU jobs and should be submitted from the repository root.

## Files

- `code/samplers.py`: HierTS, IndepTS, PoolTS, and LinUCB implementations.
- `code/simulation.py`: main simulations for Settings A/B/C/D.
- `code/make_plots.py`: paper figures and LaTeX tables.
- `code/setting_d_prime.py`: Setting D' control.
- `code/m_ablation.py`: Gibbs sweep ablation.
- `code/stan_validation.py`: sampler diagnostics and Stan cross-check.
- `results/`: saved results used in the paper.

## Notes

Generated logs, virtual environments, checkpoints, and local test outputs are
ignored by `.gitignore`.
