"""
samplers.py
===========
Shared sampler implementations imported by both simulation.py and
make_plots.py. Having a single source-of-truth eliminates the risk of
the two scripts drifting apart (which had been a real concern in the
previous codebase).

Provides:
  - HierProbitBandit  : full hierarchical model with compound-symmetry prior
  - IndepProbitBandit : per-(arm,task) independent probit
  - PooledProbitBandit: per-arm pooled probit (no task labels)
  - LinUCBBandit      : frequentist Linear UCB baseline (Li et al. 2010)

All Gibbs samplers use:
  * vectorized truncated-normal sampling (scipy supports array args natively)
  * full blocked-Gibbs Z resampling at the start of every sweep
  * MH-within-Gibbs for rho with the corrected 1/sigma^2 factor

Constants K_ARMS, N_TASKS, P_FEAT are imported by callers and passed in.
"""

import numpy as np
from scipy.stats import truncnorm
from scipy.special import ndtr



# Vectorized truncated-normal sampling
def stn_pos_vec(mus, rng=None):
    """Sample from TN(mu, 1) truncated to (0, inf) — vectorized."""
    mus = np.asarray(mus, dtype=float)
    if mus.size == 0:
        return mus
    return truncnorm.rvs(-mus, np.inf, loc=mus, scale=1.0, random_state=rng)

def stn_neg_vec(mus, rng=None):
    """Sample from TN(mu, 1) truncated to (-inf, 0) — vectorized."""
    mus = np.asarray(mus, dtype=float)
    if mus.size == 0:
        return mus
    return truncnorm.rvs(-np.inf, -mus, loc=mus, scale=1.0, random_state=rng)

def stn_pos_scalar(mu, rng=None):
    return float(truncnorm.rvs(-mu, np.inf, loc=mu, scale=1.0, random_state=rng))

def stn_neg_scalar(mu, rng=None):
    return float(truncnorm.rvs(-np.inf, -mu, loc=mu, scale=1.0, random_state=rng))



# Compound-symmetry helpers
def cs_inv(s, r, p):
    """Inverse of sigma2 * [(1-r)I + r 11^T]."""
    c = 1.0/(s*(1-r))
    d = r/(s*(1+(p-1)*r)*(1-r))
    return c*np.eye(p) - d*np.ones((p, p))

def cs_full(s, r, p):
    """Full covariance sigma2 * [(1-r)I + r 11^T]."""
    return s*((1-r)*np.eye(p) + r*np.ones((p, p)))

def rho_to_eta(rho, p):
    lo = -1.0/(p-1)
    return np.log((rho - lo)/(1 - rho))

def eta_to_rho(eta, p):
    lo = -1.0/(p-1)
    e  = np.exp(eta)
    return (lo + e)/(1 + e)



# Hierarchical Probit Bandit (full model)
class HierProbitBandit:
    """
    Full blocked Gibbs sampler for the hierarchical probit model.

    Steps per sweep:
      1. Resample all latent Z_t from their truncated-normal full
         conditionals (vectorized).
      2. Update beta_{k,j} for every (k,j).
      3. Update beta0_k for every k.
      4. Update sigma2_k for every k (conjugate IG).
      5. MH-within-Gibbs for rho_k, with 1/sigma^2 in the log full
         conditional and Jacobian correction in eta-space.
    """

    def __init__(self, rng, K, N, P,
                 lam=1.0, a_sig=1.0, b_sig=1.0, mh_sd=0.3,
                 tie_break_eps=1e-9):
        self.rng = rng
        self.K, self.N, self.P = K, N, P
        self.lam   = lam
        self.a_sig = a_sig
        self.b_sig = b_sig
        self.mh_sd = mh_sd
        self.tie_break_eps = tie_break_eps

        self.beta  = np.zeros((K, N, P))
        self.beta0 = np.zeros((K, P))
        self.sig2  = np.ones(K)
        self.rho   = np.zeros(K)

        self.X = {(k,j): [] for k in range(K) for j in range(N)}
        self.Y = {(k,j): [] for k in range(K) for j in range(N)}
        self.Z = {(k,j): np.array([]) for k in range(K) for j in range(N)}

    # bandit API 
    def select_arm(self, j, x):
        scores = np.array([ndtr(x @ self.beta[k, j]) for k in range(self.K)])
        # Tiny noise breaks first-round ties (all-zero beta -> all 0.5)
        if self.tie_break_eps > 0:
            scores = scores + self.rng.uniform(
                -self.tie_break_eps, self.tie_break_eps, size=self.K)
        return int(np.argmax(scores))

    def observe(self, k, j, x, y):
        self.X[(k,j)].append(x.copy())
        self.Y[(k,j)].append(int(y))
        mu = float(x @ self.beta[k, j])
        z  = (stn_pos_scalar(mu, self.rng) if y == 1
              else stn_neg_scalar(mu, self.rng))
        self.Z[(k,j)] = np.append(self.Z[(k,j)], z)

    # Gibbs internals
    def _resample_Z(self):
        for k in range(self.K):
            for j in range(self.N):
                if len(self.X[(k,j)]) == 0:
                    continue
                Xkj    = np.asarray(self.X[(k,j)])
                Ykj    = np.asarray(self.Y[(k,j)])
                mu_all = Xkj @ self.beta[k, j]
                pos = (Ykj == 1)
                neg = ~pos
                new_Z = np.empty_like(mu_all)
                if pos.any():
                    new_Z[pos] = stn_pos_vec(mu_all[pos], self.rng)
                if neg.any():
                    new_Z[neg] = stn_neg_vec(mu_all[neg], self.rng)
                self.Z[(k,j)] = new_Z

    def gibbs_step(self):
        rng = self.rng
        K, N, P = self.K, self.N, self.P

        # Step 1: latent Z
        self._resample_Z()

        # Step 2: beta_{k,j}
        for k in range(K):
            Sinv = cs_inv(self.sig2[k], self.rho[k], P)
            for j in range(N):
                if len(self.X[(k,j)]) == 0:
                    self.beta[k, j] = rng.multivariate_normal(
                        self.beta0[k], cs_full(self.sig2[k], self.rho[k], P))
                    continue
                Xkj  = np.asarray(self.X[(k,j)])
                Zkj  = np.asarray(self.Z[(k,j)])
                Prec = Xkj.T @ Xkj + Sinv
                Cov  = np.linalg.inv(Prec)
                mu_p = Cov @ (Xkj.T @ Zkj + Sinv @ self.beta0[k])
                self.beta[k, j] = rng.multivariate_normal(mu_p, Cov)

        # Step 3: beta0_k
        for k in range(K):
            Sinv = cs_inv(self.sig2[k], self.rho[k], P)
            Prec = N * Sinv + self.lam * np.eye(P)
            Cov  = np.linalg.inv(Prec)
            mu_p = Cov @ (Sinv @ self.beta[k].sum(axis=0))
            self.beta0[k] = rng.multivariate_normal(mu_p, Cov)

        # Step 4: sigma2_k  (conjugate IG)
        for k in range(K):
            d_kj = self.beta[k] - self.beta0[k]
            Rinv = cs_inv(1.0, self.rho[k], P)
            quad = float(sum(d_kj[j] @ Rinv @ d_kj[j] for j in range(N)))
            self.sig2[k] = 1.0 / rng.gamma(
                self.a_sig + N * P / 2.0,
                1.0 / (self.b_sig + quad / 2.0))

        # Step 5: rho_k via MH on eta-space  (with 1/sigma^2 factor)
        for k in range(K):
            rc = self.rho[k]
            ec = rho_to_eta(rc, P)
            ep = ec + rng.normal(0, self.mh_sd)
            rp = eta_to_rho(ep, P)

            def lfc(r, _k=k):
                if r <= -1.0/(P-1) or r >= 1:
                    return -np.inf
                d_kj = self.beta[_k] - self.beta0[_k]
                Rinv = cs_inv(1.0, r, P)
                quad = sum(d_kj[j] @ Rinv @ d_kj[j] for j in range(N))
                # log|R(r)| only — sigma^2 part is constant in r
                log_det_R = (P-1)*np.log(1-r) + np.log(1 + (P-1)*r)
                return -N/2.0 * log_det_R - quad / (2.0 * self.sig2[_k])

            lo = -1.0/(P-1)
            log_jac = (np.log(rp - lo) + np.log(1 - rp)
                       - np.log(rc - lo) - np.log(1 - rc))
            if np.log(rng.uniform() + 1e-300) < lfc(rp) - lfc(rc) + log_jac:
                self.rho[k] = rp

    def update(self, n):
        for _ in range(n):
            self.gibbs_step()



# Independent Probit Bandit (per-arm, per-task)
class IndepProbitBandit:
    """No information sharing across tasks. Independent N(0, I) prior."""

    def __init__(self, rng, K, N, P, tie_break_eps=1e-9):
        self.rng = rng
        self.K, self.N, self.P = K, N, P
        self.tie_break_eps = tie_break_eps
        self.beta = np.zeros((K, N, P))
        self.X = {(k,j): [] for k in range(K) for j in range(N)}
        self.Y = {(k,j): [] for k in range(K) for j in range(N)}
        self.Z = {(k,j): np.array([]) for k in range(K) for j in range(N)}

    def select_arm(self, j, x):
        scores = np.array([ndtr(x @ self.beta[k, j]) for k in range(self.K)])
        if self.tie_break_eps > 0:
            scores = scores + self.rng.uniform(
                -self.tie_break_eps, self.tie_break_eps, size=self.K)
        return int(np.argmax(scores))

    def observe(self, k, j, x, y):
        self.X[(k,j)].append(x.copy())
        self.Y[(k,j)].append(int(y))
        mu = float(x @ self.beta[k, j])
        z  = (stn_pos_scalar(mu, self.rng) if y == 1
              else stn_neg_scalar(mu, self.rng))
        self.Z[(k,j)] = np.append(self.Z[(k,j)], z)

    def _resample_Z(self):
        for k in range(self.K):
            for j in range(self.N):
                if len(self.X[(k,j)]) == 0: continue
                Xkj = np.asarray(self.X[(k,j)])
                Ykj = np.asarray(self.Y[(k,j)])
                mu_all = Xkj @ self.beta[k, j]
                pos = (Ykj == 1); neg = ~pos
                new_Z = np.empty_like(mu_all)
                if pos.any(): new_Z[pos] = stn_pos_vec(mu_all[pos], self.rng)
                if neg.any(): new_Z[neg] = stn_neg_vec(mu_all[neg], self.rng)
                self.Z[(k,j)] = new_Z

    def gibbs_step(self):
        rng = self.rng
        self._resample_Z()
        for k in range(self.K):
            for j in range(self.N):
                if len(self.X[(k,j)]) == 0:
                    self.beta[k, j] = rng.normal(0, 1, self.P)
                    continue
                Xkj  = np.asarray(self.X[(k,j)])
                Zkj  = np.asarray(self.Z[(k,j)])
                Prec = Xkj.T @ Xkj + np.eye(self.P)
                Cov  = np.linalg.inv(Prec)
                self.beta[k, j] = rng.multivariate_normal(Cov @ (Xkj.T @ Zkj), Cov)

    def update(self, n):
        for _ in range(n):
            self.gibbs_step()



# Pooled Probit Bandit (per-arm, all tasks pooled)
class PooledProbitBandit:
    """Ignores task labels. One probit model per arm."""

    def __init__(self, rng, K, N, P, tie_break_eps=1e-9):
        self.rng = rng
        self.K, self.N, self.P = K, N, P
        self.tie_break_eps = tie_break_eps
        self.beta = np.zeros((K, P))
        self.X = {k: [] for k in range(K)}
        self.Y = {k: [] for k in range(K)}
        self.Z = {k: np.array([]) for k in range(K)}

    def select_arm(self, j, x):
        scores = np.array([ndtr(x @ self.beta[k]) for k in range(self.K)])
        if self.tie_break_eps > 0:
            scores = scores + self.rng.uniform(
                -self.tie_break_eps, self.tie_break_eps, size=self.K)
        return int(np.argmax(scores))

    def observe(self, k, j, x, y):
        self.X[k].append(x.copy())
        self.Y[k].append(int(y))
        mu = float(x @ self.beta[k])
        z  = (stn_pos_scalar(mu, self.rng) if y == 1
              else stn_neg_scalar(mu, self.rng))
        self.Z[k] = np.append(self.Z[k], z)

    def _resample_Z(self):
        for k in range(self.K):
            if len(self.X[k]) == 0: continue
            Xk = np.asarray(self.X[k])
            Yk = np.asarray(self.Y[k])
            mu_all = Xk @ self.beta[k]
            pos = (Yk == 1); neg = ~pos
            new_Z = np.empty_like(mu_all)
            if pos.any(): new_Z[pos] = stn_pos_vec(mu_all[pos], self.rng)
            if neg.any(): new_Z[neg] = stn_neg_vec(mu_all[neg], self.rng)
            self.Z[k] = new_Z

    def gibbs_step(self):
        rng = self.rng
        self._resample_Z()
        for k in range(self.K):
            if len(self.X[k]) == 0:
                self.beta[k] = rng.normal(0, 1, self.P)
                continue
            Xk = np.asarray(self.X[k])
            Zk = np.asarray(self.Z[k])
            Prec = Xk.T @ Xk + np.eye(self.P)
            Cov  = np.linalg.inv(Prec)
            self.beta[k] = rng.multivariate_normal(Cov @ (Xk.T @ Zk), Cov)

    def update(self, n):
        for _ in range(n):
            self.gibbs_step()



# LinUCB baseline (Li et al. 2010)
class LinUCBBandit:
    """
    Disjoint LinUCB: one ridge regression per (arm, task) pair, with the
    standard upper confidence bound exploration term

        a_t = argmax_k  x^T theta_{k,j} + alpha * sqrt(x^T A_{k,j}^{-1} x).

    Reference: Li, Chu, Langford, Schapire, "A contextual-bandit approach
    to personalized news article recommendation", WWW 2010.
    Used here as a non-Bayesian comparator. Treats binary rewards as
    real-valued — standard practice in the LinUCB literature.
    """
    def __init__(self, rng, K, N, P, alpha=1.0):
        self.rng = rng
        self.K, self.N, self.P = K, N, P
        self.alpha = alpha
        self.A = {(k,j): np.eye(P) for k in range(K) for j in range(N)}
        self.b = {(k,j): np.zeros(P) for k in range(K) for j in range(N)}

    def select_arm(self, j, x):
        scores = np.empty(self.K)
        for k in range(self.K):
            A_inv = np.linalg.inv(self.A[(k, j)])
            theta = A_inv @ self.b[(k, j)]
            mean  = float(x @ theta)
            ucb   = self.alpha * float(np.sqrt(max(x @ A_inv @ x, 0.0)))
            scores[k] = mean + ucb
        return int(np.argmax(scores))

    def observe(self, k, j, x, y):
        self.A[(k, j)] += np.outer(x, x)
        self.b[(k, j)] += float(y) * x

    def update(self, n=None):
        # No MCMC — closed-form ridge update happens in observe().
        pass
