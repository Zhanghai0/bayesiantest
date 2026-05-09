// hierarchical_probit_bandit.stan
//
// Hierarchical Bayesian Probit Bandit — Stan implementation
// Used for SAMPLER VALIDATION only (not the online bandit loop).
//
// CHANGE vs. previous version: NON-CENTERED PARAMETERIZATION.
//   The previous version sampled beta_{k,j} directly from a Normal whose
//   covariance is sigma_k^2 R(rho_k). When sigma_k is small, the posterior
//   geometry exhibits the classic Neal-funnel pathology and NUTS reports
//   a substantial fraction of divergent transitions (~12% on the
//   T=200 dataset).
//
//   The non-centered parameterization (NCP) reparameterizes
//        beta_{k,j} = beta0_k + sigma_k * L(rho_k) z_{k,j},
//
//   where z_{k,j} ~ N(0, I_p) is sampled directly and L is the Cholesky
//   factor of R(rho_k). This decouples the posterior geometry of z from
//   sigma, removing the funnel and typically eliminating divergences.
//
// Model (with NCP):
//   z_{k,j} ~ N(0, I_p)                          [auxiliary]
//   beta_{k,j} = beta0_k + sigma_k * L(rho_k) z_{k,j}   [transformed]
//   Y_t ~ Bernoulli(Phi(X_t' beta_{k,j}))
//   beta0_k ~ N(0, lam^{-1} I)
//   sigma2_k ~ IG(a_sig, b_sig)
//   rho_k ~ Uniform(-1/(p-1), 1)

data {
  int<lower=1> T;
  int<lower=1> K;
  int<lower=1> N;
  int<lower=1> P;

  matrix[T, P] X;
  array[T] int<lower=1, upper=K> arm;
  array[T] int<lower=1, upper=N> task;
  array[T] int<lower=0, upper=1> Y;

  real<lower=0> lam;
  real<lower=0> a_sig;
  real<lower=0> b_sig;
}

parameters {
  // Non-centered auxiliary variables (the actual sampled quantities).
  array[K, N] vector[P] z;
  array[K] vector[P] beta0;
  vector<lower=0>[K] sigma2;
  // rho_raw: unconstrained, mapped to (-1/(P-1), 1) via sigmoid.
  vector[K] rho_raw;
}

transformed parameters {
  real lo = -1.0 / (P - 1.0);
  vector[K] rho;
  vector<lower=0>[K] sigma_k;          // stdev (positive square root)
  array[K, N] vector[P] beta;          // implied task-specific coefficients

  for (k in 1:K) {
    rho[k]     = lo + (1.0 - lo) * inv_logit(rho_raw[k]);
    sigma_k[k] = sqrt(sigma2[k]);
  }

  // Build L(rho_k) for each arm (compound symmetry Cholesky).
  // R(rho) = (1-rho) I + rho 1 1';  P x P PSD when rho in (lo, 1).
  for (k in 1:K) {
    matrix[P, P] R_k;
    for (i in 1:P) {
      for (j_ in 1:P) {
        R_k[i, j_] = (i == j_) ? 1.0 : rho[k];
      }
    }
    matrix[P, P] L_k = cholesky_decompose(R_k);

    for (j_ in 1:N) {
      beta[k, j_] = beta0[k] + sigma_k[k] * (L_k * z[k, j_]);
    }
  }
}

model {
  // Auxiliary z's: standard normal (the heart of the NCP).
  for (k in 1:K) {
    for (j_ in 1:N) {
      z[k, j_] ~ std_normal();
    }
  }

  // Hyperpriors.
  for (k in 1:K) {
    beta0[k] ~ normal(0, 1.0 / sqrt(lam));
    sigma2[k] ~ inv_gamma(a_sig, b_sig);
  }

  // Make rho ~ Uniform(lo, 1) by adding the log-Jacobian of the sigmoid map.
  // (The inv_logit map's Jacobian gives an implicit prior proportional to
  // 1/[(rho-lo)(1-rho)] ; multiplying by (rho-lo)(1-rho) makes it uniform.)
  for (k in 1:K) {
    target += log(rho[k] - lo) + log(1 - rho[k]);
  }

  // Likelihood.
  for (t in 1:T) {
    int k  = arm[t];
    int j_ = task[t];
    real eta = dot_product(X[t], beta[k, j_]);
    Y[t] ~ bernoulli(Phi(eta));
  }
}

generated quantities {
  vector[T] p_pred;
  vector[T] log_lik;
  for (t in 1:T) {
    int k  = arm[t];
    int j_ = task[t];
    real eta = dot_product(X[t], beta[k, j_]);
    p_pred[t]  = Phi(eta);
    log_lik[t] = bernoulli_lpmf(Y[t] | Phi(eta));
  }
}
