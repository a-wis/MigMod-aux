// This Stan program defines a simple model for migration
// estimation that combines sending and receiving country
// data.
//
// Data follow a poisson distribution and a measurement model,
// missing data are imputed by a theory-driven gravity model.
// The input data are split between observed and missing, with
// appropriate indices for observed and missing.
data {
  // Data
  int<lower=0> N_obs;
  int<lower=0> N_all;
  int<lower=0> N_mis;
  int<lower=0> N_obs1;
  int<lower=0> N_mis1;
  int<lower=0> N_obs2;
  int<lower=0> N_mis2;
  int<lower=0> N_preds;
  int<lower=0> N_corr;
  int z1[N_obs1]; //immi
  int z2[N_obs2]; //emig
  int z_ind1[N_obs1];
  int z_ind2[N_obs2];
  matrix[N_obs+N_mis,N_preds] X_all;
  matrix[N_obs1,3] XL_i;
  matrix[N_obs2,3] XL_e;
  int acc_i[N_obs1];
  int acc_e[N_obs2];
  int corridor_all[N_all];
  // Priors
  real hypv_psi0;
  real hypv_psi1;
  real EL[2];
  real EH[2];
  real IL[2];
  real IH[2];
}
// The parameters accepted by the model.
parameters {
  vector[N_obs1] mu1;
  vector[N_obs2] mu2;
  vector<lower=0,upper=1>[3] lambda1;
  vector<lower=0,upper=1>[3] lambda2;
  vector[N_preds] beta;
  real<lower=0> sigma[2];
  real<lower=0> sigma_y;
  real<lower=0> sigma_psi_0;
  real<lower=0> sigma_psi_1;
  real psi_c0;
  real<lower=0,upper=1> psi_c1;
  vector[N_corr] q1raw;
  vector[N_corr] q2raw;
  vector[N_all] yraw;
}
transformed parameters {
  vector[N_all] y;
  vector[N_all] alpha_0;
  vector[N_all] alpha_1;
  vector<lower=0>[N_obs1] sigma_mu1;
  vector<lower=0>[N_obs2] sigma_mu2;
  vector<upper=0>[3] l_lambda1;
  vector<upper=0>[3] l_lambda2;
  vector[N_corr] psi_0;
  vector[N_corr] psi_1;

  //OD effects for true flow model
  psi_0 = psi_c0 + sigma_psi_0*q1raw;
  psi_1 = psi_c1 + sigma_psi_1*q2raw;
  alpha_0 = psi_0[corridor_all];
  alpha_1 = psi_1[corridor_all];
  // model for true flows - imputation (first year)
  y[1:N_corr] = alpha_0[1:N_corr] + X_all[1:N_corr,1:N_preds]*beta + sigma_y*yraw[1:N_corr];
  // model for true flows - imputation (other years)
  for (i in (N_corr+1):N_all){
    y[i] = alpha_0[i] + alpha_1[i]*y[i-N_corr] + X_all[i,1:N_preds]*beta + sigma_y*yraw[i];
  }

  // setting variance parameter for three groups of accuracy
  for (i in 1:N_obs1) sigma_mu1[i] = sigma[acc_i[i]];
  for (i in 1:N_obs2) sigma_mu2[i] = sigma[acc_e[i]];
  //log of undercounting parameters
  l_lambda1=log(lambda1);
  l_lambda2=log(lambda2);
}
// The model to be estimated. We model the output
// 'z1' and 'z2' to Poisson distributed with means 'mu1'
// and 'mu2', which then follow a log-normal distributions
// that have a common component 'y' - the true migration flows.
model {
  sigma_y ~ normal(0,5);
  sigma ~ student_t(2.5,0,25);
  sigma_psi_0 ~ normal(0,3.0);
  sigma_psi_1 ~ normal(0,1.0);
  beta ~ normal(0.0,5.0);
  lambda1[1] ~ beta(IL[2],IL[1]); //immigration
  lambda1[2] ~ beta(IL[2],IL[1]); //immigration
  lambda1[3] ~ beta(IH[2],IH[1]); //immigration
  lambda2[1] ~ beta(EL[2],EL[1]); //immigration
  lambda2[2] ~ beta(EL[2],EL[1]); //immigration
  lambda2[3] ~ beta(EH[2],EH[1]); //immigration

  psi_c0 ~ normal(0,hypv_psi0);
  psi_c1 ~ normal(0,hypv_psi1);

  q1raw ~ normal(0,1);
  q2raw ~ normal(0,1);
  yraw ~ normal(0,1);

  mu1 ~ normal(y[z_ind1] + XL_i*l_lambda1, sigma_mu1);
  mu2 ~ normal(y[z_ind2] + XL_e*l_lambda2, sigma_mu2);

  z1 ~ poisson_log(mu1);
  z2 ~ poisson_log(mu2);

}
generated quantities {
  vector[N_all] yl;

  yl = exp(y);
}
