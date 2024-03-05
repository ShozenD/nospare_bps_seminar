data {
  int<lower=0> T;              // Number of time points
  int<lower=0> J;              // Number of factors
  int<lower=0> F;              // Number of features
  real<lower=0> l;             // Leverage
  vector[T] Y;                  // Observed data
  matrix[T, F] X; 
  matrix[J, T] MF;
  matrix[J, T] VF;              // external covariate matrix for modeling weight (in-sample)
  // vector[J] MF[T];             // Mean of state vector for each time, for each factor
  // vector[J] VF[T];             // Variance of state vector for each time, for each factor
  int <lower=0> T_out;          // Number of out-sample time points
  // vector[J] MF_out[T_out];     // Mean of state vector for each time, for each factor (out-sample)
  // vector[J] VF_out[T_out];     // Variance of state vector for each time, for each factor (out-sample)
  matrix[J, T_out] MF_out;
  matrix[J, T_out] VF_out;
}

transformed data {
  // vector[J] TP[T]; // Array of vectors, each with an additional element for the intercept
  //vector[J] TP_out[T_out]; // Array of vectors, each with an additional element for the intercept
  // TP = MF;
  //TP_out = MF_out;
  matrix[J, T] TP = MF; // Transformed covariates for the in-sample data
  matrix[J, T_out] TP_out = MF_out; // Transformed covariates for the out-sample data
}

parameters {
  matrix[J, T] beta;              // State vector for each time, for each factor
  real<lower=0> sigma;            // Standard deviation of observation noise
  vector<lower=0>[J] tau;        // Standard deviations for the J noise sources
}

model {
  // 事前分布
  target += normal_lpdf(sigma | 0, 1);
  target += inv_gamma_lpdf(tau | 0.5, 0.5);

  for (t in 1:T) {
    // beta[t] ~ multi_normal(MF[t], diag_matrix(VF[t]));
    target += normal_lpdf(beta[:,t] | MF[:,t], VF[:,t]);
    // TP[t] ~ normal(MF[t], VF[t]);  // Assuming v_t[t] is a vector of variances for each factor at time t
    target += normal_lpdf(TP[:,t] | MF[:,t], VF[:,t]);

    if (t > 1) {
      target += normal_lpdf(beta[:,t] | beta[:,t-1], tau); // System evolution
    }

    // Likelihood
    // Y[t] ~ normal(dot_product(TP[t], beta[t]) * l, sigma);
    target += normal_lpdf(Y[t] | dot_product(TP[:,t], beta[:,t]) * l, sigma);
  }
}

generated quantities {
  vector[T] y_pred;                // Predicted values
  vector[T] y_var;                 // Predicted variance
  vector[T] y_pred_out;
  vector[T] y_var_out;

  for (t in 1:T) {
    y_pred[t] = dot_product(TP[:,t], beta[:,t]) * l; // Calculate predicted values
    y_var[t] = dot_product(VF[:,t], square(TP[:,t])) * square(l) + square(sigma);
  }

  for (t_out in 1:T_out) {
    // Assuming you have a way to generate TP_out[t_out], the transformed covariates for the out-sample data
    y_pred_out[t_out] = dot_product(TP_out[:,t_out], beta[:,T]) * l; // Last beta is used for prediction
    y_var_out[t_out] = dot_product(VF_out[:, t_out], square(TP_out[:, t_out])) * square(l) + square(sigma);
  }
}
