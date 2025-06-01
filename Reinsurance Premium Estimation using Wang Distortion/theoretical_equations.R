# Implementation of the theoretical equations in my thesis

# Simulating data from the Burr distribution
# The comparison with the original rburr function seems to be shape1 = lambda, 
# shape2 = tau and scale = alpha^(1 / tau).
rburr_custom <- function(n, alpha, lambda, tau) {
  U <- runif(n)
  X <- (alpha * (U^(- 1 / lambda) - 1))^(1 / tau)
  return(X)
}

# Tail probability calculator
tburr <- function(q, alpha, lambda, tau) {
  tail_prob <- (alpha / (alpha + q^tau))^lambda
  return(tail_prob)
}

tburr_distort <- function(q, alpha, lambda, tau, g_func) {
  distorted_tail_prob <- g_func(tburr(q, alpha, lambda, tau))
  return(distorted_tail_prob)
}

integrate_tburr <- function(x_pn, alpha, lambda, tau, g_func) {
  integrand <- function(q) {
    tburr_distort(q, alpha, lambda, tau, g_func)
  }

  result <- integrate(integrand, lower = x_pn, upper = Inf)
  return(result$value)
}


# Function for calculating the Hill estimator for a specific k.
hill_estimator <- function(k, sorted_data, n, X_kn) {
  gamma_k <- mean(log(sorted_data[(n-k+1):n]) - log(X_kn))
  return(gamma_k)
}

E_func <- function(s, k, sorted_data, N, X_kn) {
  value <- mean((sorted_data[(N-k+1):(N)] / X_kn)^s)
  return(value)
}

M_func <- function(alpha, k, sorted_data, N, X_kn) {
  value <- mean((log(sorted_data[(N-k+1):(N)]) - log(X_kn))^alpha)
  return(value)
}

T_func <- function(tau, k, sorted_data, N, X_kn) {
  M_1  <- M_func(1, k, sorted_data, N, X_kn)
  M_2  <- M_func(2, k, sorted_data, N, X_kn)
  M_3  <- M_func(3, k, sorted_data, N, X_kn)
  
  if (tau == 0) {
    numerator   <- log(M_1) - 1/2 * log(M_2 / 2)
    denominator <- 1/2 * log(M_2 / 2) - 1/3 * log(M_3 / 6)
  } else {
    numerator   <- M_1^(tau) - (M_2 / 2)^(tau / 2)
    denominator <- (M_2 / 2)^(tau / 2) - (M_3 / 6)^(tau / 3)
  }
  
  T_val <- numerator / denominator
  return(T_val)
}

rho_est <- function(tau, k, sorted_data, N, X_kn) {
  T_val   <- T_func(tau, k, sorted_data, N, X_kn)
  rho_val <- 3 * (T_val - 1) / (T_val - 3)
  return(rho_val)
}


# Function for estimating the tail probability of the threshold x_pn 
# for a specific k.
tail_probability <- function(k, n, X_kn, x_pn, gamma_k) {
  hat_pn <- k/n * (x_pn / X_kn)^(- 1 / gamma_k)
  return(hat_pn)
}

tail_probability_ext <- function(k, n, X_kn, x_pn, gamma_k, delta, kappa) {
  y <- x_pn / X_kn
  hat_pn_ext <- k/n * (y * (1 + delta - delta*y^kappa))^(- 1 / gamma_k)
  return(hat_pn_ext)
}

# Function for calculating the net premium principle 
# for a specific threshold x_pn, tail probability and Hill estimator.
net_premium <- function(x_pn, tail_prob_x_pn, gamma_k) {
  Pi_x_pn <- x_pn * tail_prob_x_pn * (1 / (1 / gamma_k - 1))
  return(Pi_x_pn)
}

# Function for calculating the distorted premium principle at a threshold x_pn,
# given a distortion operator g, the tail probability and the Hill estimator.
distorted_premium <- function(x_pn, tail_prob, gamma_k, g_func, beta) {
  Pi_g <- - gamma_k / (gamma_k + beta) * x_pn * g_func(tail_prob)
  return(Pi_g)
}

run_simulation <- function(N, g_func, beta, alpha, lambda, tau, x_pn, true_Pi, rho_k) {
  # Sample the data and sort it
  data <- rburr_custom(N, alpha, lambda, tau)
  sorted <- sort(data)
  z_alpha <- qnorm(0.975)
  
  true_rho <- - 1 / lambda

  if (true_rho == -0.5) {
    X_kn_rho <- sorted[N-rho_k]
    rho <- rho_est(-0.3, rho_k, sorted, N, X_kn_rho)
  } else if (true_rho == -1) {
    X_kn_rho <- sorted[N-rho_k]
    rho <- rho_est(0.25, rho_k, sorted, N, X_kn_rho)
  } else {
    X_kn_rho <- sorted[N-rho_k]
    rho <- rho_est(0.8, rho_k, sorted, N, X_kn_rho)
  }
  
  # Initialize vectors for the extreme value index, tail probability 
  # and premium estimates
  values <- data.frame(
    ks = seq(1, N-1),
    gamma_k    = NA_real_,
    gamma_k_bc = NA_real_,
    hat_pn     = NA_real_,
    hat_pn_bc  = NA_real_,
    premium    = NA_real_,
    premium_bc = NA_real_,
    coverage    = rep(0, N-1),
    coverage_bc = rep(0, N-1)
  )
  
  for (k in 1:(N-1)) {
    X_kn      <- sorted[N-k]
    gamma     <- hill_estimator(k, sorted, N, X_kn)
    E_val     <- E_func(rho / gamma, k, sorted, N, X_kn)
    delta     <- gamma * (1 - 2*rho) * (1 - rho)^3 * rho^(-4) * (E_val - 1 / (1 - rho))
    gamma_bc  <- gamma - delta * rho / (1 - rho)

    tail_prob    <- tail_probability(k, N, X_kn, x_pn, gamma)
    tail_prob_bc <- tail_probability(k, N, X_kn, x_pn, gamma_bc)

    log_dn    <- log(k / (N * tail_prob))
    log_dn_bc <- log(k / (N * tail_prob_bc))

    Pi        <- distorted_premium(x_pn, tail_prob, gamma, g_func, beta)
    Pi_lower  <- Pi * exp(log_dn / sqrt(k) * beta * z_alpha)
    Pi_upper  <- Pi * exp(- log_dn / sqrt(k) * beta * z_alpha)

    Pi_bc        <- distorted_premium(x_pn, tail_prob_bc, gamma_bc, g_func, beta)
    Pi_lower_bc  <- Pi_bc * exp(- log_dn_bc / sqrt(k) * beta * (1 - rho) / rho * z_alpha)
    Pi_upper_bc  <- Pi_bc * exp(log_dn_bc / sqrt(k) * beta * (1 - rho) / rho * z_alpha)

    if (rho >= 0) {
      delta        <- NA
      E_val        <- NA
      gamma_bc     <- NA
      tail_prob_bc <- NA
      Pi_bc        <- NA
      Pi_lower_bc  <- NA
      Pi_upper_bc  <- NA
    } else if (gamma_bc <= 0) {
      gamma_bc     <- NA
      tail_prob_bc <- NA
      Pi_bc        <- NA
      Pi_lower_bc  <- NA
      Pi_upper_bc  <- NA
    } else if (gamma_bc >= - beta - 0.05) {
      Pi_bc <- NA
      Pi_lower_bc <- NA
      Pi_upper_bc <- NA
    }

    if (!is.na(Pi_lower) && !is.na(Pi_upper) && !is.na(true_Pi)) {
      if (Pi_lower <= true_Pi && true_Pi <= Pi_upper) {
        values$coverage[k] <- 1
      }
    }

    if (!is.na(Pi_lower_bc) && !is.na(Pi_upper_bc) && !is.na(true_Pi)) {
      if (Pi_lower_bc <= true_Pi && true_Pi <= Pi_upper_bc) {
        values$coverage_bc[k] <- 1
      }
    }

    values$gamma_k[k]    <- gamma
    values$hat_pn[k]     <- tail_prob
    values$premium[k]    <- Pi
    values$gamma_k_bc[k] <- gamma_bc
    values$hat_pn_bc[k]  <- tail_prob_bc
    values$premium_bc[k] <- Pi_bc
  }
  
  return(values)
}

# Function to average over m simulations and return a data.frame
average_simulation <- function(N, distortion, g, beta, alpha, lambda, tau, true_Pi, rho_k_val, x_pn, m) {
  message(glue::glue("Running simulations for N={N}, g={distortion}, beta={beta}, rho={- 1 / lambda}"))
  
  sims <- replicate(
    m,
    run_simulation(N = N, g = g, beta = beta, 
                   alpha = alpha, lambda = lambda, tau = tau, 
                   x_pn = x_pn, true_Pi, rho_k = rho_k_val),
    simplify = FALSE
  )
  
  # Stack all simulation data.frames row-wise by column `ks`
  combined <- bind_rows(sims, .id = "sim_id")  # adds a simulation ID
  
  # Group by index `ks` and average over the simulations, ignoring NAs
  sims_avg <- combined %>%
    group_by(ks) %>%
    summarise(across(-sim_id, ~mean(.x, na.rm = TRUE))) %>%
    ungroup()
  
  # Reduce the m dataframes to 1 by averaging
  #sims_avg <- as.data.frame(reduce(sims, `+`) / m)  
  
  return(sims_avg)
}


# The below two functions are only to simulate estimates of rho from the Burr distribution.
simulate_rho <- function(N, alpha, lambda, tau) {
  # Sample the data and sort it
  data   <- rburr_custom(N, alpha, lambda, tau)
  sorted <- sort(data)
  
  # Initialize dataframe for estimates of rho at different rho0 values
  values <- data.frame(
    ks = seq(1, N-1),
    rho_m35  = NA_real_,  # for rho0 = -0.5
    rho_m3   = NA_real_,  # for rho0 = -0.5
    rho_p2   = NA_real_,  # for rho0 = 0.5
    rho_p25  = NA_real_,  # for rho0 = 0.5
    rho_p75  = NA_real_,  # for rho0 = 0.5
    rho_p8   = NA_real_  # for rho0 = 0.5
  )
  
  for (k in 1:(N-1)) {
    X_kn <- sorted[N - k]
    
    # Compute all rho estimates
    estimates <- c(
      rho_m35  = rho_est(-0.35,  k, sorted, N, X_kn),
      rho_m3   = rho_est(-0.3,  k, sorted, N, X_kn),
      rho_p2   = rho_est(0.2,   k, sorted, N, X_kn),
      rho_p25  = rho_est(0.25,   k, sorted, N, X_kn),
      rho_p75  = rho_est(0.75,   k, sorted, N, X_kn),
      rho_p8   = rho_est(0.8,   k, sorted, N, X_kn)
    )
    
    # Set values to NA if estimate >= 0
    estimates[estimates >= 0] <- NA_real_
    
    # Assign filtered estimates back to dataframe
    values[k, names(estimates)] <- estimates
  }
  
  return(values)
}

# Function to average over m simulations and return a data.frame
average_rho <- function(N, alpha, lambda, tau, m) {
  message(glue::glue("Running rho simulations for N={N}, alpha={alpha}, lambda={lambda}, tau={tau}"))
  
  # Use future_map with seeds
  sims <- replicate(
    m,
    simulate_rho(N = N, alpha = alpha, lambda = lambda, tau = tau),
    simplify = FALSE
  )
  
  # Stack all simulation data.frames row-wise by column `ks`
  combined <- bind_rows(sims, .id = "sim_id")  # adds a simulation ID
  
  # Group by index `ks` and average over the simulations, ignoring NAs
  sims_avg <- combined %>%
    group_by(ks) %>%
    summarise(across(-sim_id, ~mean(.x, na.rm = TRUE))) %>%
    ungroup()
  
  return(sims_avg)
}

estimate_rho <- function(data) {
  N      <- length(data)
  sorted <- sort(data)
  
  values <- data.frame(
    ks       = seq(1, N-1),
    rho_m05  = NA_real_,
    rho_m025 = NA_real_,
    rho_0    = NA_real_,
    rho_025  = NA_real_,
    rho_05   = NA_real_
  )
  
  for (k in 1:(N-1)) {
    X_kn <- sorted[N-k]
    values$rho_m05[k]  <- rho_est(-0.5, k, sorted, N, X_kn)
    values$rho_m025[k] <- rho_est(-0.25, k, sorted, N, X_kn)
    values$rho_0[k]    <- rho_est(0, k, sorted, N, X_kn)
    values$rho_025[k]  <- rho_est(0.25, k, sorted, N, X_kn)
    values$rho_05[k]   <- rho_est(0.5, k, sorted, N, X_kn)
  }
  
  return(values)
}



calculate_estimates <- function(data, x_pn, g_func, beta, rho) {
  # Extract length, sort the data and define the confidence interval length
  N <- length(data)
  sorted <- sort(data)
  z_alpha    <- qnorm(0.975)
  
  # Initialize vectors for the extreme value index, tail probability 
  # and premium estimates
  results <- data.frame(
    ks      = seq(1, N-1),
    gamma_k = NA_real_,
    gamma_k_lower = NA_real_,
    gamma_k_upper = NA_real_,
    gamma_k_bc = NA_real_,
    gamma_k_bc_lower = NA_real_,
    gamma_k_bc_upper = NA_real_,
    hat_pn  = NA_real_,
    hat_pn_lower  = NA_real_,
    hat_pn_upper  = NA_real_,
    hat_pn_bc  = NA_real_,
    hat_pn_bc_lower  = NA_real_,
    hat_pn_bc_upper  = NA_real_,
    premium = NA_real_,
    premium_lower = NA_real_,
    premium_upper = NA_real_,
    premium_bc = NA_real_,
    premium_bc_lower = NA_real_,
    premium_bc_upper = NA_real_
  )
  
  for (k in 1:(N-1)) {
    X_kn    <- sorted[N-k]
    # Hill estimator and confidence intervals
    gamma <- hill_estimator(k, sorted, N, X_kn)
    gamma_lower <- gamma - (z_alpha * gamma) / sqrt(k)
    gamma_upper <- gamma + (z_alpha * gamma) / sqrt(k)
    
    # Tail probability estimator and confidence intervals
    tail_prob <- tail_probability(k, N, X_kn, x_pn, gamma)
    log_dn    <- log(k / (N*tail_prob))
    tail_prob_lower <- tail_prob * exp(- log_dn * z_alpha / sqrt(k))
    tail_prob_upper <- tail_prob * exp(log_dn * z_alpha / sqrt(k))
    
    # Premium estimator and confidence interval
    Pi <- distorted_premium(x_pn, tail_prob, gamma, g_func, beta)
    Pi_lower <- Pi * exp(log_dn / sqrt(k) * beta * z_alpha)
    Pi_upper <- Pi * exp(- log_dn / sqrt(k) * beta * z_alpha)
    
    # For the bias corrected version
    delta     <- gamma * (1 - 2*rho) * (1 - rho)^3 * rho^(-4) * (E_func(rho / gamma, k, sorted, N, X_kn) - 1 / (1 - rho))
    gamma_bc  <- gamma - delta * rho / (1 - rho)
    gamma_bc_lower <- gamma_bc * ( 1 + (1 - rho) / rho * z_alpha / sqrt(k))
    gamma_bc_upper <- gamma_bc * ( 1 - (1 - rho) / rho * z_alpha / sqrt(k))
    
    tail_prob_bc <- tail_probability(k, N, X_kn, x_pn, gamma_bc)
    log_dn_bc    <- log(k / (N * tail_prob_bc))
    tail_prob_bc_lower <- tail_prob_bc * exp((1-rho)/rho * log_dn_bc / sqrt(k) * z_alpha)
    tail_prob_bc_upper <- tail_prob_bc * exp(- (1-rho)/rho * log_dn_bc / sqrt(k) * z_alpha)
    
    Pi_bc <- distorted_premium(x_pn, tail_prob_bc, gamma_bc, g_func, beta)
    Pi_bc_lower <- Pi_bc * exp(- beta * (1-rho) / rho * log_dn_bc / sqrt(k) * z_alpha)
    Pi_bc_upper <- Pi_bc * exp(beta * (1-rho) / rho * log_dn_bc / sqrt(k) * z_alpha)
    
    # Update data frame with new values
    results$ks[k] <- k
    results$gamma_k[k] <- gamma
    results$gamma_k_lower[k] <- gamma_lower
    results$gamma_k_upper[k] <- gamma_upper
    results$hat_pn[k] <- tail_prob
    results$hat_pn_lower[k] <- tail_prob_lower
    results$hat_pn_upper[k] <- tail_prob_upper
    results$premium[k] <- Pi
    results$premium_lower[k] <- Pi_lower
    results$premium_upper[k] <- Pi_upper
    
    results$gamma_k_bc[k] <- gamma_bc
    results$gamma_k_bc_lower[k] <- gamma_bc_lower
    results$gamma_k_bc_upper[k] <- gamma_bc_upper
    results$hat_pn_bc[k] <- tail_prob_bc
    results$hat_pn_bc_lower[k] <- tail_prob_bc_lower
    results$hat_pn_bc_upper[k] <- tail_prob_bc_upper
    results$premium_bc[k] <- Pi_bc
    results$premium_bc_lower[k] <- Pi_bc_lower
    results$premium_bc_upper[k] <- Pi_bc_upper
  }
  
  return(results)
}