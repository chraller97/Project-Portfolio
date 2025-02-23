# This functions contains all functions needed for analyzing and cleaning
# the Taiwan Earthquake Data using EVT.
#
# Author:        Christian Aaris Lærkedahl Ørnskov
# Creation Date: 2024-04-17


# ------------------------------------------------------------------------------
# Cleaning the data
# ------------------------------------------------------------------------------
clean_data <- function(df) {
  #' This function cleans the data frame and calculates arrival times as the 
  #' number of days since first arrival.
  #' 
  #' @param df The data frame to clean
  #' @return   Returns the cleaned data frame

  # Convert time-column to standard unit and sort to increasing dates
  df$time    <- as.POSIXct(df$time, format = "%Y-%m-%dT%H:%M", tz = Sys.timezone())
  df         <- df[order(df$time), ]
  
  # If several arrivals occur at the exact same time, keep only the one with highest magnitude
  df         <- df %>% group_by(time) %>% slice(which.max(mag))
  
  # Add "arrival" column as time in days since first arrival that is set to 0.
  N          <- length(df$time)
  df$arrival <- rep(0, N)
  for (i in 2:N) {
    df$arrival[i] <- time_length(interval(df$time[1], df$time[i]), unit = "days")
  }
  
  return(df[, c("time", "arrival", "mag")])
}


# ------------------------------------------------------------------------------
# Generate Proposal values
# ------------------------------------------------------------------------------
proposal <- function(params, weights) {
  #' This function generates new proposal value based on the current values
  #' 
  #' @param params  The current parameters (sigma, xi, mu, beta, k, c, p) 
  #'    in that order.
  #' @param weights The weights (standard deviation) for the error term of 
  #'    the proposal values.
  #' @return        A vector of proposal values.
  
  proposal_values <- rnorm(7, mean = params, sd = weights)
  
  return(proposal_values)
}


# ------------------------------------------------------------------------------
# The full Metropolis-Hastings Algorithm
# ------------------------------------------------------------------------------
MH <- function(data, initial_params, weights, S, n, Tend) {
  #' The Metropolis Hastings algorithm. Generates a markov chain for the 
  #' posterior distribution of each parameter given the data
  #' 
  #' @param data The data frame containing the exceedances information. 
  #'    Magnitudes should be given as 'magnitude - threshold'-
  #' @param initial_params The initial parameters to start the chain. 
  #'    These are the MLE estimates of the log-likelihood.
  #' @param weights The weights for the standard devation in the proposals.
  #' @param S       The length of the chain.
  #' @param n       Number of threshold exceedances (nr. of rows in data frame).
  #' @param Tend    The time of the last arrival.
  #' @return        A matrix containing the Markov chain of each parameter.
  
  # Unpack parameters and data
  exceedances <- data
  sigma_mle   <- initial_params[1]
  xi_mle      <- initial_params[2]
  mu_mle      <- initial_params[3]
  beta_mle    <- initial_params[4]
  k_mle       <- initial_params[5]
  c_mle       <- initial_params[6]
  p_mle       <- initial_params[7]
  
  param_chain      <- matrix(NA, nrow = S, ncol = 7)
  param_chain[1, ] <- initial_params
  
  acceptances <- 0

  # Progress bar
  pb <- txtProgressBar(min = 0, max = S, initial = 0, char = "=", style = 3)
  
  # Metropolis Hasting Loop
  for (i in 2:S) {
    # Update progress bar
    setTxtProgressBar(pb, i)
    
    # Get current parameters propose new values
    current_params  <- param_chain[i-1, ]
    proposed_params <- proposal(current_params, weights)
    
    # Prior densities of current and proposal parameters
    prior_current  <- sum(dnorm(current_params,  mean = initial_params, sd = 10, log = TRUE))
    prior_proposed <- sum(dnorm(proposed_params, mean = initial_params, sd = 10, log = TRUE))
    
    # Log-likelihood of current and proposal parameters
    ll_current    <- - model_loglikelihood(current_params,  exceedances, n, Tend)
    ll_proposed   <- - model_loglikelihood(proposed_params, exceedances, n, Tend)
    
    # Metropolis Hasting Ratio
    R                <- exp((prior_proposed + ll_proposed) - (prior_current + ll_current))
    
    # If the ratio is not finite, set it to 0 (no update)
    R[!is.finite(R)] <- 0
    
    # Define acceptance probability
    accept_prob      <- min(R, 1)
    
    # Accept or not (boolean)
    accept           <- runif(1) < accept_prob
    
    # Update chain with current or proposed values
    if (accept) {
      param_chain[i, ] <- proposed_params
      acceptances      <- acceptances + 1
    } else {
      param_chain[i, ] <- current_params
    }
  }
  
  # The acceptance ratio of the algorithm
  acceptance_ratio <- acceptances / S
  print(paste("Acceptance ratio was: ", acceptance_ratio))
  
  # Return Markov chain
  return(param_chain)
}

# ------------------------------------------------------------------------------
# Functions for the log-likelihood of the model
# ------------------------------------------------------------------------------
intensity_measure <- function(tau, data, params) {
  #' This function calculates the intensity measure at a specific arrival time
  #' tau based on the exceedances data and the parameters
  #' 
  #' @param tau    The time at which to calculate the intensity measure.
  #' @param data   The data frame containing exceedances information.
  #' @param params Vector containing (mu, beta, k, c, and p) in that order.
  #' @return       The intensity measure at the given time tau.
  
  arrivals    <- data$arrival[data$arrival < tau]
  magnitudes  <- data$mag[data$arrival < tau]
  
  exp_term    <- exp(params[2] * magnitudes)
  time_term   <- (tau - arrivals + params[4])^(1-params[5]) - params[4]^(1-params[5])
  integrals   <- sum(exp_term * time_term)
  compensator <- params[1]*tau + (params[3] / (1-params[5])) * integrals
  
  return(compensator)
}

intensity_density <- function(tau, data, params) {
  #' This function calculates the conditional intensity at a given time tau 
  #' based on the exceedances data and parameters.
  #' 
  #' @param tau The time at which to calculate the conditional intensity.
  #' @param data The data frame containing exceedances information.
  #' @param params Vector containing (mu, beta, k, c, and p) in that order.
  #' @return The conditional intensity at the given time tau.

  arrivals   <- data$arrival
  magnitudes <- data$mag
  
  exp_term <- exp(params[2] * magnitudes[arrivals < tau])
  conditional_intensity <- params[1] + sum( exp_term * params[3] / (tau - arrivals[arrivals < tau] + params[4])^params[5])
  
  return(conditional_intensity)
}

model_loglikelihood <- function(params, data, n, Tend) {
  #' This function calculates the log-likelihood of the marked point process 
  #' model by individually calculating each part of the log-likelihood
  #' 
  #' @param params Vector containing the log of the parameters:
  #'  (sigma, xi, mu, beta, k, c, p) in that order.
  #' @param data   The data frame containing the exceedances information.
  #' @param n      Length of the data frame.
  #' @param Tend   The last arrival time of the exceedances.
  #' @return       The negative log-likelihood of the model (for the MLE estimation).

  # Unpack data
  exceedances <- data

  # Convert to actual parameters
  sigma <- exp(params[1])
  xi    <- -exp(params[2])
  mu    <- exp(params[3])
  beta  <- exp(params[4])
  k     <- exp(params[5])
  c     <- exp(params[6])
  p     <- exp(params[7])
  
  # Check if the term inside the log is non-positive, if so, return Inf
  term_inside_log <- 1 + xi * exceedances$mag / sigma
  if (any(term_inside_log <= 0)) {
    return(Inf)  # Return Inf if invalid
  }
  
  # Negative intensity measure
  neg_intensity_measure <- - intensity_measure(Tend, exceedances, c(mu, beta, k, c, p))
  
  # Sum of log intensities
  intensity             <- sapply(exceedances$arrival, intensity_density, data = exceedances, params = c(mu, beta, k, c, p))
  sum_log_intensity     <- sum(log(intensity))
  
  # Sum of log of GPD densities
  sum_log_density_gpd   <- - n * log(sigma) - sum((1 + 1/xi) * log(1 + xi * exceedances$mag / sigma))
  
  # Log-likelihood
  loglikelihood <- neg_intensity_measure + sum_log_intensity + sum_log_density_gpd
  
  return(-loglikelihood)
}


# ------------------------------------------------------------------------------
# Functions for analyzing posterior distributions
# ------------------------------------------------------------------------------
posterior_means <- function(df, burn) {
  #' Calculates the posterior mean of a data frame containing the markov chains.
  #' 
  #' @param df   The data frame containing the markov chains.
  #' @param burn The burn-in period before the posterior distribution.
  #' @return     The posterior mean of each parameter as a vector.
  
  N <- nrow(df)
  posterior_distribution <- df[burn:N, ]
  means <- colMeans(posterior_distribution)
  
  return(means)
}

posterior_sds <- function(df, burn) {
  #' Calculates the posterior mean of a data frame containing the markov chains.
  #' 
  #' @param df   The data frame containing the markov chains.
  #' @param burn The burn-in period before the posterior distribution.
  #' @return     The posterior standard deviation of each parameter as a vector.
  
  N <- nrow(df)
  posterior_distribution <- df[burn:N, ]
  std   <- sapply(posterior_distribution, sd)
  
  return(std)
}

transform_arrivals <- function(arrivals, params) {
  #' Transforms the arrival times according to the random time change theorem.
  #' 
  #' @param arrivals The arrival times to transform.
  #' @param params   The parameters to use as input in the intensity_measure
  
  parameters  <- as.matrix(params[3:7])
  n           <- length(arrivals)
  transform <- rep(0, length = n)
  for (i in 2:n) {
    transform[i] <- intensity_measure(arrivals[i], exceedances, parameters)
  }
  
  return(transform)
}


# ------------------------------------------------------------------------------
# Functions for plotting results
# ------------------------------------------------------------------------------
# Plot the MCMC chain for a parameter
chain_plot <- function(df, threshold) {
  #' Function to plot the Markov chain of all parameters for one run of 
  #' the MH-algrotihm.
  #' 
  #' @param df        The data frame of parameter values.
  #' @param threshold The threshold used in the chain.
  #' @return          A facet plot of the chain of all parameters.

  x_length   <- length(df[, 1])
  y_length   <- length(df[1, ])
  melted_df  <- melt(df)
  
  facet_plot <- melted_df %>% ggplot(aes(x = rep(seq(1, x_length, 1), y_length), 
                                         y = value)) +
    geom_point(aes(color = "#F8766D"), size = 1) +
    facet_wrap(~ variable, scales = "free_y", ncol = 3) +
    theme(legend.position  = "none",
          strip.background = element_rect(colour = "black",
                                          fill   = "#619CFF")) +
    labs(x     = "Index", 
         y     = "Value",
         title = paste("Markov Chain for all parameters at threshold: ", threshold))
  
  return(facet_plot)
}

statistic_plot <- function(df, thresholds, statistic) {
  #' Function to create a facet_wrap plot of the posterior means over 
  #' different thresholds
  #' 
  #' @param df         The data frame of posterior means (each row is a thresholds)
  #' @param thresholds The vector of thresholds
  #' @return           A plot of the posterior means of each parameter 
  #'    for each threshold
  
  x_length  <- length(thresholds)
  y_length  <- ncol(df)
  melted_df <- melt(df)
  
  mean_plot <- melted_df %>% ggplot(aes(x = rep(thresholds, y_length), y = value)) +
    geom_line(aes(color = "#F8766D"), size = 1) +
    facet_wrap(~ variable, scales = "free_y", ncol = 3) +
    theme(legend.position="none",
          strip.background=element_rect(colour="black",
                                        fill="#619CFF")) +
    labs(x = "Threshold", y = "Value",
         title = paste("Posterior", statistic, "for different thresholds"))
  return(mean_plot)
}

plot_transformed <- function(transformed_arrivals, threshold) {
  #' Plots the transformed arrival to see if they look like a 
  #' homogeneous Poisson process with unit rate
  #' @param transformed_arrivals The transformed arrivals to be plotted.
  #' @param threshold            The threshold that calculated the parameters.
  #' @return                     The plot of transformed arrival times.
  
  n           <- length(transformed_arrivals)
  cum_arrival <- cumsum(rep(1, n))
  df          <- data.frame(x = transformed_arrivals, y = cum_arrival)
  the_plot    <- df %>% ggplot(aes(x = x, y = y)) +
    geom_step(direction = "hv", size = 0.5) +
    geom_segment(aes(xend = transformed_arrivals, y = -1, yend = 1), size = 0.5) +
    labs(x = "Transformed", 
         y = "Cumulative Arrivals", 
         title = paste("Transformed cumulative arrivals at threshold: ", threshold)) +
    theme_minimal()
  
  return(the_plot)
}

qq_plot_transformed <- function(transformed_arrivals, threshold) {
  #' Plots the transformed arrival to see if they look like a 
  #' homogeneous Poisson process with unit rate
  #' @param transformed_arrivals The transformed arrivals to be plotted.
  #' @param threshold            The threshold that calculated the parameters.
  #' @return The qq plot of transformed arrival times against the exponential 
  #'    distribution with rate = 1.
  
  n                <- length(transformed_arrivals)
  quantiles        <- seq(1/n, n/(n+1), length.out = n-1)
  theoretical      <- qexp(quantiles, rate = 1)
  sample_quantiles <- diff(transformed_arrivals)
  
  df <- data.frame(Theoretical = theoretical, Sample = sample_quantiles)
  
  qq_plot <- ggplot(df, aes(sample = Sample)) +
    stat_qq(distribution = qexp, dparams = list(rate = 1)) +
    geom_abline(slope = 1, intercept = 0, color = "red") +
    labs(title = paste("Q-Q Plot for exponential Distribution at threshold:", threshold),
         x = "Theoretical Quantiles",
         y = "Sample Quantiles") +
    theme_minimal()
  
  return(qq_plot)
}

qq_plot_gpd <- function(mags, params, threshold) {
  #' Plots the QQ plot for the Generalized Pareto Distribution (GPD)
  #' 
  #' @param mags   The magnitudes of the observed data.
  #' @param params The scale and shape parameter of the GPD (sigma, xi).
  #' @return       The QQ plot comparing the observed data to the theoretical GPD.
  
  params <- as.numeric(params)
  
  # Compute the theoretical quantiles of the GPD
  quantiles_gpd <- qgpd(ppoints(length(mags)), scale = params[1], shape = params[2])
  observed      <- sort(mags)
  
  # Create a dataframe for ggplot
  df <- data.frame(Theoretical = quantiles_gpd, Observed = observed)

  # Create the QQ plot using ggplot2
  qq_plot <- ggplot(df, aes(x = Theoretical, y = Observed)) +
                geom_point() +
                geom_abline(slope = 1, intercept = 0, color = "red") +
                labs(title = paste("QQ Plot of observed vs. GPD at threshold: ", threshold),
                     x = "Theoretical Quantiles (GPD)",
                     y = "Sample Quantiles (Observed Data)") +
                theme_minimal()
  
  return(qq_plot)
}






# ------------------------------------------------------------------------------
# Older version for updating the parameters separately in the MH-algorithm (below)
# ------------------------------------------------------------------------------

# # Markov Chains
# sigmas <- c(sigma_mle+1)
# xis    <- c(xi_mle+1)
# mus    <- c(mu_mle+1)
# betas  <- c(beta_mle+1)
# ks     <- c(k_mle+1)
# cs     <- c(c_mle+1)
# ps     <- c(p_mle+1)

# # Previous value - theta(t)
# sigma <- sigmas[i]
# xi    <- xis[i]
# mu    <- mus[i]
# beta  <- betas[i]
# k     <- ks[i]
# c     <- cs[i]
# p     <- ps[i]

# # Proposal value - theta^*
# proposed <- proposal(c(sigma, xi, mu, beta, k, c, p), weights)
# c(sigma_star, xi_star, mu_star, beta_star, k_star, c_star, p_star) %<-% proposed

# # Prior densities theta(t)
# p_sigma <- dnorm(sigma, mean = sigma_mle, sd = 10, log = TRUE)
# p_xi    <- dnorm(xi,    mean = xi_mle,    sd = 10, log = TRUE)
# p_mu    <- dnorm(mu,    mean = mu_mle,    sd = 10, log = TRUE)
# p_beta  <- dnorm(beta,  mean = beta_mle,  sd = 10, log = TRUE)
# p_k     <- dnorm(k,     mean = k_mle,     sd = 10, log = TRUE)
# p_c     <- dnorm(c,     mean = c_mle,     sd = 10, log = TRUE)
# p_p     <- dnorm(p,     mean = p_mle,     sd = 10, log = TRUE)
# 
# # Prior densities theta^*
# p_sigma_star <- dnorm(sigma_star, mean = sigma_mle, sd = 10, log = TRUE)
# p_xi_star    <- dnorm(xi_star,    mean = xi_mle,    sd = 10, log = TRUE)
# p_mu_star    <- dnorm(mu_star,    mean = mu_mle,    sd = 10, log = TRUE)
# p_beta_star  <- dnorm(beta_star,  mean = beta_mle,  sd = 10, log = TRUE)
# p_k_star     <- dnorm(k_star,     mean = k_mle,     sd = 10, log = TRUE)
# p_c_star     <- dnorm(c_star,     mean = c_mle,     sd = 10, log = TRUE)
# p_p_star     <- dnorm(p_star,     mean = p_mle,     sd = 10, log = TRUE)

# # Proposal densities theta(t)
# g_sigma <- dnorm(sigma, mean = sigma_star, sd = w_sigma, log = TRUE)
# g_xi    <- dnorm(xi,    mean = xi_star,    sd = w_xi,    log = TRUE)
# g_mu    <- dnorm(mu,    mean = mu_star,    sd = w_mu,    log = TRUE)
# g_beta  <- dnorm(beta,  mean = beta_star,  sd = w_beta,  log = TRUE)
# g_k     <- dnorm(k,     mean = k_star,     sd = w_k,     log = TRUE)
# g_c     <- dnorm(c,     mean = c_star,     sd = w_c,     log = TRUE)
# g_p     <- dnorm(p,     mean = p_star,     sd = w_p,     log = TRUE)
# 
# # Proposal densities theta^*
# g_sigma_star <- dnorm(sigma_star, mean = sigma, sd = w_sigma, log = TRUE)
# g_xi_star    <- dnorm(xi_star,    mean = xi,    sd = w_xi,    log = TRUE)
# g_mu_star    <- dnorm(mu_star,    mean = mu,    sd = w_mu,    log = TRUE)
# g_beta_star  <- dnorm(beta_star,  mean = beta,  sd = w_beta,  log = TRUE)
# g_k_star     <- dnorm(k_star,     mean = k,     sd = w_k,     log = TRUE)
# g_c_star     <- dnorm(c_star,     mean = c,     sd = w_c,     log = TRUE)
# g_p_star     <- dnorm(p_star,     mean = p,     sd = w_p,     log = TRUE)

# # Log likelihood (adjust with negative)
# ll            <- - model_loglikelihood(c(sigma, xi, mu, beta, k, c, p), exceedances, n, Tend)
# ll_sigma_star <- - model_loglikelihood(c(sigma_star, xi, mu, beta, k, c, p), exceedances, n, Tend)
# ll_xi_star    <- - model_loglikelihood(c(sigma, xi_star, mu, beta, k, c, p), exceedances, n, Tend)
# ll_mu_star    <- - model_loglikelihood(c(sigma, xi, mu_star, beta, k, c, p), exceedances, n, Tend)
# ll_beta_star  <- - model_loglikelihood(c(sigma, xi, mu, beta_star, k, c, p), exceedances, n, Tend)
# ll_k_star     <- - model_loglikelihood(c(sigma, xi, mu, beta, k_star, c, p), exceedances, n, Tend)
# ll_c_star     <- - model_loglikelihood(c(sigma, xi, mu, beta, k, c_star, p), exceedances, n, Tend)
# ll_p_star     <- - model_loglikelihood(c(sigma, xi, mu, beta, k, c, p_star), exceedances, n, Tend)

#   # Metropolis-Hastings Ratio
#   R_sigma <- exp((p_sigma_star + ll_sigma_star) - (p_sigma + ll))
#   R_xi    <- exp((p_xi_star    + ll_xi_star)    - (p_xi    + ll))
#   R_mu    <- exp((p_mu_star    + ll_mu_star)    - (p_mu    + ll))
#   R_beta  <- exp((p_beta_star  + ll_beta_star)  - (p_beta  + ll))
#   R_k     <- exp((p_k_star     + ll_k_star)     - (p_k     + ll))
#   R_c     <- exp((p_c_star     + ll_c_star)     - (p_c     + ll))
#   R_p     <- exp((p_p_star     + ll_p_star)     - (p_p     + ll))
# 
#   # Condition for non-valid set of parameters
#   if(is.na(R_sigma)) {
#     R_sigma <- 0
#   }
#   if(is.na(R_xi)) {
#     R_xi <- 0
#   }
#   if(is.na(R_mu)) {
#     R_mu <- 0
#   }
#   if(is.na(R_beta)) {
#     R_beta <- 0
#   }
#   if(is.na(R_k)) {
#     R_k <- 0
#   }
#   if(is.na(R_c)) {
#     R_c <- 0
#   }
#   if(is.na(R_p)) {
#     R_p <- 0
#   }
# 
#   # Acceptance probability
#   prob_sigma <- min(R_sigma, 1)
#   prob_xi    <- min(R_xi,  1)
#   prob_mu    <- min(R_mu,  1)
#   prob_beta  <- min(R_beta, 1)
#   prob_k     <- min(R_k,   1)
#   prob_c     <- min(R_c,   1)
#   prob_p     <- min(R_p,   1)
# 
#   c(u_sigma, u_xi, u_mu, u_beta, u_k, u_c, u_p) %<-% rep(runif(1), 7)
# 
#   # Update Markov Chains with new values based on acceptance probability
#   if (u_sigma <= prob_sigma) {
#     sigmas[i+1] <- sigma_star
#   } else {
#     sigmas[i+1] <- sigma
#   }
# 
#   if (u_xi <= prob_xi) {
#     xis[i+1] <- xi_star
#   } else {
#     xis[i+1] <- xi
#   }
# 
#   if (u_mu <= prob_mu) {
#     mus[i+1] <- mu_star
#   } else {
#     mus[i+1] <- mu
#   }
# 
#   if (u_beta <= prob_beta) {
#     betas[i+1] <- beta_star
#   } else {
#     betas[i+1] <- beta
#   }
# 
#   if (u_k <= prob_k) {
#     ks[i+1] <- k_star
#   } else {
#     ks[i+1] <- k
#   }
# 
#   if (u_c <= prob_c) {
#     cs[i+1] <- c_star
#   } else {
#     cs[i+1] <- c
#   }
# 
#   if (u_p <= prob_p) {
#     ps[i+1] <- p_star
#   } else {
#     ps[i+1] <- p
#   }
# }
# 
# sample <- matrix(c(sigmas, xis, mus, betas, ks, cs, ps), ncol = 7)

# return(sample)







