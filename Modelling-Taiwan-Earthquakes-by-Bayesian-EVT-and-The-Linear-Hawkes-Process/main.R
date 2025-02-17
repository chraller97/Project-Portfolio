# This is the main script for analyzing the Taiwan Earthquake Data using EVT
#
# Author:        Christian Aaris Lærkedahl Ørnskov
# Creation Date: 2024-04-17
#
# Earthquake data From Rectangle Drawn By:
#
# Data from USCG
# North: 25,78
# East:  122,61
# South: 21,36
# West:  119,38

# Dependencies and packages
library(tidyverse)
library(ismev)
library(evd)
library(stelfi)
library(truncnorm)
library(kableExtra)
library(pracma)
library(PtProcess)
library(reshape2)
library(zeallot)
library(lubridate)
library(goftest)
source("function_module.R")

# Load and clean data
data  <- read.csv("taiwan_earthquakes2.csv", sep = ";", header = TRUE)
data  <- clean_data(data)
N     <- length(data$arrival)

# Specify thresholds
thresholds   <- seq(5.0, 6.6, 0.1)
N_thresholds <- length(thresholds)

# Pre-allocate space for different results
samples           <- vector("list", N_thresholds)
posterior_mean    <- as.data.frame(matrix(NA, nrow = N_thresholds, ncol = 8))
posterior_sd      <- as.data.frame(matrix(NA, nrow = N_thresholds, ncol = 8))
transformed       <- vector("list", N_thresholds)
transformed_plots <- vector("list", N_thresholds)
qq_transformed    <- vector("list", N_thresholds)
qq_gpd            <- vector("list", N_thresholds)

# Run MH-algorithm
for (i in 1:N_thresholds) {
  cat("Iteration: ", i, "\n")

  # Extract exceedances for the current threshold
  threshold       <- thresholds[i]
  exceedances     <- data[data$mag > threshold, ]
  exceedances$mag <- exceedances$mag - threshold
  
  # Number of exceedances
  n_u             <- nrow(exceedances)
  
  # Number of years between first and last exceedance
  n_years         <- time_length(interval(exceedances$time[1], exceedances$time[n_u]), unit = "year")
  
  # Average exceedances per year
  n_y             <- n_u / n_years
  
  # Average percentage of exceedances
  zeta            <- n_u / N
  
  # Time of last exceedance
  Tend            <- tail(exceedances$arrival, n = 1)
  
  # Maximum likelihood estimation of theta for Empirical Bayes
  initial_params <- c(-0.5, -2.3, -5.8,  0.9, -5.3, -2.3,  0.7)
  theta          <- nlm(f         = model_loglikelihood, 
                        p         = initial_params, 
                        data      = exceedances,
                        n         = n_u,
                        Tend      = Tend,
                        iterlim   = 500)$estimate
  
  # Define length of chain and burn-in period
  S          <- 25000
  burn       <- 10000
  
  # Weights for the proposals
  weights <- c(0.1, 0.02, 0.15, 0.2, 0.2, 0.01, 0.05)

  # Begin the algorithm
  print("Begin MH-algorithm")
  chains <- MH(exceedances, theta, weights, S, n_u, Tend)

  # Convert chains to data frame and convert to true parameter values.
  chains <- data.frame(sigma =   exp(chains[, 1]),
                       xi    = - exp(chains[, 2]),
                       mu    =   exp(chains[, 3]),
                       beta  =   exp(chains[, 4]),
                       k     =   exp(chains[, 5]),
                       c     =   exp(chains[, 6]),
                       p     =   exp(chains[, 7]))
  
  # Calculate the 100-year return level
  chains$z100 <- rep(0, S)
  for (j in 1:S) {
    chains$z100[j] <- threshold + qgpd(1 - 1/100, scale = chains$sigma[j], shape = chains$xi[j])
  }
  
  # Insert data frame for this threshold into the list of data frame
  samples[[i]] <- chains
  
  posterior_mean[i, ] <- posterior_means(chains, burn)
  posterior_sd[i, ]   <- posterior_sds(chains, burn)
  
  transformed[[i]]       <- transform_arrivals(exceedances$arrival, posterior_mean[i, ])
  transformed_plots[[i]] <- plot_transformed(transformed[[i]], thresholds[i])
  qq_transformed[[i]]    <- qq_plot_transformed(transformed[[i]], threshold)
  
  qq_gpd[[i]]         <- qq_plot_gpd(exceedances$mag, posterior_mean[i, 1:2], threshold)
  
  print("Goodness of fit test of the estiamted parameters of the GPD")
  ad_test <- ad.test(exceedances$mag, 
                     pgpd, 
                     scale = posterior_mean[i, 1], 
                     shape = posterior_mean[i, 2])
  print(ad_test)
  
  cat("\n")
}

# Column names for the posterior mean an standard deviation
colnames(posterior_mean) <- c("sigma", "xi", "mu", "beta", "k", "c", "p", "z100")
colnames(posterior_sd) <- c("sigma", "xi", "mu", "beta", "k", "c", "p", "z100")


# Create and plot a facet_wrap of the markov chains for each threshold
chain_plots <- vector("list", length = N_thresholds)
for (i in 1:N_thresholds) {
  chain_plots[[i]] <- chain_plot(samples[[i]], thresholds[i])
}


# Posterior mean and sd plots for each parameter and threshold
mean_plot <- statistic_plot(posterior_mean, thresholds, "means")
sd_plot   <- statistic_plot(posterior_sd, thresholds, "standard deviations")


# Use a Kolmogorov-Smirnov test for the transformed arrivals at each threshold
for (i in 1:N_thresholds) {
  diff_transformed <- diff(transformed[[i]])
  ks               <- ks.test(diff_transformed, "pexp", rate = 1)
  print("===============================================================================")
  print(paste("The Kolmogorov-Smirnov test of the transformed arrival at threshold: ", thresholds[i]))
  print(ks)
}

for (i in 1:N_thresholds) {
  mags <- data$mag[data$mag > thresholds[i]] - thresholds[i]
  print("===============================================================================")
  ad_test <- ad.test(mags, 
                     pgpd, 
                     scale = posterior_mean[i, 1], 
                     shape = posterior_mean[i, 2])
  print(paste("The Anderson-Darling test of the magnitudes at threshold: ", thresholds[i]))
  print(ad_test)
}




