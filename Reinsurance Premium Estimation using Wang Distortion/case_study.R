# Case Study
#
# Author: Christian Aaris Lærkedahl Ørnskov
# Thesis: Reinsurance Premium Estimation using Wang Distortion
# Dataset: Norwegain Fire Insurance from CASDatasets adjusted for 2012 values.

# --------------- Settings & Packages ------------------------------------------
library(tidyverse)   # For general data analysis and visualization
library(CASdatasets) # For the dataset
library(moments)     # For the functions skewness and kurtosis
source("theoretical_equations.R")

theme_set(theme_grey(base_family = "serif"))
options(scipen = 999)



# --------------- Initial data analysis ----------------------------------------
# --------------- Chapter 2 in thesis   ----------------------------------------
# Load data, extract 2012 value, convert to million NKR 
# and extract the length of the dataset
data("norfire")
norway <- norfire$Loss2012 / 1000
norfire$Loss2012 <- norfire$Loss2012 / 1000
N <- length(norway)

# Compute initial descriptive statistics
descriptive_stats <- norfire %>% 
  summarize(Minimum     = min(Loss2012),
            Median      = median(Loss2012),
            Quantile75  = quantile(Loss2012, probs = 0.75),
            Quantile95  = quantile(Loss2012, probs = 0.95),
            Quantile99  = quantile(Loss2012, probs = 0.99),
            Maximum     = max(Loss2012),
            Expectation = mean(Loss2012),
            Variance    = var(Loss2012),
            Skewness    = skewness(Loss2012),
            Kurtosis    = kurtosis(Loss2012))
#descriptive_stats

# Create a histogram of the log-transformed data
histogram <- ggplot(norfire, aes(x = log(Loss2012))) +
  geom_histogram(bins = 40, color = "black") +
  labs(title = "Histogram of the logarithmic claim sizes", 
       x = "log(Claim size)", 
       y = "Counts")
#histogram
#ggsave(file = "histogram.pdf", plot = histogram, width = 7, height = 7 / 1.618)

# Create QQ-plots against the standard normal & standard exponential distribution 
qq_norm <- ggplot(norfire, aes(sample = Loss2012)) +
  stat_qq(distribution = qnorm, size = 1.5, alpha = 0.7) +                    # QQ plot for the normal distribution
  geom_abline(slope = 1, intercept = 0, color = "darkred", linetype = "dashed") + # Reference line
  labs(title = "Normal QQ-plot", 
       x = "Theoretical quantiles", 
       y = "Claim size quantiles")

qq_exp <- ggplot(norfire, aes(sample = Loss2012)) +
  stat_qq(distribution = function(p) {qexp(p, rate = 1)}, alpha = 0.7) +      # QQ plot for the exponential distribution
  geom_abline(slope = 1, intercept = 0, color = "darkred", linetype = "dashed") + # Reference line
  labs(title = "Exponential QQ-plot", 
       x = "Theoretical quantiles", 
       y = "Claim size quantiles")
qq_norm_exp <- qq_norm + qq_exp
#qq_norm_exp
#ggsave(file = "qqplots.pdf", plot = qq_norm_exp, width = 7, height = 3.6)

# Calculate the trivial empirical net premium estimates.
ts           <- seq(250, 1000, 1)
empirical_Pi <- sapply(ts, function(x) { mean(pmax(0, norway - x))})

empirical_Pi_data <- data.frame(t = ts, premium = empirical_Pi)
empirical_Pi_plot <- empirical_Pi_data %>% 
  ggplot(aes(x = t, y = premium)) +
    #geom_point() +
    geom_line(color = "black", linewidth = 0.5) +
    geom_vline(xintercept = descriptive_stats$Maximum, linetype = "dashed") +
    annotate("text", x = descriptive_stats$Maximum, y = max(empirical_Pi), 
           label = "max.", vjust = 0, hjust = -0.25, size = 3, color = "black") +
    labs(title = "Empirical estimates of the net premium principle for different thresholds",
         x = "Threshold",
         y = "Estimated premium")
empirical_Pi_plot
#ggsave(file = "Pi_from_ECDF.pdf", plot = empirical_Pi_plot, width = 7, height = 7 / 1.618)



# --------------- Analysis under the Heavy-tailed framework --------------------
# --------------- Chapter 3 in thesis ------------------------------------------
# Create a QQ-plot for the Pareto-type distribution 
# (log-transformed data against the standard exponential distribution)
k                <- 752
log_sort         <- log(sort(norway))
pareto_qq_data   <- data.frame(
  theoretical = - log(1 - seq(1, N) / (N + 1)), 
  observed    = log_sort
)

# Fit the Pareto tail using the top k order statistics, and extract the slope
fit   <- lm(observed ~ theoretical, data = pareto_qq_data[(N - k):N, ])
slope <- coef(fit)[2]

# Compute start and end points for the fitted line
x_start <- - log((k + 1) / (N + 1))
x_end   <- - log(1 / (N + 1))
y_start <- log_sort[N - k]
y_end   <- slope * (x_end - x_start) + y_start

qq_pareto  <- ggplot(pareto_qq_data, aes(x = theoretical, y = observed)) +
  geom_point(size = 0.5) +
  annotate("segment", x = x_start, y = y_start, xend = x_end, yend = y_end,
               linetype = "dashed", linewidth = 0.5, color = "red") +  
  labs(title = "Pareto QQ-plot",
       x = "Theoretical quantiles",
       y = "Observed quantiles")
#qq_pareto
#slope
#ggsave(file = "pareto_qq.pdf", plot = qq_pareto, width = 7, height = 7 / 1.618)

# Create a plot of the estimates of rho as a function of k
rho <- estimate_rho(norway)
rho_plot_cs <- rho[7000:(N-1), ] %>% 
  ggplot(aes(x = ks, y = rho_0)) +
  geom_line(linewidth = 0.25, color = "steelblue") +
  geom_line(aes(y = rho_m025), linewidth = 0.25, color = "darkred") +
  geom_line(aes(y = rho_025), linewidth = 0.25, color = "forestgreen") +
  labs(title = "Estimates of rho as a function of k",
       x = "k",
       y = "rho")
rho_plot_cs
ggsave(file = "rho_plot_case_study.pdf", plot = rho_plot_cs, width = 7, height = 7 / 1.618)

# Visual inspection of the plot of rho, suggests to fix rho = rho_k[8500]
rho_val <- rho$rho_0[9000]

# Create a plot of the Hill estimator as a function of k
x_pn     <- 500
g_net    <- function(x) { x }
beta_net <- -1

estimates_net <- calculate_estimates(norway, x_pn, g_net, beta_net, rho_val)

hill_plot <- estimates_net[25:2000, ] %>% 
  ggplot(aes(x = ks, y = gamma_k)) +
  geom_line(linewidth = 0.25, color = "steelblue") +
  geom_line(aes(y = gamma_k_bc), linewidth = 0.25, color = "darkred") +
  geom_line(aes(y = gamma_k_lower), linewidth = 0.15, color = "steelblue", linetype = "dotted") +
  geom_line(aes(y = gamma_k_upper), linewidth = 0.15, color = "steelblue", linetype = "dotted") +
  geom_line(aes(y = gamma_k_bc_lower), linewidth = 0.15, color = "darkred", linetype = "dotted") +
  geom_line(aes(y = gamma_k_bc_upper), linewidth = 0.15, color = "darkred", linetype = "dotted") +
  labs(title = "Extreme value index as a function of k",
       x = "k",
       y = "Extreme value index")
hill_plot
ggsave(file = "hill_plot.pdf", plot = hill_plot, width = 7, height = 7 / 1.618)

tail_prob_plot <- estimates_net[25:2000,] %>%
  ggplot(aes(x = ks, y = hat_pn)) +
  geom_line(linewidth = 0.25, color = "steelblue") +
  geom_line(aes(y = hat_pn_bc), linewidth = 0.25, color = "darkred") +
  geom_line(aes(y = hat_pn_lower), linewidth = 0.15, color = "steelblue", linetype = "dotted") +
  geom_line(aes(y = hat_pn_upper), linewidth = 0.15, color = "steelblue", linetype = "dotted") +
  geom_line(aes(y = hat_pn_bc_lower), linewidth = 0.15, color = "darkred", linetype = "dotted") +
  geom_line(aes(y = hat_pn_bc_upper), linewidth = 0.15, color = "darkred", linetype = "dotted") +
  labs(title = "Tail probability as a function of k",
       x = "k",
       y = "Tail probability")
tail_prob_plot
ggsave(file = "tail_prob_plot.pdf", plot = tail_prob_plot, width = 7, height = 7 / 1.618)



# --------------- Analysis under the Heavy-tailed framework --------------------
# --------------- Chapter 3 in thesis ------------------------------------------
# Define the distortion operators, dual-power and proportional hazard 
# premium principle and their beta value
g_dual_power    <- function(x) { 1 - (1 - x)^(1 / 0.85)}
beta_dual_power <- -1

g_prop_hazard    <- function(x) { x^0.85 }
beta_prop_hazard <- -1

# Calculate the premiums for different distortion operators
estimates_dual_power  <- calculate_estimates(norway, x_pn, g_dual_power, beta_dual_power, rho_val)
estimates_prop_hazard <- calculate_estimates(norway, x_pn, g_prop_hazard, beta_prop_hazard, rho_val)

# Construct the net premium principle plot
net_premium_plot <- estimates_net[25:2000, ] %>%
  ggplot(aes(x = ks, y = premium)) +
  geom_line(linewidth = 0.25, color = "steelblue") +
  geom_line(aes(y = premium_bc), linewidth = 0.25, color = "darkred") +
  geom_line(aes(y = premium_lower), linewidth = 0.15, color = "steelblue", linetype = "dotted") +
  geom_line(aes(y = premium_upper), linewidth = 0.15, color = "steelblue", linetype = "dotted") +
  geom_line(aes(y = premium_bc_lower), linewidth = 0.15, color = "darkred", linetype = "dotted") +
  geom_line(aes(y = premium_bc_upper), linewidth = 0.15, color = "darkred", linetype = "dotted") +
  labs(title = "Net premium principle as a function of k",
       x = "k",
       y = "Premium")
net_premium_plot
ggsave(file = "net_premium_plot.pdf", plot = net_premium_plot, width = 7, height = 7 / 1.618)


full_premium_plot <- estimates_net[25:2000, ] %>%
  ggplot(aes(x = ks, y = premium)) +
  geom_line(linewidth = 0.25, color = "forestgreen") +
  geom_line(aes(y = premium_bc), linewidth = 0.25, color = "forestgreen", linetype = "dotted") +
  geom_line(aes(y = estimates_dual_power[25:2000, ]$premium), linewidth = 0.25, color = "darkred") +
  geom_line(aes(y = estimates_dual_power[25:2000, ]$premium_bc), linewidth = 0.25, color = "darkred", linetype = "dotted") +
  geom_line(aes(y = estimates_prop_hazard[25:2000, ]$premium), linewidth = 0.25, color = "steelblue") +
  geom_line(aes(y = estimates_prop_hazard[25:2000, ]$premium_bc), linewidth = 0.25, color = "steelblue", linetype = "dotted") +
  labs(title = "Different premium principles as a function of k",
       x = "k",
       y = "Premium")
full_premium_plot
ggsave(file = "full_premium_plot.pdf", plot = full_premium_plot, width = 7, height = 7 / 1.618)

trivial_premium = 1/N * sum(max(norway - x_pn, 0))
trivial_premium
