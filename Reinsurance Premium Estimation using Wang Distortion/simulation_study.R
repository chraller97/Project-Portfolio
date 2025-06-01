# Simulation study for my thesis
# Preliminary setup
library(tidyverse)
library(evd)
library(actuar)
library(patchwork)
source("theoretical_equations.R")

theme_set(theme_grey(base_family = "serif"))
options(scipen = 999)

# Check code for mistakes.
# Bias from many sources.
# Alternate methods for bias reduction.
# E[X^beta | X > U(1/p)]


# -------------------- Simulation Setup ----------------------------------------
## Define the design parameters 
m            <- 2500
x_pn         <- 5
true_gamma   <- 0.25

# Define the sample sizes and the corresponding fixed k for each.
sample_sizes <- tibble(
  N         = c(1000, 250),
  rho_k_val = c(990, 245)
)

## Define the 3 different distortion operators
distortions <- tibble(
  distortion = c("Net", "Prop. Hazard", "Dual-power"),
  g          = list(function(x) x, 
                    function(x) x^0.8, 
                    function(x) 1 - (1 - x)^1.25),
  beta       = c(-1, -0.8, -1)
)

## Define the 3 different Burr distributions
distributions <- tibble(
  distribution = c("Burr distribution with rho = -0.5", 
                    "Burr distribution with rho = -1", 
                    "Burr distribution with rho = -2"),
  alpha        = 1,
  lambda       = c(2, 1, 0.5),
  tau          = c(2, 4, 8)
) %>%
  mutate(true_hat_pn = tburr(x_pn, alpha, lambda, tau))

## Create all possible combinations of distribution, distortion and sample size
# and calculate the corresponding true premium
parameter_grid <- crossing(distributions, distortions, sample_sizes) %>%
  rowwise() %>%
  mutate(true_premium = integrate_tburr(x_pn, alpha, lambda, tau, g)) %>%
  ungroup()


# -------------------- Simulation ----------------------------------------------
## For each row in parameter_grid, call average_simulation(...)

results <- parameter_grid %>%
  mutate(
    simulation = pmap(
      list(N, distortion, g, beta, alpha, lambda, tau, true_premium, rho_k_val),
      ~ average_simulation(..1, ..2, ..3, ..4, ..5, ..6, ..7, ..8, ..9, x_pn = x_pn, m = m)
    )
  )

## Extract the data to make plots for gamma_k and hat_pn for each distribution
all_sims <- results %>%
  select(distribution, distortion, N, true_hat_pn, true_premium, simulation) %>%
  unnest(simulation) %>%
  mutate(
    estimate = list(c("gamma_k", 
                      "gamma_k_bc", 
                      "hat_pn", 
                      "hat_pn_bc", 
                      "premium", 
                      "premium_bc", 
                      "coverage", 
                      "coverage_bc")),
    .by = c(distribution, distortion, N, ks)
  ) %>%
  filter(ks>= 25, ks <= 0.5*N) %>%
  pivot_longer(cols = c(gamma_k, 
                        gamma_k_bc, 
                        hat_pn, 
                        hat_pn_bc, 
                        premium, 
                        premium_bc, 
                        coverage, 
                        coverage_bc), 
               names_to  = "type", 
               values_to = "value"
  ) %>%
  mutate(type = recode(type,
                       gamma_k     = "EVI", 
                       gamma_k_bc  = "EVI (bc)", 
                       hat_pn      = "Tail prob.", 
                       hat_pn_bc   = "Tail prob. (bc)",
                       premium     = "Premium", 
                       premium_bc  = "Premium (bc)",
                       coverage    = "Coverage", 
                       coverage_bc = "Coverage (bc)")
  ) %>%
  select(-estimate)



# -------------------- Visualization -------------------------------------------
## Plot parameter estimates for one distribution
plot_parameter <- function(dist_name) {
  # Extract data relevant to the distribution
  data    <- all_sims %>% filter(distribution == dist_name, distortion == "Net")
  true_pn <- data %>% pull(true_hat_pn) %>% unique()
  
  # Define shared y-axis limits
  evi_data  <- data %>% filter(type %in% c("EVI", "EVI (bc)"))
  tail_data <- data %>% filter(type %in% c("Tail prob.", "Tail prob. (bc)"))
  
  evi_y_range  <- range(c(evi_data$value, true_gamma), na.rm = TRUE)
  tail_y_range <- range(c(tail_data$value, true_pn), na.rm = TRUE)
  
  # Split into 4 individual plots and combine later
  p1 <- evi_data %>%
    filter(N == 250) %>%
    ggplot(aes(x = ks, y = value, color = type)) +
    geom_line(linewidth = 0.3) +
    geom_hline(yintercept = true_gamma, linetype = "dashed") +
    scale_color_manual(values = c("EVI" = "steelblue", "EVI (bc)" = "darkred")) +
    #scale_x_continuous(expand = expansion(add = c(0, 0)), limits = c(0, NA)) +
    scale_y_continuous(limits = evi_y_range) +
    labs(x = "k", y = "Extreme value index") +
    theme(legend.position = "none")
  
  p2 <- evi_data %>%
    filter(N == 1000) %>%
    ggplot(aes(x = ks, y = value, color = type)) +
    geom_line(linewidth = 0.3) +
    geom_hline(yintercept = true_gamma, linetype = "dashed") +
    scale_color_manual(values = c("EVI" = "steelblue", "EVI (bc)" = "darkred")) +
    #scale_x_continuous(expand = expansion(add = c(0, 0)), limits = c(0, NA)) +
    scale_y_continuous(limits = evi_y_range) +
    labs(x = "k", y = "Extreme value index") +
    theme(legend.position = "none")
  
  p3 <- tail_data %>%
    filter(N == 250) %>%
    ggplot(aes(x = ks, y = value, color = type)) +
    geom_line(linewidth = 0.3) +
    geom_hline(yintercept = true_pn, linetype = "dashed") +
    scale_color_manual(values = c("Tail prob." = "steelblue", "Tail prob. (bc)" = "darkred")) +
    #scale_x_continuous(expand = expansion(add = c(0, 0)), limits = c(0, NA)) +
    scale_y_continuous(limits = tail_y_range) +
    labs(x = "k", y = "Tail probability") +
    theme(legend.position = "none")
  
  p4 <- tail_data %>%
    filter(N == 1000) %>%
    ggplot(aes(x = ks, y = value, color = type)) +
    geom_line(linewidth = 0.3) +
    geom_hline(yintercept = true_pn, linetype = "dashed") +
    scale_color_manual(values = c("Tail prob." = "steelblue", "Tail prob. (bc)" = "darkred")) +
    #scale_x_continuous(expand = expansion(add = c(0, 0)), limits = c(0, NA)) +
    scale_y_continuous(limits = tail_y_range) +
    labs(x = "k", y = "Tail probability") +
    theme(legend.position = "none")
  
  # Combine and add title with dynamic distribution name
  grid_plot <- (p1 | p2) / (p3 | p4) +
    plot_annotation(tag_levels = "a")
  
  return(grid_plot)
}

plot_premium <- function(dist_name) {
  data <- all_sims %>%
    filter(distribution == dist_name, type %in% c("Premium", "Premium (bc)")) %>%
    mutate(value = as.numeric(value))
  
  # Colors for the two types (not distortion!)
  type_colors <- c("Premium" = "steelblue", "Premium (bc)" = "darkred")
  
  # Plotting function for a single distortion
  plot_per_distortion <- function(distortion_type) {
    subset <- data %>% filter(distortion == distortion_type)
    true_premium_val <- subset %>% select(true_premium) %>% unique()
    true_premium_val <- as.numeric(true_premium_val)
    
    y_range <- range(c(subset$value, true_premium_val), na.rm = TRUE)
    
    p1 <- subset %>% filter(N == 250) %>%
      ggplot(aes(x = ks, y = value, color = type)) +
      geom_line(linewidth = 0.4) +
      geom_hline(yintercept = true_premium_val, linetype = "dashed", color = "black") +
      scale_color_manual(values = type_colors) +
      scale_y_continuous(limits = y_range) +
      labs(title = paste(distortion_type, ", N = 250", sep = ""), y = "Premium", x = "k")
    
    p2 <- subset %>% filter(N == 1000) %>%
      ggplot(aes(x = ks, y = value, color = type)) +
      geom_line(linewidth = 0.4) +
      geom_hline(yintercept = true_premium_val, linetype = "dashed", color = "black") +
      scale_color_manual(values = type_colors) +
      scale_y_continuous(limits = y_range) +
      labs(title = paste(distortion_type, ", N = 1000", sep = ""), y = "Premium", x = "k")
    
    return(p1 | p2)
  }
  
  # Combine all plots
  plot_net          <- plot_per_distortion("Net")
  plot_prop_hazard  <- plot_per_distortion("Prop. Hazard")
  plot_dual_power   <- plot_per_distortion("Dual-power")
  
  final_plot <- plot_net / plot_prop_hazard / plot_dual_power +
    plot_layout(guide = "collect") & theme(legend.position = "none")
  
  return(final_plot)
}

plot_coverage <- function(dist_name) {
  data <- all_sims %>%
    filter(distribution == dist_name, type %in% c("Coverage", "Coverage (bc)")) %>%
    mutate(value = as.numeric(value))
  
  expected_coverage <- 0.95
  
  # Define fixed colors and linetypes for each distortion
  colors    <- c("Net" = "forestgreen", "Prop. Hazard" = "steelblue", "Dual-power" = "darkred")
  linetypes <- c("Non BC" = "solid", "BC" = "dotted")
  
  # Function to make the coverage probability plot for different N
  plot_for_N <- function(N_val) {
    # Extract specific data based on N_val
    data <- data %>% filter(N == N_val)
    p <- ggplot() +
      geom_line(data = data %>% filter(type == "Coverage", distortion == "Net"),
                aes(x = ks, y = value),
                color = colors["Net"],
                linetype = linetypes["Non BC"],
                linewidth = 0.4) +
      geom_line(data = data %>% filter(type == "Coverage (bc)", distortion == "Net"),
                aes(x = ks, y = value),
                color = colors["Net"],
                linetype = linetypes["BC"],
                linewidth = 0.4) +
      geom_line(data = data %>% filter(type == "Coverage", distortion == "Prop. Hazard"),
                aes(x = ks, y = value),
                color = colors["Prop. Hazard"],
                linetype = linetypes["Non BC"],
                linewidth = 0.4) +
      geom_line(data = data %>% filter(type == "Coverage (bc)", distortion == "Prop. Hazard"),
                aes(x = ks, y = value),
                color = colors["Prop. Hazard"],
                linetype = linetypes["BC"],
                linewidth = 0.4) +
      geom_line(data = data %>% filter(type == "Coverage", distortion == "Dual-power"),
                aes(x = ks, y = value),
                color = colors["Dual-power"],
                linetype = linetypes["Non BC"],
                linewidth = 0.4) +
      geom_line(data = data %>% filter(type == "Coverage (bc)", distortion == "Dual-power"),
                aes(x = ks, y = value),
                color = colors["Dual-power"],
                linetype = linetypes["BC"],
                linewidth = 0.4) +
      geom_hline(yintercept =  expected_coverage, 
                 linetype = "dashed", 
                 color = "black") +
      coord_cartesian(ylim = c(0, 1)) +
      labs(x = "k", y = "Coverage probability") +
      theme(legend.position = "none")
    return(p)
  }
  
  p1 <- plot_for_N(250)
  p2 <- plot_for_N(1000)
  
  grid_plot <- (p1 | p2) +
    plot_annotation(tag_levels = "a")
  return(grid_plot)
}

# Create and save plots for each distribution
dist_names <- unique(distributions$distribution)
parameter_plots <- map(dist_names, plot_parameter)
parameter_plots

premium_plots <- map(dist_names, plot_premium)
premium_plots

coverage_plots <- map(dist_names, plot_coverage)
coverage_plots

ggsave(file = "param_estimates_rho_0.5.pdf", plot = parameter_plots[[1]], device = cairo_pdf, width = 8.5, height = 7)
ggsave(file = "param_estimates_rho_1.pdf", plot = parameter_plots[[2]], device = cairo_pdf, width = 8.5, height = 7)
ggsave(file = "param_estimates_rho_2.pdf", plot = parameter_plots[[3]], device = cairo_pdf, width = 8.5, height = 7)

ggsave(file = "premium_estimates_rho_0.5.pdf", plot = premium_plots[[1]], device = cairo_pdf, width = 8.5, height = 10.5)
ggsave(file = "premium_estimates_rho_1.pdf", plot = premium_plots[[2]], device = cairo_pdf, width = 8.5, height = 10.5)
ggsave(file = "premium_estimates_rho_2.pdf", plot = premium_plots[[3]], device = cairo_pdf, width = 8.5, height = 10.5)

ggsave(file = "coverage_probability_rho_0.5.pdf", plot = coverage_plots[[1]], device = cairo_pdf, width = 8.5, height = 4)
ggsave(file = "coverage_probability_rho_1.pdf", plot = coverage_plots[[2]], device = cairo_pdf, width = 8.5, height = 4)
ggsave(file = "coverage_probability_rho_2.pdf", plot = coverage_plots[[3]], device = cairo_pdf, width = 8.5, height = 4)

# # Below is a list of distortion operators
# g_power_1        <- function(x) { x                 }
# g_power_0.33     <- function(x) { x^{1/3}             }
# g_dual_power     <- function(x) { 1 - (1 - x)^3 }
# 
# x_vals <- seq(0, 1, length.out = 2000)
# 
# g_data <- tibble(
#   x = x_vals,
#   'g(x) = x'             = g_power_1(x_vals),
#   'g(x) = x^{1/3}'       = g_power_0.33(x_vals),
#   'g(x) = 1 - (1 - x)^3' = g_dual_power(x_vals),
# )
# 
# g_long <- g_data %>%
#   pivot_longer(-x, names_to = "Function", values_to = "y")
# 
# distortion_plot <- ggplot(g_long, aes(x = x, y = y, color = Function, linetype = Function)) +
#   geom_line(linewidth = 1) +
#   labs(title = "Different distortion operators on [0, 1]",
#        x = "x",
#        y = "g(x)",
#        color = "Function") +
#   scale_color_manual(values = c(
#     "g(x) = x"             = "forestgreen",   # prop hazard
#     "g(x) = x^{1/3}"       = "steelblue", # net premium (power function)
#     "g(x) = 1 - (1 - x)^3" = "darkred"      # dual power
#   )) +
#   scale_linetype_manual(values = c(
#     "g(x) = x"             = "solid",       # net premium
#     "g(x) = x^{1/3}"       = "dashed",      # prop hazard
#     "g(x) = 1 - (1 - x)^3" = "dotdash"      # dual power
#   )) +
#   theme(legend.position = "none")
# distortion_plot
# ggsave(file = "distortion_plot.pdf", plot = distortion_plot, width = 7, height = 7 / 1.618)
# 
# # 
# #This part estimates rho given m and the three distribution and produces a plot for the estimates
# rho_param_grid <- crossing(distributions[-5], sample_sizes)
# 
# rhos <- rho_param_grid %>%
#   mutate(
#     simulation = pmap(
#       list(N, alpha, lambda, tau),
#       ~ average_rho(..1, ..2, ..3, ..4, m = 2500)
#     )
#   )
# 
# # Plot function for rho
# plot_rhos <- function(rho_data) {
#   data <- rho_data %>%
#     select(distribution, simulation, N) %>%
#     unnest(cols = simulation) %>%
#     filter(ks >= 0.7*N)
# 
#   dist_names <- unique(rho_data$distribution)
#   color_map <- setNames(
#     c("steelblue", "darkred", "forestgreen")[seq_along(dist_names)],
#     dist_names
#   )
# 
#   p1 <- data %>% filter(N == 250) %>%
#     ggplot(aes(x = ks, y = rho_m3, color = distribution)) +
#     geom_line(linewidth = 0.1) +
#     geom_hline(yintercept = -0.5, color = "steelblue",   linetype = "dashed") +
#     geom_hline(yintercept = -1,   color = "darkred",     linetype = "dashed") +
#     geom_hline(yintercept = -2,   color = "forestgreen", linetype = "dashed") +
#     scale_color_manual(values = color_map) +
#     coord_cartesian(ylim = c(-4, 0)) +
#     labs(x = "k", y = "rho") +
#     theme(legend.position = "none")
# 
#   p2 <- data %>% filter(N == 1000) %>%
#     ggplot(aes(x = ks, y = rho_m3, color = distribution)) +
#     geom_line(linewidth = 0.1) +
#     geom_hline(yintercept = -0.5, color = "steelblue",   linetype = "dashed") +
#     geom_hline(yintercept = -1,   color = "darkred",     linetype = "dashed") +
#     geom_hline(yintercept = -2,   color = "forestgreen", linetype = "dashed") +
#     scale_color_manual(values = color_map) +
#     coord_cartesian(ylim = c(-4, 0)) +
#     labs(x = "k", y = "rho") +
#     theme(legend.position = "none")
# 
#   p3 <- data %>% filter(N == 250) %>%
#     ggplot(aes(x = ks, y = rho_p25, color = distribution)) +
#     geom_line(linewidth = 0.1) +
#     geom_hline(yintercept = -0.5, color = "steelblue",   linetype = "dashed") +
#     geom_hline(yintercept = -1,   color = "darkred",     linetype = "dashed") +
#     geom_hline(yintercept = -2,   color = "forestgreen", linetype = "dashed") +
#     scale_color_manual(values = color_map) +
#     coord_cartesian(ylim = c(-4, 0)) +
#     labs(x = "k", y = "rho") +
#     theme(legend.position = "none")
# 
#   p4 <- data %>% filter(N == 1000) %>%
#     ggplot(aes(x = ks, y = rho_p25, color = distribution)) +
#     geom_line(linewidth = 0.1) +
#     geom_hline(yintercept = -0.5, color = "steelblue",   linetype = "dashed") +
#     geom_hline(yintercept = -1,   color = "darkred",     linetype = "dashed") +
#     geom_hline(yintercept = -2,   color = "forestgreen", linetype = "dashed") +
#     scale_color_manual(values = color_map) +
#     coord_cartesian(ylim = c(-4, 0)) +
#     labs(x = "k", y = "rho") +
#     theme(legend.position = "none")
# 
#   p5 <- data %>% filter(N == 250) %>%
#     ggplot(aes(x = ks, y = rho_p8, color = distribution)) +
#     geom_line(linewidth = 0.1) +
#     geom_hline(yintercept = -0.5, color = "steelblue",   linetype = "dashed") +
#     geom_hline(yintercept = -1,   color = "darkred",     linetype = "dashed") +
#     geom_hline(yintercept = -2,   color = "forestgreen", linetype = "dashed") +
#     scale_color_manual(values = color_map) +
#     coord_cartesian(ylim = c(-4, 0)) +
#     labs(x = "k", y = "rho") +
#     theme(legend.position = "none")
# 
#   p6 <- data %>% filter(N == 1000) %>%
#     ggplot(aes(x = ks, y = rho_p8, color = distribution)) +
#     geom_line(linewidth = 0.1) +
#     geom_hline(yintercept = -0.5, color = "steelblue",   linetype = "dashed") +
#     geom_hline(yintercept = -1,   color = "darkred",     linetype = "dashed") +
#     geom_hline(yintercept = -2,   color = "forestgreen", linetype = "dashed") +
#     scale_color_manual(values = color_map) +
#     coord_cartesian(ylim = c(-4, 0)) +
#     labs(x = "k", y = "rho") +
#     theme(legend.position = "none")
# 
#   p <- (p1 | p2) / (p3 | p4) / (p5 | p6) +
#     plot_annotation(tag_levels = "a")
#   return(p)
# }
# rho_plot <- plot_rhos(rhos)
# rho_plot
# ggsave(file = "rho_plot.pdf", plot = rho_plot, width = 8.5, height = 10.5)





