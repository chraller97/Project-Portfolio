function [price_CV] = lsmCV(S0, S, K, r, sigma, T, M, N, b)
%LSM Calculates the value of an American option (put), with the given parameters,
%using the Least Squares Monte Carlo (LSM) method by Longstaff and Schwartz
%(2001) and reducing variance through Control Variates.

% Calculate the step-szie and discount value
dt = T / M;
disc = exp(-r * dt);
disc_european = exp(-r * T);

% Calculate true black-scholes value for the European counterpart
sqrtT = sqrt(T);
d1 = (log(S0 / K) + (r + 1/2 * sigma^2)*T) / (sigma * sqrtT);
d2 = d1 - sigma * sqrtT;
european_bs = disc_european * K * normcdf(-d2) - S0 * normcdf(-d1);

% Set number of paths to be used for calculating beta
nbeta = floor(0.1 * N);
S_beta = S(1:nbeta, :);
payoff_beta = max(K - S_beta, 0);
cash_flow_beta = payoff_beta(:, M);

% Backwards recursion
for i = (M-1):-1:1
    cash_flow_beta = cash_flow_beta * disc;      
    exercise_beta = payoff_beta(:, i);       
    itm = exercise_beta > 0;                  

    [C, ~, mu] = polyfit(S_beta(itm, i), cash_flow_beta(itm), b);    
    Vcont_beta = polyval(C, S_beta(itm, i), [], mu);                 
    
    % Update cash_flow with exercise value where necessary
    optimal_beta = Vcont_beta < exercise_beta(itm);
    itm(itm) = optimal_beta;
    cash_flow_beta(itm) = exercise_beta(itm);
end
% American and European put prices
P_AM_beta = disc * cash_flow_beta;
P_EU_beta = disc_european * payoff_beta(:, M);

% Calculate beta
covariance = cov(P_AM_beta, P_EU_beta);
beta = - covariance(1, 2) / covariance(2, 2);

% Begin the control variate estimation
% The option payoff for each time step of each path
S_cv = S(nbeta+1:N, :);
exercise_value = max(K - S_cv, 0);
cash_flow = exercise_value(:, M);

% Backwards recursion
for i = (M-1):-1:1
    cash_flow = cash_flow * disc;           % Discount the cash-flow
    exercise = exercise_value(:, i);        % Value from immediate exercise
    itm = exercise > 0;                   % Paths whose immediate exercise is in the money

    [C, ~, mu] = polyfit(S_cv(itm, i), cash_flow(itm), b);     % Regression step
    Vcont = polyval(C, S_cv(itm, i), [], mu);                  % Approximated expected continuation value
    
    % Update cash_flow with exercise value where necessary
    optimal = Vcont < exercise(itm);
    itm(itm) = optimal;
    cash_flow(itm) = exercise(itm);
end

% Discount backwards and take the mean
price = mean(disc .* cash_flow);

% European price
price_eu = mean(disc_european * exercise_value(:, M));

price_CV = price + beta * (price_eu - european_bs);

end