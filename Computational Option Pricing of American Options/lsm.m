function [prices] = lsm(S, K, r, T, M, b)
%LSM Calculates the value of an American option (put), with the given parameters,
%using the Least Squares Monte Carlo (LSM) method by Longstaff and Schwartz
%(2001)

% Calculate the step-szie and discount value
dt = T / M;
disc = exp(-r * dt);

% The option payoff for each time step of each path
exercise_value = max(K - S, 0);
cash_flow = exercise_value(:, M);

% Backwards recursion
for i = (M-1):-1:1
    cash_flow = cash_flow * disc;           % Discount the cash-flow
    exercise = exercise_value(:, i);        % Value from immediate exercise
    itm = exercise > 0;                   % Paths whose immediate exercise is in the money

    [C, ~, mu] = polyfit(S(itm, i), cash_flow(itm), b);     % Regression step
    Vcont = polyval(C, S(itm, i), [], mu);                  % Approximated expected continuation value
    
    % Update cash_flow with exercise value where necessary
    optimal = Vcont < exercise(itm);
    itm(itm) = optimal;
    cash_flow(itm) = exercise(itm);
end

% Discount backwards and take the mean
prices = mean(disc .* cash_flow);

end