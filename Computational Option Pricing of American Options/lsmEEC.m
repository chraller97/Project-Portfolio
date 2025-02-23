function [price, curve, t, stock] = lsmEEC(S0, K, r, sigma, T, M, N, b)
%LSMEEC Calculates the value of an American option (put), with the given parameters,
%using the Least Squares Monte Carlo (LSM) method by Longstaff and Schwartz
%(2001) alongside estimating the Earl-Exercise curve.

% Calculate the step-szie and discount value
dt = T / M;
disc = exp(-r * dt);
t = dt:dt:T;

% Simulating N trajectories of the underlying asset S
S = em2(S0, r, sigma, T, M, N);

% The option payoff for each time step of each path
exercise_value = max(K - S, 0);
cash_flow = exercise_value(:, M);

EarlyExercise = zeros(N, M);
EarlyExercise(:, M) = cash_flow > 0;

% Backwards recursion
for i = (M-1):-1:1
    cash_flow = cash_flow * disc;           % Discount the cash-flow
    exercise = exercise_value(:, i);        % Value from immediate exercise
    itm = logical(exercise);                   % Paths whose immediate exercise is in the money

    [C, ~, mu] = polyfit(S(itm, i), cash_flow(itm), b);     % Regression step
    Vcont = polyval(C, S(itm, i), [], mu);                  % Approximated expected continuation value
    
    % Update cash_flow with exercise value where necessary
    optimal = Vcont < exercise(itm);
    itm(itm) = optimal;
    cash_flow(itm) = exercise(itm);
    EarlyExercise(:, i) = itm;
    EarlyExercise(itm, (i+1):M) = 0;
end

% Discount backwards and take the mean
price = mean(disc .* cash_flow);

% Random price path
Nidx = randsample(N, 1);
Midx = find(EarlyExercise(Nidx, :));
stock = S(Nidx, 1:Midx);

% This part attempts to construct an early exercise curve
S(EarlyExercise == 0) = 0;

curve = sum(S, 1) ./ sum(S~=0);

end