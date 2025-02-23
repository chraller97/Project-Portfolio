% This is the main script for my bachelor "A study of different numerical methods for American option pricing"
clc; close all;

% Initial values
r = 0.06;           % Interest rate for risk-free bond
sigma = 0.25;       % Constant volatility of the asset
S0 = 50;           % Initial value of the asset at t = 0
K = 55;            % Agreed upon strike price
T = 2;             % Time of maturity
putcall = 0;       % Type of option - either put = 0 or call = 1
M = 32;          % Number of steps over time T

% ------------------------------------------------------------------------------
% This part plots the resulting S and V for the binomial method
% Note: the binomial() function needs to be changed to return S, V and t
% ------------------------------------------------------------------------------
%{
[S, V, t] = binomial(r, sigma, S0, K, T, 0, M);

deltaT = T/M;
figure(2)
hold on
for k = 1:M
    plot(S(1:k, k), t(k), "black+")
end
grid on
xlabel("Asset price S")
ylabel("Time t")
%title("Evolution of asset prices over time for the binomial method")
%saveas(2, "BinomS.png")

figure(3)
for k = 1:M
    plot3(S(1:k, k), repelem(t(k), k), V(1:k, k), "redo")
    hold on
    plot(S(1:k, k), t(k), "black+")
end
grid on
xlabel("Asset price S")
ylabel("Time t")
zlabel("Option price V")
%saveas(3, "BinomV.png")
%}



% ----------------------------------------------------------------
% This part studies the computational time of the binomial method
% ----------------------------------------------------------------
%{
Ms = 10:200:2010;
l = length(Ms);
times = zeros(1, l);

for i = 1:l
    disp(i)
    f = @() binomial(r, sigma, S0, K, T, 0, Ms(i));
    times(i) = timeit(f);
end

C = polyfit(Ms, times, 2);
times_fitted = polyval(C, Ms);
plot(Ms, times, "blueo")
hold on
plot(Ms, times_fitted, 'g')

% Confidence intervals
fit(Ms', times', "poly2")
%}



% --------------------------------------------------------------
% This part studies the convergence of the binomial method
% --------------------------------------------------------------
%{
true_val = binomial(r, sigma, S0, K, T, 0, 32500);

Ms = 10:100:10010;
dt = T./Ms;
l = length(Ms);
values = zeros(1, l);

for i = 1:l
    disp(i)
    values(i) = binomial(r, sigma, S0, K, T, 0, Ms(i));
end

err = abs(values - true_val);

C = polyfit(log(dt), log(err), 1);
convOrder = C(1);
fitted = exp(polyval(C, log(dt)));

figure(1)
loglog(dt, err)
hold on
grid on
plot(dt, fitted, "g")
fplot(@(x) x, [min(dt) max(dt)], "r")
%}



% -------------------------------------------------------------------------
% This part tests effects of b and M on convergence in the lsm algorithm
% -------------------------------------------------------------------------
%{
Ms = [50 100 150 200 250 300 350 400];
Ns = 500000;
bs = [3 5 7 9 11];
Ml = length(Ms);
bl = length(bs);
nsim = 10;

values = zeros(Ml, bl);

for i = 1:Ml
    for j = 1:bl
        disp([i j]);
        vals = zeros(1, nsim);
        for s = 1:nsim
            S = em2(S0, r, sigma, T, Ms(i), Ns);
            vals(s) = lsm(S, K, r, T, Ms(i), bs(j));
        end
        values(i, j) = mean(vals);
    end
end

err = abs(values - true_val);

figure(1)
hold on
for i = 1:bl
    plot(Ms, err(:, i), "DisplayName", strcat("b = ", num2str(bs(i))))
end
%}
% Result is: set b = 5 and M = 150 for the specific example



% -------------------------------------------------------------------------
% This part tests the computational time of the LSM algorithm
% -------------------------------------------------------------------------
%{
b = 5;
M = 150;
T = 4;
Ns = 5000:5000:350000;
l = length(Ns);
times = zeros(2, l);

for i = 1:l
    disp(i)
    S = em2(S0, r, sigma, T, M, Ns(i));
    f1 = @() lsm(S, K, r, T, M, b);
    f2 = @() lsmCV(S0, S, K, r, sigma, T, M, Ns(i), b);
    times(1, i) = timeit(f1);
    times(2, i) = timeit(f2);
end

C1 = polyfit(Ns, times(1, :), 1);
C2 = polyfit(Ns, times(2, :), 1);
times_fitted1 = polyval(C1, Ns);
times_fitted2 = polyval(C2, Ns);

plot(Ns, times(1, :), "Marker", "-", "DisplayName", "Original lsm")
hold on
plot(Ns, times(2, :), "Marker", "-", "DisplayName", "Control Variate lsm")

% Confidence intervals
fit1 = fit(Ns', times(1, :)', "poly1");
fit2 = fit(Ns', times(2, :)', "poly1");
%}



% -------------------------------------------------------------------------
% This part tests the convergence of the LSM algorithm
% -------------------------------------------------------------------------
%{
% First determine the "true" value of convergence
nsim = 100;
M = 150;
N = 500000;
b = 5;
V0_estimates = zeros(1, nsim);
parfor i = 1:nsim
    disp(i)
    S = em2(S0, r, sigma, T, M, N);
    V0_estimates(i) = lsmCV(S0, S, K,r,sigma, T, M, N, b);
end
true_val = mean(V0_estimates);
%}

% Actual test
M = 150;
b = 5;
Ns = 10000:10000:350000;
nsim = 200;
Nl = length(Ns);

values = zeros(2, Nl);
variances = zeros(2, Nl);

for i = 1:Nl
    disp(i)
    vals = zeros(2, nsim);
    for k = 1:nsim
        S = em2(S0, r, sigma, T, M, Ns(i));

        val = lsm(S, K, r, T, M, b);
        
        valCV = lsmCV(S0, S, K, r, sigma, T, M, Ns(i), b);
        
        vals(1, k) = val;
        vals(2, k) = valCV;
    end
    values(:, i) = mean(vals, 2);
    variances(:, i) = var(vals, 0, 2);
end

variance_reduction = variances(2, :) ./ variances(1, :);

err = abs(values - true_val);

C1 = polyfit(log(1 ./ Ns), log(err(1, :)), 1);
C2 = polyfit(log(1 ./ Ns), log(err(2, :)), 1);
fit1 = exp(polyval(C1, log(1 ./ Ns)));
fit2 = exp(polyval(C2, log(1 ./ Ns)));

% Confidence intervals
% fit(log(1 ./ Ns)', log(err(1, :))', "poly1");
% fit(log(1 ./ Ns)', log(err(2, :))', "poly1");

figure(1)
hold on
plot(Ns, err(1, :), "Displayname", "Original lsm")
plot(Ns, err(2, :), "Displayname", "Control Variate")

figure(3)
hold on
plot(Ns, variances(1, :), "DisplayName", "Original lsm")
plot(Ns, variances(2, :), "DisplayName", "Control Variate lsm")

figure(4)
hold on
plot(Ns, variance_reduction)

figure(5)
loglog(1 ./ Ns, err(1, :), "DisplayName", "Original lsm")
hold on
loglog(1 ./ Ns, err(2, :), "DisplayName", "Control Variate lsm")
plot(1 ./ Ns, fit1, "DisplayName", "Original lsm fit")
plot(1 ./ Ns, fit2, "DisplayName", "Control Variate lsm fit")

figure(6)
hold on
plot(Ns, values(1, :), "DisplayName", "Original lsm")
plot(Ns, values(2, :), "DisplayName", "Control Variate lsm")
yline(true_val)



% -------------------------------------------------------------------------
% This part estimates the early-exercise curve for an option with the lsm
% method
% -------------------------------------------------------------------------

%{
[V, curve, t, stock] = lsmEEC(50, 60, 0.06, 0.25, 4, 150, 3250000, 5);

figure(1)
plot(curve, t, "green")
hold on
plot(stock, t(1:length(stock)), "Color", "magenta")
grid on
line([S0, S0], [0, 4], "Color", "blue")
line([K, K], [0, 4], "Color", "red")
yline(4, "black")
ylim([0 4.5])
%}



% -------------------------------------------------------------------------
% This part compares (err, time) for all three methods
% -------------------------------------------------------------------------
%{
%true_val_B = binomial(r, sigma, S0, K, T, 0, 32500);

Ms = 10:40:5010;
l = length(Ms);
bin_values = zeros(1, l);
bin_times = zeros(1, l);
nsim = 10;

for i = 1:l
    disp(i)
    time = zeros(1, nsim);
    vals = zeros(1, nsim);
    for k = 1:nsim
        tic
        val = binomial(r, sigma, S0, K, T, 0, Ms(i));
        t = toc;
        time(k) = t;
        vals(k) = val;
    end
    bin_times(i) = mean(time);
    bin_values(i) = mean(vals);
end

bin_err = abs(bin_values - true_val_B);

figure(1)
loglog(bin_err, bin_times, "DisplayName", "Binomial")
%}


