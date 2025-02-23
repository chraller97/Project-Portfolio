function [S] = em(S0, r, sigma, T, M, N)
%EM2 Simulates N trajectory/realization of S from 0 to T given the inputs
%using the euler-maruyama method for SDEs and assuming the underlying asset
%follows a geometric brownian motion

deltaT = T / M;
sqrtDeltaT = sqrt(deltaT);
 
S = zeros(N, M+1);
S(:, 1) = S0;

Z = randn(N, M);
dW = [zeros(N, 1), Z .* sqrtDeltaT];

for j = 2:(M+1)
    S(:, j) = S(:, j-1) + r * S(:, j-1) * deltaT + sigma * S(:, j-1) .* dW(:, j);
end
S = S(:, 2:M+1);

end