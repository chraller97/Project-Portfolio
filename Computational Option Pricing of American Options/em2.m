function [S] = em2(S0, r, sigma, T, M, N)
%EM2 Simulates N trajectory/realization of S from 0 to T given the inputs
%using the euler-maruyama method for SDEs and assuming the underlying asset
%follows a geometric brownian motion

deltaT = T / M;
sqrtDeltaT = sqrt(deltaT);
t = deltaT:deltaT:T;

Z = sqrtDeltaT * randn(N, M);
dW = cumsum(Z, 2);

S = S0*exp((r - sigma^2/2)*t + sigma*dW);

end