function [S, t] = upgraded_em(S0, t0, T, M, funcA, parA, funcB, parB)
%EM Simulates a trajectory/realization of S from t0 to T given the inputs
%using the euler-maruyama method for SDEs.
deltaT = (T - t0) / M;
sqrtDeltaT = sqrt(deltaT);
 
S = zeros(1, M);
t = zeros(1, M);
S(1) = S0;
t(1) = t0;

Z = normrnd(0, 1, [1, M-1]);
dW = Z .* sqrtDeltaT;

for j = 2:M
    t(j) = t(j-1) + deltaT;
    
    parA(end) = S(j-1);
    a = funcA(parA);
    parB(end) = S(j-1);
    b = funcB(parB);
    
    S(j) = S(j-1) + a * deltaT + b * dW(j-1);
end



end

