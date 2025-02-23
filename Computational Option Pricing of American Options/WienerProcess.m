function [W, t] = WienerProcess(W0, M, T)
%WIENERPROCESS Returns a wiener process evolving from 0 to T. By definition
%W0 = 0, but an option is included to set W0 = constant

% Determine time-step
deltaT = T/M;
sqrtDeltaT = sqrt(deltaT);

% Pre-allocate space for W and t
W = zeros(1, M);
t = zeros(1, M);

% Initial value
t(1) = 0;
W(1) = W0;
Z = normrnd(0, 1, [1, M-1]);

for j = 2:M
    t(j) = t(j-1) + deltaT; % update t
    W(j) = W(j-1) + Z(j-1) * sqrtDeltaT; % Calculate next value of W
end

end

