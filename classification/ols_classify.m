function [OUT] = ols_classify(DATA,PAR)

% --- OLS classifier test ---
%
%   [OUT] = ols_classify(DATA,PAR)
%
%   Input:
%       DATA.
%           input = attributes matrix [p x N]
%           output = labels matrix [c x N]
%       PAR.
%           A = transformation matrix [c x p]
%   Output:
%       OUT.
%           y_h = classifier's output [c x N]
%           Mconf = classifier's confusion matrix [c x c]
%           acerto = classifier's accuracy rate [cte]

%% INITIALIZATIONS

MATin = DATA.input;                 % input matrix
MATout = DATA.output;               % output matrix

A = PAR.A;                          % transformation matrix 

[Nc,N] = size(MATout);              % Number of classes and samples

MATin = [ones(1,N) ; MATin];        % add bias to input matrix [x0 = +1]

Mconf = zeros(Nc,Nc);               % Confusion Matrix

%% ALGORITHM

% Function output

y_h = A * MATin;

% Calculate success rate and confusion matrix

for t = 1:N,
    [~,iT2] = max(MATout(:,t));     % Desired Output index
    [~,iY_h] = max(y_h(:,t));       % Estimated Output index
    Mconf(iT2,iY_h) = Mconf(iT2,iY_h) + 1;
end

acerto = sum(diag(Mconf)) / N;

%% FILL OUTPUT STRUCTURE

OUT.y_h = y_h;
OUT.Mconf = Mconf;
OUT.acerto = acerto;

%% END