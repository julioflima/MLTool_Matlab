function [FPE] = index_fpe(DATA,PAR)

% ---  Calculate Akaike's Final Prediction Error index for Clustering ---
%
%   [FPE] = index_fpe(DATA,PAR)
%
%   Input:
%       DATA.
%           dados = input matrix [k x N]
%       PAR.
%           C = prototypes [k x Neu]
%   Output:
%       FPE = FPE index

%% INIT

% Load Data

data = DATA.dados;
[k,N] = size(data);

% Load Parameters

[~,c] = size(PAR.C);

%% ALGORITHM

p = k*c;
RSS = PAR.SSE(end);
FPE = N*log(RSS/N) + N*log((N+p)/(N-p));

%% FILL OUTPUT STRUCTURE

% Dont Need

%% END