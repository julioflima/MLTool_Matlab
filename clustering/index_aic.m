function [AIC] = index_aic(DATA,PAR)

% ---  Calculate Akaike's index for Clustering ---
%
%   [AIC] = index_aic(DATA,PAR)
%
%   Input:
%       DATA.
%           dados = input matrix [k x N]
%       PAR.
%           C = prototypes [k x Neu]
%   Output:
%       AIC = AIC index

%% INIT

% Load Data

dados = DATA.dados;
[k,N] = size(dados);

% Load Parameters

[~,c] = size(PAR.C);

%% ALGORITHM

p = k*c;
RSS = PAR.SSE(end);
AIC = N*log(RSS/N) + 2*p;

%% FILL OUTPUT STRUCTURE

% Dont Need

%% END