function [BIC] = index_bic(DATA,PAR)

% ---  Calculate BIC's index for Clustering ---
%
%   [BIC] = index_bic(DATA,PAR)
%
%   Input:
%       DATA.
%           dados = input matrix [k x N]
%       PAR.
%           C = prototypes [k x Neu]
%   Output:
%       BIC = BIC index

%% INIT

% Load Data

dados = DATA.dados;
[k,N] = size(dados);

% Load Parameters

[~,c] = size(PAR.C);

%% ALGORITHM

p = k*c;
RSS = PAR.SSE(end);
BIC = N*log(RSS/N) + p*log(N);

%% FILL OUTPUT STRUCTURE

% Dont Need

%% END