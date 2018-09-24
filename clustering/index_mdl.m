function [MDL] = index_mdl(DATA,PAR)

% ---  Calculate MDL's index for Clustering ---
%
%   [MDL] = mdl_index(DATA,PAR)
%
%   Input:
%       DATA.
%           dados = input matrix [k x N]
%       PAR.
%           C = prototypes [k x Neu]
%   Output:
%       MDL = MDL index

%% INIT

% Load Data

data = DATA.dados;
[k,N] = size(data);

% Load Parameters

[~,c] = size(PAR.C);

%% ALGORITHM

p = k*c;
RSS = PAR.SSE(end);
MDL = N*log(RSS/N) + (p/2)*log(N);

%% FILL OUTPUT STRUCTURE

% Dont Need

%% END