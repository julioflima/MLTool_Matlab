function [PARout] = kmeans_label(DATA,OUT_CL)

% --- k-means Labeling Function ---
%
%   [PARout] = kmeans_label(DATA,OUT_CL)
% 
%   Input:
%       DATA.
%           input = input matrix [p x N]
%           output = output matrix [1 x N]
%       OUT_CL.
%           C = prototypes [p x Neu]
%           index = [1 x N]
%           SSE = [1 x Nep]
%           lbl = type of labeling [cte]
%   Output:
%       PARout.
%           C = prototypes [p x Neu]
%           index = [1 x N]
%           SSE = [1 x Nep]
%           label = class of each neuron [1 x Neu]

%% ALGORITHM

[PARout] = prototypes_label(DATA,OUT_CL);

%% END