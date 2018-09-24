function [PARout] = som1d_label(DATA,OUT_CL)

% --- SOM 1D Labeling Function ---
%
%   [PARout] = som1d_label(DATA,OUT_CL)
% 
%   Input:
%       DATA.
%           input = input matrix [p x N]
%           ouput = output matrix [c x N]
%       OUT_CL.
%           C = prototypes [p x Neu]
%           index = [1 x N]
%           SSE = [1 x Nep]
%   Output:
%       PARout.
%           C = prototypes [p x Neu]
%           index = [1 x N]
%           SSE = [1 x Nep]
%           label = class of each neuron [1 x Neu]

%% ALGORITHM

[PARout] = prototypes_label(DATA,OUT_CL);

%% END