function [OUT] = kmeans_classify(DATA,PAR)

% --- k-means classify function ---
%
%   [OUT] = kmeans_classify(DATA,PAR)
% 
%   Input:
%       DATA.
%           input = input matrix [p x N]
%           output = output matrix [1 x N]
%       PAR.
%           C = prototypes [p x Neu]
%           index = [1 x N]
%           SSE = [1 x Nep]
%           label = class of each neuron [1 x Neu]
%   Output:
%       OUT.
%           y_h = classifier's output [1 x N]
%           Mconf = classifier's confusion matrix [c x c]
%           acerto = classifier's accuracy rate [int]

%% ALGORITHM

[OUT] = prototypes_class(DATA,PAR);

%% END