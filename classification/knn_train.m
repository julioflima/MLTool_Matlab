function [PARout] = knn_train(DATA,PAR)

% --- KNN classifier training ---
%
%   [PARout] = knn_train(DATA,PAR)
%
%   Input:
%       DATA.
%           input = training attributes [p x N]
%           output = training labels [c x N]
%       PAR.
%           k = number of nearest neighbors
%   Output:
%       PARout.
%           k = number of nearest neighbors
%           dados = training attributes [p x N]
%           alvos = training labels [c x N]

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(PAR))),
    PARaux.k = 3;      	% Number of nearest neighbors
    PAR = PARaux;
    
else
    if (~(isfield(PAR,'k'))),
        PAR.k = 3;
    end
end

%% ALGORITHM

PARout = PAR;
PARout.input = DATA.input;
PARout.output = DATA.output;

%% THEORY

% ToDo - All

%% END