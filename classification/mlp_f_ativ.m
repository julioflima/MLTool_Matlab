function [Yi] = mlp_f_ativ(Ui,option)

% --- MLP Activation Function ---
%
%	[Yi] = mlp_f_ativ(Ui,option)
%
%   input:
%       Ui = neuron activation
%       option = type of activation function
%           = 1: Sigmoidal -> output: [0 e 1]
%           = 2: Hyperbolic Tangent -> output: [-1 e +1]
%   Output:
%       Yi = result of non-linear function

%% ALGORITHM

switch option
    case (1)    % [0 e 1]
        Yi = 1./(1+exp(-Ui));
    case (2)    % [-1 e +1]
        Yi = (1-exp(-Ui))./(1+exp(-Ui));
    otherwise
        Yi = Ui;
        disp('Invalid activation function option')
end

%% END