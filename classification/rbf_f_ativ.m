function [Zi] = rbf_f_ativ(ui,ri,ativ)

% --- RBF Activation Function ---
%
%	[Zi] = rbf_f_ativ(Ui,ativ)
%
%   input:
%       Ui = neuron output [cte]
%       ativ = type of activation function
%           = 1: gaussian -> 
%           = 2: multiquadratic ->
%           = 3: inverse multiquadratic -> 
%   Output:
%       Yi = result of activation function

%% ALGORITHM

switch ativ
    case (1)    % Guassian
        Zi = exp(-(ui^2)/(2*(ri^2)));
    case (2)    % multiquadratic
        Zi = sqrt(ri^2 + ui^2);
    case (3)    % inverse multiquadratic
        Zi = 1/sqrt(ri^2 + ui^2);
    otherwise
        disp('Invalid activation function option')
end

%% END