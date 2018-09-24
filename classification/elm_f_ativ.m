function [Yi] = elm_f_ativ(Ui,option)

% --- ELM Classifier activation function  ---
%
%   [Yi] = elm_f_ativ(Ui,option)
%
%   Input:
%       Ui = activation result
%       option = activation function type
%           1 -> sigmoide [0 e 1]
%           2 -> hyperbolic tangent [-1 e +1]
%   Output:
%       Yi = output of non-linear function
%

%% ALGORITHM

switch option
    case (1)
        Yi = 1./(1+exp(-Ui));
    case (2)
        Yi = (1-exp(-Ui))./(1+exp(-Ui));
    otherwise
        Yi = Ui;
        disp('Type a correct option. There wasn`t activation')
end

%% END