function [PARout] = elm_train(DATA,PAR)

% --- ELM classifier training function ---
%
%   [PARout] = elm_train(DATA,PAR)
%
%   Input:
%       DATA.
%           input = attributes [p x N]
%           output = labels [c x N]
%       PAR.
%           Nh = number of hidden neurons
%           Nlin = Non-linearity 
%               1 -> sigmoide [0 e 1]
%               2 -> hyperbolic tangent [-1 e +1]
%   Output:
%       PARout.
%           W = Weight Matrix (Hidden Layer)
%           M = Weight Matrix (Output Layer)

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(PAR))),
    PARaux.Nh = 25;         % Number of hidden neurons
    PARaux.Nlin = 2;        % Non-linearity
    PAR = PARaux;
    
else
    if (~(isfield(PAR,'Nh'))),
        PAR.Nh = 25;
    end
    if (~(isfield(PAR,'Nlin'))),
        PAR.Nlin = 2;
    end
end

%% INITIALIZATION

MATin = DATA.input;             % Input Matrix
MATout = DATA.output;           % Output Matrix

Nh = PAR.Nh;                    % number of hidden neurons
Nlin = PAR.Nlin;                % Non-linearity 

[p,N] = size(MATin);            % Size of input matrix

W = 0.01*(2*rand(Nh,p+1)-1);	% Weights of Hidden layer [-0.01,0.01]

Z = zeros(Nh+1,N);              % Activation matrix of hidden neurons

%% ALGORITHM

% X = input vector [p+1 x 1]
% d = desirable output [cte]

for t = 1:N,
    X  = [1; MATin(:,t)];    	% Add bias (x0 = +1) to input vector
    Ui = W*X;                   % Activation of hidden layer neurons
    Yi = elm_f_ativ(Ui,Nlin);	% Non-linear function
    Z(:,t) = [1; Yi];           % Add bias to activation vector
end

% Output Layer Weight Matrix Calculation (by Pseudo-inverse)

M = MATout*pinv(Z); % M = T1/Z;

%% FILL OUTPUT STRUCTURE

PARout = PAR;
PARout.W = W;
PARout.M = M;

%% THEORY

% ToDo - All

%% END