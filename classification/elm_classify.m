function [OUT] = elm_classify(DATA,PAR)

% --- ELM classifier test ---
%
%   [OUT] = elm_classify(DATA,PAR)
%
%   Input:
%       DATA.
%           input = attributes [p x N]
%           output = labels [c x N]
%       PAR.
%           W = Weight Matrix (Hidden Layer)
%           M = Weight Matrix (Output Layer)
%           Nh = number of hidden neurons
%           Nlin = Non-linearity 
%               1 -> sigmoide [0 e 1]
%               2 -> hyperbolic tangent [-1 e +1]
%   Output:
%       OUT.
%           y_h = classifier's output [c x N]
%           Mconf = classifier's confusion matrix [c x c]
%           acerto = classifier's accuracy rate [int]

%% INIT

MATin = DATA.input;         % Input data of the problem
MATout = DATA.output;      	% Output data of the problem

W = PAR.W;                  % Weight Matrix (Hidden Layer)
M = PAR.M;                  % Weight Matrix (Output Layer)
Nlin = PAR.Nlin;            % non-linearity
Nh = PAR.Nh;              	% number of hidden neurons

[Nc,N] = size(MATout);      % number of classes and samples

Z = zeros(Nh+1,N);        	% Activation matrix of hidden neurons

Mconf = zeros(Nc,Nc);       % classifier's confusion matrix [c x c]

%% ALGORITHM

% X = input vector [p+1 x 1]
% d = desirable output [cte]

% Calculate estimate function

for t = 1:N,
    X  = [+1; MATin(:,t)];    	% Add bias (x0 = +1) to input vector
    Ui = W*X;                   % Activation of hidden layer neurons   
    Yi = elm_f_ativ(Ui,Nlin);   % Non-linear function
    Z(:,t) = [+1; Yi];          % Add bias to activation vector
end

y_h = M*Z;

% Calculate success rate and confusion matrix

for t = 1:N,
    [~,iT2] = max(MATout(:,t));     % Desired Output index
    [~,iY_h] = max(y_h(:,t));       % Estimated Output index
    Mconf(iT2,iY_h) = Mconf(iT2,iY_h) + 1;
end

acerto = sum(diag(Mconf)) / N;

%% FILL OUTPUT STRUCTURE

OUT.y_h = y_h;
OUT.Mconf = Mconf;
OUT.acerto = acerto;

%% END