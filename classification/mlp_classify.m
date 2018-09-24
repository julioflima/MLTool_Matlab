function [OUT] = mlp_classify(DATA,PAR)

% --- MLP Classifier Test ---
%
%   [OUT] = mlp_classify(DATA,PAR)
%
%   Input:
%       DATA.
%           input = attributes [p x N]
%           output = labels [c x N]
%       PAR.
%           W = Hidden layer weight Matrix     [Nh x p+1]
%           M = Output layer weight Matrix     [No x Nh+1]
%           Nlin = Non-linearity
%               1 -> sigmoidal 
%               2 -> tg. hiperbolica
%   Output:
%       OUT.
%           y_h = classifier's output               [c x N]
%           Mconf = classifier's confusion matrix   [c x c]
%           acerto = classifier's accuracy rate     [int]
%           MQEte = mean quantizarion error per test epoch
%               (learning curve)

%% INITIALIZATIONS

% Get data
MATin = DATA.input;       	% Get attribute matrix
MATout = DATA.output;    	% Get label matrix

% Get parameters
W = PAR.W;                  % Hidden layer weight Matrix
M = PAR.M;                  % Output layer weight Matrix
Nlin = PAR.Nlin;            % Non-linearity
[Nc,N] = size(MATout);      % Number of classes and samples

% Initialize Outputs
y_h = zeros(Nc,N);          % Estimated output
MQEte = 0;                  % Quantization error
Mconf = zeros(Nc,Nc);       % Confusion Matrix

% Add bias to input matrix
MATin = [ones(1,N);MATin];  % x0 = +1

%% ALGORITHM

% Generate outputs

for t = 1:N,

    % HIDDEN LAYER
    X = MATin(:,t);            	% Get input sample
    Ui = W*X;                   % Activation of hidden neurons
    Yi = mlp_f_ativ(Ui,Nlin);   % Non-linear function
    
    % OUTPUT LAYER
    Y = [+1; Yi];               % Vetor de entrada + bias (y0 = +1)
    Uk = M*Y;                   % Ativações dos neuronios da camada de saida
    Ok = mlp_f_ativ(Uk,Nlin);   % Função nao-linear
    
    y_h(:,t) = Ok;              % Hold neural net output
    
    Ek = MATout(:,t) - Ok;   	% Quantization error
    
    MQEte = MQEte + sum(Ek.^2); % Global quantization erroR

end

% Calculate success rate and confusion matrix

for t = 1:N,
    [~,iT2] = max(MATout(:,t));   	% Desired Output index
    [~,iY_h] = max(y_h(:,t));     	% Estimated Output index
    Mconf(iT2,iY_h) = Mconf(iT2,iY_h) + 1;
end

acerto = sum(diag(Mconf)) / N;

%% FILL OUTPUT STRUCTURE

OUT.y_h = y_h;
OUT.Mconf = Mconf;
OUT.acerto = acerto;
OUT.MQEte = MQEte;

%% END