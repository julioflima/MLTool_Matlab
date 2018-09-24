function [OUT] = ps_classify(DATA,PAR)

% --- PS classifier test ---
%
%   [OUT] = ps_classify(DATA,PAR)
%
%   Input:
%       DATA.
%           dados = attributes [p x N]
%           alvos = labels [c x N]
%       PAR.
%           W = transformation matrix [c x p]
%   Output:
%       OUT.
%           y_h = classifier's output [c x N]
%           Mconf = classifier's confusion matrix [c x c]
%           acerto = classifier's accuracy rate [int]
%           MQEte = mean quantization error of test

%% INICIALIZAÇÕES

Q = DATA.input;         % input matrix
T2 = DATA.output;       % output matrix

W = PAR.W;              % Weight matrix

[Nc,N] = size(T2);      % Number of samples and classes

Q = [ones(1,N) ; Q];    % add bias to input matrix [x0 = +1]

y_h = zeros(Nc,N);      % Estimated Matrix

MQEte = 0;              % Mean Quantization Error
Mconf = zeros(Nc,Nc);   % Confusion Matrix

%% ALGORITMO

for t = 1:N,
    
	X = Q(:,t);         % Get Sample
    U = W*X;            % Neuron Output
    Y = sign(U);        % Neuron Activation
    y_h(:,t) = Y;       % Accumulate output
    
    Ek = T2(:,t) - Y;   % Quantization errror
    
    MQEte = MQEte + 0.5*sum(Ek.^2); % Mean Quantization Error
    
end

% Confusion Matrix

for t = 1:N,
    
    [~, iT2] = max(T2(:,t));    % Desired Output index
    [~, iY_h] = max(y_h(:,t));  % Estimated Output index
    Mconf(iT2,iY_h) = Mconf(iT2,iY_h) + 1;
end

% Accuracy

acerto = sum(diag(Mconf))/N;

%% FILL OUTPUT STRUCTURE

OUT.y_h = y_h;
OUT.Mconf = Mconf;
OUT.acerto = acerto;
OUT.MQEte = MQEte;

%% END