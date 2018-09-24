function [PARout] = ps_train(DATA,PAR)

% --- PS classifier training ---
%
%   [PARout] = ps_train(DATA,PAR)
%
%   Input:
%       DATA.
%           input = attributes                          [p x N]
%           output = labels                             [c x N]
%       PAR.
%           Ne = maximum number of epochs               [cte]
%           eta = learning rate                         [0.01 0.1]
%   Output:
%       PARout.
%           W = Regression / Classification Matrix      [c x p+1]
%           MQEtr = mean quantization error of training	[1 x Ne]

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(PAR))),
    PARaux.Ne = 200;       	% maximum number of training epochs
    PARaux.eta = 0.05;    	% Learning step
    PAR = PARaux;
    
else
    if (~(isfield(PAR,'Ne'))),
        PAR.Ne = 200;
    end
    if (~(isfield(PAR,'eta'))),
        PAR.eta = 0.05;
    end
end

%% INITIALIZATIONS

% Data matrix
MATin = DATA.input;         % input matrix
MATout = DATA.output;       % output matrix

% Hyperparameters Init
Ne = PAR.Ne;                % maximum number of epochs
eta = PAR.eta;              % learning rate

% Problem Init
[Nc,~] = size(MATout);      % Classes of especific problem
No = Nc;                    % Number of neurons
[p,N] = size(MATin);        % Number of samples and parameters
MQEtr = zeros(1,Ne);        % Learning Curve

% Weight Matrix Init
W = 0.01*rand(No,p+1);      % Input layer -> output layer

MATin = [ones(1,N);MATin];  % add bias to input matrix [x0 = +1]

%% ALGORITHM

for ep = 1:Ne,   % for each epoch

    % Shuffle Data
    I = randperm(N); 
    MATin = MATin(:,I); 
    MATout = MATout(:,I);   
    
    % Init sum of quadratic errors
    SQE = 0;
    
    for t = 1:N,   % 1 epoch
        
        % Calculate Output
        X = MATin(:,t);         % Get sample
        U = W*X;                % Neurons Output
        Y = sign(U);            % Neurons Activation Function
        
        % Error Calculation
        Ek = MATout(:,t) - Y;    	% Quantization error or each neuron
        SQE = SQE + sum(Ek.^2);     % sum of quadratic errors
        
        % Atualização dos Pesos
        W = W + eta*Ek*X';
        
    end % end of one epoch
    
    % Mean squared error for each epoch
    MQEtr(ep) = SQE/N;

end % end of all epochs

%% FILL OUTPUT STRUCTURE

PARout = PAR;
PARout.W = W;
PARout.MQEtr = MQEtr;

%% THEORY

% ToDo - All

%% END