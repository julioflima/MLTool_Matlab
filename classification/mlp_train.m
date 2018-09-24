function [PARout] = mlp_train(DATA,PAR)

% --- MLP Classifier Training ---
%
%   [PARout] = mlp_train(DATA,PAR)
%
%   Input:
%       DATA.
%           input = attributes [p x N]
%           output = labels [c x N]
%       PAR.
%           Ne = maximum number of epochs	[cte]
%           Nh = number of hidden neurons  	[cte]
%           eta = learning rate           	[0.01 0.1]
%           mom = moment factor            	[0.5 1.0]
%           Nlin = non-linearity           	[cte]
%               = 1 -> sigmoidal                             
%               = 2 -> hyperbolic tangent
%   Output:
%       PARout.
%           W = Weight Matrix: Inputs to Hidden Layer           [Nh x p+1]
%           M = Weight Matrix: Hidden layer to output layer     [No x Nh+1]
%           MQEtr = Mean Quantization Error                     [Ne x 1]
%               (learning curve)

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(PAR))),
    PARaux.Nh = 10;       	% Number of hidden neurons
    PARaux.Ne = 200;     	% Maximum training epochs
    PARaux.eta = 0.05;     	% Learning Step
    PARaux.mom = 0.75;    	% Moment Factor
    PARaux.Nlin = 2;       	% Non-linearity
    PAR = PARaux;
    
else
    if (~(isfield(PAR,'Nh'))),
        PAR.Nh = 10;
    end
    if (~(isfield(PAR,'Ne'))),
        PAR.Ne = 200;
    end
    if (~(isfield(PAR,'eta'))),
        PAR.eta = 0.05;
    end
    if (~(isfield(PAR,'mom'))),
        PAR.mom = 0.75;
    end
    if (~(isfield(PAR,'Nlin'))),
        PAR.Nlin = 2;
    end
end

%% INITIALIZATIONS

% Data Initialization
MATin = DATA.input;         % Input Matrix
MATout = DATA.output;       % Output Matrix

% Hyperparameters Initialization
Ne = PAR.Ne;                % Number of training epochs
Nh = PAR.Nh;                % Number of hidden layer
eta = PAR.eta;              % learning rate 
mom = PAR.mom;              % Moment Factor
Nlin = PAR.Nlin;            % Non-linearity

% Problem Initialization
[No,~] = size(MATout);      % Number of Classes and Output Neurons
[p,N] = size(MATin);        % attributes and samples
MQEtr = zeros(1,Ne);        % Mean Quantization Error

% Weight matrix Initialization
W = 0.01*rand(Nh,p+1);      % Hidden Neurons weights
W_old = W;                  % necessary for moment factor
M = 0.01*rand(No,Nh+1);     % Output Neurons weights
M_old = M;                  % necessary for moment factor

% Add bias to input matrix
MATin = [ones(1,N);MATin];  % x0 = +1

%% ALGORITHM

for ep = 1:Ne,   % for each epoch

    % Shuffle Data
    I = randperm(N);        
    MATin = MATin(:,I);     
    MATout = MATout(:,I);   
    
    SQE = 0; % Init sum of quadratic errors
    
    for t = 1:N,   % for each sample
            
        % HIDDEN LAYER
        X = MATin(:,t);             % Get input sample
        Ui = W * X;                 % Activation of hidden neurons 
        Yi = mlp_f_ativ(Ui,Nlin);	% Non-linear function
        
        % OUTPUT LAYER
        Y = [+1; Yi];               % build input of output layer
        Uk = M * Y;                 % Activation of output neurons
        Ok = mlp_f_ativ(Uk,Nlin);	% Non-linear function
        
        % ERROR CALCULATION
        Ek = MATout(:,t) - Ok;   	% error between desired output and estimation
        SQE = SQE + sum(Ek.^2);     % sum of quadratic error
        
        % LOCAL GRADIENTS - OUTPUT LAYER
        Dk = mlp_f_gradlocal(Ok,Nlin);	% derivative of activation function
        DDk = Ek.*Dk;                  	% local gradient (output layer)
        
        % LOCAL GRADIENTS -  HIDDEN LAYER
        Di = mlp_f_gradlocal(Yi,Nlin); 	% derivative of activation function
        DDi = Di.*(M(:,2:end)'*DDk);    % local gradient (hidden layer)
        
        % WEIGHTS ADJUSTMENT - OUTPUT LAYER
        MM_aux = M;                             % Hold current weights
        M = M + eta*DDk*Y' + mom*(M - M_old);   % Update current weights
        M_old = MM_aux;                         % Hold last weights
        
        % WEIGHTS ADJUSTMENT - HIDDEN LAYER
        WW_aux = W;                             % Hold current weights
        W = W + eta*DDi*X' + mom*(W - W_old);   % Update current weights
        W_old = WW_aux;                         % Hold last weights
        
    end   % end of epoch
        
        % Mean Squared Error of epoch
        MQEtr(ep) = SQE/N;
        
end   % end of all epochs

%% FILL OUTPUT STRUCTURE

PARout = PAR;
PARout.W = W;
PARout.M = M;
PARout.MQEtr = MQEtr;

%% THEORY

% ToDo - All

%% END