function [PARout] = ols_train(DATA,PAR)

% --- OLS classifier training ---
%
%   [PARout] = ols_train(DATA,PAR)
%
%   Input:
%       DATA.
%           input = input matrix [p x N]
%           output = output matrix [c x N]
%       PAR.
%           aprox = type aproximation
%               1 -> A = T*pinv(P);
%               2 -> A = T*P'/(P*P');
%               3 -> A = T/P;
%   Output:
%       PARout.
%           A = linear transformation matrix [c x p]

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(PAR))),
    PARaux.aprox = 1;       % Neurons' labeling function
    PAR = PARaux;
    
else
    if (~(isfield(PAR,'aprox'))),
        PAR.aprox = 1;
    end
end

%% INITIALIZATIONS

MATin = DATA.input;             % Input matrix
MATout = DATA.output;       	% Output matrix

[~,N] = size(MATout);           % Number of samples

MATin = [ones(1,N) ; MATin];	% add bias to input matrix [x0 = +1]

aprox = PAR.aprox;            	% Type of approximation

%% ALGORITHM

if aprox == 1,
    A = MATout*pinv(MATin);
elseif aprox == 2,
    A = MATout*MATin'/(MATin*MATin');
elseif aprox == 3,
    A = MATout/MATin;
else
    disp('type a valid option: 1 or 2');
    A = [];
end

%% FILL OUTPUT STRUCTURE

PARout = PAR;
PARout.A = A;

%% THEORY

% Considering T = MATout the output matrix
% Considering P = MATin  the input matrix
% Considering A the linear transformation between input and output

%    T    =    A    *    P
% [c x N] = [c x p] * [p x N]
%
% A*P = T
% A*P*P' = T*P'
% A*(P*P')*(P*P')-1 = T*P'*(P*P')-1
% A = T*P'*(P*P')-1
% 
% Pseudo-inverse: P'*(P*P')-1

%    T    =    P    *    A
% [N x c] = [N x p] * [p x c]
% 
% P*A = T
% P'*P*A = P'*T
% (P'*P)-1*(P'*P)*A = (P'*P)-1*P'*T
% A = (P'*P)-1*P'*T
%
% Pseudo-inverse: (P'*P)-1*P'

%% END