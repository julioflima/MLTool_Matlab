function [PARout] = rbf_train(DATA,PAR)

% --- RBF Classifier Training ---
%
%   [PARout] = rbf_train(DATA,PAR)
%
%   Input:
%       DATA.
%           input = attributes [p x N]
%           output = labels [c x N]
%       PAR.
%           Nh = number of hidden neurons   [q]
%           init = type of centroids initialization   [cte]
%           rad = type of radius / spread calculation [cte]
%           out = way to calculate output weights     [cte]
%           ativ = activation function type           [cte]
%   Output:
%       PARout.
%           W = Hidden layer weight Matrix 	[p x q]
%           r = radius of each rbf         	[1 x q]
%           M = Output layer weight Matrix	[c x q+1]

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(PAR))),
    PARaux.Nh = 10;  	% number of hidden neurons
    PARaux.init = 2; 	% centroids initialization type
    PARaux.rad = 2;  	% radius / spread type
    PARaux.out = 1;    	% how to calculate output weights
    PARaux.ativ = 1;   	% activation function type
    PAR = PARaux;
else
    if (~(isfield(PAR,'Nh'))),
        PAR.Nh = 10;
    end
    if (~(isfield(PAR,'init'))),
        PAR.init = 2;
    end
    if (~(isfield(PAR,'rad'))),
        PAR.rad = 2;
    end
    if (~(isfield(PAR,'out'))),
        PAR.out = 1;
    end
    if (~(isfield(PAR,'ativ'))),
        PAR.ativ = 1;
    end
end

%% ALGORITHM

% First Step: Centroids Determination

W = rbf_f_cent(DATA,PAR);           % select centroids of each rbf

% Second Step: Radius / spread of each centroid

r = rbf_f_radius(W,PAR);            % select spread of each rbf

% Third Step: Output layer weight Matrix

M = rbf_f_outweig(DATA,PAR,W,r);    % calculate output weights matrix

%% FILL OUTPUT STRUCTURE

PARout = PAR;
PARout.W = W;
PARout.r = r;
PARout.M = M;

%% THEORY

% Radial Basis Function
% Only one hidden layer
% Hidden layer neurons -> activation functions = radial basis functions 
% Output layer neurons -> activation functions = linear

%% END