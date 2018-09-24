function [M] = rbf_f_outweig(DATA,PAR,W,r)

% --- Calculates Output Matrix Weights ---
%
%   [M] = rbf_f_outweig(DATA,PAR,W,r)
%
%   Input:
%       DATA.
%           input = input matrix [p x N]
%       PAR.
%           Nh = number of hidden neurons (centroids)
%           init = how will be the prototypes' initial values
%               1: Forgy Method (randomly choose k observations from data set)
%               2: Vector Quantization (kmeans)
%   Output:
%       C = prototypes matrix [p x k]

%% INITIALIZATIONS

% Get Data

MATin = DATA.input;
MATout = DATA.output;
[k,N] = size(MATout);

% Get Parameters

q = PAR.Nh;
out_type = PAR.out;
ativ = PAR.ativ;

% Initialize Output Matrix Weights

M = zeros(k,q+1);

%% ALGORITHM

% Calculates hidden layer activations

Z = zeros(q+1,N);               % Output of hidden layer
for t = 1:N
    x = MATin(:,t);             % get sample
    z_vec = zeros(q,1);      	% init output of basis functions
    for i = 1:q,
        Ci = W(:,i);            % Get rbf center
        ui = sum((x - Ci).^2);  % Calculates distance
        ri = r(i);              % get radius
        z_vec(i) = rbf_f_ativ(ui,ri,ativ);
    end
    z_vec = z_vec/sum(z_vec);  	% normalize outputs
    Z(:,t) = [1;z_vec];        	% add bias to z vector
end

% Calculates

if (out_type == 1) % Through OLS
    % M = D*Z'*inv(Z*Z');           % direct calculus of inverse
    % M = D/Z;                      % QR factorization
    M = MATout*pinv(Z);          	% uses SVD to estimate M matrix

elseif (out_type == 2) % Through LMS (Adaline)
    % ToDo - all
    % Obs: "Output Layer Output" belongs to real numbers)
    
else
    disp('Unknown initialization. Output Weights = 0.');
    
end

%% END