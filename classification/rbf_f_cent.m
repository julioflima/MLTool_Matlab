function [C] = rbf_f_cent(DATA,PAR)

% --- Initialize centroids for RBF ---
%
%   [C] = rbf_f_init(DATA,PAR)
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

input = DATA.input;
[p,N] = size(input);

% Get Parameters

init = PAR.init;
k = PAR.Nh;

% Initialize prototypes with 0

C = zeros(p,k);

%% ALGORITHM

if (init == 1)
    % Randomly choose one
    I = randperm(N);
    C = input(:,I(1:k));

elseif (init == 2)
    % Vector Quantization (kmeans)
    Hp.k = k;
    OUT = kmeans_cluster(DATA,Hp);
    C = OUT.C;
else
    disp('Unknown initialization. Prototypes = 0.');
    
end

%% END