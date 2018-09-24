function [CH] = index_ch(DATA,PAR)

% ---  Calculate CH index for Clustering ---
%
%   [CH] = index_ch(DATA,PAR)
%
%   Input:
%       DATA.
%           dados = input matrix [p x N]
%       PAR.
%           index = index of each sample [1 x N]
%   Output:
%       CH = CH index

%% INIT

% Load Data

data = DATA.dados;
[p,N] = size(data);

% Load Parameters

labels = PAR.index;
k = length(find(unique(labels)));

% Init Clusters

clusters = cell(1,k);
for l=1:k,
    J = find(labels == l);
    VJ = data(:,J);
    clusters{l} = VJ';
end

% Init Aux Variables

M = mean(data');	% Centroid of the whole dataset
Ni = 0;             % accumulates number of samples per cluster
Wq = zeros(p);      % Initial value of within-cluster scatter matrix
Bq = zeros(p);      % Initial value of between-cluster scatter matrix

%% ALGORITHM

if k == 1
    CH = NaN;
else
    for j=1:k,
        nj = size(clusters{j});	% Number of samples of cluster j
        mj = mean(clusters{j});	% Mean of cluster j
        Cj = clusters{j};
        Swj = cov(Cj,1);        % Within-cluster scatter matrix of cluster j
        Sbj = (mj-M)'*(mj-M);	% Between-cluster scatter matrix of cluster j
        Wq = Wq + nj(1)*Swj;
        Bq = Bq + nj(1)*Sbj;
        Ni = Ni + nj(1);
    end
    
    Wq = Wq/Ni;  % Final within-cluster scatter matrix for K-Means algorithm
    Bq = Bq/Ni;  % Final between-cluster scatter matrix for K-Means algorithm
    
    CH = (trace(Bq)/(k-1)) / (trace(Wq)/(N-k));
end

%% FILL OUTPUT STRUCTURE

% Dont Need

%% END