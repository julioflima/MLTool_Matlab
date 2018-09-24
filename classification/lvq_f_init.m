function [codebook] = lvq_f_init(DATA,PAR)

% --- Initialize prototypes for clustering ---
%
%   [codebook] = prototypes_init(DATA,PAR)
%
%   Input:
%       DATA.
%           input = input matrix [p x N]
%           output = output matrix [1 x N]
%       PAR.
%           k = number of prototypes
%           init = how will be the prototypes' initial values
%               1: C = zeros (mean of normalized data)
%               2: Forgy Method (randomly choose k observations from data set)
%               3: Randomly assign a cluster to each observation, than
%                  update clusters' centers
%               4: prototype's values randomly choosed between min and
%                  max values of data's atributtes.
%   Output:
%       codebook.
%           C = prototypes matrix [p x k]
%           labels = prototypes labels [1 x k]

%% INITIALIZATIONS

% Get Data

input = DATA.input;
[p,N] = size(input);
output = DATA.output;
Nc = length(unique(output));

% Get Parameters

init = PAR.init;
k = PAR.k;

% Init output

C = zeros(p,k);
label = zeros(1,k);

%% ALGORITHM
   

if (init == 1)
    % Calculates prototypes per class
    ppc = floor(k/Nc);
    % init counter
    cont = 0;
    % assign labels to prototypes 
    for i = 1:Nc,
        if i ~= Nc,
            label(cont+1:cont+ppc) = i;
        else
            label(cont+1:end) = i;
        end
        cont = cont + ppc;
    end
    
elseif (init == 2)
    I = randperm(N);
    C = input(:,I(1:k));
    label = output(:,I(1:k));
    
elseif (init == 3)
    % Initialize number of samples for each cluster
    n_samples = zeros(1,k);
    % initialize randomly each sample index
    I = rand(1,N);
    index = ceil(k*I);
    % calculate centroids
    for i = 1:N,
        n_samples(index(i)) = n_samples(index(i)) + 1;
        C(:,index(i)) = C(:,index(i)) + input(:,i);
    end
    for i = 1:k,
        C(:,i) = C(:,i) / n_samples(i);
    end
    
    % ToDo - Add label
    
else
    disp('Unknown initialization. Prototypes = 0.');
end

%% FILL OUTPUT STRUCTURE

codebook.C = C;
codebook.label = label;

%% END