function [C] = prototypes_init(DATA,PAR)

% --- Initialize prototypes for clustering ---
%
%   [C] = prototypes_init(DATA,PAR)
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
%       C = prototypes matrix [p x k] or [p x k1 x k2]

%% INITIALIZATIONS

% Get Data

input = DATA.input;
[p,N] = size(input);

% Get Parameters

init = PAR.init;
k = PAR.k;

%% ALGORITHM

% For Algotithms with 1-D Grid

if (length(k) == 1),
    
    C = zeros(p,k);
    
    if (init == 1) 
        % does nothing

    elseif (init == 2)
        I = randperm(N);
        C = input(:,I(1:k));
    
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
        
    else
        disp('Unknown initialization. Prototypes = 0.');

    end
    
% For Algotithms with 2-D Grid

elseif (length(k) == 2),
    
    C = zeros(p,k(1),k(2));
    
    if (init == 1)
        % does nothing

    elseif (init == 2)
        I = randperm(N);
        aux = 1;
        for i = 1:k(1),
            for j = 1:k(2),
                C(:,i,j) = input(:,I(aux));
                aux = aux + 1;
            end
        end

    elseif (init == 3)
        %ToDo - mean of randomly choosen data
        
    else
        disp('Unknown initialization. Prototypes = 0.');    

    end
    
end

%% END