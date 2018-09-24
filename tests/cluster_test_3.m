function [OUT] = cluster_test_3(DATA,PAR)

% --- Clustering General Function ---
%
%   [OUT] = clustering(DATA,PAR)
%
%   Input:
%       DATA.
%           input = input matrix [pxN]
%           output = output matrix [cxN]
%       PAR.
%           Kmax = max number of clusters
%           nrep = number of repetitions with each cluster quantity
%           alg = which algorithm to use
%               1: Kmeans
%               2: wta
%               3: som-1d
%               4: som-2d
%               5: Mixture of Gaussians
%           ind = validation index
%               1: AIC
%               2: BIC
%               3: CH
%               4: DB
%               5: Dunn
%               6: FPE
%               7: MDL
%               8: Silhuette
%           "others" = parameters for each specific algorithm
%   Output:
%       OUT.
%           C = centroids
%           index = index of each sample
%           val_index = validation index results
%           MSQE = mean squared error

%% INITIALIZATION

% Define clustering algorithm
switch PAR.alg
    case 1,
        func_alg = @kmeans_train;
    case 2,
        func_alg = @wta_train;
    case 3,
        func_alg = @som1d_train;
    case 4,
        func_alg = @som2d_train;
    case 5,
        func_alg = @mog_train;
    otherwise,
        disp('Unknown function')
end

% Define validation_index
switch(PAR.ind)
    case 1,
        func_ind = @index_aic;
    case 2,
        func_ind = @index_bic;
    case 3,
        func_ind = @index_ch;
    case 4,
        func_ind = @index_db;
    case 5,
        func_ind = @index_dunn;
    case 6,
        func_ind = @index_fpe;
    case 7,
        func_ind = @index_mdl;
    case 8,
        func_ind = @index_silhouette;
    otherwise
        disp('Unknown validation index')
end

% Init Structures

val_ind = zeros(1,PAR.Kmax);
val_ind(1) = NaN; % Dont Have a Validation Index for 1 cluster
best_results = cell(1,PAR.Kmax);
MSQEs = zeros(1,PAR.Kmax);

%% ALGORITHM

for i = 2:PAR.Kmax,
    
    PAR.k = i;      % Define current Cluster
    MSQE_min = Inf; % Init min Mean Squared Quantization Error
        
    for j = 1:PAR.nrep,
        
        [OUT_CL] = func_alg(DATA,PAR);
        [MSQE] = prototypes_mse(OUT_CL.C,DATA,OUT_CL);
        
        % Get best distribution
        if(MSQE < MSQE_min),
            MSQE_min = MSQE;
            best_CL = OUT_CL;
        end
        
    end
    
    % Hold Best result of a particular number of clusters
    best_results{i} = best_CL;
    MSQEs(i) = MSQE_min;
    
    % Calculate and Hold Validation Index
	index = func_ind(DATA,best_CL);
    val_ind(i) = index;

end

% pegar a melhor configuracao entre os "Ks"

%% FILL OUTPUT STRUCTURE

OUT.MSQEs = MSQEs;
OUT.val_ind = val_ind;
OUT.best_results = best_results;

%% END