function [DB] = index_db(DATA,PAR)

% ---  Calculate Davies-Bouldin index for Clustering ---
%
%   [DB] = index_db(DATA,PAR)
%
%   Input:
%       DATA.
%           dados = input matrix [p x N]
%       PAR.
%           index = index of each sample [1 x N]
%           C = prototypes [p x k]
%   Output:
%       DB = DB index

%% INIT

% Load Data

data = DATA.dados;

% Load Parameters

labels = PAR.index;
C = PAR.C;
[~,k] = size(C);

% Init Aux Variables

clusters = cell(1,k);
for l = 1:k,
    J = find(labels == l);
    VJ = data(:,J);
    clusters{l} = VJ;
end

%% ALGORITHM

if k == 1,
    DB = NaN;
else
    DB = 0;             % this value will accumalate
    sigma = zeros(1,k); % find standard deviation of each cluster
    
    for l = 1:k
        % Find distance of objects at each cluster to its centroid
        [~,qtd] = size(clusters{l});
        centroid = C(:,l);
        dispersion = zeros(1,qtd);
        for samplePosition = 1:qtd
            sample = clusters{l}(:,samplePosition);
            dispersion(samplePosition) = sum((centroid-sample).^2);
        end
        % Find sigma
        sigma(l) = sqrt(sum(dispersion)/qtd);
    end
    
    % Find max((sigma_k+sigma_l/d_kl))
    for l = 1:k
        aux_value = zeros(1,k);
        centroid_l = C(:,l);
        for m = 1:k
            if m ~= l
                centroid_m = C(:,m);
                d_kl = sqrt(sum((centroid_l - centroid_m).^2));
                aux_value(m) = (sigma(l) + sigma(m))/d_kl;
            end
        end
        %Find DB index
        DB = DB +(max(aux_value))/k;
    end
end

%% FILL OUTPUT STRUCTURE

% Dont Need

%% END