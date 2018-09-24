function [S] = index_silhouette(DATA,PAR)

% ---  Calculate silhuette index for Clustering ---
%
%   [S] = index_silhouette(C,PAR,N)
%
%   Input:
%       DATA.
%           dados = input matrix [p x N]
%       PAR.
%           index = index of each sample [1 x N]
%   Output:
%       S = silhouette index

%% INIT

% Load Data

data = DATA.dados;
[~, N] = size(data);

% Load Parameters

labels = PAR.index;
k = length(find(unique(labels)));

% Init Aux Variables

clusters = cell(1,k);
cohesion = cell(1,k);
separation = cell(1,k);
samplesQtd = zeros(1,k);

for l=1:k,
    J = find(labels == l);
    VJ = data(:,J);
    clusters{l} = VJ;
    samplesQtd(l) = length(J); 
    cohesion{l} = zeros(1,samplesQtd(l));
    separation{l} = zeros(1,samplesQtd(l));
end

%% ALGORITHM

if k == 1,
    S = NaN;
else
    S = 0; %This value will accumalate
    
    for l = 1:k,
        for m = 1:samplesQtd(l)
            a = clusters{l}(:,m);
            
            %Find Cohesion
            for n = 1:samplesQtd(l)
                if m ~= n
                    b = clusters{l}(:,n);
                    cohesion{l}(m) = cohesion{l}(m) + (sqrt(sum((a-b).^2))/(samplesQtd(l) - 1));
                end
            end
            
            %Find Separation
            separation_aux = zeros(1,k);
            for o = 1:k
                if o == l
                    separation_aux(o) = inf;
                else
                    for n = 1:samplesQtd(o)
                        b = clusters{o}(:,n);
                        separation_aux(o) = separation_aux(o) + (sqrt(sum((a-b).^2))/samplesQtd(o));
                    end
                end
            end
            separation{l}(m) = min(separation_aux);
            
            %Find Silhouette Coefficient
            dif_value = separation{l}(m) - cohesion{l}(m);
            max_value = max(separation{l}(m),cohesion{l}(m));
            s = (dif_value/max_value)/N;
            S = S + s;
            
        end
    end
end

%% FILL OUTPUT STRUCTURE

% Dont Need

%% END