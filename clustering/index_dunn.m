function [DUNN] = index_dunn(DATA,PAR)

% ---  Calculate Dunn index for Clustering ---
%
%   [DUNN] = index_dunn(DATA,PAR)
%
%   Input:
%       DATA.
%           dados = input matrix [p x N]
%       PAR.
%           index = index of each sample [1 x N]
%           C = prototypes [p x k]
%   Output:
%       DUNN = DUNN index

%% INIT

% Load Data

data = DATA.dados;

% Load Parameters

labels = PAR.index;
C = PAR.C;
[~,k] = size(C);

% Init Aux Variables

clusters = cell(1,k);
samplesQtd = zeros(1,k);

for l = 1:k,
    J = find(labels == l);
    VJ = data(:,J);
    clusters{l} = VJ;
    samplesQtd(l) = length(J);
end

%% ALGORITHM

if k == 1,
    DUNN = NaN;
else
    %Find centroids combination
    centroidsCombination = combvec(1:k,1:k);
    comb_aux = centroidsCombination(1,:) - centroidsCombination(2,:);
    comb_aux = find(comb_aux ~= 0);
    comb_qtd = length(comb_aux);
    centroidsCombination_aux = zeros(2,comb_qtd);
    for i = 1:comb_qtd
        centroidsCombination_aux(:,i) = centroidsCombination(:,comb_aux(i));
    end
    centroidsCombination = centroidsCombination_aux;
    
    %Find min(Sigma(Si,Sj))
    num = inf; %min(Sigma(Si,Sj))
    [~,comb_qtd] = size(centroidsCombination);
    for i = 1:comb_qtd
        for j = 1:samplesQtd(centroidsCombination(1,i))
            Si = clusters{centroidsCombination(1,i)}(:,j);
            for l = 1:samplesQtd(centroidsCombination(2,i))
                Sj = clusters{centroidsCombination(2,i)}(:,l);
                deltaSiSj = sqrt(sum((Si-Sj).^2));
                if deltaSiSj < num
                    num = deltaSiSj;
                end
            end
        end
    end
    
    den = 0;
    for i = 1:k
        for l = 1:samplesQtd(i)
            x = clusters{i}(:,l);
            for m = 1:samplesQtd(i)
                if m ~= l
                    y = clusters{i}(:,m);
                    Sl = sqrt(sum((x-y).^2));
                    if Sl > den
                        den = Sl;
                    end
                end
            end
        end
    end
    
    DUNN = num/den;
end

%% FILL OUTPUT STRUCTURE

% Dont Need

%% END