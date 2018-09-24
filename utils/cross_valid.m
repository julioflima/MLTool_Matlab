function [accuracy] = cross_valid(DATA,HP,CVp,f_train,f_class)

% --- Cross Validation Function ---
%
%   [accuracy] = cross_valid(DATA,HP,CVp,f_train,f_class)
%
%   Input:
%       DATA.
%           dados = Matrix of training attributes [pxN]
%           alvos = Matrix of training labels [cxN]
%       HP = set of HyperParameters to be tested
%       CVp.
%           on = indicates that cross validation is used
%           Nfold = number of partitions
%       f_train = handler for classifier's training function
%       f_class = handler for classifier's classification function       
%   Output:
%       accuracy = mean accuracy for data set and parameters

%% INIT

P = DATA.dados;                 % Attributes Matrix [pxN]
T1 = DATA.alvos;                % labels Matriz [cxN]

[~, Ntrain] = size(P);          % Number of samples

Nfold = CVp.fold;               % Number of folds
part = floor(Ntrain/Nfold);     % Size of each data partition

accuracy = 0;                   % Init accurary

%% ALGORITHM

if CVp.on == 1,
    
    for fold = 1:Nfold;

        % Define Data division

        if fold == 1,
            DATAtr.dados = P(:,part+1:end);
            DATAtr.alvos = T1(:,part+1:end);
            DATAts.dados = P(:,1:part);
            DATAts.alvos = T1(:,1:part);
        elseif fold == Nfold,
            DATAtr.dados = P(:,1:(Nfold-1)*part);
            DATAtr.alvos = T1(:,1:(Nfold-1)*part);
            DATAts.dados = P(:,(Nfold-1)*part+1:end);
            DATAts.alvos = T1(:,(Nfold-1)*part+1:end);
        else
            DATAtr.dados = [P(:,1:(fold-1)*part) P(:,fold*part+1:end)];
            DATAtr.alvos = [T1(:,1:(fold-1)*part) T1(:,fold*part+1:end)];
            DATAts.dados = P(:,(fold-1)*part+1:fold*part);
            DATAts.alvos = T1(:,(fold-1)*part+1:fold*part);
        end

        % Training of classifier

        [PARo] = f_train(DATAtr,HP);

        % Test of classifier

        [OUT] = f_class(DATAts,PARo);

        % Acc Accuracy rate

        accuracy = accuracy + OUT.acerto;

    end

    accuracy = accuracy/CVp.fold;
    
else
    
    accuracy = 0;

end

%% END