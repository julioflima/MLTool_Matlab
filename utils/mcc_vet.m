function [MccVect] = mcc_vet(OUTclass)

% --- Matthews Correlation Coeficient Vector for a classifier ---
%
%   [MccVect] = mcc_vet(OUTclass)
%
%   Input:
%       OUTclass = cell containing the outputs from the classifier, for
%       each turn of classification
%
%   Output: 
%       MccVect = vector containing the Matthews Correlation Coeficient 
%       Vector from the classifier, for each turn of classification

%% INITIALIZATIONS

[Nr,~] = size(OUTclass);
MccVect = zeros(Nr,1);

%% ALGORITHM

for i = 1:Nr,
   MccVect(i) = OUTclass{i,1}.mcc;
end

%% END