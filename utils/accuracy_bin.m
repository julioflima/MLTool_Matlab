function [AccVect] = accuracy_bin(OUTclass)

% --- Accuracy Vector for a classifier ---
%
%   [AccVect] = accuracy_mult(OUTclass)
%
%   Input:
%       OUTclass = cell containing the outputs from the classifier, for
%       each turn of classification
%
%   Output: 
%       AccVect = vector containing the accuracy from the classifier, 
%       for each turn of classification, considering a binary problem
%

%% INITIALIZATIONS

[Nr,~] = size(OUTclass);
AccVect = zeros(Nr,1);

%% ALGORITHM

for i = 1:Nr,
   AccVect(i) = OUTclass{i,1}.nf;
end

%% END