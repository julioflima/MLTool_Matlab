function [coeficient] = CoefVar(OUTclass)
%
% --- Calculates Coeficient of Variation ---
%
% [coeficient] = CoefVar(OUTclass)
%
%   Input:
%       OUTclass = cell containing the outputs from the classifier, for
%       each turn of classification
%
%   Output:
%       coeficient = result

%% INIT

[Nr,~] = size(OUTclass);
AccVect = zeros(Nr,1);

%% ALGORITHM

for i = 1:Nr,
   AccVect(i) = OUTclass{i,1}.acerto; 
end

med = mean(AccVect);
dp = std(AccVect);

coeficient = 100*dp/med;

%% THEORY

% Ref: https://en.wikipedia.org/wiki/Coefficient_of_variation

%% END