function [coeficient] = Mcc(conf_matrix)
%
% Calculates Matthews Correlation Coefficient
%
% [mcc_out] = McNemar(mcc_in)
%
% - Input:
%       conf_matrix = from a binary classification problem
% - Output:
%       coeficient = between [-1 +1]
%
% Ref: https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
%

%% ININICALIZAÇÕES

TP = conf_matrix(1,1);
TN = sum(sum(conf_matrix(2:end,2:end)));
FP = sum(conf_matrix(1,2:end));
FN = sum(conf_matrix(2:end,1));

sum1 = TP + FP;
sum2 = TP + FN;
sum3 = TN + FP;
sum4 = TN + FN;

%% ALGORITMO

if (sum1 == 0 || sum2 == 0 || sum3 == 0 || sum4 == 0),
    den = 1;
else
    den = sum1*sum2*sum3*sum4;
    den = den^0.5;
end

coeficient = (TP*TN - FP*FN)/den; 

end