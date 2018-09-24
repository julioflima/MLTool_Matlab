function [zscore,Mat_nemar] = McNemar(out1,out2,labels)
%
% Fun��o para calcular McNemar Test
% Ultima modifica��o: 12/08/2015
%
% [zscore,Mat_nemar] = McNemar(out1,out2,labels)
%
% - Entradas:
%       out1 = saidas do classificador 1
%       out2 = saidas do classificador 2
%       labels = rotulo da amostra
% - Sa�das
%       zscore = qu�o significativa � a diferen�a
%       Mat_nemar = matriz que mede diferen�as / semelhan�as

%% ININICALIZA��ES

[amostras,~] = size(labels);
[Nc1,~] = size(out1);
[Nc2,~] = size(out2);

zscore = 0;
Mat_nemar = [];

%% ALGORITMO

% ToDo - All

for i = 1:amostras,
    zscore = Nc1 + Nc2;
end

end
