function [zscore,Mat_nemar] = McNemar(out1,out2,labels)
%
% Função para calcular McNemar Test
% Ultima modificação: 12/08/2015
%
% [zscore,Mat_nemar] = McNemar(out1,out2,labels)
%
% - Entradas:
%       out1 = saidas do classificador 1
%       out2 = saidas do classificador 2
%       labels = rotulo da amostra
% - Saídas
%       zscore = quão significativa é a diferença
%       Mat_nemar = matriz que mede diferenças / semelhanças

%% ININICALIZAÇÕES

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
