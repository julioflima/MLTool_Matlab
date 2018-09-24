function [ soma ] = cell_sum( celula )
%
% função para soma de matrizes em celulas
% Ultima modificação: 12/08/2014
%
% [ soma ] = cell_sum( celula )
%
% - Entradas:
%       celula = contém matrizes por repetição
% - Saídas
%       soma = somatório das matrizes

%% INICIALIZAÇÕES

[repet,~] = size(celula);
[Nc,~] = size(celula{1,1});
soma = zeros(Nc);

%% ALGORITMO

for i = 1:repet,
    soma = soma + celula{repet};
end

end
