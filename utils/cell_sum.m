function [ soma ] = cell_sum( celula )
%
% fun��o para soma de matrizes em celulas
% Ultima modifica��o: 12/08/2014
%
% [ soma ] = cell_sum( celula )
%
% - Entradas:
%       celula = cont�m matrizes por repeti��o
% - Sa�das
%       soma = somat�rio das matrizes

%% INICIALIZA��ES

[repet,~] = size(celula);
[Nc,~] = size(celula{1,1});
soma = zeros(Nc);

%% ALGORITMO

for i = 1:repet,
    soma = soma + celula{repet};
end

end
