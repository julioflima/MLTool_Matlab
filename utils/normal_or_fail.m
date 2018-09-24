function [acerto] = normal_or_fail(Mconf)
%
% Função para cálculo da taxa de acerto entre falha/não-falha
% Ultima modificação: 12/08/2014
%
% [acerto] = normal_or_fail(Mconf)
%
% - Entradas:
%       Mconf = matriz de confusão completa
% - Saídas
%       acerto = taxa de acerto entre falha e não falha

%% INICIALIZAÇÕES

n_amostras = sum(sum(Mconf));   %número de amostras testadas

%% ALGORITMO

if (n_amostras == 0),
    acerto = -1;
else
    n_acerto = n_amostras - sum(Mconf(1,:)) - sum(Mconf(:,1)) + 2*Mconf(1,1);
    acerto = n_acerto/n_amostras;
end

%% END