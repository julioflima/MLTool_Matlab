function [acerto] = normal_or_fail(Mconf)
%
% Fun��o para c�lculo da taxa de acerto entre falha/n�o-falha
% Ultima modifica��o: 12/08/2014
%
% [acerto] = normal_or_fail(Mconf)
%
% - Entradas:
%       Mconf = matriz de confus�o completa
% - Sa�das
%       acerto = taxa de acerto entre falha e n�o falha

%% INICIALIZA��ES

n_amostras = sum(sum(Mconf));   %n�mero de amostras testadas

%% ALGORITMO

if (n_amostras == 0),
    acerto = -1;
else
    n_acerto = n_amostras - sum(Mconf(1,:)) - sum(Mconf(:,1)) + 2*Mconf(1,1);
    acerto = n_acerto/n_amostras;
end

%% END