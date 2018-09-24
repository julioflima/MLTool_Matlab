function [OUT] = reject_opt2(DATAts,OUTts,REJp)

% --- Classifier's Reject Option ---
%
%   [OUT] = reject_opt2(DATAts,OUTts,REJp)
%
%   Entradas:
%       out = Celula contendo saida da rede e rotulos
%           out{1} = y_h    [Nc x Ntst]
%           out{2} = T2     [Nc x Ntst]
%           out{3} = rot_T2 [1 x Ntst]
%       wr = custo de rejeicao
%   Saidas
%       R = taxa de padrões rejeitados
%       E = erros de classificação
%       B = limiar de rejeição otimo
%       Mconf = nova matriz de confusao
%       acerto = Nova taxa de acerto
%       rejected = Matriz [Nc x Nr] indicando amostras rejeitados

%% INICIALIZAÇÕES

% Inicializa Entradas

wr = REJp.w;                % Custo de rejeição
y_h = OUTts.y_h;            % Saidas do Classificador
T2 = DATAts.input;          % Rotulos reais das amostras
[Nc,tst] = size(T2);        % Quantidade de amostras de Teste

% Inicializa Saidas

Mconf = zeros(Nc,Nc);       % Nova matriz de confusao

% Variacao do limiar de rejeicao

Bi = 0.25;
dB = 0.05;
Bf = 1.00;

% Inicializa Risco empirico (1 para cada limiar de rejeição)

Rh = zeros(1,16);

%% ALGORITMO

% Encontra Limiar Otimo

j = 0;      % index do R_hatched

for Btst = Bi:dB:Bf,

Nr = 0;     % no de padroes rejeitados
Ne = 0;     % no de padroes classificados erroneamente
j = j+1;    % incrementa index

for i = 1:tst,
    % se saida menor que limiar, Incrementa no de padroes rejeitados
    if (max(y_h(:,i)) < Btst),
        Nr = Nr+1;
    else
        [~,rot] = max(T2(:,i));
        [~,est] = max(y_h(:,i));
        % se ocorre erro, incrementa no de padroes classificados errados
        if (rot~=est)
            Ne = Ne+1;
        end
    end
end

Rb = Nr/tst;        % taxa de rejeicao para limiar especifico
Eb = Ne/(tst - Nr); % erros de classificacao para limiar especifico

Rh(j) = wr*Rb + Eb;

end

Bo = min(Rh);

% Inicializa amostras rejeitadas

n_rejected = length(find(max(y_h) < Bo)); % numero de amostras rejeitadas
rejected = cell(n_rejected,2);            % guarda amostras rejeitadas

% Calculo das saidas

Nr = 0;         % no de padroes rejeitados
Ne = 0;         % no de padroes classificados erroneamente

for i = 1:tst,
        % se saida menor que limiar, Incrementa no de padroes rejeitados
    if (max(y_h(:,i)) < Bo),
        Nr = Nr+1;
        rejected{Nr,1} = y_h(:,i);
        rejected{Nr,2} = T2(:,i);
    else
        [~,rot] = max(T2(:,i));
        [~,est] = max(y_h(:,i));
        % Calcula nova matriz de confusao
        Mconf(rot,est) = Mconf(rot,est) + 1;
        if (rot~=est)
            Ne = Ne+1;
        end
    end
end

R = Nr/tst;         % taxa de rejeicao
E = Ne/(tst - Nr);  % erros de classificacao
B = Bo;             % limiar de rejeicao

% Nova taxa de acerto

if (Bo == 1),
    acerto = -1;
else
    acerto = (sum(diag(Mconf)))/tst;
end

%% FILL OUTPUT STRUCTURE

OUT.R = R;
OUT.E = E;
OUT.B = B;
OUT.Mconf = Mconf;
OUT.acerto = acerto;
OUT.rejected = rejected;

%% END