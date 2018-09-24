function [OUTrj] = reject_opt(DATA,OUTts,REJp,OPTION)

% --- Classifier's Reject Option ---
%
%   [OUTrj] = reject_opt(DATA,OUTts,REJp,OPTION)
%
%   Input:
%       out = Celula contendo saida da rede e rotulos
%           out{1} = y_h    [Nc x Ntst]
%           out{2} = T2     [Nc x Ntst]
%           out{3} = rot_T2 [1 x Ntst]
%       reject = faixa de valores que serão rejeitados [-reject +reject]
%       problem = como o problema é tratado
%   Output:
%       rejected = Matriz [Nr x 2] indicando amostras rejeitados
%           rejected(r,1) = saida da rede
%           rejected(r,2) = rotulo
%       reject_ratio = razão entra amostras rejeitadas e total de amostras
%       Mconf = Matriz de confusão para valores não rejeitados
%       acerto = taxa de acerto sem amostras rejeitadas

%% INICIALIZAÇÕES

problem = OPTION.prob;      % problem definition
reject_band = REJp.band;    % threshold of rejection
y_h = OUTts.y_h;            % classifier output
T2 = DATA.alvos;            % real output
[Nc,tst] = size(T2);        % Number of classes and samples

% Inicializa Variaveis Auxiliares

i_real = zeros(1,tst);      % Indice da label real
i_out = zeros(1,tst);       % Indice da label de saída da rede

% Inicializa Saidas

n_rejected = length(find(max(y_h) < reject_band));
rejected = cell(n_rejected,2);  % Célula que armazena amostras rejeitadas 
                                % Contêm rótulos rejeitados e rótulos reais

%% CONFIGURA ALVOS

% Configura labels de saída

switch problem,
    case 1,
        Mconf = zeros(Nc); % Matriz de confusao
        [~,i_real] = max(T2);
        [~,i_out] = max(y_h);
    case 2,
        Mconf = zeros(Nc); % Matriz de confusao
        [~,i_real] = max(T2);
        [~,i_out] = max(y_h);
    case 3,
        Mconf = zeros(Nc); % Matriz de confusao
        [~,i_real] = max(T2);
        [~,i_out] = max(y_h);
    case 4,
        Mconf = zeros(Nc); % Matriz de confusao
        [~,i_real] = max(T2);
        [~,i_out] = max(y_h);
    case 5,
        Mconf = zeros(Nc); % Matriz de confusao
        [~,i_real] = max(T2);
        [~,i_out] = max(y_h);
    case 6,
        Mconf = zeros(Nc); % Matriz de confusao
        [~,i_real] = max(T2);
        [~,i_out] = max(y_h);
    case 7,
        Mconf = zeros(Nc); % Matriz de confusao
        [~,i_real] = max(T2);
        [~,i_out] = max(y_h);
    case 8,
        Mconf = zeros(Nc); % Matriz de confusao
        [~,i_real] = max(T2);
        [~,i_out] = max(y_h);
    case 9,
        Mconf = zeros(7); % Matriz de confusao
        for i = 1:tst,
            if(y_h(1,i) >= 0),
                i_out(i) = 1;
            elseif ((y_h(1,i) < 0) && (y_h(1,i) >= -0.3))
                i_out(i) = 2;
            elseif ((y_h(1,i) < -0.3) && (y_h(1,i) >= -0.4))
                i_out(i) = 5;
            elseif ((y_h(1,i) < -0.4) && (y_h(1,i) >= -0.5))
                i_out(i) = 3;
            elseif ((y_h(1,i) < -0.5) && (y_h(1,i) >= -0.6))
                i_out(i) = 6;
            elseif ((y_h(1,i) < -0.6) && (y_h(1,i) >= -0.8))
                i_out(i) = 4;
            elseif (y_h(1,i) < -0.8)
                i_out(i) = 7;
            end
            
            switch T2(1,i),
                case 1
                    i_real(i) = 1;
                case -0.3
                    i_real(i) = 2;
                case -0.5
                    i_real(i) = 3;
                case -0.8
                    i_real(i) = 4;
                case -0.4
                    i_real(i) = 5;
                case -0.6
                    i_real(i) = 6;
                case -0.9
                    i_real(i) = 7;
            end
        end
        
    case 10,
        Mconf = zeros(7); % Matriz de confusao
        for i = 1:tst,
            if(y_h(1,i) >= 0),
                i_out(i) = 1;
            elseif ((y_h(1,i) < 0) && (y_h(1,i) >= -0.5))
                i_out(i) = 2;
            elseif ((y_h(1,i) < -0.5) && (y_h(1,i) >= -0.6))
                i_out(i) = 5;
            elseif ((y_h(1,i) < -0.6) && (y_h(1,i) >= -0.8))
                i_out(i) = 3;
            elseif ((y_h(1,i) < -0.8) && (y_h(1,i) >= -0.85))
                i_out(i) = 6;
            elseif ((y_h(1,i) < -0.85) && (y_h(1,i) >= -0.9))
                i_out(i) = 4;
            elseif (y_h(1,i) < -0.9)
                i_out(i) = 7;
            end
            
            switch T2(1,i),
                case 1
                    i_real(i) = 1;
                case -0.5
                    i_real(i) = 2;
                case -0.8
                    i_real(i) = 3;
                case -0.9
                    i_real(i) = 4;
                case -0.6
                    i_real(i) = 5;
                case -0.85
                    i_real(i) = 6;
                case -0.95
                    i_real(i) = 7;
            end            
        end
        
    case 11,
        Mconf = zeros(7); % Matriz de confusao
        for i = 1:tst,
            if(y_h(1,i) >= 0),
                i_out(i) = 1;
            elseif ((y_h(1,i) < 0) && (y_h(1,i) >= -0.5))
                i_out(i) = 2;
            elseif ((y_h(1,i) < -0.5) && (y_h(1,i) >= -0.6))
                i_out(i) = 3;
            elseif ((y_h(1,i) < -0.6) && (y_h(1,i) >= -0.8))
                i_out(i) = 4;
            elseif ((y_h(1,i) < -0.8) && (y_h(1,i) >= -0.85))
                i_out(i) = 5;
            elseif ((y_h(1,i) < -0.85) && (y_h(1,i) >= -0.9))
                i_out(i) = 6;
            elseif (y_h(1,i) < -0.9)
                i_out(i) = 7;
            end
            
            switch T2(1,i),
                case 1
                    i_real(i) = 1;
                case -0.5
                    i_real(i) = 2;
                case -0.6
                    i_real(i) = 3;
                case -0.8
                    i_real(i) = 4;
                case -0.85
                    i_real(i) = 5;
                case -0.9
                    i_real(i) = 6;
                case -0.95
                    i_real(i) = 7;
            end                
        end
        
    case 12,
        Mconf = zeros(7); % Matriz de confusao

    otherwise
        disp('Wrong Problem Type')
end

%% ALGORITMO

n_rejected = 0;

for i = 1:tst,
    if (max(y_h(:,i)) < reject_band),
        n_rejected = n_rejected+1;
        rejected{n_rejected,1} = y_h(:,i);
        rejected{n_rejected,2} = T2(:,i);
    else
        Mconf(i_real(i),i_out(i)) = Mconf(i_real(i),i_out(i)) + 1;
    end
end

% Calcula razao de amostras rejeitadas e taxa de acerto

reject_ratio = n_rejected/tst;

if (reject_ratio == 1),
    acerto = -1;
else
    acerto = (sum(diag(Mconf)))/tst;
end

%% FILL OUTPUT STRUCTURE

OUTrj.rejected = rejected;
OUTrj.reject_ratio = reject_ratio;
OUTrj.Mconf = Mconf;
OUTrj.acerto = acerto;

%% END