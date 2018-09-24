function [DATAout] = hold_out(DATAin,OPTIONS)

% --- Separates data for training and test ---
%
%   [DATAout] = hold_out(DATAin,OPTIONS)
%
%   Input:
%       DATAin.Dn = matriz com vetores de atributos [pxN]
%       DATAin.alvos = matriz com rotulos de cada amostra [cxN]
%       DATAin.rot = rotulos com apenas 1 valor (1 à Nc) [1xN]
%       OPTIONS.ptrn = % de dados para treinamento
%       OPTIONS.hold = metodo de hold out definido
%           1 = pega todos os dados, divide 80% para treino e 20% para tst
%           2 = pega 80% da classe com menos amostras, e a mesma quantidade
%           para as outras classes, e coloca para treino. o resto para tst.
%   Output:
%       DATAout.
%           P = amostras de treino
%           T1 = rotulos de treino
%           rot_T1 = rotulos de treino, com valores de 1 a Nc
%           Q = amostras de teste
%           T2 = rotulos de teste
%           rot_T2 = rotulos de teste, com valores de 1 a Nc

%% INICIALIZAÇÕES

Mho = OPTIONS.hold;
ptrn = OPTIONS.ptrn;
input = DATAin.input;
output = DATAin.output;
lbl = DATAin.lbl;

% Number of classes and samples
[Nc,N] = size(output);
flag_seq = 0;
if Nc == 1,
    % informs that labels are sequential
    flag_seq = 1;
    % calculates one more time the number of classes
    Nc = length(unique(output));
end

% Inicialização das saídas
P = [];
T1 = [];
rot_T1 = [];
Q = [];
T2 = [];
rot_T2 = [];

%% ALGORITMO

switch (Mho)
    
%------------- HOLD OUT -> 80 x 20 --------------%

case(1)

% Embaralha entradas e saidas (dados e alvos)
I = randperm(N);        
Dn = input(:,I);
output = output(:,I);
lbl = lbl(:,I);

% Numero de Vetores para treinamento
J = floor(ptrn*N);

% Entradas e Saidas para Treinamento
P = Dn(:,1:J); 
T1 = output(:,1:J);
rot_T1 = lbl(:,1:J);

% Entradas e Saidas para Teste
Q = Dn(:,J+1:end); 
T2 = output(:,J+1:end);
rot_T2 = lbl(:,J+1:end);

%-------- HOLD OUT -> TREINO EQUILIBRADO --------%

case(2)

% Initialize auxiliary variables

dados_classe = cell(Nc,1);
alvos_classe = cell(Nc,1);
rot_classe = cell(Nc,1);
for i = 1:Nc,
    dados_classe{i} = [];
    alvos_classe{i} = [];
    rot_classe{i} = [];
end

% Separate data of each class

for i = 1:N,
    % current sample
    dado = input(:,i);
    alvo = output(:,i);
    rot_atual = lbl(:,i);
    % define class
    if (flag_seq == 1),
        classe = alvo;
    else
        classe = find(alvo>0);
    end
    % adiciona amostra à matriz correspondente
    dados_classe{classe} = [dados_classe{classe} dado];
    alvos_classe{classe} = [alvos_classe{classe} alvo];
    rot_classe{classe} = [rot_classe{classe} rot_atual];
end

% Calcula minima quantidade de amostras em uma classe
for i = 1:Nc,
    if (i == 1),
        % inicializa minimo numero de amostras
        [~,Nmin] =  size(dados_classe{i});
    else
        % verifica menor numero
        [~,n] =  size(dados_classe{i});
        if (n < Nmin),
            Nmin = n;
        end
    end
end

% Quantidade de amostras, para treinamento, de cada classe
J = floor(ptrn*Nmin);

for i = 1:Nc,
    % quantidade de amostras da classe i
    [~,n] =  size(dados_classe{i});
    % embaralha amostras da classe i
    I = randperm(n);
    % Amostras de treinamento e teste
    P = [P dados_classe{i}(:,I(1:J))];
    Q = [Q dados_classe{i}(:,I(J+1:end))];
    % Alvos de treinamento e teste
    T1 = [T1 alvos_classe{i}(:,I(1:J))];
    T2 = [T2 alvos_classe{i}(:,I(J+1:end))];
    % Rotulos de treinamento e teste
    rot_T1 = [rot_T1 rot_classe{i}(:,I(1:J))];
    rot_T2 = [rot_T2 rot_classe{i}(:,I(J+1:end))];
end

%-------------- NENHUMA DAS OPÇÕES --------------%

otherwise

    disp('Digite uma opção correta. Os dados não foram separados.')

end

%% FILL STRUCTURE

DATAout.DATAtr.input = P;
DATAout.DATAtr.output = T1;
DATAout.DATAtr.lbl = rot_T1;

DATAout.DATAts.input = Q;
DATAout.DATAts.output = T2;
DATAout.DATAts.lbl = rot_T2;

%% END