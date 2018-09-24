%% TESTE DO CLASSIFICADOR PS-KSOM
% Primal Space Kernel Self-Organising Map

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

OPT.prob = 7;               % Which problem will be solved / which data set will be used
OPT.prob2 = 1;              % When it needs more specific data set
OPT.norm = 3;               % Normalization definition
OPT.lbl = 0;                % Data labeling definition
OPT.Nr = 50;                % Number of repetitions of each algorithm
OPT.hold = 1;               % Hold out method
OPT.ptrn = 0.8;             % Percentage of samples for training
OPT.file = 'fileX.mat';     % file where all the variables will be saved  

%% DATA LOADING AND PRE-PROCESSING

DATA = data_class_loading(OPT);     % Load Data Set

DATA = normalize(DATA,OPT);         % normalize the attributes' matrix

DATA = label_adjust(DATA,OPT);      % adjust labels for the problem

%% HOLD OUT / CROSS VALIDATION / TRAINING / TEST

% Divide dados entre treino e teste

[DATAho] = hold_out(DATA,OPT);
DATAtr = DATAho.DATAtr;
DATAts = DATAho.DATAts;

Min = DATAtr.input;
MATout = DATAtr.output;

[~,N] = size(Min);

% Seleciona uma quantidade menor de dados de treinamento

M = 50;    % Matriz reduzida - no caso, peguei todos os dados de treino

% OBS: Verificar metodo de selecao de prototipos, via entropia 
%      implementado pelo daniel!!!

I = randperm(235);
Min = Min(:,I);
MATout = MATout(:,I);

Mred = Min(:,1:M);

% Cria Matriz de Kernel

sig2 = 2;

[~,c] = size(Mred);

Kmat = zeros(c,c);

for i = 1:c,
    for j = i:c,
        Kmat(i,j) = exp(-norm(Mred(:,i)-Mred(:,j))^2/(2*sig2));
        Kmat(j,i) = Kmat(i,j);
    end
end

% Auto Decomposicao da matriz de kernel
% U1 autovetores (mat full), D1 autovalores (mat diag)

[U_aux,D_aux] = eig(Kmat);  
vec_aux = diag(D_aux);

% Inverte ordem dos autovetores e autovalores
% (deixa do maior para o menor)

[l1,c1] = size(vec_aux);
[l2,c2] = size(U_aux);
eig_val = zeros(l1,c1);
eig_vec = zeros(l2,c2);

for j = 1:M,
    eig_vec(:,j) = U_aux(:,M-j+1); 
    eig_val(j) = vec_aux(M-j+1);    
end

% Gera matriz de amostras projetadas em phi
% (ordem dos loops iguais a do Daniel)

Mphi1 = zeros(M,N);

for i = 1:M,
   for n = 1:N,
       Xn = Min(:,n);
       sum_aux = 0;
       for m = 1:M,
           Zm = Mred(:,m);
           sum_aux = sum_aux + eig_vec(m,i)*exp(-(norm(Xn-Zm))^2/(2*sig2));
       end
       Mphi1(i,n) = sum_aux / sqrt(eig_val(i));
   end
end

% Gera matriz de amostras projetadas em phi
% (ordem dos loops modificada -> n mais externo)

Mphi = zeros(M,N);

for n = 1:N,
    Xn = Min(:,n);
    for i = 1:M,
        sum_aux = 0;
        for m = 1:M,
            Zm = Mred(:,m);
            sum_aux = sum_aux + eig_vec(m,i)*exp(-(norm(Xn-Zm))^2/(2*sig2));
        end
        Mphi(i,n) = sum_aux / sqrt(eig_val(i));
    end
end

%% END
