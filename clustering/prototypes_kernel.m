function [K] = prototypes_kernel(dados,alvos,PAR)

% --- Kernel Function for Prototypes ---
%
% [K] = svm_f_kernel(treino,alvotr,SvmPar)
%
%   Entradas:
%       treino = matriz de treinamento      [Nxp]
%       alvotr = saidas de treinameno       [Nx1]
%       PAR.
%           tipo (qual função base do kernel)
%               1: Gaussiano
%               2: Polinomial
%               3: MLP
%   Saidas:
%      K = Kernel Matrix

%% INICIALIZAÇÕES

[trn,~] = size(dados); % Quantidade de amostras
Ktype = PAR.Ktype;      % Tipo de Kernel
K = zeros(trn,trn);     %inicializa matriz de kernel

%% ALGORITMO

% KERNEL GAUSSIANO

if Ktype == 1,
    sig2 = PAR.sig2;     %sig2 = 1 (default)
    for i=1:trn,
        for j=1:trn,
            K(i,j) = alvos(i)*alvos(j)*...
            exp(-norm(dados(j,:)-dados(i,:)).^2/sig2);
        end
    end
    
% KERNEL POLINOMIAL

elseif Ktype == 2, 
    ord = PAR.ord;       %ord = 2 (default)
    for i=1:trn,
        for j=1:trn,
            K(i,j) = (dados(j,:)*dados(i,:)'+1)^ord;
        end
    end
    
% KERNEL MLP

elseif Ktype == 3,
    k_mlp = PAR.k_mlp;   %k_mlp = 0.1 (default)
    teta = PAR.teta;     %teta = 0.1 (default)
    for i=1:trn,
        for j=1:trn,
            K(i,j) = tanh(k_mlp*dados(j,:)*dados(i,:)'+teta);
        end
    end

% OPÇÃO INVALIDA DE KERNEL
    
else
    disp('digite uma opção válida de kernel');    

end

%% END