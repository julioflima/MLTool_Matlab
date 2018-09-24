function [K] = svm_f_kernel(PAR)

% --- SVM and LSSVM Kernel Function ---
%
% [K] = svm_f_kernel(PAR)
%
%   Input:
%       PAR.
%           P = atributtes matrix 	[N x p]
%           T1 = labels vector     	[N x 1]
%           Ktype = kernel type
%               1 -> Gaussian
%               2 -> Polinomial
%               3 -> MLP
%           sig2 (gaussian kernel variance)
%           ord (polinomial kernel order)
%           k_mlp / .teta (mlp kernel parameters)
%   Output:
%      K = Kernel Matrix

%% INITIALIZATION

% Get data
Mtr = PAR.P;
T1 = PAR.T1;

% Get parameters
[N,~] = size(Mtr);  	% Number of samples
Ktype = PAR.Ktype;      % Kernel type

% Initialize Output
K = zeros(N,N);         % kernel matrix

%% ALGORITHM

% GAUSSIAN KERNEL

if Ktype == 1,
    sig2 = PAR.sig2;    % variance (ex: sig2 = 1);
    for i = 1:N,
        for j = i:N,
            K(i,j) = T1(i)*T1(j) * exp(-norm(Mtr(j,:)-Mtr(i,:)).^2/sig2);
            K(j,i) = K(i,j);
        end
    end
    
% POLYNOMIAL KERNEL

elseif Ktype == 2, 
    ord = PAR.ord;       % order (ex: ord = 2)
    for i = 1:N,
        for j = i:N,
            K(i,j) = (Mtr(j,:) * Mtr(i,:)'+1)^ord;
            K(j,i) = K(i,j);
        end
    end
    
% MLP KERNEL

elseif Ktype == 3,
    k_mlp = PAR.k_mlp;   % (ex: k_mlp = 0.1)
    teta = PAR.teta;     % (ex: teta = 0.1)
    for i = 1:N,
        for j = i:N,
            K(i,j) = tanh(k_mlp*Mtr(j,:)*Mtr(i,:)'+teta);
            K(j,i) = K(i,j);
        end
    end

% INVALID KERNEL
    
else
    disp('invalid kernel option');

end

%% END