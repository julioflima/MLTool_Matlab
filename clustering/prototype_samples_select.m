function [Mred] = prototype_samples_select(DATA,PAR)

% --- Sample prototypes selection ---
%
%   [Mred] = prototype_samples_select(DATA,PAR)
%
%   Input:
%       DATA.
%           input = input matrix [p x N]
%       PAR.
%           M = number of samples used to estimate kernel matrix
%           s = prototype selection type
%           sig2 = Variance (gaussian, log, cauchy kernel)
%   Output:
%       Mred = reduced matrix with choosen samples [p x M]

%% INITIALIZATION

M = PAR.M;           % number of samples used to estimate kernel matrix
s = PAR.s;         	 % prototype selection type
sig2 = PAR.sig2;     % variance (gaussian kernel)
v = PAR.v;           % accuracy parameter (level of sparsity)

MATin = DATA.input;  % Input matrix
[~,N] = size(MATin); % Total of samples

Kmat = zeros(M,M);   % Kernel Matrix

%% ALGORITHM

% Shuffle data
I = randperm(N);
MATin = MATin(:,I);

% Randomly select from input
if (s == 1)
    % data is already shuffled
    Mred = MATin(:,1:M);

% Renyi's entropy method
elseif (s == 2),
    
    n_it = 100; % Maximum number of iterations 
    Mred = MATin(:,1:M); % get first M samples
    Mrem = MATin(:,M+1:end); % remaining samples
    
    % initial entropy calculation
    for i = 1:M,
        for j = i:M,
            Kmat(i,j) = exp(-norm(Mred(:,i)-Mred(:,j))^2/(2*sig2));
            Kmat(j,i) = Kmat(i,j);
        end
    end
    entropy = ones(1,M)*Kmat*ones(M,1);
    
    % choose prototype samples
    for k = 1:n_it,
        % randomly select sample from reduced matrix
        I = randperm(M);
        red_s = Mred(:,I(1));
        % randomly select sample from remaining matrix
        J = randperm(N-M);
        rem_s = Mrem(:,J(1));
        % construct new reduced matrix
        Mred_new = Mred;
        Mred_new(:,I(1)) = rem_s;
        % construct new remaining matrix
        Mrem_new = Mrem;
        Mrem_new(:,J(1)) = red_s;
        % Calculate new entropy
        for i = 1:M,
            for j = i:M,
                Kmat(i,j) = exp(-norm(Mred_new(:,i)-Mred_new(:,j))^2/(2*sig2));
                Kmat(j,i) = Kmat(i,j);
            end
        end
        entropy_new = ones(1,M)*Kmat*ones(M,1);
        % replace old matrix for new ones
        if (entropy_new > entropy),
            entropy = entropy_new;
            Mred = Mred_new;
            Mrem = Mrem_new;
        end
    end
    
% Approximate Linear Dependency (ALD) Method    
elseif (s == 3),
    % init dictionary
    Md = MATin(:,1);
    for t = 2:N,
        % get new sample
        Xt = MATin(:,t);
        [~,m] = size(Md);
        % calculate Kmat t-1
        Kmat = zeros (m,m);
        for i = 1:m,
            for j = i:m,
                Kmat(i,j) = exp(-norm(Md(:,i)-Md(:,j))^2/(2*sig2));
                Kmat(j,i) = Kmat(i,j);
            end
        end
        Kmat = Kmat + 0.01; % avoid inverse problems
        % Calculate k t-1
        kt = zeros(m,1);
        for i = 1:m,
            kt(i) = exp(-norm(Md(:,i)-Xt)^2/(2*sig2));
        end
        % Calculate Ktt
        ktt = exp(-norm(Xt-Xt)^2/(2*sig2));
        % Calculate coefficients
        at = Kmat\kt;
        % Calculate delta
        delta = ktt - kt'*at;
        % Expand or not dictionary
        if (delta > v),
            Md = [Md, Xt];
        else
            
        end
    end
    Mred = Md;
end

%% FILL OUTPUT STRUCTURE



%% END