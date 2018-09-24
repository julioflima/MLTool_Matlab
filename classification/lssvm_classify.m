function [OUT] = lssvm_classify(DATA,PAR)

% --- LSSVM classifier test ---
%
%   [OUT] = lssvm_classify(DATA,PAR)
%
%   Input:
%       DATA.
%           dados = attributes of test data [p x N]
%           alvos = labels of test data [c x N]
%       PAR.
%           P    = dados de treinamento             [p x Ntr]
%           T1   = alvos dos dados de treinamento   [c x Ntr]
%           alpha = lagrange multipliers
%           b0 = bias (hipersurface limiar)
%           Ktype = kernel type
%               1 -> Gaussian
%               2 -> Polinomial
%               3 -> MLP
%           sig2 (gaussian kernel variance)
%           ord (polinomial kernel order)
%           k_mlp / teta (mlp kernel parameters)
%   Output:
%       OUT.
%           y_h = classifier's output [c x N]
%           Mconf = classifier's confusion matrix [c x c]
%           acerto = classifier's accuracy rate [int]

%% INITIALIZATIONS

% Get Testing Data
Q = DATA.input;
T2 = DATA.output;

% Get Training Data
P = PAR.P;
T1 = PAR.T1;

% Adjust for the pattern [N x p] and [N x 1]
P = P';
T1 = T1(1,:)';
Q = Q';
T2 = T2(1,:)';

% Get Parameters
alpha = PAR.alpha;
b0 = PAR.b0;
sig2 = PAR.sig2;

% Number of samples
[Ntr,~] = size(P);      % training
[Nts,~] = size(Q);      % testing 

% Initialize Variables
y_h = zeros(Nts,1);     % 
y_h_s = zeros(Nts,1);   % 
T2_s = sign(T2);        % 
Mconf = zeros(2,2);     % 

%% ALGORITHM

% Calculate estimated output

for i = 1:Nts,

    % inicialize kernel
    K_ts1 = zeros(Ntr,1);
    K_ts = zeros(Ntr,1);
    
    % update kernel
    for j = 1:Ntr,
        K_ts1(j,1) = exp(-norm(Q(i,:)-P(j,:)).^2/2*sig2);
        K_ts(j,1) = alpha(j,:)*T1(j,:)*K_ts1(j,1);
    end
    
   	% output and signal function
    y_h(i,1) = sum(K_ts) + b0;
    y_h_s(i,1) = sign(y_h(i,1));
end

% Calculate Confusion Matrix and Accuracy

for j = 1:Nts,
    if T2_s(j) == 1,
        lin = 1;
    else
        lin = 2;
    end
    if y_h_s(j) == 1,
        col = 1;
    else
        col = 2;
    end
    Mconf(lin,col) = Mconf(lin,col) + 1;
end

acerto = sum(diag(Mconf)) / Nts;

y_h = [y_h,-y_h]';

%% FILL OUTPUT STRUCTURE

OUT.y_h = y_h;
OUT.Mconf = Mconf;
OUT.acerto = acerto;

%% END