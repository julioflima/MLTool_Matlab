function [PARout] = lssvm_train(DATA,PAR)

% --- LSSVM classifier training ---
%
%   [PARout] = lssvm_train(DATA,PAR)
%
%   Input:
%       DATA.
%           input = attributes [p x N]
%           output = labels [c x N]
%       PAR.
%           C = regularization parameter
%           Ktype = kernel type
%               1 -> Gaussian
%               2 -> Polinomial
%               3 -> MLP
%           sig2 (gaussian kernel variance)
%           ord (polinomial kernel order)
%           k_mlp / .teta (mlp kernel parameters)
%   Output:
%       PARout.
%           alpha = langrage multiplier (indicates support vectors)
%           b0 = optimum bias
%           nsv = support vector number
%           P = attributes from training [p x N]
%           T1 = labels from training [c x N]

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(PAR))),
    PARaux.C = 0.5;    	% Regularization Constant
    PARaux.Ktype = 1;  	% Kernel Type (gaussian = 1)
    PARaux.sig2 = 128; % Variance (gaussian kernel)
    PAR = PARaux;
else
    if (~(isfield(PAR,'C'))),
        PAR.C = 0.5;
    end
    if (~(isfield(PAR,'Ktype'))),
        PAR.Ktype = 1;
    end
    if (~(isfield(PAR,'sig2'))),
        PAR.sig2 = 128;
    end

end

%% INITIALIZATIONS

% Get Data (and save it)
P = DATA.input;
T1 = DATA.output;
PAR.P = P;
PAR.T1 = T1;

% Adjust samples [N x p]; and labels [N x 1] for binary problem (-1 or +1)
P = P';
T1 = T1(1,:)';

% General Parameters
[N,~] = size(P);
C = PAR.C;
Ktype = PAR.Ktype;

% Initialize Kernel and Omega matrix
K = zeros(N,N);
Omega = zeros(N,N);

%% ALGORITHM

if Ktype == 1, % GAUSSIAN KERNEL

sig2 = PAR.sig2;
    
for i=1:N,
    for j=1:N,
        K(i,j) = exp(-norm(P(j,:)-P(i,:)).^2/2*sig2);
        Omega(i,j) = T1(i,1)*T1(j,1)*K(i,j);
    end
end

Omega = Omega + (1/C)*eye(N);

end

% Build A matrix and b vector in order to solve the problem Ax = b:

A(1,:) = [0 T1'];
for j=1:N,
    A(j+1,:)=[T1(j) Omega(j,:)];
end
b1 = ones(N,1);
b = [0;b1];

x_sys = linsolve(A,b);  % Solve linear equations Ax = b:

b0 = x_sys(1,1);        % find bias
alpha = x_sys(2:end,1); % find lagrange multipliers

% Determines number of support vectors
epsilon = 0;
svi = find(alpha ~= epsilon);
nsv = length(svi);

%% FILL OUTPUT STRUCTURE

PARout = PAR;
PARout.alpha = alpha;
PARout.b0 = b0;
PARout.nsv = nsv;

%% THEORY

% ToDo - All

%% END