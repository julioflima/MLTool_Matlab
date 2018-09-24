function [PARout] = svm_train(DATA,PAR)

% --- SVM classifier training ---
%
%   [PARout] = svm_train(DATA,PAR)
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
    PARaux.C = 5;           % Regularization Constant
    PARaux.Ktype = 1;   	% Kernel Type (gaussian = 1)
    PARaux.sig2 = 0.01;   	% Variance (gaussian kernel)
    PAR = PARaux;
else
    if (~(isfield(PAR,'C'))),
        PAR.C = 5;
    end
    if (~(isfield(PAR,'Ktype'))),
        PAR.Ktype = 1;
    end
    if (~(isfield(PAR,'sig2'))),
        PAR.sig2 = 0.011;
    end
end

%% INITIALIZATIONS

% Get Data
P = DATA.input;
T1 = DATA.output;

% Adjust Samples [N x p]
P = P';

% Adjust Labels [N x 1] for binary problem (-1 or +1)
T1 = T1';
T1 = T1(:,1);

% Hold Data
PAR.P = P;
PAR.T1 = T1;

% General Parameters
[N,~] = size(P);        % number of samples
C = PAR.C;              % regularization parameter
Ktype = PAR.Ktype;     	% kernel type
if Ktype == 1,
    sig2 = PAR.sig2;	% variance
end

%% ALGORITHM

% Calculate Kernel Matrix

Kmat = svm_f_kernel(PAR);

reg = 1e-10;                        % Regularization factor
Kmat = Kmat + reg*eye(size(Kmat));  % Avoid conditioning problems

% Calculate alphas in order to identify support vectors

H = Kmat;           % Kernel Matrix
f = -ones(N,1);     % Non-linear function
Aeq = T1';          % Equality Restriction
beq = 0;            % Equality Restriction
A = zeros(1,N);     % Inequality Restriction
b = 0;              % Inequality Restriction
lb = zeros(N,1);	% Minimum values for SV (Lagrange Multipliers)
ub = C*ones(N,1);	% Maximum values for SV (Lagrande Multipliers)
x0 = [];            % Dosen't indicate a initial value for x

% Quadratic optimization problem
opts = optimoptions(@quadprog,'Algorithm','interior-point-convex','Display','off');
[alpha,~,~,~,~] = quadprog(H,f,A,b,Aeq,beq,lb,ub,x0,opts);

% Get support vectors whose values are more than epsilon
epsilon = 0.001;
svi = find(alpha > epsilon);
nsv = length(svi);

% Calculate Bias (b0)

SV = P(svi,:);
out_SV = T1(svi,:);

for i = 1:nsv,
    if out_SV(i,1) == 1,
        x_mais = SV(i,:);
    else
        x_menos = SV(i,:);
    end
end

if Ktype == 1,
    
b0_aux = 0;
for j = 1:N,
   b0_aux = b0_aux + alpha(j)*T1(j)* ...
   (exp(-norm(P(j,:)-x_mais).^2/sig2) + exp(-norm(P(j,:)-x_menos).^2/sig2));
end
b0 = -0.5 * b0_aux;
    
end

%% FILL OUTPUT STRUCTURE

PARout = PAR;
PARout.alpha = alpha;
PARout.b0 = b0;
PARout.nsv = nsv;

%% THEORY

% ToDo - All

%% END