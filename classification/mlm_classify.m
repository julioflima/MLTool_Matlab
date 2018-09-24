function [OUT] = mlm_classify(DATA,PAR)

% --- MLM Classifier Test ---
%
%   [OUT] = mlm_classify(DATA,PAR)
%
%   Input:
%       DATA.
%           input = test attributes [p x N]
%           output = test labels [c x N]
%       PAR.
%           B = Regression matrix           [Npt x Npt]
%           Rx = Input reference points   	[Npt x p]
%           Ty = Output reference points   	[Npt x c]
%   Output:
%       OUT.
%           y_h = classifier's output [c x N]
%           Mconf = classifier's confusion matrix [c x c]
%           acerto = classifier's accuracy rate [int]

%% INICIALIZAÇÕES

% Get Data
Q = DATA.input;
T2 = DATA.output;

% Get Parameters
B = PAR.B;
Rx = PAR.Rx;
Ty = PAR.Ty;

% Adjust for the pattern [N x p] and [N x c]
Q = Q';
T2 = T2';

% Get output dimensions
[Nts,Nc] = size(T2);

% Initialize outputs
y_h = zeros(size(Q,1),size(Ty,2)); 
Mconf = zeros(Nc);

%% ALGORITMO

% Distance matrix: input to reference points

Dx = pdist2(Q,Rx,'euclidean'); % [N x k]

% Distance matrix: output to reference points

DY_h = Dx*B; % [N x k]

% Options for fsolve algorithm

opts = optimoptions('fsolve','Algorithm','levenberg-marquardt', ...
          'Display', 'off', 'FunValCheck', 'on', 'TolFun', 10e-10);

% fsolve = solve non-linear equations system

for i = 1: size(Q, 1), 
    y_h(i,:) = fsolve(@(x)(sum((Ty - repmat(x, length(Ty), 1)).^2, 2) - DY_h(i,:)'.^2).^2, zeros(1, size(Ty, 2)), opts);
end

% Calculate Confusion Matrix and Accuracy

for i = 1:Nts,
    [~,iMaxD] = max(T2(i,:));
    [~,iMaxH] = max(y_h(i,:));
    Mconf(iMaxD,iMaxH) = Mconf(iMaxD,iMaxH) + 1;
end

acerto = sum(diag(Mconf)) / Nts;
y_h = y_h';

%% FILL OUTPUT STRUCTURE

OUT.y_h = y_h;
OUT.Mconf = Mconf;
OUT.acerto = acerto;

%% END