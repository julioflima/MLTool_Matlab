function [PARout] = mlm_train(DATA,PAR)

% --- MLM Classifier Training ---
%
%   [PARout] = mlm_train(DATA,PAR)
%
%   Input:
%       DATA.
%           input = training attributes [p x N]
%           output = training labels [c x N]
%       PAR.
%           K = number of reference points
%   Output:
%       PARout.
%           B = Regression matrix           [Npt x Npt]
%           Rx = Input reference points   	[Npt x p]
%           Ty = Output reference points   	[Npt x c]

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(PAR))),
    PARaux.K = 9;       % Number of reference points
    PAR = PARaux;
    
else
    if (~(isfield(PAR,'K'))),
        PAR.K = 9;      % Number of reference points
    end
end

%% INITIALIZATIONS

% Get Data
P = DATA.input;
T1 = DATA.output;

% Put in [N x p] and [N x c] pattern
P = P';
T1 = T1';

% Number of samples
[rx,~] = size(P);    % training
[ry,~] = size(T1);   % test

% Get hyperparameter
K = PAR.K;

%% ALGORITHM

% Aleatory selects reference points

I = randperm(rx);
I = I(1:K);

Rx = P(I,:);
[rr,~] = size(Rx);

Ty = T1(I,:);
[rt,~] = size(Ty);

% Input distance matrix of reference points

Dx = zeros(rx,rr);      %initialize Dx [N x K]

for n = 1:rx
    xi = P(n,:);        % get input sample
    for K = 1:rr
        mk = Rx(K,:);   % get input reference point
        Dx(n,K) = pdist([xi;mk],'euclidean'); % Calculates distance
    end
end

% Output distance matrix of reference points

Dy = zeros(ry,rt);      % initialize Dy [N x K]

for n = 1:ry,
    yi = T1(n,:);       % get output sample
    for K = 1:rt,
        tk = Ty(K,:);   % get output reference point
        Dy(n,K) = pdist([yi;tk],'euclidean'); % Calculates distance
    end
end

% Calculates Regeression Matrix [K x K]

B = pinv(Dx)*Dy;
% B = (Dx' * Dx) \ Dx' * Dy;
% B = inv(Dx' * Dx) * Dx' * Dy;

%% FILL OUTPUT STRUCTURE

PARout = PAR;
PARout.B = B;
PARout.Rx = Rx;
PARout.Ty = Ty;

%% THEORY

% Dy = Dx * B
% Dx'*Dx*B = Dx'*Dy
% inv(Dx'*Dx)*(Dx'*Dx)*B = inv(Dx'*Dx)*Dx'*Dy
% I*B = inv(Dx'*Dx)*Dx'*Dy
% B = inv(Dx'*Dx)*Dx'*Dy

%% END