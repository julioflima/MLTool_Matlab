function [PARout] = gauss_train(DATA,PAR)

% --- Gaussian Classifier Training ---
%
%   [PARout] = gauss_train(DATA,PAR)
%
%   Input:
%       DATA.
%           input = attributes [p x N]
%           output = labels [c x N]
%       PAR. 
%           type = type of gaussian classifier
%   Output:
%       PARout.
%           Ni = number of samples per class            (Nc x 1)
%           med_ni = centroid of each class             (Nc x p)
%           Ci = covariance matrix of each class        (Nc x p x p)

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(PAR))),
    PARaux.type = 2;          % Type of classificer
    PAR = PARaux;
    
else
    if (~(isfield(PAR,'type'))),
        PAR.type = 2;
    end
end

%% INITIALIZATIONS

% Data input and output
MATin = DATA.input;
MATout = DATA.output;

% Number or classes
[Nc,~] = size(MATout);

% Input matrix with [N x p] dimension
MATin = MATin';
[N,p] = size(MATin);

% Output Matrix with [N x 1] dimension
MATout = MATout';
Mout_aux = zeros(N,1);
for i = 1:N,
    Mout_aux(i) = find(MATout(i,:) > 0);
end
MATout = Mout_aux;

%% ALGORITHM

Ni = zeros(Nc,1);     	% Number of samples per class
med_ni = zeros(Nc,p);   % Mean of samples for each class

for i = 1:N,
    med_ni(MATout(i),:) = med_ni(MATout(i),:) + MATin(i,:); % sample sum accumulator
    Ni(MATout(i)) = Ni(MATout(i)) + 1;                      % number of samples accumulator
end

for i = 1:Nc,
    med_ni(i,:) = med_ni(i,:)/Ni(i);	% calculate mean
end

% Calculte Covariance Matrix for each class [(X-mu)*(X-mu)']/N

% Initialize matrix
Ci = cell(1,Nc);
for i = 1:Nc,
    Ci{i} = zeros(p,p);
end

% Calculate iteratively
for i = 1:N,
    Ci{MATout(i)} = Ci{MATout(i)} + (MATin(i,:)-med_ni(MATout(i),:))'*(MATin(i,:)-med_ni(MATout(i),:));
end

% Divide by number of elements
for i = 1:Nc,
    Ci{i} = Ci{i}/Ni(i);
end

%% FILL OUTPUT STRUCTURE

PARout = PAR;
PARout.Ni = Ni;
PARout.med_ni = med_ni;
PARout.Ci = Ci;

%% THEORY

% ToDo - All

%% END