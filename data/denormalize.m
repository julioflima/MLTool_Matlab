function [DATAout] = denormalize(DATAin,OPTION)

% --- Normalize the raw data ---
%
%   [Xnorm] = denormalize(DATAin,option)
%
%   Input:
%       DATAin.
%           dados = data matrix [pxN]
%       OPTION.
%           norm = how will be the normalization
%               1: denormalize between [0, 1]
%               2: denormalize between  [-1, +1]
%               3:denormalize by mean and standard deviation
%                   Xnorm = (X-Xmean)/(std)
%   Output:
%       DATAout.
%           dados = denormalized matrix [pxN]

%% INITIALIZATIONS

option = OPTION.norm;   % gets normalization option from structure
dados = DATAin.dados;   % gets and data from structure - [pxN]

[p,N] = size(dados);    % number of samples and attributes
Xmin = DATAin.Xmin;     % minimum value of each attribute
Xmax = DATAin.Xmax;     % maximum value of each attribute
Xmed = DATAin.Xmed;     % mean of each attribute
dp = DATAin.dp;         % standard deviation of each attribute

%% ALGORITHM

dados_norm = zeros(p,N); % initialize data

switch option
    case (1)    % denormalize between [0 e 1]
        for i = 1:p,
            for j = 1:N,
                dados_norm(i,j) = dados(i,j)*(Xmax(i)-Xmin(i)) + Xmin(i); 
            end
        end
    case (2)    % denormalize between [-1 e +1]
        for i = 1:p,
            for j = 1:N,
                dados_norm(i,j) = 0.5*(dados(i,j) + 1)*(Xmax(i)-Xmin(i)) + Xmin(i); 
            end
        end
    case (3)    % denormalize by the mean and standard deviation
        for i = 1:p,
            for j = 1:N,
                dados_norm(i,j) = dados(i,j)*dp(i) + Xmed(i);
            end
        end
    otherwise
        dados_norm = dados;
        disp('Choose a correct option. Data was not normalized.')
end

%% FILL OUTPUT STRUCTURE

DATAin.dados = dados_norm;

DATAout = DATAin;

%% END