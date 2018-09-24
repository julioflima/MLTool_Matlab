function [DATAout] = normalize(DATAin,OPTION)

% --- Normalize the raw data ---
%
%   [DATAout] = normalize(DATAin,option)
%
%   Input:
%       DATAin.input = data matrix [pxN]
%       OPTION.norm = how will be the normalization
%           = 1 -> normalize between [0, 1]
%           = 2 -> normalize between  [-1, +1]
%           = 3 -> normalize by mean and standard deviation
%                   Xnorm = (X-Xmean)/(std)
%   Output:
%       DATAout.input = normalized matrix [pxN]

%% INITIALIZATIONS

option = OPTION.norm;   % gets normalization option from structure
dados = DATAin.input';  % gets and transpose data from structure - [Nxp]

[N,p] = size(dados);    % number of samples and attributes
Xmin = min(dados)';     % minimum value of each attribute
Xmax = max(dados)';     % maximum value of each attribute
Xmed = mean(dados)';    % mean of each attribute
dp = std(dados)';       % standard deviation of each attribute

%% ALGORITHM

dados_norm = zeros(N,p); % initialize data

switch option
    case (1)    % normalize between [0 e 1]
        for i = 1:N,
            for j = 1:p,
                dados_norm(i,j) = (dados(i,j) - Xmin(j))/(Xmax(j)-Xmin(j)); 
            end
        end
    case (2)    % normalize between [-1 e +1]
        for i = 1:N,
            for j = 1:p,
                dados_norm(i,j) = 2*(dados(i,j) - Xmin(j))/(Xmax(j)-Xmin(j)) - 1; 
            end
        end
    case (3)    % normalize by the mean and standard deviation
        for i = 1:N,
            for j = 1:p,
                dados_norm(i,j) = (dados(i,j) - Xmed(j))/dp(j); 
            end
        end
    otherwise
        dados_norm = dados;
        disp('Choose a correct option. Data was not normalized.')
end

dados_norm = dados_norm'; % transpose data for -> [p x N]

%% FILL OUTPUT STRUCTURE

DATAin.input = dados_norm;

DATAin.Xmin = Xmin;
DATAin.Xmax = Xmax;
DATAin.Xmed = Xmed;
DATAin.dp = dp;

DATAout = DATAin;

%% END