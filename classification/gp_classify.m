function [OUT] = gp_classify(DATA,PAR)

% --- Teste do Classificador GP ---
%
%   Input:
%       DATA.
%           dados = attributes [p x N]
%           alvos = labels [c x N]
%       PAR.
%           l2 = 
%           K = 
%           sig2 = 
%   Output:
%       OUT.
%           y_h = classifier's output [c x N]
%           Mconf = classifier's confusion matrix [c x c]
%           acerto = classifier's accuracy rate [int]

%% INICIALIZAÇÕES

% ToDo - All (delete line above)

DATA.dados = PAR;

%% ALGORITMO

% ToDo - All

%% FILL OUTPUT STRUCTURE

OUT.y_h = DATA.alvos;
OUT.Mconf = zeros(2,2);
OUT.acerto = 0;

%% END