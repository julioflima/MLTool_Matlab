function [OUT] = ksom_ef_classify(DATA,PAR)

% --- KSOM-EF Classify Function ---
%
%   [OUT] = ksom_ef_classify(DATA,PAR)
%
%   Input:
%       DATA.
%           input = input matrix [p x N]
%           output = output matrix [1 x N]
%       PAR.
%           C = prototypes [p x Nlin x Ncol]
%           index = [2 x N]
%           SSE = [1 x Nep]
%           label = class of each neuron [Nlin x Ncol]
%   Output:
%       OUT.
%           y_h = classifier's output [1 x N]
%           Mconf = classifier's confusion matrix [c x c]
%           acerto = classifier's accuracy rate [int]

%% INITIALIZATION

% Get Data
input = DATA.input;
data_lbl = DATA.output;
[~,N] = size(input);
Nc = length(unique(data_lbl));

% Get Prototypes and its labels
C = PAR.C;
label = PAR.label;

% Init outputs
y_h = zeros(1,N);
Mconf = zeros(Nc,Nc);
acerto = 0;

%% ALGORITHM

for i = 1:N,
    sample = input(:,i);                        % test sample
    win = prototypes_win_k(C,sample,PAR);       % winner neuron index
    y_h(i) = label(win(1),win(2));              % Update function's output
    Mconf(data_lbl(i),y_h(i)) = Mconf(data_lbl(i),y_h(i)) + 1;
end

acerto = acerto + trace(Mconf)/sum(sum(Mconf));

%% FILL OUTPUT STRUCTURE

OUT.y_h = y_h;
OUT.Mconf = Mconf;
OUT.acerto = acerto;

%% END