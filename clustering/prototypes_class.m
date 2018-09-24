function [OUT] = prototypes_class(DATA,PAR)

% --- Prototype-Based Classify Function ---
%
%   [OUT] = prototypes_class(DATA,PAR)
% 
%   Input:
%       DATA.
%           input = input matrix [p x N]
%           output = output matrix [1 x N]
%       PAR.
%           C = prototypes [p x Neu]
%           label = class of each neuron [1 x Neu]
%           index = [1 x N]
%           SSE = [1 x Nep]
%   Output:
%       OUT.
%           y_h = classifier's output [1 x N]
%           Mconf = classifier's confusion matrix [c x c]
%           acerto = classifier's accuracy rate [int]

%% INITIALIZATION

% Get Data

data_in = DATA.input;
data_lbls = DATA.output;

[~,N] = size(data_in);
Nc = length(unique(data_lbls));

% Get prototypes and its labels

C = PAR.C;
prot_lbls = PAR.label;

% Init Output

y_h = zeros(1,N);
Mconf = zeros(Nc,Nc);
acerto = 0;

%% ALGORITHM

for i = 1:N,
    sample = data_in(:,i);              % test sample
    win = prototypes_win(C,sample,PAR); % Winner Neuron index
    y_h(i) = prot_lbls(win);        	% Update function's output
    Mconf(data_lbls(i),prot_lbls(win)) = Mconf(data_lbls(i),prot_lbls(win)) + 1;
end

acerto = acerto + trace(Mconf)/sum(sum(Mconf));

%% FILL OUTPUT STRUCTURE

OUT.y_h = y_h;
OUT.Mconf = Mconf;
OUT.acerto = acerto;

%% END