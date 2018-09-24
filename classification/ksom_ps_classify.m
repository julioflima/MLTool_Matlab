function [OUT] = ksom_ps_classify(DATA,PAR)

% --- KSOM-PS Classify Function ---
%
%   [OUT] = ksom_ps_classify(DATA,PAR)
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
%           Mred = matrix with M samples of training data [p x M]
%           eig_vec = eigenvectors of reduced kernel matrix [M x M]
%           eig_val = eigenvalues of reduced kernel matrix [1 x M]
%   Output:
%       OUT.
%           y_h = classifier's output [1 x N]
%           Mconf = classifier's confusion matrix [c x c]
%           acerto = classifier's accuracy rate [int]

%% INITIALIZATION

% Get Data
MATin = DATA.input;
data_lbl = DATA.output;

% Number of Samples and Classes
[~,N] = size(MATin);
Nc = length(unique(data_lbl));

% Get Prototypes and its labels
C = PAR.C;
label = PAR.label;

% Get auxiliary Variables
Mred = PAR.Mred;
[~,M] = size(Mred);
eig_vec = PAR.eig_vec;
eig_val = PAR.eig_val;
sig2 = PAR.sig2;

% Init outputs
y_h = zeros(1,N);
Mconf = zeros(Nc,Nc);
acerto = 0;

%% ALGORITHM

% Map samples to feature space

Mphi = zeros(M,N);

for n = 1:N,
    Xn = MATin(:,n);
    for i = 1:M,
        sum_aux = 0;
        for m = 1:M,
            Zm = Mred(:,m);
            k_xz = exp(-(norm(Xn-Zm))^2/(2*sig2)); % kernel between Xn an Zm
            sum_aux = sum_aux + eig_vec(m,i)*k_xz; % calculate sum
        end
        Mphi(i,n) = sum_aux / sqrt(eig_val(i)); % atributte of mapped vector
    end
end

MATin = Mphi;

% Classify at mapped space

for i = 1:N,
    sample = MATin(:,i);                        % test sample
    win = prototypes_win(C,sample,PAR);       % winner neuron index
    y_h(i) = label(win(1),win(2));            % Update function's output
    Mconf(data_lbl(i),y_h(i)) = Mconf(data_lbl(i),y_h(i)) + 1;
end

acerto = acerto + trace(Mconf)/sum(sum(Mconf));

%% FILL OUTPUT STRUCTURE

OUT.y_h = y_h;
OUT.Mconf = Mconf;
OUT.acerto = acerto;

%% END