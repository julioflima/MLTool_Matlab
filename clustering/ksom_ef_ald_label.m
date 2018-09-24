function [PARout] = ksom_ef_ald_label(DATA,OUT_CL)

% --- KSOM-EF Labeling Function ---
%
%   [PARout] = ksom_ef_label(DATA,OUT_CL)
% 
%   Input:
%       DATA.
%           input = input matrix [p x N]
%           output = output matrix [c x N]
%       OUT_CL.
%           lbl = Neurons' labeling function
%           Kt = Type of Kernel
%           sig2 = Variance (gaussian, log, cauchy kernel)
%           C = prototypes [p x Nlin x Ncol]
%           index = [2 x N]
%           SSE = [1 x Nep]
%   Output:
%       PARout.
%           C = prototypes [p x Nlin x Ncol]
%           index = [2 x N]
%           SSE = [1 x Nep]
%           label = class of each neuron [Nlin x Ncol]

%% INITIALIZATION

% Get data and classes
input = DATA.input;
data_lbl = DATA.output;

% Get number of samples and classes
[~,N] = size(input);
Nc = length(unique(data_lbl));

% Get prototypes
C = OUT_CL.C;
[~,Nlin,Ncol] = size(C);
lbl_type = OUT_CL.lbl;
kernel_type = OUT_CL.Kt;
sig2 = OUT_CL.sig2;

% Init other parameters
votes = zeros(Nlin,Ncol,Nc);        % for voronoi labeling
mean_dist = zeros (Nlin,Ncol,Nc);   % for average distance labeling
count = zeros(Nlin,Ncol,Nc);        % for average distance labeling
min_dist = 0;                       % for minimum distance labeling

% Init Output
label = zeros(Nlin,Ncol);

%% ALGORITHM

if lbl_type == 1, % Voronoi Method
    
    %votacao para rotular os neuronios
    for i = 1:N,
        sample = input(:,i);                        % data sample
        win = prototypes_win_k(C,sample,OUT_CL);   	% winner neuron index
        votes(win(1),win(2),data_lbl(i)) = votes(win(1),win(2),data_lbl(i)) + 1;
    end
    
    %rotulacao dos neuronios
    for i=1:Nlin,
        for j=1:Ncol,
            [~,class] = max(votes(i,j,:));          % get class with max number of votes
            label(i,j) = class(1);                  % label neuron
        end
    end

elseif lbl_type == 2, % Average Method
    
    for i = 1:N,
        sample = input(:,i);
        win = prototypes_win_k(C,sample,OUT_CL);
        neuron = C(:,win(1),win(2));

        if kernel_type == 1,
        	dist_curr = 2 - 2*exp(-sum((sample - neuron).^2) / (2*sig2));
       	elseif kernel_type == 2,
        	dist_curr  = (2*sig2)/(sig2+1) + (2*sig2)/(sig2+sum((sample - neuron).^2));
       	elseif kernel_type == 3,
        	dist_curr = -2*log(1+(sum((sample - neuron).^2)/sig2));
        end
        
        count(win(1),win(2),data_lbl(i)) =  count(win(1),win(2),data_lbl(i)) + 1;
        mean_dist(win(1),win(2),data_lbl(i)) = ...
            (mean_dist(win(1),win(2),data_lbl(i))*(count(win(1),win(2),data_lbl(i)) - 1) + dist_curr) ...
            / count(win(1),win(2),data_lbl(i));
    end
    
    for i = 1:Nlin,
        for j = 1:Ncol,
            min_dist = 0;
            for c = 1:Nc,
                if (min_dist == 0 || mean_dist(i,j,c) < min_dist),
                    min_dist = mean_dist(i,j,c);
                    label(i,j) = c;
                end
            end
        end
    end    
    
elseif lbl_type == 3, % Minimum Distance Method
    
    for i = 1:Nlin,
        for j = 1:Ncol,
            for n = 1:N,
                % Get Sample, neuron and current distance
                sample = input(:,n);
                neuron = C(:,i,j);
              	if kernel_type == 1,
                    dist_curr = 2 - 2*exp(-sum((sample - neuron).^2) / (2*sig2));
                elseif kernel_type == 2,
                    dist_curr  = (2*sig2)/(sig2+1) + (2*sig2)/(sig2+sum((sample - neuron).^2));
                elseif kernel_type == 3,
                    dist_curr = -2*log(1+(sum((sample - neuron).^2)/sig2));
                end
                % set label and minimal distance
                if n == 1,
                    min_dist = dist_curr;
                    label(i,j) = data_lbl(n);
                else
                    if dist_curr < min_dist,
                        min_dist = dist_curr;
                        label(i,j) = data_lbl(n);
                    end
                end
            end
        end
    end    
    
end

%% FILL OUTPUT STRUCTURE

PARout = OUT_CL;       % Get parameters from training data
PARout.label = label;  % Set neuron's labels

%% END