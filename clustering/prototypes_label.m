function [PARout] = prototypes_label(DATA,OUT_CL)

% --- Clusters' Labeling Function ---
%
%   [PARout] = prototypes_label(DATA,OUT_CL)
% 
%   Input:
%       DATA.
%           input = input matrix [p x N]
%           output = output matrix [1 x N]
%       OUT_CL.
%           C = prototypes [p x k]
%           index = [1 x N]
%           SSE = [1 x Nep]
%           lbl = type of labeling [cte]
%   Output:
%       PARout.
%           C = prototypes [p x k]
%           label = class of each neuron [1 x k]
%           index = [1 x N]
%           SSE = [1 x Nep]

%% INITIALIZATION

% Get data and classes
dados = DATA.input;
data_lbl = DATA.output;

% Get number of samples and classes
[~,N] = size(dados);
Nc = length(unique(data_lbl));

% Get prototypes and labeling type
C = OUT_CL.C;
[~,k] = size(C);
lbl_type = OUT_CL.lbl;

% Init other parameters
votes = zeros(k,Nc);     % for voronoi labeling
mean_dist = zeros(k,Nc); % for average distance labeling
count = zeros(k,Nc);     % for average distance labeling
min_dist = 0;            % for minimum distance labeling

% Init Output
label = zeros(1,k);

%% ALGORITHM

if lbl_type == 1, % Voronoi Method
    
    % Fill voting matrix
    for i = 1:N,
        sample = dados(:,i);                        % data sample
        win = prototypes_win(C,sample,OUT_CL);      % winner neuron index
        votes(win,data_lbl(i)) = votes(win,data_lbl(i)) + 1;  % add to voting matrix
    end
    
    % Set Labels
    for i = 1:k,
        [~,class] = max(votes(i,:));                % get class with max number of votes
        label(i) = class(1);                        % label neuron
    end
    
elseif lbl_type == 2, % Average Method
    
	for i = 1:N,
        sample = dados(:,i);
        win = prototypes_win(C,sample,OUT_CL);
        neuron = C(:,win);
        dist_curr = sum((sample - neuron).^2);
        
        count(win,data_lbl(i)) = count(win,data_lbl(i)) + 1;
        mean_dist(win,data_lbl(i)) = ...
         (mean_dist(win,data_lbl(i))*(count(win,data_lbl(i))-1) + dist_curr) ...
            /count(win,data_lbl(i));
	end
    
    for i = 1:k,
        min_dist = 0;
        for c = 1:Nc,
            if(min_dist == 0 || mean_dist(i,c) < min_dist),
                min_dist = mean_dist(i,c);
                label(i) = c;
            end
        end
    end
    
elseif lbl_type == 3, % Minimum Distance Method
    
    for i = 1:k,
        for n = 1:N,
            % Get Sample, neuron and current distance
            sample = dados(:,n);
            neuron = C(:,i);                        
            dist_curr = sum((sample - neuron).^2);
            % set label and minimal distance
            if n == 1,
                min_dist = dist_curr;
                label(i) = data_lbl(n);
            else
                if dist_curr < min_dist,
                    min_dist = dist_curr;
                    label(i) = data_lbl(n);
                end
            end
        end
    end
    
end

%% FILL OUTPUT STRUCTURE

PARout = OUT_CL;       % Get parameters from training data
PARout.label = label;  % Set neuron's labels

%% END