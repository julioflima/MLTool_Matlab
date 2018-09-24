function [PARout] = kmeans_train(DATA,PAR)

% --- k-means based classifier training ---
%
%   [PARout] = kmeans_train(DATA,PAR)
%
%   Input:
%       DATA.
%           input = input matrix [p x N]
%           output = output matrix [1 x N]
%       PAR.
%           ep = max number of epochs [cte]
%           k = number of prototypes [cte]
%           init = type of initialization for prototypes [cte]
%               1: C = zeros
%               2: C = randomly picked from data
%               3: C = mean of randomly choosen data
%           dist = type of distance [cte]
%               0: dot product
%               2: euclidean distance
%           lbl = type of labeling [cte]
%           Von = enable or disable video
%   Output:
%       PARout.
%       	C = clusters centroids (prototypes) [pxk]
%           label = class of each neuron [1 x k]
%           index = cluster index for each sample [1xN]
%           SSE = Sum of Squared Errors for each epoch [1 x ep]
%           M = frame structure (can be played with 'video function')

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(PAR))),
    PARaux.ep = 200;    % max number of epochs
    PARaux.k = 20;      % number of clusters (prototypes)
    PARaux.init = 2;    % type of initialization
    PARaux.dist = 2;    % type of distance
    PARaux.lbl = 1;     % Neurons' labeling function
    PARaux.Von = 0;     % disable video
    PAR = PARaux;
else
    if (~(isfield(PAR,'ep'))),
        PAR.ep = 200;
    end
    if (~(isfield(PAR,'k'))),
        PAR.k = 20;
    end
    if (~(isfield(PAR,'init'))),
        PAR.init = 2;
    end
    if (~(isfield(PAR,'dist'))),
        PAR.dist = 2;
    end
    if (~(isfield(PAR,'lbl'))),
        PAR.lbl = 1;
    end
    if (~(isfield(PAR,'Von'))),
        PAR.Von = 0;
    end
end

%% ALGORITHM

OUT_CL = kmeans_cluster(DATA,PAR);
PARout = kmeans_label(DATA,OUT_CL);

%% THEORY

% ToDo - All

%% END