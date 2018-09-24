function [PARout] = kmeans_cluster(DATA,PAR)

% --- k-means clustering function ---
%
%   [PARout] = kmeans_cluster(DATA,PAR)
%
%   Input:
%       DATA.
%           input = input matrix [p x N]
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
%           Von = enable or disable video
%   Output:
%       PARout.
%       	C = clusters centroids (prototypes) [pxk]
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
    if (~(isfield(PAR,'lbl'))),
        PAR.lbl = 1;
    end
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
    if (~(isfield(PAR,'Von'))),
        PAR.Von = 0;
    end
end

%% INITIALIZATIONS

% Get Data

input = DATA.input;
[~,N] = size(input);

% Get Hyperparameters

% lbl = PAR.lbl;
Nep = PAR.ep;
k = PAR.k;
% init = PAR.init;
% dist = PAR.dist;
Von = PAR.Von;

% Init output variables

C = prototypes_init(DATA,PAR);
index = zeros(1,N);
SSE = zeros(1,Nep);

M = struct('cdata',cell(1,Nep),'colormap', cell(1,Nep));

%% ALGORITHM

for ep = 1:Nep,

    % Save frame of the current epoch
    if (Von),
        M(ep) = prototypes_frame(C,DATA);
    end

    % Flag to verify if there was any change of data labeling
    change_flag = 0;

    % Initialize the number of samples for each centroid
    n_samples = zeros(1,k);

    % Assign data points to each cluster
    for i = 1:N,

        % Calculate closer centroid to sample
        sample = input(:,i);
        current_index = prototypes_win(C,sample,PAR);

        % Update index and signs that a change occured
        if (index(i) ~= current_index),
            index(i) = current_index;
            change_flag = 1;
        end

    end

    % Calculate SSE for the current loop
    SSE(ep) = prototypes_sse(C,DATA,PAR);

    % If there wasn't any change, break the loop
    if (change_flag == 0),
        break;
    end

    % Calculate new centroids
    for i = 1:N,
        n_samples(index(i)) = n_samples(index(i)) + 1;
        C(:,index(i)) = C(:,index(i)) + input(:,i);
    end
    for i = 1:k,
        % Randomly assigns a sample to a cluster centroid
        if n_samples(i) == 0
            I = randperm(N);
            C(:,i) = input(:,I(1));
        % Calculate mean of samples belonging to the cluster
        else
            C(:,i) = C(:,i) / n_samples(i);
        end
    end

end

%% FILL OUTPUT STRUCTURE

PARout = PAR;
PARout.C = C;
PARout.index = index;
PARout.SSE = SSE(1:ep);
PARout.M = M(1:ep);

%% END