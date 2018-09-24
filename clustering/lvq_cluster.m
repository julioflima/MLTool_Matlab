function [PARout] = lvq_cluster(DATA,PAR)

% --- LVQ Clustering Function ---
%
%   [PARout] = lvq_cluster(DATA,PAR)
%
%   Input:
%       DATA.
%           input = input matrix [p x N]
%           output = output matrix [c x N]
%       PAR.
%           ep = max number of epochs [cte]
%           k = number of prototypes (neurons) [1 x 1]
%           init = type of initialization for prototypes [cte]
%               1: C = zeros
%               2: C = randomly picked from data
%               3: C = mean of randomly choosen data
%           dist = type of distance [cte]
%               0: dot product
%               2: euclidean distance
%           learn = type of learning step [cte]
%               1: N = No (constant)
%               2: N = No*(1-(t/tmax))
%               3: N = No/(1+t)
%               4: N = No*((Nt/No)^(t/tmax))
%           No = initial learning step [cte]
%           Nt = final learning step [cte]
%   Output:
%       PARout.
%       	C = clusters centroids (prototypes) [p x k]
%           index = cluster index for each sample [1 x N]
%           SSE = Sum of Squared Errors for each epoch [1 x ep]
%           M = frame structure (can be played with 'video function')

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(PAR))),
    PARaux.ep = 200;        % max number of epochs
    PARaux.k = 20;          % number of neurons (prototypes)
    PARaux.init = 02;   	% neurons' initialization
    PARaux.dist = 02;      	% type of distance
    PARaux.learn = 02;    	% type of learning step
    PARaux.No = 0.7;      	% initial learning step
    PARaux.Nt = 0.01;      	% final   learning step
    PARaux.Von = 0;         % disable video
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
    if (~(isfield(PAR,'learn'))),
        PAR.learn = 2;
    end
    if (~(isfield(PAR,'No'))),
        PAR.No = 0.7;
    end
    if (~(isfield(PAR,'Nt'))),
        PAR.Nt = 0.01;
    end
    if (~(isfield(PAR,'Von'))),
        PAR.Von = 0;
    end
end

%% INITIALIZATION

% Get data 

input = DATA.input;
output = DATA.output;
[~,N] = size(input);

% Get parameters

Nep = PAR.ep;
% k = PAR.k;
% init = PAR.init;
% dist = PAR.dist;
learn = PAR.learn;
No = PAR.No;
Nt = PAR.Nt;
Von = PAR.Von;

% Init aux variables

tmax = N*Nep;           % max number of iterations
t = 0;                 	% count iterations

% Init Outputs

codeBook = lvq_f_init(DATA,PAR);
C = codeBook.C;
label = codeBook.label;
index = zeros(1,N);
SSE = zeros(1,Nep);

M = struct('cdata',cell(1,Nep),'colormap', cell(1,Nep));

%% ALGORITHM

% Update prototypes (just winner for each iteration)
for ep = 1:Nep,
    
    % Save frame of the current epoch
    if (Von),
        M(ep) = prototypes_frame(C,DATA);
    end
    
    % shuffle data
    I = randperm(N);
    input = input(:,I);
    output = output(:,I);
    
    % Get Winner Neuron, update Learning Step, update prototypes
    for i = 1:N,

        t = t+1;                                    % Uptade Iteration
        sample = input(:,i);                        % Training sample
        win = prototypes_win(C,sample,PAR);         % Winner Neuron index
        n = prototypes_learn(learn,tmax,t,No,Nt);   % Learning Step
        
        % Update Winner Neuron (Prototype)
        if label(win) == output(i),
            C(:,win) = C(:,win) +  n * (input(:,i) - C(:,win));
        else
            C(:,win) = C(:,win) -  n * (input(:,i) - C(:,win));
        end
    end
    
    % SSE (one epoch)
    SSE(ep) = prototypes_sse(C,DATA,PAR);
    
end

% Assign indexes
for i = 1:N,
    sample = DATA.input(:,i);               % not shuffled data
    win = prototypes_win(C,sample,PAR);     % Winner Neuron index
    index(:,i) = win;                       % save index for sample
end

%% FILL OUTPUT STRUCTURE

% Get Previous Parameters
PARout = PAR;

% Get Output Parameters
PARout.C = C;
PARout.label = label;
PARout.index = index;
PARout.SSE = SSE;
PARout.M = M;

%% END