function [PARout] = som2d_train(DATA,PAR)

% --- SOM 2D based classifier training ---
%
%   [PARout] = som2d_train(DATA,PAR)
%
%   Input:
%       DATA.
%           input = input matrix [p x N]
%           output = output matrix [1 x N]
%       PAR.
%           ep = max number of epochs [cte]
%           k = number of prototypes (neurons) [1 x 2]
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
%           Nn = number of neighbors [cte]
%           neig = type of neighborhood function [cte]
%               1: if winner, h = 1, else h = 0.
%               2: if neighbor, h = exp (-(||ri -ri*||^2)/(V^2))
%                    where: V = Vo*((Vt/Vo)^(t/tmax))
%               3: decreasing function 1
%           Vo = initial neighborhood parameter [cte]
%           Vt = final neighborhood parameter [cte]
%           lbl = type of labeling [cte]
%           Von = enable or disable video
%   Output:
%       PARout.
%           C = clusters centroids (prototypes) [p x k(1) x k(2)]
%           label = class of each neuron [Nlin x Ncol]
%           index = cluster index for each sample [2 x N]
%           SSE = Sum of Squared Errors for each epoch [1 x ep]
%           M = frame structure (can be played with 'video function'

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(PAR))),
    PARaux.ep = 200;       	% max number of epochs
    PARaux.k = [5 4];      	% number of neurons (prototypes)
    PARaux.init = 02;     	% neurons' initialization
    PARaux.dist = 02;      	% type of distance
    PARaux.learn = 02;     	% type of learning step
    PARaux.No = 0.7;       	% initial learning step
    PARaux.Nt = 0.01;      	% final learnin step
    PARaux.Nn = 01;      	% number of neighbors
    PARaux.neig = 03;      	% type of neighborhood function
    PARaux.Vo = 0.8;      	% initial neighborhood constant
    PARaux.Vt = 0.3;      	% final neighborhood constant
    PARaux.lbl = 1;         % Neurons' labeling function
    PARaux.Von = 0;         % disable video 
    PAR = PARaux;
else
    if (~(isfield(PAR,'ep'))),
        PAR.ep = 200;
    end
    if (~(isfield(PAR,'k'))),
        PAR.k = [5 4];
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
    if (~(isfield(PAR,'Nn'))),
        PAR.Nn = 1;
    end
    if (~(isfield(PAR,'neig'))),
        PAR.neig = 2;
    end
    if (~(isfield(PAR,'Vo'))),
        PAR.Vo = 0.8;
    end
    if (~(isfield(PAR,'Vt'))),
        PAR.Vt = 0.3;
    end
    if (~(isfield(PAR,'lbl'))),
        PAR.lbl = 1;
    end
    if (~(isfield(PAR,'Von'))),
        PAR.Von = 0;
    end
end

%% ALGORITHM

[OUT_CL] = som2d_cluster(DATA,PAR);
[PARout] = som2d_label(DATA,OUT_CL);

%% FILL OUTPUT STRUCTURE

%% END