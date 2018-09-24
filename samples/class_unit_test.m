%% Machine Learning ToolBox

% Classification Algorithms - Unit Test
% Author: David Nascimento Coelho
% Last Update: 2018/02/04

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% GENERAL DEFINITIONS

% General options' structure

OPT.prob = 06;              % Which problem/data set will be solved/
OPT.prob2 = 1;              % When it needs a more specific data set
OPT.hp = 4;                 % Chooses or not hiperparameters
OPT.norm = 3;               % Normalization definition
OPT.lbl = 0;                % Labeling definition
OPT.Nr = 02;                % Number of repetitions of the algorithm
OPT.hold = 2;               % Hold out method
OPT.ptrn = 0.8;             % Percentage of samples for training
OPT.file = 'fileX.mat';     % file where all the variables will be saved

%% CHOOSE ALGORITHM

% Handlers for classifiers functions

class_train = @wta_train;
class_test = @wta_classify;

%% CHOOSE HYPERPARAMETERS

if OPT.hp == 1,
% choose all hyperparameters
WTAp.ep = 200;       	% max number of epochs
WTAp.k = 20;          	% number of neurons (prototypes)
WTAp.init = 02;     	% neurons' initialization
WTAp.dist = 02;        	% type of distance
WTAp.learn = 02;      	% type of learning step
WTAp.No = 0.7;        	% initial learning step
WTAp.Nt = 0.01;      	% final   learning step
WTAp.lbl = 1;        	% Neurons' labeling function
WTAp.Von = 0;          	% disable video

Hp = WTAp;          	% Hyperparameters structure
clear WTAp;           	% Clear previous structure

elseif OPT.hp == 2,
% WTAp.ep = 200;       	% max number of epochs
% WTAp.k = 20;        	% number of neurons (prototypes)
% WTAp.init = 02;     	% neurons' initialization
% WTAp.dist = 02;    	% type of distance
% WTAp.learn = 02;    	% type of learning step
% WTAp.No = 0.7;      	% initial learning step
% WTAp.Nt = 0.01;      	% final   learning step
% WTAp.lbl = 1;        	% Neurons' labeling function
WTAp.Von = 0;          	% disable video
    
Hp = WTAp;             	% Hyperparameters structure
clear WTAp;            	% Clear previous structure

elseif OPT.hp == 3,
% use default hyperparameters
Hp = [];              	% Empty Matrix of Hyperparameters

elseif OPT.hp == 4,
% use default hyperparameters
Hp = struct();       	% Empty structure of Hyperparameters

end

%% DATA LOADING AND PRE-PROCESSING

DATA = data_class_loading(OPT);     % Load Data Set

Nc = length(unique(DATA.output));	% get number of classes

[p,N] = size(DATA.input);           % get number of attributes and samples

DATA = normalize(DATA,OPT);         % normalize the attributes' matrix

DATA = label_adjust(DATA,OPT);      % adjust labels for the problem

%% SHUFFLE AND HOLD OUT

% Shuffle data

I = randperm(N);
DATA.input = DATA.input(:,I); 
DATA.output = DATA.output(:,I);
DATA.lbl = DATA.lbl(:,I);

% Divide data between training and testing

[DATAho] = hold_out(DATA,OPT);
DATAtr = DATAho.DATAtr;
DATAts = DATAho.DATAts;

%% TRAINING AND TEST

display('begin')
display(datestr(now));

[Hp] = class_train(DATAtr,Hp);      % Calculate parameters
[OUTtr] = class_test(DATAtr,Hp);    % Results with training data
[OUTts] = class_test(DATAts,Hp);    % Results with test data

display('finish')
display(datestr(now));

%% RESULTS / STATISTICS

OUTtr.Mconf,
OUTtr.acerto,
OUTts.Mconf,
OUTts.acerto,

%% GRAPHICS



%% END
