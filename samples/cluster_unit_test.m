%% Machine Learning ToolBox

% Clustering Algorithms - Unit Test
% Author: David Nascimento Coelho
% Last Update: 2018/01/25

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% GENERAL DEFINITIONS

% General options' structure

OPT.prob = 06;              % Which problem will be solved / which data set will be used
OPT.prob2 = 1;              % When it needs a more specific data set
OPT.hp = 1;                 % Uses or not default hiperparameters
OPT.norm = 3;               % Normalization definition
OPT.lbl = 0;                % Labeling definition
OPT.Nr = 02;                % Number of repetitions of the algorithm
OPT.hold = 2;               % Hold out method
OPT.ptrn = 0.8;             % Percentage of samples for training
OPT.file = 'fileX.mat';     % file where all the variables will be saved

% Figure XXX parameters' structure

OPT_FIG2.fig = 2;           % Define figure number
OPT_FIG2.Xaxis = 1;         % Attribute to be plotted at x axis
OPT_FIG2.Yaxis = 2;         % Attribute to be plotted at y axis

% Figure XXX parameters' structure

OPT_FIG3.fig = 3;           % Define figure number
OPT_FIG3.Xaxis = 1;         % Attribute to be plotted at x axis
OPT_FIG3.Yaxis = 2;         % Attribute to be plotted at y axis

%% CHOOSE ALGORITHM

% Handlers for classifiers functions

cluster_alg = @som1d_cluster;
label_alg = @som1d_label;

%% CHOOSE HYPERPARAMETERS

if OPT.hp == 0,
    % use default hyperparameters
    Hp = struct();              % Empty Hyperparameters structure
else
    % choose hyperparameters
    PAR_SOM1d.lbl = 1;          % Neurons' labeling function
    PAR_SOM1d.ep = 200;       	% max number of epochs
    PAR_SOM1d.k = 05;         	% number of neurons (prototypes)
    PAR_SOM1d.init = 02;      	% neurons' initialization
    PAR_SOM1d.dist = 02;       	% type of distance
    PAR_SOM1d.learn = 02;     	% type of learning step
    PAR_SOM1d.No = 0.7;        	% initial learning step
    PAR_SOM1d.Nt = 0.01;      	% final learning step
    PAR_SOM1d.Nn = 01;         	% number of neighbors
    PAR_SOM1d.neig = 02;       	% type of neighbor function
    PAR_SOM1d.Vo = 0.8;        	% initial neighbor constant
    PAR_SOM1d.Vt = 0.3;       	% final neighbor constant
    
    Hp = PAR_SOM1d;        	% Hyperparameters structure
end

%% DATA LOADING AND PRE-PROCESSING

DATA = data_class_loading(OPT);     % Load Data Set

Nc = length(unique(DATA.output));	% get number of classes

[p,N] = size(DATA.input);           % get number of attributes and samples

DATA = normalize(DATA,OPT);         % normalize the attributes' matrix

DATA = label_adjust(DATA,OPT);      % adjust labels for the problem

%% CLUSTERING

[OUT_CL] = cluster_alg(DATA,Hp);
[OUT_CL] = label_alg(DATA,OUT_CL);

%% RESULTS / STATISTICS

% Quantization error
figure(1);
hold on
title ('SSE Curve');
xlabel('Epochs');
ylabel('SSE');
axis ([0 length(OUT_CL.SSE) min(OUT_CL.SSE)-0.1 max(OUT_CL.SSE)+0.1]);
plot(1:length(OUT_CL.SSE),OUT_CL.SSE);
hold off

%% GRAPHICS

% Clusters' Prototypes and Data
figure(2)
plot_clusters_and_data(DATA,OUT_CL,OPT_FIG2);

% Clusters' Grid and Data
figure(3)
plot_grid_and_data(DATA,OUT_CL,OPT_FIG3);

% Labeled Neurons' Grid
figure(4)
plot_labeled_neurons(OUT_CL);

% See Clusters Movie
figure(5)
movie(OUT_CL.M)

%% END
