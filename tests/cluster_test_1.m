%% Machine Learning ToolBox

% Clustering Example
% Author: David Nascimento Coelho
% Last Update: 2017/04/25

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% GENERAL DEFINITIONS

% General options' structure

OPT.prob = 4;               % Which problem will be solved / which data set will be used
OPT.norm = 3;               % Normalization definition
OPT.lbl = 0;                % Labeling definition
OPT.Nr = 02;                % Number of repetitions of each algorithm
OPT.hold = 1;               % Hold out method
OPT.ptrn = 0.8;             % Percentage of samples for training
OPT.file = 'file1.mat';     % file where all the variables will be saved  

% Figure 02 parameters' structure

OPT_FIG2.fig = 2;           % Define figure number
OPT_FIG2.Xaxis = 1;         % Attribute to be plotted at x axis
OPT_FIG2.Yaxis = 2;         % Attribute to be plotted at y axis

% Figure 03 parameters' structure

OPT_FIG3.fig = 3;           % Define figure number
OPT_FIG3.Xaxis = 1;         % Attribute to be plotted at x axis
OPT_FIG3.Yaxis = 2;         % Attribute to be plotted at y axis

%% DATA LOADING AND PRE-PROCESSING

DATA = data_class_loading(OPT.prob);  % Load Data Set

Nc = length(unique(DATA.rot));	% get number of classes

[p,N] = size(DATA.dados);   	% get number of attributes and samples

DATA = normalize(DATA,OPT);     % Normalize Data

DATA = label_adjust(DATA,OPT);  % adjust labels for the problem

%% HIPERPARAMETERS - DEFAULT

% Clustering General Hyperparameters

PAR_AUX.Kmax = 15;	% Max number of clusters (prototypes)
PAR_AUX.nrep = 05;	% number of repetitions with each quantity of clusters
PAR_AUX.alg = 01;	% Clustering Algorithm
PAR_AUX.ind = 01;	% Validation index.

% Clustering Specific Hyperparameters

switch (PAR_AUX.alg),
    case 1,
        % Kmeans
        PAR_KM.ep = 200;        % max number of epochs
        PAR_KM.k = 08;          % number of clusters (prototypes)
        PAR_KM.init = 02;       % type of initialization
        PAR_KM.dist = 02;       % type of distance
        PAR_CL = PAR_KM;
    case 2,
        % WTA
        PAR_WTA.ep = 200;       % max number of epochs
        PAR_WTA.k = 04;         % number of neurons (prototypes)
        PAR_WTA.init = 02;      % neurons' initialization
        PAR_WTA.dist = 02;      % type of distance
        PAR_WTA.learn = 02;     % type of learning step
        PAR_WTA.No = 0.7;       % initial learning step
        PAR_WTA.Nt = 0.01;      % final   learning step
        PAR_CL = PAR_WTA;
    case 3,
        % SOM-1D
        PAR_SOM1d.ep = 200;     % max number of epochs
        PAR_SOM1d.k = 20;       % number of neurons (prototypes)
        PAR_SOM1d.init = 02;    % neurons' initialization
        PAR_SOM1d.dist = 02;    % type of distance
        PAR_SOM1d.learn = 02;   % type of learning step
        PAR_SOM1d.No = 0.7;     % initial learning step
        PAR_SOM1d.Nt = 0.01;    % final learning step
        PAR_SOM1d.Nn = 01;      % number of neighbors
        PAR_SOM1d.neig = 02;    % type of neighbor function
        PAR_SOM1d.Vo = 0.8;     % initial neighbor constant
        PAR_SOM1d.Vt = 0.3;     % final neighbor constant
        PAR_CL = PAR_SOM1d;
    case 4,
        % SOM-2D
        PAR_SOM2d.ep = 200;     % max number of epochs
        PAR_SOM2d.k = [5 4];	% number of neurons (prototypes)
        PAR_SOM2d.init = 02;	% neurons' initialization
        PAR_SOM2d.dist = 02;	% type of distance
        PAR_SOM2d.learn = 02;	% type of learning step
        PAR_SOM2d.No = 0.7;     % initial learning step
        PAR_SOM2d.Nt = 0.01;    % final learnin step
        PAR_SOM2d.Nn = 01;      % number of neighbors
        PAR_SOM2d.neig = 03;	% type of neighborhood function
        PAR_SOM2d.Vo = 0.8;     % initial neighborhood constant
        PAR_SOM2d.Vt = 0.3;     % final neighborhood constant
        PAR_CL = PAR_SOM2d;
    case 5,
        % GD-KSOM
        PAR_ksom_gd.ep = 200;	% max number of epochs
        PAR_ksom_gd.k = [5 4];  % number of neurons (prototypes)
        PAR_ksom_gd.init = 02;	% neurons' initialization
        PAR_ksom_gd.dist = 02;	% type of distance
        PAR_ksom_gd.learn = 02;	% type of learning step
        PAR_ksom_gd.No = 0.7; 	% initial learning step
        PAR_ksom_gd.Nt = 0.01;	% final learning step
        PAR_ksom_gd.Nn = 01;	% number of neighbors
        PAR_ksom_gd.neig = 03;	% type of neighbor function
        PAR_ksom_gd.Vo = 0.8;	% initial neighbor constant
        PAR_ksom_gd.Vt = 0.3;	% final neighbor constant
        PAR_ksom_gd.Kt = 3;     % Type of Kernel
        PAR_ksom_gd.sig2 = 0.5; % Variance (gaussian, log, cauchy kernel)
        PAR_CL = PAR_ksom_gd;
end

% Build Complete Hyperparameters Structure

PAR_CL.Kmax = PAR_AUX.Kmax;
PAR_CL.nrep = PAR_AUX.nrep;
PAR_CL.alg = PAR_AUX.alg;
PAR_CL.ind = PAR_AUX.ind;

%% ALGORITHM

% Divide Data between training and test

[DATAho] = hold_out(DATA,OPT);

DATAtr.dados = DATAho.P;
DATAtr.alvos = DATAho.T1;
DATAtr.rot = DATAho.rot_T1;

DATAts.dados = DATAho.Q;
DATAts.alvos = DATAho.T2;
DATAts.rot = DATAho.rot_T2;

% Clustering General Function
% [OUT] = clustering(DATA,PAR_CL);

switch (PAR_AUX.alg),
    case 1,
        % Kmeans
        [OUT_CL] = kmeans_train(DATAtr,PAR_KM);
        [PAR] = kmeans_label(DATAtr,OUT_CL);
        [OUT_CLASS] = kmeans_classify(DATAts,PAR);
    case 2,
        % WTA
        [OUT_CL] = wta_train(DATAtr,PAR_WTA);
        [PAR] = wta_label(DATAtr,OUT_CL);
        [OUT_CLASS] = wta_classify(DATAts,PAR);
    case 3,
        % SOM-1D
        [OUT_CL] = som1d_train(DATAtr,PAR_SOM1d);
        [PAR] = som1d_label(DATAtr,OUT_CL);
        [OUT_CLASS] = som1d_classify(DATAts,PAR);
    case 4,
        % SOM-2D
        [OUT_CL] = som2d_train(DATAtr,PAR_SOM2d);
        [PAR] = som2d_label(DATAtr,OUT_CL);
        [OUT_CLASS] = som2d_classify(DATAts,PAR);
    case 5,
        % LVQ -> ToDo
    case 6,
        % KSOM-GD
        [OUT_CL] = ksom_gd_train(DATAtr,PAR_ksom_gd);
        [PAR] = ksom_gd_label(DATAtr,OUT_CL);
        [OUT_CLASS] = ksom_gd_classify(DATAts,PAR);
    case 7,
        % KSOM-EF -> ToDo
    case 8,
        % KSOM-PS -> ToDo
    case 9,
        % MoG -> ToDo
end

acertao = OUT_CLASS.Mconf(1,1)+sum(sum(OUT_CLASS.Mconf(2:end,2:end)));
acertao = acertao/59;

%% CLASSIFICATION RESULTS



%% CLUSTER RESULTS - INDIVIDUAL

% Clusters' Prototypes and Data
% figure(2)
% plot_clusters_and_data(DATA,OUT_CL,OPT_FIG2);

% Clusters' Grid and Data
% figure(3)
% plot_grid_and_data(DATA,OUT_CL,OPT_FIG3);

% Labeled Neurons' Grid
% figure(4)
% plot_labeled_neurons(PAR);

% Quantization error
% figure(5);
% hold on
% title ('SSE Curve');
% xlabel('Epochs');
% ylabel('SSE');
% axis ([0 length(OUT_CL.SSE) min(OUT_CL.SSE)-0.1 max(OUT_CL.SSE)+0.1]);
% plot(1:length(OUT_CL.SSE),OUT_CL.SSE);
% hold off

% See Clusters Movie
% figure(6)
% movie(OUT_CL.M)

%% CLUSTER RESULTS - GENERAL

% Validation Index (2:Kmax)
% figure(7)
% plot(1:PAR_CL.Kmax,OUT_CL.val_ind);

% DATA and Prototypes (2:Kmax)
% figure(8)
% for i = 8:PAR_CL.Kmax,
%     OPT_FIG.fig = i;
%     plot_clusters(DATA,OUT_CL.best_results{i},OPT_FIG);
% end

%% SAVE DATA

% Denormalized data
% DATA = denormalize(DATA,OPT_GEN);

% Save to file .mat
% save(OPT.file);

% Save Video
% movie2avi(OUT_CL.M,'clustering.avi','compression','none')

%% END