%% SOM Sample

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% GENERAL DEFINITIONS

% General options' structure

OPT.prob = 7;               % Which problem will be solved / which data set will be used
OPT.prob2 = 1;              % When it needs more specific data set
OPT.norm = 3;               % Normalization definition
OPT.lbl = 0;                % Data labeling definition
OPT.Nr = 50;                % Number of repetitions of each algorithm
OPT.hold = 1;               % Hold out method
OPT.ptrn = 0.8;             % Percentage of samples for training
OPT.file = 'fileX.mat';     % file where all the variables will be saved  

% Cross Validation hiperparameters

CVp.on = 0;                 % If 1, includes cross validation
CVp.fold = 5;               % Number of folds for cross validation

% Reject Option hiperparameters

REJp.on = 0;                % Includes reject option
REJp.band = 0.3;            % Range of rejected values [-band +band]
REJp.w = 0.25;              % Rejection cost

% Prototypes' labeling definition

prot_lbl = 1;               % = 1 / 2 / 3

%% DATA LOADING AND PRE-PROCESSING

DATA = data_class_loading(OPT); % Load Data Set

Nc = length(unique(DATA.rot));  % get number of classes

[p,N] = size(DATA.dados);   	% get number of attributes and samples

DATA = normalize(DATA,OPT);     % normalize the attributes' matrix

DATA = label_adjust(DATA,OPT);  % adjust labels for the problem

%% HIPERPARAMETERS - DEFAULT

SOM2p_acc = cell(OPT.Nr,1);	 % Init of Acc Hyperparameters of SOM-2D
PAR_SOM2d.lbl = prot_lbl;	 % Neurons' labeling function
PAR_SOM2d.ep = 200;          % max number of epochs
PAR_SOM2d.k = [5 4];         % number of neurons (prototypes)
PAR_SOM2d.init = 02;         % neurons' initialization
PAR_SOM2d.dist = 02;         % type of distance
PAR_SOM2d.learn = 02;        % type of learning step
PAR_SOM2d.No = 0.7;          % initial learning step
PAR_SOM2d.Nt = 0.01;         % final learnin step
PAR_SOM2d.Nn = 01;      	 % number of neighbors
PAR_SOM2d.neig = 03;         % type of neighborhood function
PAR_SOM2d.Vo = 0.8;          % initial neighborhood constant
PAR_SOM2d.Vt = 0.3;          % final neighborhood constant

%% CLASSIFIERS' RESULTS INIT

som2_out_tr = cell(OPT.Nr,1);	% Acc of training data output
som2_out_ts = cell(OPT.Nr,1);	% Acc of test data output
som2_Mconf_sum = zeros(Nc,Nc);  % Aux var for mean confusion matrix calc

%% HOLD OUT / CROSS VALIDATION / TRAINING / TEST

hold_acc = cel{OPT.Nr,1};       % Init hold_out acc

for r = 1:OPT.Nr,

% %%%%%%%%% DISPLAY REPETITION AND DURATION %%%%%%%%%%%%%%

display(r);
display(datestr(now));

% %%%%%%%%%%%%%%%%%%%% HOLD OUT %%%%%%%%%%%%%%%%%%%%%%%%%%

[DATAho] = hold_out(DATA,OPT);

hold_acc{r} = DATAho;
DATAtr = DATAho.DATAtr;
DATAts = DATAho.DATAts;

% %%%%%%%%%%%%%% CLASSIFIERS' TRAINING %%%%%%%%%%%%%%%%%%%

[OUT_CL] = som2d_train(DATAtr,PAR_SOM2d);
[PAR_SOM2d] = som2d_label(DATAtr,OUT_CL);

% %%%%%%%%%%%%%%%%% CLASSIFIERS' TEST %%%%%%%%%%%%%%%%%%%%

% SOM 2D

[OUTtr] = som2d_classify(DATAtr,PAR_SOM2d);
OUTtr.nf = normal_or_fail(OUTtr.Mconf);
som2_out_tr{r,1} = OUTtr; % training set results

[OUTts] = som2d_classify(DATAts,PAR_SOM2d);
OUTts.nf = normal_or_fail(OUTts.Mconf);
som2_out_ts{r,1} = OUTts; % test set results

SOM2p_acc{r} = PAR_SOM2d; % hold parameters
som2_Mconf_sum = som2_Mconf_sum + OUTts.Mconf; % hold confusion matrix

end

%% STATISTICS

% Mean Confusion Matrix

som2_Mconf_sum = som2_Mconf_sum / OPT.Nr; 

%% GRAPHICS

% Init labels' cell and Inicializa boxplot matrix

labels = {};

Mat_boxplot1 = []; % Train Multiclass
Mat_boxplot2 = []; % Train Binary
Mat_boxplot3 = []; % Test Multiclass
Mat_boxplot4 = []; % Test Binary

[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'SOM 2D'};
Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(som2_out_tr)];
Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(som2_out_tr)];
Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(som2_out_ts)];
Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(som2_out_ts)];

% BOXPLOT 1
figure; boxplot(Mat_boxplot1, 'label', labels);
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
ylabel('Accuracy')              % label eixo y
xlabel('Classifiers')           % label eixo x
title('Classifiers Comparison') % Titulo
axis ([0 n_labels+1 0 1.05])	% Eixos

hold on
media1 = mean(Mat_boxplot1);    % Taxa de acerto média
max1 = max(Mat_boxplot4);       % Taxa máxima de acerto
plot(media1,'*k')
hold off

% BOXPLOT 2
figure; boxplot(Mat_boxplot2, 'label', labels);
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
ylabel('Accuracy')              % label eixo y
xlabel('Classifiers')           % label eixo x
title('Classifiers Comparison') % Titulo
axis ([0 n_labels+1 0 1.05])	% Eixos

hold on
media2 = mean(Mat_boxplot2);    % Taxa de acerto média
max2 = max(Mat_boxplot4);       % Taxa máxima de acerto
plot(media2,'*k')
hold off

% BOXPLOT 3
figure; boxplot(Mat_boxplot3, 'label', labels);
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
ylabel('Accuracy')              % label eixo y
xlabel('Classifiers')           % label eixo x
title('Classifiers Comparison') % Titulo
axis ([0 n_labels+1 0 1.05])	% Eixos

hold on
media3 = mean(Mat_boxplot3);    % Taxa de acerto média
max3 = max(Mat_boxplot4);       % Taxa máxima de acerto
plot(media3,'*k')
hold off

% BOXPLOT 4
figure; boxplot(Mat_boxplot4, 'label', labels);
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
ylabel('Accuracy')              % label eixo y
xlabel('Classifiers')           % label eixo x
title('Classifiers Comparison') % Titulo
axis ([0 n_labels+1 0 1.05])	% Eixos

hold on
media4 = mean(Mat_boxplot4);    % Taxa de acerto média
max4 = max(Mat_boxplot4);       % Taxa máxima de acerto
plot(media4,'*k')
hold off

%% Best and Worst Mconf

results_to_test = som2_out_ts;

x1 = accuracy_bin(results_to_test);

[~,max_mconf] = max(x1);
[~,min_mconf] = min(x1);

Mconf_max = results_to_test{max_mconf,1}.Mconf;
Mconf_min = results_to_test{min_mconf,1}.Mconf;

%% SAVE DATA

save(OPT.file);

%% END