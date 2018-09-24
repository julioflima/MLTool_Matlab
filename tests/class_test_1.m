%% Machine Learning ToolBox

% Classification Algorithms - General Tests
% Author: David Nascimento Coelho
% Last Update: 2018/01/16

close;      % Close all windows
clear;      % Clear all variables
clc;        % Clear command window

format long e;  % Output data style (float)

%% GENERAL DEFINITIONS

% General options' structure

OPT.prob = 6;               % Which problem will be solved / used
OPT.prob2 = 1;              % More details about a specific data set
OPT.norm = 3;               % Normalization definition
OPT.lbl = 1;                % Labeling definition
OPT.Nr = 02;                % Number of repetitions of each algorithm
OPT.hold = 2;               % Hold out method
OPT.ptrn = 0.8;             % Percentage of samples for training
OPT.file = 'fileX.mat';     % file where all the variables will be saved

% Cross Validation hiperparameters

CVp.on = 0;                 % If 1, includes cross validation
CVp.fold = 5;               % Number of folds for cross validation

% Reject Option hiperparameters

REJp.on = 0;                % If 1, Includes reject option
REJp.band = 0.3;            % Range of rejected values [-band +band]
REJp.w = 0.25;              % Rejection cost

%% DATA LOADING AND PRE-PROCESSING

DATA = data_class_loading(OPT);     % Load Data Set

Nc = length(unique(DATA.output));   % get number of classes

[p,N] = size(DATA.input);           % get number of attributes and samples

DATA = normalize(DATA,OPT);         % normalize the attributes' matrix

DATA = label_adjust(DATA,OPT);      % adjust labels for the problem

%% HIPERPARAMETERS - DEFAULT

% If an specific parameter is not set, the algorithm
% uses a default value.

OLSp_acc = cell(OPT.Nr,1);  % Init of Acc Hyperparameters of OLS
OLSp.on = 1;                % Run the classifier
OLSp.aprox = 1;             % Type of aproximation

% BAYp_acc = cell(OPT.Nr,1);  % Init of Acc Hyperparameters of Bayes
% BAYp.on = 1;                % Run the classifier
% BAYp.type = 2;              % Type of classificer
% 
% PSp_acc = cell(OPT.Nr,1);   % Init of Acc Hyperparameters of PS
% PSp.on = 1;                 % Run the classifier
% PSp.Ne = 200;               % maximum number of training epochs
% PSp.eta = 0.05;             % Learning step
% 
% MLPp_acc = cell(OPT.Nr,1);  % Init of Acc Hyperparameters of MLP
% MLPp.on = 1;                % Run the classifier
% MLPp.Nh = 10;               % Number of hidden neurons
% MLPp.Ne = 200;              % Maximum training epochs
% MLPp.eta = 0.05;            % Learning Step
% MLPp.mom = 0.75;            % Moment Factor
% MLPp.Nlin = 2;              % Non-linearity
% 
% ELMp_acc = cell(OPT.Nr,1);  % Acc Hyperparameters of ELM
% ELMp.on = 1;                % Run the classifier
% ELMp.Nh = 25;               % Number of hidden neurons
% ELMp.Nlin = 2;              % Non-linearity
% 
% SVMp_acc = cell(OPT.Nr,1);  % Init of Acc Hyperparameters of SVM
% SVMp.on = 1;                % Run the classifier
% SVMp.C = 5;                 % Regularization Constant
% SVMp.Ktype = 1;             % Kernel Type (gaussian = 1)
% SVMp.sig2 = 0.01;           % Variance (gaussian kernel)
% 
% LSSVMp_acc = cell(OPT.Nr,1);% Init of Acc Hyperparameters of LSSVM
% LSSVMp.on = 1;              % Run the classifier
% LSSVMp.C = 0.5;             % Regularization Constant
% LSSVMp.Ktype = 1;           % Kernel Type (gaussian = 1)
% LSSVMp.sig2 = 128;          % Variance (gaussian kernel)
% 
% MLMp_acc = cell(OPT.Nr,1);  % Init of Acc Hyperparameters of MLM
% MLMp.on = 1;                % Run the classifier
% MLMp.K = 09;                % Number of reference points

%% HIPERPARAMETERS - GRID FOR CROSS VALIDATION

% Get Default Hyperparameters

OLScv = OLSp;

% BAYcv = BAYp;
% 
% PScv = PSp;
% 
% MLPcv = MLPp;
% 
% ELMcv = ELMp;
% 
% SVMcv = SVMp;
% 
% LSSVMcv = LSSVMp;
% 
% MLMcv = MLMp;

% Set Variable HyperParameters

if CVp.on == 1,

OLScv;

BAYcv;

PScv;

MLPcv.Nh = 2:20;
    
ELMcv.Nh = 10:30;
    
SVMcv.C = [0.5 5 10 15 25 50 100 250 500 1000];
SVMcv.sig2 = [0.01 0.05 0.1 0.5 1 5 10 50 100 500];
    
LSSVMcv.C = [2^-5 2^-4 2^-3 2^-2 2^-1 2^0 2^1 2^2 2^3 2^4 2^5 2^6 2^7 2^8 2^9 2^10 2^11 2^12 2^13 2^14 2^15 2^16 2^17 2^18 2^19 2^20];
LSSVMcv.sig2 = [2^-10 2^-9 2^-8 2^-7 2^-6 2^-5 2^-4 2^-3 2^-2 2^-1 2^0 2^1 2^2 2^3 2^4 2^5 2^6 2^7 2^8 2^9 2^10];
    
MLMcv.K = 2:15;
   
end

%% CLASSIFIERS' RESULTS INIT

hold_acc = cell(OPT.Nr,1);      % Acc of labels and data division

ols_out_tr = cell(OPT.Nr,1);	% Acc of training data output
ols_out_ts = cell(OPT.Nr,1);	% Acc of test data output
ols_out_rj = cell(OPT.Nr,1);	% Acc of reject option output
ols_Mconf_sum = zeros(Nc,Nc);   % Aux var for mean confusion matrix calc
% 
% bay_out_tr = cell(OPT.Nr,1);	% Acc of training data output
% bay_out_ts = cell(OPT.Nr,1);	% Acc of test data output
% bay_out_rj = cell(OPT.Nr,1);	% Acc of reject option output
% bay_Mconf_sum = zeros(Nc,Nc);   % Aux var for mean confusion matrix calc
% 
% ps_out_tr = cell(OPT.Nr,1);     % Acc of training data output
% ps_out_ts = cell(OPT.Nr,1);     % Acc of test data output
% ps_out_rj = cell(OPT.Nr,1);     % Acc of reject option output
% ps_Mconf_sum = zeros(Nc,Nc);    % Aux var for mean confusion matrix calc
% 
% mlp_out_tr = cell(OPT.Nr,1);	% Acc of training data output
% mlp_out_ts = cell(OPT.Nr,1);	% Acc of test data output
% mlp_out_rj = cell(OPT.Nr,1);	% Acc of reject option output
% mlp_Mconf_sum = zeros(Nc,Nc);   % Aux var for mean confusion matrix calc
% 
% elm_out_tr = cell(OPT.Nr,1);	% Acc of training data output
% elm_out_ts = cell(OPT.Nr,1);	% Acc of test data output
% elm_out_rj = cell(OPT.Nr,1);	% Acc of reject option output
% elm_Mconf_sum = zeros(Nc,Nc);   % Aux var for mean confusion matrix calc
% 
% svm_out_tr = cell(OPT.Nr,1);	% Acc of training data output
% svm_out_ts = cell(OPT.Nr,1);	% Acc of test data output
% svm_out_rj = cell(OPT.Nr,1);	% Acc of reject option output
% svm_Mconf_sum = zeros(Nc,Nc);   % Aux var for mean confusion matrix calc
% 
% lssvm_out_tr = cell(OPT.Nr,1);	% Acc of training data output
% lssvm_out_ts = cell(OPT.Nr,1);	% Acc of test data output
% lssvm_out_rj = cell(OPT.Nr,1);	% Acc of reject option output
% lssvm_Mconf_sum = zeros(Nc,Nc); % Aux var for mean confusion matrix calc
% 
% mlm_out_tr = cell(OPT.Nr,1);	% Acc of training data output
% mlm_out_ts = cell(OPT.Nr,1);	% Acc of test data output
% mlm_out_rj = cell(OPT.Nr,1);	% Acc of reject option output
% mlm_Mconf_sum = zeros(Nc,Nc);   % Aux var for mean confusion matrix calc

%% HOLD OUT / CROSS VALIDATION / TRAINING / TEST

for r = 1:OPT.Nr,

% %%%%%%%%% DISPLAY REPETITION AND DURATION %%%%%%%%%%%%%%

display(r);
display(datestr(now));

% %%%%%%%%%%%%%% SHUFFLE AND HOLD OUT %%%%%%%%%%%%%%%%%%%%

% Shuffle data

I = randperm(N);
DATA.input = DATA.input(:,I); 
DATA.output = DATA.output(:,I);
DATA.lbl = DATA.lbl(:,I);

% Divide data between training and test

[DATAho] = hold_out(DATA,OPT);

hold_acc{r} = DATAho;
DATAtr = DATAho.DATAtr;
DATAts = DATAho.DATAts;

% %%%%%%%%%%%%%%%% CROSS VALIDATION %%%%%%%%%%%%%%%%%%%%%

% Cross validation with grid search method

% [OLSp] = cross_valid_gs(DATAtr,CVp,OLScv,@ols_train,@ols_classify);
% 
% [BAYp] = cross_valid_gs(DATAtr,CVp,BAYcv,@gauss_train,@gauss_classify);
% 
% [PSp] = cross_valid_gs(DATAtr,CVp,PScv,@ps_train,@ps_classify);
% 
% [MLPp] = cross_valid_gs(DATAtr,CVp,MLPcv,@mlp_train,@mlp_classify);
% 
% [ELMp] = cross_valid_gs(DATAtr,CVp,ELMcv,@elm_train,@elm_classify);
% 
% [SVMp] = cross_valid_gs(DATAtr,CVp,SVMcv,@svm_train,@svm_classify);
% 
% [LSSVMp] = cross_valid_gs(DATAtr,CVp,LSSVMcv,@lssvm_train,@lssvm_classify);
% 
% [MLMp] = cross_valid_gs(DATAtr,CVp,MLMcv,@mlm_train,@mlm_classify);
% 

% %%%%%%%%%%%%%% CLASSIFIERS' TRAINING %%%%%%%%%%%%%%%%%%%

[OLSp] = ols_train(DATAtr,OLSp);

% [BAYp] = gauss_train(DATAtr,BAYp);
% 
% [PSp] = ps_train(DATAtr,PSp);
% 
% [MLPp] = mlp_train(DATAtr,MLPp);
%  
% [ELMp] = elm_train(DATAtr,ELMp);
% 
% [SVMp] = svm_train(DATAtr,SVMp);
%  
% [LSSVMp] = lssvm_train(DATAtr,LSSVMp);
%  
% [MLMp] = mlm_train(DATAtr,MLMp);
%  

% %%%%%%%%%%%%%%%%% CLASSIFIERS' TEST %%%%%%%%%%%%%%%%%%%%

% OLS

[OUTtr] = ols_classify(DATAtr,OLSp);
OUTtr.nf = normal_or_fail(OUTtr.Mconf);
OUTtr.mcc = Mcc(OUTtr.Mconf);
ols_out_tr{r,1} = OUTtr;

[OUTts] = ols_classify(DATAts,OLSp);
OUTts.nf = normal_or_fail(OUTts.Mconf);
OUTts.mcc = Mcc(OUTts.Mconf);
ols_out_ts{r,1} = OUTts;

[OUTrj] = reject_opt2(DATAts,OUTts,REJp);
OUTrj.nf = normal_or_fail(OUTrj.Mconf);
OUTrj.mcc = Mcc(OUTrj.Mconf);
ols_out_rj{r,1} = OUTrj;

OLSp_acc{r} = OLSp;
ols_Mconf_sum = ols_Mconf_sum + OUTts.Mconf;
% 
% % BAYES
% 
% [OUTtr] = gauss_classify(DATAtr,BAYp);
% OUTtr.nf = normal_or_fail(OUTtr.Mconf);
% OUTtr.mcc = Mcc(OUTtr.Mconf);
% bay_out_tr{r,1} = OUTtr;
% 
% [OUTts] = gauss_classify(DATAts,BAYp);
% OUTts.nf = normal_or_fail(OUTts.Mconf);
% OUTts.mcc = Mcc(OUTts.Mconf);
% bay_out_ts{r,1} = OUTts;
% 
% [OUTrj] = reject_opt2(DATAts,OUTts,REJp);
% OUTrj.nf = normal_or_fail(OUTrj.Mconf);
% OUTrj.mcc = Mcc(OUTrj.Mconf);
% bay_out_rj{r,1} = OUTrj;
% 
% BAYp_acc{r} = BAYp;
% bay_Mconf_sum = bay_Mconf_sum + OUTts.Mconf;
% 
% % PS
% 
% [OUTtr] = ps_classify(DATAtr,PSp);
% OUTtr.nf = normal_or_fail(OUTtr.Mconf);
% OUTtr.mcc = Mcc(OUTtr.Mconf);
% ps_out_tr{r,1} = OUTtr;
% 
% [OUTts] = ps_classify(DATAts,PSp);
% OUTts.nf = normal_or_fail(OUTts.Mconf);
% OUTts.mcc = Mcc(OUTts.Mconf);
% ps_out_ts{r,1} = OUTts;
% 
% [OUTrj] = reject_opt2(DATAts,OUTts,REJp);
% OUTrj.nf = normal_or_fail(OUTrj.Mconf);
% OUTrj.mcc = Mcc(OUTrj.Mconf);
% ps_out_rj{r,1} = OUTrj;
% 
% PSp_acc{r} = PSp;
% ps_Mconf_sum = ps_Mconf_sum + OUTts.Mconf;
% 
% % MLP
% 
% [OUTtr] = mlp_classify(DATAtr,MLPp);
% OUTtr.nf = normal_or_fail(OUTtr.Mconf);
% OUTtr.mcc = Mcc(OUTtr.Mconf);
% mlp_out_tr{r,1} = OUTtr;
% 
% [OUTts] = mlp_classify(DATAts,MLPp);
% OUTts.nf = normal_or_fail(OUTts.Mconf);
% OUTts.mcc = Mcc(OUTts.Mconf);
% mlp_out_ts{r,1} = OUTts;
% 
% [OUTrj] = reject_opt2(DATAts,OUTts,REJp);
% OUTrj.nf = normal_or_fail(OUTrj.Mconf);
% OUTrj.mcc = Mcc(OUTrj.Mconf);
% mlp_out_rj{r,1} = OUTrj;
% 
% MLPp_acc{r} = MLPp;
% mlp_Mconf_sum = mlp_Mconf_sum + OUTts.Mconf;
% 
% % ELM
% 
% [OUTtr] = elm_classify(DATAtr,ELMp);
% OUTtr.nf = normal_or_fail(OUTtr.Mconf);
% OUTtr.mcc = Mcc(OUTtr.Mconf);
% elm_out_tr{r,1} = OUTtr;
% 
% [OUTts] = elm_classify(DATAts,ELMp);
% OUTts.nf = normal_or_fail(OUTts.Mconf);
% OUTts.mcc = Mcc(OUTts.Mconf);
% elm_out_ts{r,1} = OUTts;
% 
% [OUTrj] = reject_opt2(DATAts,OUTts,REJp);
% OUTrj.nf = normal_or_fail(OUTrj.Mconf);
% OUTrj.mcc = Mcc(OUTrj.Mconf);
% elm_out_rj{r,1} = OUTrj;
% 
% ELMp_acc{r} = ELMp;
% elm_Mconf_sum = elm_Mconf_sum + OUTts.Mconf;
% 
% % SVM
% 
% [OUTtr] = svm_classify(DATAtr,SVMp);
% OUTtr.nf = normal_or_fail(OUTtr.Mconf);
% OUTtr.mcc = Mcc(OUTtr.Mconf);
% svm_out_tr{r,1} = OUTtr;
% 
% [OUTts] = svm_classify(DATAts,SVMp);
% OUTts.nf = normal_or_fail(OUTts.Mconf);
% OUTts.mcc = Mcc(OUTts.Mconf);
% svm_out_ts{r,1} = OUTts;
% 
% [OUTrj] = reject_opt2(DATAts,OUTts,REJp);
% OUTrj.nf = normal_or_fail(OUTrj.Mconf);
% OUTrj.mcc = Mcc(OUTrj.Mconf);
% svm_out_rj{r,1} = OUTrj;
% 
% SVMp_acc{r} = SVMp;
% svm_Mconf_sum = svm_Mconf_sum + OUTts.Mconf;
% 
% % LSSVM
% 
% [OUTtr] = lssvm_classify(DATAtr,LSSVMp);
% OUTtr.nf = normal_or_fail(OUTtr.Mconf);
% OUTtr.mcc = Mcc(OUTtr.Mconf);
% lssvm_out_tr{r,1} = OUTtr;
% 
% [OUTts] = lssvm_classify(DATAts,LSSVMp);
% OUTts.nf = normal_or_fail(OUTts.Mconf);
% OUTts.mcc = Mcc(OUTts.Mconf);
% lssvm_out_ts{r,1} = OUTts;
% 
% [OUTrj] = reject_opt2(DATAts,OUTts,REJp);
% OUTrj.nf = normal_or_fail(OUTrj.Mconf);
% OUTrj.mcc = Mcc(OUTrj.Mconf);
% lssvm_out_rj{r,1} = OUTrj;
% 
% LSSVMp_acc{r} = LSSVMp;
% lssvm_Mconf_sum = lssvm_Mconf_sum + OUTts.Mconf;
% 
% % MLM
% 
% [OUTtr] = mlm_classify(DATAtr,MLMp);
% OUTtr.nf = normal_or_fail(OUTtr.Mconf);
% OUTtr.mcc = Mcc(OUTtr.Mconf);
% mlm_out_tr{r,1} = OUTtr;
% 
% [OUTts] = mlm_classify(DATAts,MLMp);
% OUTts.nf = normal_or_fail(OUTts.Mconf);
% OUTts.mcc = Mcc(OUTts.Mconf);
% mlm_out_ts{r,1} = OUTts;
% 
% [OUTrj] = reject_opt2(DATAts,OUTts,REJp);
% OUTrj.nf = normal_or_fail(OUTrj.Mconf);
% OUTrj.mcc = Mcc(OUTrj.Mconf);
% mlm_out_rj{r,1} = OUTrj;
% 
% MLMp_acc{r} = MLMp;
% mlm_Mconf_sum = mlm_Mconf_sum + OUTts.Mconf;

end

%% STATISTICS

% Mean Confusion Matrix

ols_Mconf_mean = ols_Mconf_sum / OPT.Nr;
% bay_Mconf_mean = bay_Mconf_sum / OPT.Nr;
% ps_Mconf_mean = ps_Mconf_sum / OPT.Nr;
% mlp_Mconf_mean = mlp_Mconf_sum / OPT.Nr;
% elm_Mconf_mean = elm_Mconf_sum / OPT.Nr;
% svm_Mconf_mean = svm_Mconf_sum / OPT.Nr;
% lssvm_Mconf_mean = lssvm_Mconf_sum / OPT.Nr;
% mlm_Mconf_mean = mlm_Mconf_sum / OPT.Nr;

%% GRAPHICS

% Init labels' cell and Init boxplot matrix

labels = {};

Mat_boxplot1 = []; % Train Multiclass
Mat_boxplot2 = []; % Train Binary
Mat_boxplot3 = []; % Test Multiclass
Mat_boxplot4 = []; % Test Binary
Mat_boxplot5 = []; % Reject Option Multiclass
Mat_boxplot6 = []; % Reject Option Binary
Mat_boxplot7 = []; % Matthews Correlation from training
Mat_boxplot8 = []; % Matthews Correlation from test
Mat_boxplot9 = []; % Matthews Correlation from reject option

if OLSp.on == 1,
[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'OLS'};
Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(ols_out_tr)];
Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(ols_out_tr)];
Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(ols_out_ts)];
Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(ols_out_ts)];
Mat_boxplot5 = [Mat_boxplot5 accuracy_mult(ols_out_rj)];
Mat_boxplot6 = [Mat_boxplot6 accuracy_bin(ols_out_rj)];
Mat_boxplot7 = [Mat_boxplot7 mcc_vet(ols_out_tr)];
Mat_boxplot8 = [Mat_boxplot8 mcc_vet(ols_out_ts)];
Mat_boxplot9 = [Mat_boxplot9 mcc_vet(ols_out_rj)];

end

% if BAYp.on == 1,
% [~,n_labels] = size(labels);
% n_labels = n_labels+1;
% labels(1,n_labels) = {'BAY'};
% Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(bay_out_tr)];
% Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(bay_out_tr)];
% Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(bay_out_ts)];
% Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(bay_out_ts)];
% Mat_boxplot5 = [Mat_boxplot5 accuracy_mult(bay_out_rj)];
% Mat_boxplot6 = [Mat_boxplot6 accuracy_bin(bay_out_rj)];
% Mat_boxplot7 = [Mat_boxplot7 mcc_vet(bay_out_tr)];
% Mat_boxplot8 = [Mat_boxplot8 mcc_vet(bay_out_ts)];
% Mat_boxplot9 = [Mat_boxplot9 mcc_vet(bay_out_rj)];
% end
% 
% if PSp.on == 1,
% [~,n_labels] = size(labels);
% n_labels = n_labels+1;
% labels(1,n_labels) = {'PS'};
% Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(ps_out_tr)];
% Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(ps_out_tr)];
% Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(ps_out_ts)];
% Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(ps_out_ts)];
% Mat_boxplot5 = [Mat_boxplot5 accuracy_mult(ps_out_rj)];
% Mat_boxplot6 = [Mat_boxplot6 accuracy_bin(ps_out_rj)];
% Mat_boxplot7 = [Mat_boxplot7 mcc_vet(ps_out_tr)];
% Mat_boxplot8 = [Mat_boxplot8 mcc_vet(ps_out_ts)];
% Mat_boxplot9 = [Mat_boxplot9 mcc_vet(ps_out_rj)];
% 
% end
% 
% if MLPp.on == 1,
% [~,n_labels] = size(labels);
% n_labels = n_labels+1;
% labels(1,n_labels) = {'MLP'};
% Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(mlp_out_tr)];
% Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(mlp_out_tr)];
% Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(mlp_out_ts)];
% Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(mlp_out_ts)];
% Mat_boxplot5 = [Mat_boxplot5 accuracy_mult(mlp_out_rj)];
% Mat_boxplot6 = [Mat_boxplot6 accuracy_bin(mlp_out_rj)];
% Mat_boxplot7 = [Mat_boxplot7 mcc_vet(mlp_out_tr)];
% Mat_boxplot8 = [Mat_boxplot8 mcc_vet(mlp_out_ts)];
% Mat_boxplot9 = [Mat_boxplot9 mcc_vet(mlp_out_rj)];
% end
% 
% if ELMp.on == 1,
% [~,n_labels] = size(labels);
% n_labels = n_labels+1;
% labels(1,n_labels) = {'ELM'};
% Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(elm_out_tr)];
% Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(elm_out_tr)];
% Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(elm_out_ts)];
% Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(elm_out_ts)];
% Mat_boxplot5 = [Mat_boxplot5 accuracy_mult(elm_out_rj)];
% Mat_boxplot6 = [Mat_boxplot6 accuracy_bin(elm_out_rj)];
% Mat_boxplot7 = [Mat_boxplot7 mcc_vet(elm_out_tr)];
% Mat_boxplot8 = [Mat_boxplot8 mcc_vet(elm_out_ts)];
% Mat_boxplot9 = [Mat_boxplot9 mcc_vet(elm_out_rj)];
% end
% 
% if SVMp.on == 1,
% [~,n_labels] = size(labels);
% n_labels = n_labels+1;
% labels(1,n_labels) = {'SVM'};
% Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(svm_out_tr)];
% Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(svm_out_tr)];
% Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(svm_out_ts)];
% Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(svm_out_ts)];
% Mat_boxplot5 = [Mat_boxplot5 accuracy_mult(svm_out_rj)];
% Mat_boxplot6 = [Mat_boxplot6 accuracy_bin(svm_out_rj)];
% Mat_boxplot7 = [Mat_boxplot7 mcc_vet(svm_out_tr)];
% Mat_boxplot8 = [Mat_boxplot8 mcc_vet(svm_out_ts)];
% Mat_boxplot9 = [Mat_boxplot9 mcc_vet(svm_out_rj)];
% end
% 
% if LSSVMp.on == 1,
% [~,n_labels] = size(labels);
% n_labels = n_labels+1;
% labels(1,n_labels) = {'LSSVM'};
% Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(lssvm_out_tr)];
% Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(lssvm_out_tr)];
% Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(lssvm_out_ts)];
% Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(lssvm_out_ts)];
% Mat_boxplot5 = [Mat_boxplot5 accuracy_mult(lssvm_out_rj)];
% Mat_boxplot6 = [Mat_boxplot6 accuracy_bin(lssvm_out_rj)];
% Mat_boxplot7 = [Mat_boxplot7 mcc_vet(lssvm_out_tr)];
% Mat_boxplot8 = [Mat_boxplot8 mcc_vet(lssvm_out_ts)];
% Mat_boxplot9 = [Mat_boxplot9 mcc_vet(lssvm_out_rj)];
% 
% end
% 
% if MLMp.on == 1,
% [~,n_labels] = size(labels);
% n_labels = n_labels+1;
% labels(1,n_labels) = {'MLM'};
% Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(mlm_out_tr)];
% Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(mlm_out_tr)];
% Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(mlm_out_ts)];
% Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(mlm_out_ts)];
% Mat_boxplot5 = [Mat_boxplot5 accuracy_mult(mlm_out_rj)];
% Mat_boxplot6 = [Mat_boxplot6 accuracy_bin(mlm_out_rj)];
% Mat_boxplot7 = [Mat_boxplot7 mcc_vet(mlm_out_tr)];
% Mat_boxplot8 = [Mat_boxplot8 mcc_vet(mlm_out_ts)];
% Mat_boxplot9 = [Mat_boxplot9 mcc_vet(mlm_out_rj)];
% end

% BOXPLOT 1
figure; boxplot(Mat_boxplot1,'label', labels);
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
ylabel('Accuracy')              % label eixo y
xlabel('Classificadores')       % label eixo x
title('Taxa de Classificação')  % Titulo
axis ([0 n_labels+1 0 1.05])	% Eixos

hold on
media1 = mean(Mat_boxplot1);    % Taxa de acerto média
plot(media1,'*k')
hold off

% BOXPLOT 2
figure; boxplot(Mat_boxplot2, 'label', labels);
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
ylabel('Accuracy')              % label eixo y
xlabel('Classificadores')       % label eixo x
title('Taxa de Classificação')  % Titulo
axis ([0 n_labels+1 0 1.05])	% Eixos

hold on
media2 = mean(Mat_boxplot2);    % Taxa de acerto média
plot(media2,'*k')
hold off

% BOXPLOT 3
figure; boxplot(Mat_boxplot3, 'label', labels);
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
ylabel('Accuracy')              % label eixo y
xlabel('Classificadores')       % label eixo x
title('Taxa de Classificação')  % Titulo
axis ([0 n_labels+1 0 1.05])	% Eixos

hold on
media3 = mean(Mat_boxplot3);    % Taxa de acerto média
plot(media3,'*k')
hold off

% BOXPLOT 4
figure; boxplot(Mat_boxplot4, 'label', labels);
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
ylabel('Accuracy')              % label eixo y
xlabel('Classificadores')       % label eixo x
title('Taxa de Classificação')  % Titulo
axis ([0 n_labels+1 0 1.05])	% Eixos

hold on
media4 = mean(Mat_boxplot4);    % Taxa de acerto média
plot(media4,'*k')
hold off

% BOXPLOT 5
figure; boxplot(Mat_boxplot5, 'label', labels);
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
ylabel('Accuracy')              % label eixo y
xlabel('Classificadores')       % label eixo x
title('Taxa de Classificação')  % Titulo
axis ([0 n_labels+1 0 1.05])	% Eixos

hold on
media5 = mean(Mat_boxplot5);    % Taxa de acerto média
plot(media5,'*k')
hold off

% BOXPLOT 6
figure; boxplot(Mat_boxplot6, 'label', labels);
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
ylabel('Accuracy')              % label eixo y
xlabel('Classificadores')       % label eixo x
title('Taxa de Classificação')  % Titulo
axis ([0 n_labels+1 0 1.05])	% Eixos

hold on
media6 = mean(Mat_boxplot6);    % Taxa de acerto média
plot(media6,'*k')
hold off

% BOXPLOT 7
figure; boxplot(Mat_boxplot7, 'label', labels);
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
ylabel('MCC')                   % label eixo y
xlabel('Classificadores')       % label eixo x
title('Train MCC')              % Titulo
axis ([0 n_labels+1 -1.05 1.05])% Eixos

hold on
media7 = mean(Mat_boxplot7);    % Taxa de acerto média
plot(media7,'*k')
hold off

% BOXPLOT 8
figure; boxplot(Mat_boxplot8, 'label', labels);
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
ylabel('MCC')                   % label eixo y
xlabel('Classificadores')       % label eixo x
title('Test MCC')               % Titulo
axis ([0 n_labels+1 -1.05 1.05])% Eixos

hold on
media8 = mean(Mat_boxplot8);    % Taxa de acerto média
plot(media8,'*k')
hold off

% BOXPLOT 9
figure; boxplot(Mat_boxplot9, 'label', labels);
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
ylabel('MCC')                   % label eixo y
xlabel('Classificadores')       % label eixo x
title('Rejection Mcc')          % Titulo
axis ([0 n_labels+1 -1.05 1.05])% Eixos

hold on
media9 = mean(Mat_boxplot9);    % Taxa de acerto média
plot(media9,'*k')
hold off

%% SAVE DATA

save(OPT.file);

%% END