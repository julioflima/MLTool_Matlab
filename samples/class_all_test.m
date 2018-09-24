%% Machine Learning ToolBox

% Classification Algorithms - General Tests
% Author: David Nascimento Coelho
% Last Update: 2018/02/04

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% GENERAL DEFINITIONS

% General options' structure

OPT.prob = 11;              % Which problem will be solved / used
OPT.prob2 = 1;              % More details about a specific data set
OPT.norm = 3;               % Normalization definition
OPT.Nr = 02;                % Number of repetitions of each algorithm
OPT.hold = 2;               % Hold out method
OPT.ptrn = 0.8;             % Percentage of samples for training
OPT.file = 'fileX.mat';     % file where all the variables will be saved

%% DATA LOADING AND PRE-PROCESSING

DATA = data_class_loading(OPT);     % Load Data Set

Nc = length(unique(DATA.output));   % get number of classes

[p,N] = size(DATA.input);           % get number of attributes and samples

DATA = normalize(DATA,OPT);         % normalize the attributes' matrix

%% HIPERPARAMETERS - DEFAULT

BAYp.type = 2;          % Type of classificer

OLSp.aprox = 1;       	% Type of aproximation

PSp.Ne = 200;          	% maximum number of training epochs
PSp.eta = 0.05;       	% Learning step

ELMp.Nh = 25;       	% Number of hidden neurons
ELMp.Nlin = 2;         	% Non-linearity

MLPp.Nh = 10;       	% Number of hidden neurons
MLPp.Ne = 200;        	% Maximum training epochs
MLPp.eta = 0.05;     	% Learning Step
MLPp.mom = 0.75;      	% Moment Factor
MLPp.Nlin = 2;       	% Non-linearity

RBFp.Nh = 10;           % number of hidden neurons
RBFp.init = 2;        	% centroids initialization type
RBFp.rad = 2;          	% radius / spread type
RBFp.out = 1;         	% how to calculate output weights
RBFp.ativ = 1;        	% activation function type

SVMp.C = 5;           	% Regularization Constant
SVMp.Ktype = 1;        	% Kernel Type (gaussian = 1)
SVMp.sig2 = 0.01;     	% Variance (gaussian kernel)

LSSVMp.C = 0.5;       	% Regularization Constant
LSSVMp.Ktype = 1;      	% Kernel Type (gaussian = 1)
LSSVMp.sig2 = 128;     	% Variance (gaussian kernel)

% GPp.l2 = 2;          	% GP constant 1
% GPp.K = 1;          	% Kernel Type (gaussian = 1)
% GPp.sig2 = 2;        	% GP constant 2

MLMp.K = 09;            % Number of reference points

KNNp.k = 3;           	% Number of nearest neighbors

KMp.ep = 200;        	% max number of epochs
KMp.k = 20;           	% number of clusters (prototypes)
KMp.init = 02;       	% type of initialization
KMp.dist = 02;        	% type of distance
KMp.lbl = 1;           	% Neurons' labeling function
KMp.Von = 0;          	% disable video

WTAp.ep = 200;       	% max number of epochs
WTAp.k = 20;          	% number of neurons (prototypes)
WTAp.init = 02;     	% neurons' initialization
WTAp.dist = 02;        	% type of distance
WTAp.learn = 02;      	% type of learning step
WTAp.No = 0.7;        	% initial learning step
WTAp.Nt = 0.01;      	% final   learning step
WTAp.lbl = 1;        	% Neurons' labeling function
WTAp.Von = 0;          	% disable video

LVQp.ep = 200;       	% max number of epochs
LVQp.k = 20;           	% number of neurons (prototypes)
LVQp.init = 02;       	% neurons' initialization
LVQp.dist = 02;        	% type of distance
LVQp.learn = 02;       	% type of learning step
LVQp.No = 0.7;         	% initial learning step
LVQp.Nt = 0.01;      	% final   learning step
LVQp.Von = 0;         	% disable video

SOM1Dp.ep = 200;     	% max number of epochs
SOM1Dp.k = 20;        	% number of neurons (prototypes)
SOM1Dp.init = 02;     	% neurons' initialization
SOM1Dp.dist = 02;     	% type of distance
SOM1Dp.learn = 02;    	% type of learning step
SOM1Dp.No = 0.7;      	% initial learning step
SOM1Dp.Nt = 0.01;      	% final learning step
SOM1Dp.Nn = 01;       	% number of neighbors
SOM1Dp.neig = 02;     	% type of neighbor function
SOM1Dp.Vo = 0.8;       	% initial neighbor constant
SOM1Dp.Vt = 0.3;      	% final neighbor constant
SOM1Dp.lbl = 1;        	% Neurons' labeling function
SOM1Dp.Von = 0;      	% disable video

SOM2Dp.ep = 200;       	% max number of epochs
SOM2Dp.k = [5 4];      	% number of neurons (prototypes)
SOM2Dp.init = 02;     	% neurons' initialization
SOM2Dp.dist = 02;      	% type of distance
SOM2Dp.learn = 02;     	% type of learning step
SOM2Dp.No = 0.7;       	% initial learning step
SOM2Dp.Nt = 0.01;      	% final learnin step
SOM2Dp.Nn = 01;      	% number of neighbors
SOM2Dp.neig = 03;      	% type of neighborhood function
SOM2Dp.Vo = 0.8;      	% initial neighborhood constant
SOM2Dp.Vt = 0.3;      	% final neighborhood constant
SOM2Dp.lbl = 1;         % Neurons' labeling function
SOM2Dp.Von = 0;         % disable video

KSOMGDp.ep = 200;    	% max number of epochs
KSOMGDp.k = [5 4];   	% number of neurons (prototypes)
KSOMGDp.init = 02;   	% neurons' initialization
KSOMGDp.dist = 02;    	% type of distance
KSOMGDp.learn = 02;   	% type of learning step
KSOMGDp.No = 0.7;     	% initial learning step
KSOMGDp.Nt = 0.01;   	% final learning step
KSOMGDp.Nn = 01;     	% number of neighbors
KSOMGDp.neig = 03;   	% type of neighbor function
KSOMGDp.Vo = 0.8;    	% initial neighbor constant
KSOMGDp.Vt = 0.3;     	% final neighbor constant
KSOMGDp.Kt = 1;       	% Type of Kernel
KSOMGDp.sig2 = 0.5;   	% Variance (gaussian, log, cauchy kernel)
KSOMGDp.lbl = 1;        % Neurons' labeling function
KSOMGDp.Von = 0;        % disable video

KSOMEFp.ep = 200;      	% max number of epochs
KSOMEFp.k = [5 4];    	% number of neurons (prototypes)
KSOMEFp.init = 02;     	% neurons' initialization
KSOMEFp.dist = 02;    	% type of distance
KSOMEFp.learn = 02;  	% type of learning step
KSOMEFp.No = 0.7;     	% initial learning step
KSOMEFp.Nt = 0.01;    	% final learning step
KSOMEFp.Nn = 01;      	% number of neighbors
KSOMEFp.neig = 03;    	% type of neighbor function
KSOMEFp.Vo = 0.8;     	% initial neighbor constant
KSOMEFp.Vt = 0.3;      	% final neighbor constant
KSOMEFp.Kt = 1;        	% Type of Kernel
KSOMEFp.sig2 = 0.5;   	% Variance (gaussian, log, cauchy kernel)
KSOMEFp.lbl = 01;     	% Neurons' labeling function
KSOMEFp.Von = 0;      	% disable video

KSOMPSp.ep = 200;    	% max number of epochs
KSOMPSp.k = [5 4];    	% number of neurons (prototypes)
KSOMPSp.init = 02;     	% neurons' initialization
KSOMPSp.dist = 02;    	% type of distance
KSOMPSp.learn = 02;    	% type of learning step
KSOMPSp.No = 0.7;     	% initial learning step
KSOMPSp.Nt = 0.01;    	% final learnin step
KSOMPSp.Nn = 01;      	% number of neighbors
KSOMPSp.neig = 03;    	% type of neighborhood function
KSOMPSp.Vo = 0.8;      	% initial neighborhood constant
KSOMPSp.Vt = 0.3;      	% final neighborhood constant
KSOMPSp.M = 50;      	% samples used to estimate kernel matrix
KSOMPSp.Kt = 1;       	% Type of Kernel (gaussian)
KSOMPSp.sig2 = 2;     	% Variance (gaussian, log, cauchy kernel)
KSOMPSp.lbl = 1;     	% Neurons' labeling function
KSOMPSp.Von = 0;      	% enable or disable video display
KSOMPSp.s = 1;      	% prototype selection type
    
%% HIPERPARAMETERS - GRID FOR CROSS VALIDATION

% Get Default Hyperparameters

BAYcv = BAYp;

OLScv = OLSp;

PScv = PSp;

ELMcv = ELMp;

MLPcv = MLPp;

RBFcv = RBFp;

SVMcv = SVMp;

LSSVMcv = LSSVMp;

% GPcv = GPp;

MLMcv = MLMp;

KNNcv = KNNp;

KMcv = KMp;

WTAcv = WTAp;

LVQcv = LVQp;

SOM1Dcv = SOM1Dp;

SOM2Dcv = SOM2Dp;

KSOMGDcv = KSOMGDp;

KSOMEFcv = KSOMEFp;

KSOMPScv = KSOMPSp;

% Set Variable Hyperparameters

BAYcv;

OLScv;

PScv;

ELMcv.Nh = 10:30;
    
MLPcv.Nh = 2:20;

RBFcv.Nh = 2:20;

SVMcv.C = [0.5 5 10 15 25 50 100 250 500 1000];
SVMcv.sig2 = [0.01 0.05 0.1 0.5 1 5 10 50 100 500];
    
LSSVMcv.C = [2^-5 2^-4 2^-3 2^-2 2^-1 2^0 2^1 2^2 2^3 2^4 2^5 2^6 2^7 2^8 2^9 2^10 2^11 2^12 2^13 2^14 2^15 2^16 2^17 2^18 2^19 2^20];
LSSVMcv.sig2 = [2^-10 2^-9 2^-8 2^-7 2^-6 2^-5 2^-4 2^-3 2^-2 2^-1 2^0 2^1 2^2 2^3 2^4 2^5 2^6 2^7 2^8 2^9 2^10];
    
% GPcv;

MLMcv.K = 2:15;
   
KNNcv.k = 1:10;

KMcv.k = 2:20;

WTAcv.k = 2:20;

LVQcv.k = 2:20;

SOM1Dcv.k = 2:20;

SOM2Dcv;

KSOMGDcv;

KSOMEFcv;

KSOMPScv;

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

% Hold out

[DATAho] = hold_out(DATA,OPT);

% Data for prototype based classifiers

DATAtr_sq = DATAho.DATAtr;          % sequential
DATAts_sq = DATAho.DATAts;          % sequential

% Data for decision boundaries classifiers

OPT.lbl = 1;                            % Labeling definition
DATAtr = label_adjust(DATAtr_sq,OPT);   % (-1 ; ... ; +1)
DATAts = label_adjust(DATAts_sq,OPT);   % (-1 ; ... ; +1)

% %%%%%%%%%%%%%%%% CROSS VALIDATION %%%%%%%%%%%%%%%%%%%%%

% with grid search method

% [BAYp] = cross_valid_gs(DATAtr,CVp,BAYcv,@gauss_train,@gauss_classify);
% 
% [OLSp] = cross_valid_gs(DATAtr,CVp,OLScv,@ols_train,@ols_classify);
% 
% [PSp] = cross_valid_gs(DATAtr,CVp,PScv,@ps_train,@ps_classify);
% 
% [ELMp] = cross_valid_gs(DATAtr,CVp,ELMcv,@elm_train,@elm_classify);
% 
% [MLPp] = cross_valid_gs(DATAtr,CVp,MLPcv,@mlp_train,@mlp_classify);
% 
% [RBFp] = cross_valid_gs(DATAtr,CVp,RBFcv,@rbf_train,@rbf_classify);
% 
% [SVMp] = cross_valid_gs(DATAtr,CVp,SVMcv,@svm_train,@svm_classify);
% 
% [LSSVMp] = cross_valid_gs(DATAtr,CVp,LSSVMcv,@lssvm_train,@lssvm_classify);
% 
% [GPp] = cross_valid_gs(DATAtr,CVp,GPcv,@gp_train,@gp_classify);
% 
% [MLMp] = cross_valid_gs(DATAtr,CVp,MLMcv,@mlm_train,@mlm_classify);
% 
% [KNNp] = cross_valid_gs(DATAtr_sq,CVp,KNNcv,@knn_train,@knn_classify);
% 
% [KMp] = cross_valid_gs(DATAtr_sq,CVp,KMcv,@kmeans_train,@kmeans_classify);
%
% [WTAp] = cross_valid_gs(DATAtr_sq,CVp,WTAcv,@wta_train,@wta_classify);
%
% [LVQp] = cross_valid_gs(DATAtr_sq,CVp,LVQcv,@lvq_train,@lvq_classify);
% 
% [SOM1Dp] = cross_valid_gs(DATAtr_sq,CVp,SOM1Dcv,@som1d_train,@som1d_classify);
% 
% [SOM2Dp] = cross_valid_gs(DATAtr_sq,CVp,SOM2Dcv,@som2d_train,@som2d_classify);
% 
% [KSOMGDp] = cross_valid_gs(DATAtr_sq,CVp,KSOMGDp,@ksom_gd_train,@ksom_gd_classify);
% 
% [KSOMEFp] = cross_valid_gs(DATAtr_sq,CVp,KSOMEFp,@ksom_ef_train,@ksom_ef_classify);
% 
% [KSOMPSp] = cross_valid_gs(DATAtr_sq,CVp,KSOMPSp,@ksom_ps_train,@ksom_ps_classify);

% %%%%%%%%%%%%%% CLASSIFIERS' TRAINING %%%%%%%%%%%%%%%%%%%

[BAYp] = gauss_train(DATAtr,BAYp);

[OLSp] = ols_train(DATAtr,OLSp);

[PSp] = ps_train(DATAtr,PSp);

[ELMp] = elm_train(DATAtr,ELMp);

[MLPp] = mlp_train(DATAtr,MLPp);

[RBFp] = rbf_train(DATAtr,RBFp);

[SVMp] = svm_train(DATAtr,SVMp);

[LSSVMp] = lssvm_train(DATAtr,LSSVMp);
 
% [GPp] = gp_train(DATAtr,GPp);

[MLMp] = mlm_train(DATAtr,MLMp);
 
[KNNp] = knn_train(DATAtr_sq,KNNp);

[KMp] = kmeans_train(DATAtr_sq,KMp);

[WTAp] = wta_train(DATAtr_sq,WTAp);

[LVQp] = lvq_train(DATAtr_sq,LVQp);

[SOM1Dp] = som1d_train(DATAtr_sq,SOM1Dp);

[SOM2Dp] = som2d_train(DATAtr_sq,SOM2Dp);

[KSOMGDp] = ksom_gd_train(DATAtr_sq,KSOMGDp);

[KSOMEFp] = ksom_ef_train(DATAtr_sq,KSOMEFp);

[KSOMPSp] = ksom_ps_train(DATAtr_sq,KSOMPSp);

% %%%%%%%%%%%%%%%%% CLASSIFIERS' TEST %%%%%%%%%%%%%%%%%%%%

[OUT_bay] = gauss_classify(DATAts,BAYp);

[OUT_ols] = ols_classify(DATAts,OLSp);

[OUT_ps] = ps_classify(DATAts,PSp);

[OUT_elm] = elm_classify(DATAts,ELMp);

[OUT_mlp] = mlp_classify(DATAts,MLPp);

[OUT_rbf] = rbf_classify(DATAts,RBFp);

[OUT_svm] = svm_classify(DATAts,SVMp);
 
[OUT_lssvm] = lssvm_classify(DATAts,LSSVMp);
 
% [OUT_gp] = gp_classify(DATAts,GPp);

[OUT_mlm] = mlm_classify(DATAts,MLMp);
 
[OUT_knn] = knn_classify(DATAts_sq,KNNp);

[OUT_kmeans] = kmeans_classify(DATAts_sq,KMp);

[OUT_wta] = wta_classify(DATAts_sq,WTAp);

[OUT_lvq] = lvq_classify(DATAts_sq,LVQp);

[OUT_som1d] = som1d_classify(DATAts_sq,SOM1Dp);

[OUT_som2d] = som2d_classify(DATAts_sq,SOM2Dp);

[OUT_ksom_gd] = ksom_gd_classify(DATAts_sq,KSOMGDp);

[OUT_ksom_ef] = ksom_ef_classify(DATAts_sq,KSOMEFp);

[OUT_ksom_ps] = ksom_ps_classify(DATAts_sq,KSOMPSp);

end

%% RESULTS / STATISTICS

OUT_bay,
OUT_ols,
OUT_ps,
OUT_elm,
OUT_mlp,
OUT_rbf,
OUT_svm,
OUT_lssvm,
% OUT_gp,
OUT_mlm,
OUT_knn,
OUT_kmeans,
OUT_wta,
OUT_lvq,
OUT_som1d,
OUT_som2d,
OUT_ksom_gd,
OUT_ksom_ef,
OUT_ksom_ps,

%% GRAPHICS



%% SAVE DATA



%% END