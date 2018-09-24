%% DATA LOADING AND PRE-PROCESSING

DATAg = data_motor_gen(OPT);        % generate data for motor problem
DATAg = normalize(DATAg,OPT);       % normalize the attributes' matrix
DATAg = label_adjust(DATAg,OPT);	% adjust labels for the problem
[p,N] = size(DATAg.dados);          % get number of attributes and samples
[Nc,~] = size(DATAg.alvos);         % get number of classes

%% HIPERPARAMETERS - DEFAULT

OLSp.on = 1;                % Run the classifier
OLSp.aprox = 1;             % Type of aproximation

BAYp.on = 1;                % Run the classifier
BAYp.type = 2;              % Type of classificer

PSp.on = 1;                 % Run the classifier
PSp.Ne = 300;               % maximum number of training epochs
PSp.eta = 0.05;             % Learning step

MLPp.on = 1;                % Run the classifier
MLPp.Nh = 10;               % No. de neuronios na camada oculta
MLPp.Ne = 200;              % No m�ximo de epocas de treino
MLPp.eta = 0.05;            % Passo de aprendizagem
MLPp.mom = 0.75;            % Fator de momento
MLPp.Nlin = 2;              % Nao-linearidade MLP (tg hiperb)

ELMp.on = 1;                % Run the classifier
ELMp.Nh = 25;               % No. de neuronios na camada oculta
ELMp.Nlin = 2;              % N�o linearidade ELM (tg hiperb)

SVMp.on = 1;                % Run the classifier
SVMp.C = 5;                 % constante de regulariza��o
SVMp.Ktype = 1;             % kernel gaussiano (tipo = 1)
SVMp.sig2 = 0.01;           % Variancia (kernel gaussiano)

LSSVMp.on = 1;              % Run the classifier
LSSVMp.C = 0.5;             % constante de regulariza��o
LSSVMp.Ktype = 1;           % kernel gaussiano (tipo = 1)
LSSVMp.sig2 = 128;          % Variancia (kernel gaussiano)

MLMp.on = 1;                % Run the classifier
MLMp.K = 09;                % Number of reference points

GPp.on = 1;                 % Run the classifier
GPp.l2 = 2;                 % Constante GP1
GPp.K = 1;                  % kernel gaussiano (tipo = 1)
GPp.sig2 = 2;               % Constante GP2

%% HIPERPARAMETERS - GRID SEARCH FOR CROSS VALIDATION

OLScv = OLSp;               % Constant Hyperparameters
% ToDo - type of aproximation

BAYcv = BAYp;               % Constant Hyperparameters
% ToDo - type of classifier

PScv = PSp;                 % Constant Hyperparameters
% ToDo - learning step

MLPcv = MLPp;               % Constant Hyperparameters
MLPcv.Nh = 2:20;            % Number of hidden neurons
mlp_Nh = zeros(OPT.Nr,1);   % Acc number of hidden neurons

ELMcv = ELMp;               % Constant Hyperparameters
ELMcv.Nh = 10:30;           % Number of hidden neurons
elm_Nh = zeros(OPT.Nr,1);	% Acc number of hidden neurons

SVMcv = SVMp;               % Constant Hyperparameters
SVMcv.C = [0.5 5 10 15 25 50 100 250 500 1000];
SVMcv.sig2 = [0.01 0.05 0.1 0.5 1 5 10 50 100 500];
svm_C = zeros(OPT.Nr,1);    % Acc regularization constant
svm_sig2 = zeros(OPT.Nr,1); % Acc of variance(gaussian kernel)
svm_nsv = zeros(OPT.Nr,1);  % Acc number of support vectors

LSSVMcv = LSSVMp;             % Constant Hyperparameters
LSSVMcv.C = [2^-5 2^-4 2^-3 2^-2 2^-1 2^0 2^1 2^2 2^3 2^4 2^5 2^6 2^7 2^8 2^9 2^10 2^11 2^12 2^13 2^14 2^15 2^16 2^17 2^18 2^19 2^20];
LSSVMcv.sig2 = [2^-10 2^-9 2^-8 2^-7 2^-6 2^-5 2^-4 2^-3 2^-2 2^-1 2^0 2^1 2^2 2^3 2^4 2^5 2^6 2^7 2^8 2^9 2^10];
lssvm_C = zeros(OPT.Nr,1);    % Acc regularization constant
lssvm_sig2 = zeros(OPT.Nr,1); % Acc of variance(gaussian kernel)
lssvm_nsv = zeros(OPT.Nr,1);  % Acc number of support vectors

MLMcv = MLMp;               % Constant Hyperparameters
MLMcv.K = 2:15;             % Reference points
mlm_K = zeros(OPT.Nr,1);    % Acc number of reference points

GPcv = GPp;                 % Constant Hyperparameters
% ToDo - All hyperparameters

%% CLASSIFIERS' RESULTS INIT

hold_acc = cell(OPT.Nr,1);       % Acc of labels and data division

ols_out_tr = cell(OPT.Nr,1);	% Acc of training data output
ols_out_ts = cell(OPT.Nr,1);	% Acc of test data output
ols_out_rj = cell(OPT.Nr,1);	% Acc of reject option output
ols_Mconf_sum = zeros(Nc,Nc);   % Aux var for mean confusion matrix calc

bay_out_tr = cell(OPT.Nr,1);	% Acc of training data output
bay_out_ts = cell(OPT.Nr,1);	% Acc of test data output
bay_out_rj = cell(OPT.Nr,1);	% Acc of reject option output
bay_Mconf_sum = zeros(Nc,Nc);   % Aux var for mean confusion matrix calc

ps_out_tr = cell(OPT.Nr,1);     % Acc of training data output
ps_out_ts = cell(OPT.Nr,1);     % Acc of test data output
ps_out_rj = cell(OPT.Nr,1);     % Acc of reject option output
ps_Mconf_sum = zeros(Nc,Nc);    % Aux var for mean confusion matrix calc

mlp_out_tr = cell(OPT.Nr,1);	% Acc of training data output
mlp_out_ts = cell(OPT.Nr,1);	% Acc of test data output
mlp_out_rj = cell(OPT.Nr,1);	% Acc of reject option output
mlp_Mconf_sum = zeros(Nc,Nc);   % Aux var for mean confusion matrix calc

elm_out_tr = cell(OPT.Nr,1);	% Acc of training data output
elm_out_ts = cell(OPT.Nr,1);	% Acc of test data output
elm_out_rj = cell(OPT.Nr,1);	% Acc of reject option output
elm_Mconf_sum = zeros(Nc,Nc);   % Aux var for mean confusion matrix calc

svm_out_tr = cell(OPT.Nr,1);	% Acc of training data output
svm_out_ts = cell(OPT.Nr,1);	% Acc of test data output
svm_out_rj = cell(OPT.Nr,1);	% Acc of reject option output
svm_Mconf_sum = zeros(Nc,Nc);   % Aux var for mean confusion matrix calc

lssvm_out_tr = cell(OPT.Nr,1);	% Acc of training data output
lssvm_out_ts = cell(OPT.Nr,1);	% Acc of test data output
lssvm_out_rj = cell(OPT.Nr,1);	% Acc of reject option output
lssvm_Mconf_sum = zeros(Nc,Nc); % Aux var for mean confusion matrix calc

mlm_out_tr = cell(OPT.Nr,1);	% Acc of training data output
mlm_out_ts = cell(OPT.Nr,1);	% Acc of test data output
mlm_out_rj = cell(OPT.Nr,1);	% Acc of reject option output
mlm_Mconf_sum = zeros(Nc,Nc);   % Aux var for mean confusion matrix calc

gp_out_tr = cell(OPT.Nr,1);     % Acc of training data output
gp_out_ts = cell(OPT.Nr,1);     % Acc of test data output
gp_out_rj = cell(OPT.Nr,1);     % Acc of reject option output
gp_Mconf_sum = zeros(Nc,Nc);    % Aux var for mean confusion matrix calc

%% HOLD OUT / CROSS VALIDATION / TRAINING / TEST

for r = 1:OPT.Nr,

% Display, at Command Window, each repeat

display(r);
display(datestr(now));

% %%%%%%%%%%%%%%%%%%%% HOLD OUT %%%%%%%%%%%%%%%%%%%%%%%%%%
    
[DATAho] = hold_out(DATAg,OPT);
hold_acc{1} = DATAho;

DATAtr.dados = DATAho.P;
DATAtr.alvos = DATAho.T1;
DATAtr.rot = DATAho.rot_T1;

DATAts.dados = DATAho.Q;
DATAts.alvos = DATAho.T2;
DATAts.rot = DATAho.rot_T2;

% %%%%%%%%%%%%%%%% CROSS VALIDATION %%%%%%%%%%%%%%%%%%%%%

% Shuffle Data

[~, Ntrain] = size(DATAtr.dados);
I = randperm(Ntrain);
DATAtr.dados = DATAtr.dados(:,I); 
DATAtr.alvos = DATAtr.alvos(:,I);

% OLS - ToDo - All

% BAYES - ToDo - All

% PS - ToDo - All

% [MLPp] = mlp_cv(DATAtr,MLPp,MLPcv,CVp);
% mlp_Nh(r) = MLPp.Nh;
% 
% [ELMp] = elm_cv(DATAtr,ELMp,ELMcv,CVp);
% elm_Nh(r) = ELMp.Nh;
% 
% [SVMp] = svm_cv(DATAtr,SVMp,SVMcv,CVp);
% svm_C(r) = SVMp.C;
% svm_sig2(r) = SVMp.sig2;
% 
% [LSSVMp] = lssvm_cv(DATAtr,LSSVMp,LSSVMcv,CVp);
% lssvm_C(r) = LSSVMp.C;
% lssvm_sig2(r) = LSSVMp.sig2;
% 
% [MLMp] = mlm_cv(DATAtr,MLMp,MLMcv,CVp);
% mlm_K(r) = MLMp.K;

% GP - ToDo - All

% %%%%%%%%%%%%%% CLASSIFIERS' TRAINING %%%%%%%%%%%%%%%%%%%

[OLSp] = ols_train(DATAtr,OLSp);

[BAYp] = gauss_train(DATAtr,BAYp);

[PSp] = ps_train(DATAtr,PSp);

[MLPp] = mlp_train(DATAtr,MLPp);
 
[ELMp] = elm_train(DATAtr,ELMp);

[SVMp] = svm_train(DATAtr,SVMp);
 
[LSSVMp] = lssvm_train(DATAtr,LSSVMp);
 
[MLMp] = mlm_train(DATAtr,MLMp);
 
[GPp] = gp_train(DATAtr,GPp);

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

ols_Mconf_sum = ols_Mconf_sum + OUTts.Mconf;

% BAYES

[OUTtr] = gauss_classify(DATAtr,BAYp);
OUTtr.nf = normal_or_fail(OUTtr.Mconf);
OUTtr.mcc = Mcc(OUTtr.Mconf);
bay_out_tr{r,1} = OUTtr;

[OUTts] = gauss_classify(DATAts,BAYp);
OUTts.nf = normal_or_fail(OUTts.Mconf);
OUTts.mcc = Mcc(OUTts.Mconf);
bay_out_ts{r,1} = OUTts;

[OUTrj] = reject_opt2(DATAts,OUTts,REJp);
OUTrj.nf = normal_or_fail(OUTrj.Mconf);
OUTrj.mcc = Mcc(OUTrj.Mconf);
bay_out_rj{r,1} = OUTrj;

bay_Mconf_sum = bay_Mconf_sum + OUTts.Mconf;

% PS

[OUTtr] = ps_classify(DATAtr,PSp);
OUTtr.nf = normal_or_fail(OUTtr.Mconf);
OUTtr.mcc = Mcc(OUTtr.Mconf);
ps_out_tr{r,1} = OUTtr;

[OUTts] = ps_classify(DATAts,PSp);
OUTts.nf = normal_or_fail(OUTts.Mconf);
OUTts.mcc = Mcc(OUTts.Mconf);
ps_out_ts{r,1} = OUTts;

[OUTrj] = reject_opt2(DATAts,OUTts,REJp);
OUTrj.nf = normal_or_fail(OUTrj.Mconf);
OUTrj.mcc = Mcc(OUTrj.Mconf);
ps_out_rj{r,1} = OUTrj;

ps_Mconf_sum = ps_Mconf_sum + OUTts.Mconf;

% MLP

[OUTtr] = mlp_classify(DATAtr,MLPp);
OUTtr.nf = normal_or_fail(OUTtr.Mconf);
OUTtr.mcc = Mcc(OUTtr.Mconf);
mlp_out_tr{r,1} = OUTtr;

[OUTts] = mlp_classify(DATAts,MLPp);
OUTts.nf = normal_or_fail(OUTts.Mconf);
OUTts.mcc = Mcc(OUTts.Mconf);
mlp_out_ts{r,1} = OUTts;

[OUTrj] = reject_opt2(DATAts,OUTts,REJp);
OUTrj.nf = normal_or_fail(OUTrj.Mconf);
OUTrj.mcc = Mcc(OUTrj.Mconf);
mlp_out_rj{r,1} = OUTrj;

mlp_Mconf_sum = mlp_Mconf_sum + OUTts.Mconf;

% ELM

[OUTtr] = elm_classify(DATAtr,ELMp);
OUTtr.nf = normal_or_fail(OUTtr.Mconf);
OUTtr.mcc = Mcc(OUTtr.Mconf);
elm_out_tr{r,1} = OUTtr;

[OUTts] = elm_classify(DATAts,ELMp);
OUTts.nf = normal_or_fail(OUTts.Mconf);
OUTts.mcc = Mcc(OUTts.Mconf);
elm_out_ts{r,1} = OUTts;

[OUTrj] = reject_opt2(DATAts,OUTts,REJp);
OUTrj.nf = normal_or_fail(OUTrj.Mconf);
OUTrj.mcc = Mcc(OUTrj.Mconf);
elm_out_rj{r,1} = OUTrj;

elm_Mconf_sum = elm_Mconf_sum + OUTts.Mconf;

% SVM

[OUTtr] = svm_classify(DATAtr,SVMp);
OUTtr.nf = normal_or_fail(OUTtr.Mconf);
OUTtr.mcc = Mcc(OUTtr.Mconf);
svm_out_tr{r,1} = OUTtr;

[OUTts] = svm_classify(DATAts,SVMp);
OUTts.nf = normal_or_fail(OUTts.Mconf);
OUTts.mcc = Mcc(OUTts.Mconf);
svm_out_ts{r,1} = OUTts;

[OUTrj] = reject_opt2(DATAts,OUTts,REJp);
OUTrj.nf = normal_or_fail(OUTrj.Mconf);
OUTrj.mcc = Mcc(OUTrj.Mconf);
svm_out_rj{r,1} = OUTrj;

svm_nsv(r) = SVMp.nsv;

svm_Mconf_sum = svm_Mconf_sum + OUTts.Mconf;

% LSSVM

[OUTtr] = lssvm_classify(DATAtr,LSSVMp);
OUTtr.nf = normal_or_fail(OUTtr.Mconf);
OUTtr.mcc = Mcc(OUTtr.Mconf);
lssvm_out_tr{r,1} = OUTtr;

[OUTts] = lssvm_classify(DATAts,LSSVMp);
OUTts.nf = normal_or_fail(OUTts.Mconf);
OUTts.mcc = Mcc(OUTts.Mconf);
lssvm_out_ts{r,1} = OUTts;

[OUTrj] = reject_opt2(DATAts,OUTts,REJp);
OUTrj.nf = normal_or_fail(OUTrj.Mconf);
OUTrj.mcc = Mcc(OUTrj.Mconf);
lssvm_out_rj{r,1} = OUTrj;

lssvm_nsv(r) = LSSVMp.nsv;

lssvm_Mconf_sum = lssvm_Mconf_sum + OUTts.Mconf;

% MLM

[OUTtr] = mlm_classify(DATAtr,MLMp);
OUTtr.nf = normal_or_fail(OUTtr.Mconf);
mlm_out_tr{r,1} = OUTtr;

[OUTts] = mlm_classify(DATAts,MLMp);
OUTts.nf = normal_or_fail(OUTts.Mconf);
OUTts.mcc = Mcc(OUTts.Mconf);
mlm_out_ts{r,1} = OUTts;

[OUTrj] = reject_opt2(DATAts,OUTts,REJp);
OUTrj.nf = normal_or_fail(OUTrj.Mconf);
OUTrj.mcc = Mcc(OUTrj.Mconf);
mlm_out_rj{r,1} = OUTrj;

mlm_Mconf_sum = mlm_Mconf_sum + OUTts.Mconf;

% GP

% [OUTtr] = gp_classify(DATAtr,GPp);
% OUTtr.nf = normal_or_fail(OUTtr.Mconf);
% gp_out_tr{r,1} = OUTtr;
% 
% [OUTts] = gp_classify(DATAts,GPp);
% OUTts.nf = normal_or_fail(OUTts.Mconf);
% gp_out_ts{r,1} = OUTts;
% 
% [OUTrj] = reject_opt2(DATAts,OUTts,REJp);
% OUTrj.nf = normal_or_fail(OUTrj.Mconf);
% gp_out_rj{r,1} = OUTrj;
% 
% gp_Mconf_sum = gp_Mconf_sum + OUTts.Mconf;

end

%% ESTATISTICAS

% Matrizes de confusao medias
ols_Mconf_mean = ols_Mconf_sum / OPT.Nr;
bay_Mconf_mean = bay_Mconf_sum / OPT.Nr;
ps_Mconf_mean = ps_Mconf_sum / OPT.Nr;
mlp_Mconf_mean = mlp_Mconf_sum / OPT.Nr;
elm_Mconf_mean = elm_Mconf_sum / OPT.Nr;
svm_Mconf_mean = svm_Mconf_sum / OPT.Nr;
lssvm_Mconf_mean = lssvm_Mconf_sum / OPT.Nr;
mlm_Mconf_mean = mlm_Mconf_sum / OPT.Nr;
% gp_Mconf_mean = gp_Mconf_men / OPT.Nr;

% Vetores de Matthews Correlation



%% GERA��O DOS GRAFICOS

% Inicializa c�lula de labels e matriz de acertos para testes
labels = {};
Mat_boxplot1 = [];
Mat_boxplot2 = [];
Mat_boxplot3 = [];
Mat_boxplot4 = [];
Mat_boxplot5 = [];
Mat_boxplot6 = [];

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
end

if BAYp.on == 1,
[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'BAY'};
Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(bay_out_tr)];
Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(bay_out_tr)];
Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(bay_out_ts)];
Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(bay_out_ts)];
Mat_boxplot5 = [Mat_boxplot5 accuracy_mult(bay_out_rj)];
Mat_boxplot6 = [Mat_boxplot6 accuracy_bin(bay_out_rj)];
end

if PSp.on == 1,
[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'PS'};
Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(ps_out_tr)];
Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(ps_out_tr)];
Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(ps_out_ts)];
Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(ps_out_ts)];
Mat_boxplot5 = [Mat_boxplot5 accuracy_mult(ps_out_rj)];
Mat_boxplot6 = [Mat_boxplot6 accuracy_bin(ps_out_rj)];
end

if MLPp.on == 1,
[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'MLP'};
Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(mlp_out_tr)];
Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(mlp_out_tr)];
Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(mlp_out_ts)];
Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(mlp_out_ts)];
Mat_boxplot5 = [Mat_boxplot5 accuracy_mult(mlp_out_rj)];
Mat_boxplot6 = [Mat_boxplot6 accuracy_bin(mlp_out_rj)];
end

if ELMp.on == 1,
[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'ELM'};
Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(elm_out_tr)];
Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(elm_out_tr)];
Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(elm_out_ts)];
Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(elm_out_ts)];
Mat_boxplot5 = [Mat_boxplot5 accuracy_mult(elm_out_rj)];
Mat_boxplot6 = [Mat_boxplot6 accuracy_bin(elm_out_rj)];
end

if SVMp.on == 1,
[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'SVM'};
Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(svm_out_tr)];
Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(svm_out_tr)];
Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(svm_out_ts)];
Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(svm_out_ts)];
Mat_boxplot5 = [Mat_boxplot5 accuracy_mult(svm_out_rj)];
Mat_boxplot6 = [Mat_boxplot6 accuracy_bin(svm_out_rj)];
end

if LSSVMp.on == 1,
[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'LSSVM'};
Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(lssvm_out_tr)];
Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(lssvm_out_tr)];
Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(lssvm_out_ts)];
Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(lssvm_out_ts)];
Mat_boxplot5 = [Mat_boxplot5 accuracy_mult(lssvm_out_rj)];
Mat_boxplot6 = [Mat_boxplot6 accuracy_bin(lssvm_out_rj)];
end

if MLMp.on == 1,
[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'MLM'};
Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(mlm_out_tr)];
Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(mlm_out_tr)];
Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(mlm_out_ts)];
Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(mlm_out_ts)];
Mat_boxplot5 = [Mat_boxplot5 accuracy_mult(mlm_out_rj)];
Mat_boxplot6 = [Mat_boxplot6 accuracy_bin(mlm_out_rj)];
end

% BOXPLOT 1
figure; boxplot(Mat_boxplot1, 'label', labels);
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
ylabel('Acur�cia')              % label eixo y
xlabel('Classificadores')       % label eixo x
title('Taxa de Classifica��o')  % Titulo
axis ([0 n_labels+1 0 1.05])	% Eixos

hold on
media1 = mean(Mat_boxplot1);    % Taxa de acerto m�dia
plot(media1,'*k')
hold off

% BOXPLOT 2
figure; boxplot(Mat_boxplot2, 'label', labels);
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
ylabel('Acur�cia')              % label eixo y
xlabel('Classificadores')       % label eixo x
title('Taxa de Classifica��o')  % Titulo
axis ([0 n_labels+1 0 1.05])	% Eixos

hold on
media2 = mean(Mat_boxplot2);    % Taxa de acerto m�dia
plot(media2,'*k')
hold off

% BOXPLOT 3
figure; boxplot(Mat_boxplot3, 'label', labels);
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
ylabel('Acur�cia')              % label eixo y
xlabel('Classificadores')       % label eixo x
title('Taxa de Classifica��o')  % Titulo
axis ([0 n_labels+1 0 1.05])	% Eixos

hold on
media3 = mean(Mat_boxplot3);    % Taxa de acerto m�dia
plot(media3,'*k')
hold off

% BOXPLOT 4
figure; boxplot(Mat_boxplot4, 'label', labels);
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
ylabel('Acur�cia')              % label eixo y
xlabel('Classificadores')       % label eixo x
title('Taxa de Classifica��o')  % Titulo
axis ([0 n_labels+1 0 1.05])	% Eixos

hold on
media4 = mean(Mat_boxplot4);    % Taxa de acerto m�dia
plot(media4,'*k')
hold off

% BOXPLOT 5
figure; boxplot(Mat_boxplot5, 'label', labels);
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
ylabel('Acur�cia')              % label eixo y
xlabel('Classificadores')       % label eixo x
title('Taxa de Classifica��o')  % Titulo
axis ([0 n_labels+1 0 1.05])	% Eixos

hold on
media3 = mean(Mat_boxplot5);    % Taxa de acerto m�dia
plot(media3,'*k')
hold off

% BOXPLOT 6
figure; boxplot(Mat_boxplot6, 'label', labels);
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
ylabel('Acur�cia')              % label eixo y
xlabel('Classificadores')       % label eixo x
title('Taxa de Classifica��o')  % Titulo
axis ([0 n_labels+1 0 1.05])	% Eixos

hold on
media4 = mean(Mat_boxplot6);    % Taxa de acerto m�dia
plot(media4,'*k')
hold off

%% SALVAR DADOS

save(OPT.file);

%% END