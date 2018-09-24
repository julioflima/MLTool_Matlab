function [OUT] = gauss_classify(DATA,PAR)

% --- Gaussian Classifier Test ---
%
%   [OUT] = gauss_classify(DATA,PAR)
%
%   Input:
%       DATA.
%           input = attributes [p x N]
%           output = labels [c x N]
%       PAR.
%           Ni = number of "a priori samples" per class (Nc x 1)
%           med_ni = centroid of each class             (Nc x p)
%           Ci = covariance matrix of each class        (Nc x p x p)
%           type = type of gaussian classifier
%               1 -> gi(x) = -0.5Qi(x) - 0.5ln(det(Ci)) + ln(p(Ci))
%               2 -> gi(x) = -0.5Qi(x) - 0.5ln(det(Ci))
%               3 -> gi(x) = -0.5Qi(x) (mahalanobis distance)
%            	covariance matrix is the pooled covariance matrix
%               4 -> gi(x) = ||x-mi||^2 (euclidean distance)
%   Output:
%       OUT.
%           y_h = classifier's output [c x N]
%           Mconf = classifier's confusion matrix [c x c]
%           acerto = classifier's accuracy rate [cte]

%% INITIALIZATIONS

% Get input and output
Q = DATA.input;
T2 = DATA.output;

% Get parameters
Ni = PAR.Ni;
med_ni = PAR.med_ni;
Ci = PAR.Ci;
type = PAR.type;

% Put output in [N x p] pattern
Q = Q';
[teste,par] = size(Q);

% Convert from [0 1 0] or [-1 1 -1] to sequential
T2 = T2';
Taux = zeros(teste,1);
for i = 1:teste,
    Taux(i) = find(T2(i,:) > 0);
end
T2 = Taux;

% Number of classes do the problem
[Nc,~] = size(Ni);

% Count all traning samples (a priori samples)
N = sum(Ni);

% Initialize estimated output and confusion matrix
y_h = -1*ones(teste,Nc);
Mconf = zeros(Nc,Nc);

%% ALGORITHM

if type == 1,  % Complete Classifier

% Covariance Matrix inverse
inv_Ci = cell(1,Nc);
for i = 1:Nc,
    inv_Ci{i} = pinv(Ci{i});
end

for i = 1:teste,

    % initialize discriminant function
    gi = zeros(Nc,1);
    
    for j = 1:Nc,
        % mahalanobis distance for each class
        MDj = (Q(i,:)-med_ni(j,:))*inv_Ci{j}*(Q(i,:)-med_ni(j,:))';
        % discriminant function for each class
        gi(j) = - 0.5*MDj -0.5*log(det(Ci{j})) + log(Ni(j)/N);
    end
    
    % Calculate output and confusion matrix
    [~,class] = max(gi);
    y_h(i,class) = 1;
    Mconf(T2(i),class) = Mconf(T2(i),class) + 1;
    
end

elseif type == 2, % Classifier without a Priori probability

% Covariance Matrix inverse
inv_Ci = cell(1,Nc);
for i = 1:Nc,
    inv_Ci{i} = pinv(Ci{i});
end

for i = 1:teste,

    % initialize discriminant function
    gi = zeros(Nc,1);
    for j = 1:Nc,
        % mahalanobis distance for each class
        MDj = (Q(i,:)-med_ni(j,:))*inv_Ci{j}*(Q(i,:)-med_ni(j,:))';
        % discriminant function for each class
        gi(j) = - 0.5*MDj -0.5*log(det(Ci{j}));
    end
    
    % Calculate output and confusion matrix
    [~,class] = max(gi);
    y_h(i,class) = 1;
    Mconf(T2(i),class) = Mconf(T2(i),class) + 1;
    
end
    
elseif type == 3, % Classifier with pooled covariance matrix
    
% Pooled covariance Matrix
Ci_pooled = zeros(par,par);
for j = 1:Nc,
    Ci_pooled = Ci_pooled + Ci{j}*Ni(j)/N;
end

% Pooled Covariance Matrix inverse
inv_Ci = pinv(Ci_pooled);

for i = 1:teste,
    
    % initialize discriminant function
    gi = zeros(Nc,1);
    for j = 1:Nc,
        % mahalanobis distance for each class
        MDj = (Q(i,:)-med_ni(j,:))*inv_Ci*(Q(i,:)-med_ni(j,:))';
        % discriminant function for each class
        gi(j) = - 0.5*MDj;
    end
    
    % Calculate output and confusion matrix
    [~,class] = max(gi);
    y_h(i,class) = 1;
    Mconf(T2(i),class) = Mconf(T2(i),class) + 1;

end

elseif type == 4, % Maximum likelihood Classifier
    
for i = 1:teste,
    
    % initialize discriminant function
    gi = zeros(Nc,1);
    for j = 1:Nc,
        % mahalanobis distance for each class
        MDj = (Q(i,:)-med_ni(j,:))*(Q(i,:)-med_ni(j,:))';
        % discriminant function for each class
        gi(j) = - 0.5*MDj;
    end
    
    % Calculate output and confusion matrix
    [~,class] = max(gi);
    y_h(i,class) = 1;
    Mconf(T2(i),class) = Mconf(T2(i),class) + 1;

end
    
else % invalid option
    
    disp('type a valid option: 1, 2, 3, 4');

end

% Adjust output

y_h = y_h';

% Calculate Accuracy

accuracy = sum(diag(Mconf))/teste;

%% FILL OUTPUT STRUCTURE

OUT.y_h = y_h;
OUT.Mconf = Mconf;
OUT.acerto = accuracy;

%% END