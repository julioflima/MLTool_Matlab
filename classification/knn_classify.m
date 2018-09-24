function [OUT] = knn_classify(DATA,PAR)

% --- KNN Classifier Test ---
%
%   [OUT] = knn_classify(DATA,PAR)
%
%   Input:
%       DATA.
%           input = test attributes [p x Nts]
%           output = test labels [c x Nts]
%       PAR.
%           k = number of nearest neighbors
%           input = training attributes [p x Ntr]
%           output = training labels [c x Ntr]
%   Output:
%       OUT.
%           y_h = classifier's output [c x N]
%           Mconf = classifier's confusion matrix [c x c]
%           acerto = classifier's accuracy rate [int]

%% INITIALIZATIONS

% Get testing data
Q = DATA.input';
T2 = DATA.output';

% Get training data
P = PAR.input';
T1 = PAR.output';

% Get parameters
k = PAR.k;

% Number of classes
Nc = length(unique(T2));

% Number of samples
[Ntr,~] = size(P);      % training
[Nts,~] = size(Q);      % testing 

% Initialize Outputs
Mconf = zeros(Nc,Nc);
y_h = zeros(1,Nts);

%% ALGORITHM

for i = 1:Nts,

    % distance from test sample for each training sample
    vet_dist = zeros(1,Ntr);
    for j = 1:Ntr,
        x = Q(i,:) - P(j,:);        % diference vector      
        vet_dist(j) = norm(x,2);    % quadratic norm
    end
    
    % sort distances and get nearest neighbors
    [~,aux1] = sort(vet_dist,2,'ascend');
    Knear = aux1(1:k);

    % Voting in order to find estimated label
    lbls = T1(Knear);
    votes = zeros(1,Nc);
    for aux2 = 1:k,
        votes(lbls(aux2)) = votes(lbls(aux2)) + 1;
    end
    
    % Calculate confusion matrix
    [~,y_h(i)] = max(votes);
    Mconf(T2(i),y_h(i)) = Mconf(T2(i),y_h(i)) + 1;
end

% Calculate Accuracy

acerto = sum(diag(Mconf)) / Nts;

%% FILL OUTPUT STRUCTURE

OUT.y_h = y_h;
OUT.Mconf = Mconf;
OUT.acerto = acerto;

%% END