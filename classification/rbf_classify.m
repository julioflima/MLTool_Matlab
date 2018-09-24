function [OUT] = rbf_classify(DATA,PAR)

% --- RBF Classifier Test ---
%
%   [OUT] = rbf_classify(DATA,PAR)
%
%   Input:
%       DATA.
%           input = attributes [p x N]
%           output = labels [c x N]
%       PAR.
%           W = Hidden layer weight Matrix     [Nh x p]
%           r = radius (spread) of basis       [Nh x 1]
%           M = Output layer weight Matrix     [No x Nh+1]
%   Output:
%       PARout.
%           y_h = classifier's output               [c x N]
%           Mconf = classifier's confusion matrix   [c x c]
%           acerto = classifier's accuracy rate     [int]

%% INITIALIZATION

% Get data
MATin = DATA.input;       	% Get attribute matrix
MATout = DATA.output;    	% Get label matrix
[Nc,N] = size(MATout);      % Number of classes and samples

% Get parameters
q = PAR.Nh;                 % Number of hidden neurons
ativ = PAR.ativ;            % activation function type  
W = PAR.W;                  % hidden layer weight Matrix
r = PAR.r;                  % radius of each rbf
M = PAR.M;                  % Output layer weight Matrix

% Initialize Outputs
y_h = zeros(Nc,N);          % Estimated output
Mconf = zeros(Nc,Nc);       % Confusion Matrix

%% ALGORITHM

for t = 1:N,
    
    % Get input sample
    X = MATin(:,t);
 
    % Calculate output of basis functions
    z_vec = zeros(q,1);             % init output of basis functions
    for i = 1:q,       
        ci = W(:,i);                % Get rbf center
        ui = sum((X - ci).^2);      % Calculates distance
        ri = r(i);                  % get radius
        z_vec(i) = rbf_f_ativ(ui,ri,ativ);
    end
    z_vec = z_vec/sum(z_vec);       % normalize hidden layer outputs
    z1 = [1;z_vec];                 % add bias to hidden layer outputs
   
    % Calculate output of neural net
    y_h(:,t) = M*z1;
    
end

% Calculate success rate and confusion matrix

for t = 1:N,
    [~,iT2] = max(MATout(:,t));   	% Desired Output index
    [~,iY_h] = max(y_h(:,t));     	% Estimated Output index
    Mconf(iT2,iY_h) = Mconf(iT2,iY_h) + 1;
end

acerto = sum(diag(Mconf)) / N;

%% FILL OUTPUT STRUCTURE

OUT.y_h = y_h;
OUT.Mconf = Mconf;
OUT.acerto = acerto;

%% END