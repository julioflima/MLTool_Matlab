function [SSE] = prototypes_sse(C,DATA,PAR)

% --- Calculate the Sum of Squared errors of prototypes ---
%
%   [SSE] = prototypes_sse(C,DATA,PAR)
%
%   Input:
%       C = prototypes [p x k]
%       DATA.
%           input = input matrix [p x N]
%       PAR.
%           dist = Type of distance 
%               0: dot product
%               2: euclidean distance
%   Output:
%       SSE = sum of squared errors between prototypes and data

%% INITIALIZATION

% Load Data

input = DATA.input;
[~,N] = size(input);

% Init Aux Variables

sum_acc = 0;    % accumulate value of squared error sum

%% ALGORITHM

for i = 1:N,
    sample = input(:,i);                    % get a sample
    win = prototypes_win(C,sample,PAR);     % index of closest prototype
    
    % get the closest prototype 
    if (length(win) == 1),
        prot = C(:,win);
    elseif (length(win) == 2),
        prot = C(:,win(1),win(2));
    end
    
    sum_acc = sum_acc + sum((prot - sample).^2);
end

%% FILL OUTPUT STRUCTURE

SSE = sum_acc;
    
%% END