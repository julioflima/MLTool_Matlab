function [out1] = test_function(in1,in2)

% --- Test Function
%
%

%% INIT - HYPERPARAMETERS

% Verify if the parameter structure was used

display(nargin)

if (nargin == 1),
    in2 = 1;
elseif (isempty(in2)),
    in2 = 2;
else
    in2 = 3;
end

% If not, use all default fields



% If yes, verify each field and apply default if anyone is missing



%% ALGORITHM

out1 = in1 + in2;

%% END