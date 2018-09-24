%% test script

clear;
clc;

%% Links

% https://www.mathworks.com/matlabcentral/answers/21012-default-parameter-if-the-user-input-is-empty

%% Tests

% Arguments Testing
a = test_function(4);
b = test_function(5,'');
c = test_function(6,1);

% Parameters testing
DATA.par1 = 1;
DATA.par2 = 2;
DATA.par3 = 3;
DATA.par4 = 4;
DATA.par5 = 5;

d = fieldnames(DATA);
e = isfield(DATA,'par5');
f = isfield(DATA,'par6');
g = any(strcmp('par5',fieldnames(DATA)));
h = any(strcmp('par6',fieldnames(DATA)));

i = [2 3 1 1 2 5 0];
j = unique(i);
l = [3 5 7 8 4 6 1];

figure; plot(i,l,'b.');

%% END
