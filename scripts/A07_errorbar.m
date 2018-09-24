%% ERROR BAR EXAMPLE

% Gera��o de um gr�fico para an�lise estat�stica dos dados
% �ltima Altera��o: 02/01/2014

clear;
clc;

%% Gera barra de erros

% x / y / e - devem ter a mesma dimensao (por isso a funcao "ones")
x = 1:10;
y = sin(x);
e1 = std(y)*ones(size(x));
e2 = std(y)*ones(size(x));
errorbar(x,y,e1,e2)

%% END