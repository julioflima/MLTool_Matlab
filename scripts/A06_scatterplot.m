%% SCATTER PLOT EXAMPLE

% Gera��o de um gr�fico para an�lise estat�stica dos dados
% Compara duas colunas (uma no eixo X, outra no eixo Y)
% �ltima Altera��o: 02/01/2014

clear;
clc;

%% Gerar grafico de dispers�o

x1 = (0:0.1:1)';
x2 = (0:0.2:2)';
X = [x1,x2];

scatterplot(X);
xlabel('teste1');
ylabel('teste2');
title('titulo1');

%% END