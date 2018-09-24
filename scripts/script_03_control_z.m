%% CONTROLE - MATLAB

% David Nascimento Coelho
% �ltima Revis�o: 23/05/2014

clear all; close all; clc;

nfig = 0;   % contagem do numero de figuras

%% Sistema de 1a Ordem

T = 1;  % Constante de tempo

a0 = 1;
b1 = T;
b0 = 1;

num1 = [0 a0];          % numerador do sistema de 1a ordem
den1 = [b1 b0];         % denominador do sistema de 1a ordem

sys1 = tf(num1,den1);   % fun��o de transfer�ncia 1

%% Mapeamento dos polos e zeros

pzmap(sys1)

%% Fra��es Parciais

num = [0 0 1];
den = [1 2 1];

% r1/(s-p1) + r2/(s-p2) + ... + k
[r,p,k] = residue(num,den);

%% END
