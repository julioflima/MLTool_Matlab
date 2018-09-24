--------SHORT CIRCUIT DATA BASE-----------

Este banco de dados � formado por um conjunto de vetores que representam as
harm�nicas das correntes do estator de um motor de indu��o trif�sico gaiola 
de esquilo, sujeito a diversos niveis de curto-cirtuito.

-------------data_ESPEC_1-----------------

- 6 attributes (harmonics - 0,5 1,5 2,5 3 5 7)

- 294 samples (42 - normal motors / 252 - faulty motors)
-    42 samples per class
	3 niveis de carga (0% 50% 100%)
	2 fases (fase em contato direto / fase em contato indireto com falha)
	7 velocidades conversor (30; 35; 40; 45; 50; 55; 60 Hz)

- 7 classes (0 - SF / 1 - A1 / 2 - A2 / 3 - A3
            4 - B1 / 5 - B2 / 6 - B3)

-------------data_ESPEC_2-----------------

- 6 attributes (harmonics - 0,5 1,5 2,5 3 5 7)
- 392 samples (56 - normal motors / 336 - faulty motors)
- 56 samples per class (SF, A1, A2, A3, B1, B2, B3)

-------------data_ESPEC_3-----------------

- 16 harmonicas (0,5 � 8)
- 56 amostras de cada classe (SF, A1, A2, A3, B1, B2, B3)
- 392 amostras no total

-------------data_ESPEC_4-----------------

- 16 harmonicas (0,5 � 8)
- 42 amostras de cada classe (SF, A1, A2, A3, B1, B2, B3)
- 294 amostras no total

-------------data_ESPEC_5-----------------

- 6 harmonicas (0,5 1,5 2,5 3 5 7)
- 252 dados normais 42 reais e 210 gerados
- 252 dados de falha - 42 por classe (A1, A2, A3, B1, B2, B3)
- 504 amostras no total (42 x 252)

-------------data_ESPEC_6-----------------

Cada vetor possui 11 atributos. 7 deles s�o utiliz�veis para
o treinamento e teste, e 4 s�o para definir o vetor. S�o eles:

1 - Harm�nica fundamental
2 - 2a Harm�nica
3 - 3a Harm�nica
4 - 5a Harm�nica
5 - 7a Harm�nica
6 - 0,5*Harm�nica fundamental
7 - 1,5*Harm�nica fundamental

8 - tipo de falha: 0 (sem falha), 1 (falha de alta impedancia), 2 (falha de baixa impedancia)
9 - Gravidade da falha (de acordo com o n�mero de espiras em curto: (0,1,2,3)
10 - Frequ�ncia da tens�o do conversor de frequ�ncia: (30, 35, 40, 45, 50, 55, 60)
11 - Porcentagem de carga aplicada: 0, 50, 100


------------------------------------------