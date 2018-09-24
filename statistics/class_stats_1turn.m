%% STATISTICS

Confusion Matrix

% Mconf = zeros(Nc,Nc);
% 
% for i = 1:N,
%     [~,iT2] = max(MATout(:,i));
%     [~,iY_h] = max(y_h(:,i));
%     Mconf(iT2,iY_h) = Mconf(iT2,iY_h) + 1;
% end
% 
% Accuracy
% 
% acerto = sum(diag(Mconf))/N;

% test with training data set
% test with test data set
% uses reject option
% Acc parameters and folds (training x test)

% Matriz de Confusão
%   linhas: a qual classe pertence o dado
%   colunas: qual a saida do classificador


%% END