function [Y_resu, Y_conf] = lambda_predict(X_test, param, idx_feat, X_train, Y_train)
%[Y_resu, Y_conf] = lambda_predict(X_test, param, idx_feat, X_train, Y_train)
% Make classification predictions with the lambda method.
% Inputs:
% X_test -- Test data matrix of dim (num test examples, num features).
% param -- Classifier parameters, see lambda_trainer.
% idx_feat -- Indices of the features selected.
% X_train -- Training data matrix of dim (num training examples, num features).
%         -- used by some predictors (not lambda though).
% Y_train -- Training labels (num training examples, 1).
%         -- used by some predictors (not lambda though).
% Returns:
% Y_resu -- +-1 predictions on the test data of dim (num test example).
% Y_conf -- Confidence values (e.g. absolute discriminamt values).

% Isabelle Guyon -- September 2003 -- isabelle@clopinet.com

Y_score=X_test(:,idx_feat)*param.W'+param.b;

Y_resu=sign(Y_score);
Y_conf=abs(Y_score);
