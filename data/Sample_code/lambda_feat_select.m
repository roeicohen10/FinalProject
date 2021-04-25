function idx=lambda_feat_select(X, Y, num)
%idx=lambda_feat_select(X, Y, num)
% Feature selection method that ranks according to the dot
% product with the target vector. Note that this criterion
% may not deliver good results if the features are not 
% centered and normalized with respect to the example distribution.

% Isabelle Guyon -- August 2003 -- isabelle@clopinet.com

fval=Y'*X;
[sval, si]=sort(-fval);
idx=si(1:num);