function check_data(X, pat_num, feat_num, check_sum)
%check_data(X, pat_num, feat_num, check_sum)
% Function that checks the sanity of the data.

% Isabelle Guyon -- August 2003 -- isabelle@clopinet.com

if any(size(X)~=[pat_num feat_num]), error('Wrong matrix dimension'); end
if sum(sum(X))~=check_sum, error('Bad check sum'); end