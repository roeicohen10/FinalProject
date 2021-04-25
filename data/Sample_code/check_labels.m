function check_labels(Y, pat_num, pos_num)
%check_labels(Y, pat_num, pos_num)
% Function that checks the sanity of the labels.

% Isabelle Guyon -- August 2003 -- isabelle@clopinet.com

if length(Y)~=pat_num, error('Wrong number of examples'); end
if length(find(Y>0))~=pos_num, error('Wrong number of positive examples'); end
if length(find(Y<0))~=pat_num-pos_num, error('Wrong number of negative examples'); end