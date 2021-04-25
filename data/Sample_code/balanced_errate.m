function errate=balanced_errate(Output, Target)
%errate=balanced_errate(Output, Target)
% Compute a "balanced" error rate as the average
% of the error rate of positive examples and the
% error rate of negative examples.

% Isabelle Guyon -- August 2003 -- isabelle@clopinet.com

if size(Output)~=size(Target), errate=[]; return; end

Output=full(Output);
Target=full(Target);

pos_idx=find(Target>0);
neg_idx=find(Target<0);
errate_pos=mean(Output(pos_idx)<0);
errate_neg=mean(Output(neg_idx)>0);
errate=mean([errate_pos,errate_neg]);