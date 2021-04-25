function area = auc(Output, Target)
%area = auc(Output, Target)
% Inputs:
% Output -- Vector of classifier discriminant values.
% Target -- Vector of corresponding +-1 target values.
% area -- Area under the ROC curve.

% Isabelle Guyon -- September 2003 -- isabelle@clopinet.com
% Adapted from Steve Gunn -- srg@ecs.soton.ac.uk

if size(Output)~=size(Target), area=[]; return; end

Output=full(Output);
Target=full(Target);

posidx=find(Target>0);
negidx=find(Target<0);
[p1,p2]=size(posidx);
[n1,n2]=size(negidx);
posout=repmat(Output(posidx),n2,n1);
negout=repmat(Output(negidx)',p1,p2);
rocmat=2*(negout<posout);
rocmat(negout==posout)=1;
area=sum(sum(rocmat))/(2*max(n1,n2)*max(p1,p2));
