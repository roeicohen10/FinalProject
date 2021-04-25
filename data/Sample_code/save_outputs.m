function save_outputs(filename, vector)
%save_outputs(filename, vector)
% Save the outputs of a classifier or the features for all the examples
% of a given data set, in the original order of the examples.
% Inputs:
% filename --    Name of the file to write to.
% vector --     +-1 values indicating the class membership.
%               or a confidence values, or feature indices.

% Isabelle Guyon -- August 2003 -- isabelle@clopinet.com

fp=fopen(filename, 'w');
for i=1:length(vector)
    fprintf(fp, '%g\n', vector(i));
end
fclose(fp);