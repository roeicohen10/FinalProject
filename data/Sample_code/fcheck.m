function f=fcheck(fn)
%f=fcheck(fn)
% Check that the file exists

fp=fopen(fn,'r');
if fp>0
    f=1;
    fclose(fp);
else
    f=0;
end

