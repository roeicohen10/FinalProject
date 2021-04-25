% NIPS workshop 2003 benchmark on variable and feature selection.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Matlab script to read the data, check it, compute some baseline
% performance, and save the result is the desired format.
% The three functions used for our lambda method are in canonical form: 
% lambda_feat_select, lambda_train, lambda_predict. If you put your method
% in canonical form, you should have only to change the method name to run
% this script.
% The second time the data should load faster because we save it in Matlab
% format.

% Isabelle Guyon -- August 2003 -- isabelle@clopinet.com

% DISCLAIMER: ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS" 
% THE ORGANIZERS DISCLAIMS ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
% THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. 
% IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER BENCHMARK ORGANIZERS BE LIABLE FOR ANY SPECIAL, 
% INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN CONNECTION 
% WITH THE USE OR PERFORMANCE OF BENCHMARK SOFTWARE, DOCUMENTS, MATERIALS, PUBLICATIONS, OR 
% INFORMATION MADE AVAILABLE. 

% Set parameters and directories (%%% CHANGE THAT %%%)
select_num=10;                  % Number of features to select.
dataset={'arcene', 'dexter', 'dorothea', 'gisette', 'madelon'};
dataset={'dexter'};
method='lambda';                % Your method name.
where_my_data_is='';            % This is the path to your data and results are
                                % e.g. c:/users/data/nips/ or /usr/home/iguyon/ 
                                % (do not forget the last slash)

data_dir=[where_my_data_is 'Data']; % Wehre you put the five data directories dowloaded.
output_dir=[where_my_data_is 'Results/' method]; % The outputs of a given method.
status=mkdir(where_my_data_is, ['Results/' method]);
zip_dir=[where_my_data_is 'Zipped']; % Zipped files ready to go!
status=mkdir(where_my_data_is, 'Zipped');
    
for k=1:length(dataset)

    % Input and output directories 
	data_name=dataset{k};
    input_dir=[data_dir '/' upper(data_name)];
	input_name=[input_dir '/' data_name];
	output_name=[output_dir '/' data_name];
	fprintf('\n/|\\-/|\\-/|\\-/|\\ Loading and checking dataset %s /|\\-/|\\-/|\\-/|\\\n\n', upper(data_name));
	% Data parameters and statistics
	p=read_parameters([input_name '.param']);
	fprintf('-- %s parameters and statistics -- \n\n', upper(data_name));
	print_parameters(p);
    % Read the data
    fprintf('\n-- %s loading data --\n', upper(data_name));
    X_train=[]; X_valid=[]; X_test=[]; Y_train=[]; Y_valid=[]; Y_test=[];
    if fcheck([data_dir '/' data_name '.mat']), 
        load([data_dir '/' data_name]); 
    else
        fprintf('\n');
	    % Read the labels
	    Y_train=read_labels([input_name '_train.labels']);
	    Y_valid=read_labels([input_name '_valid.labels']);  
	    Y_test=read_labels([input_name '_test.labels']);   
	    % Read the data
	    X_train=matrix_data_read([input_name '_train.data'],p.feat_num,p.train_num,p.data_type);
	    X_valid=matrix_data_read([input_name '_valid.data'],p.feat_num,p.valid_num,p.data_type);
        X_test=matrix_data_read([input_name '_test.data'],p.feat_num,p.test_num,p.data_type);
        save([data_dir '/' data_name], 'X_train', 'X_valid', 'X_test', 'Y_train', 'Y_valid', 'Y_test');
    end
    fprintf('\n-- %s data loaded --\n', upper(data_name));
	% Check the labels
	check_labels(Y_train, p.train_num, p.train_pos_num);
    if ~isempty(Y_valid), check_labels(Y_valid, p.valid_num, p.valid_pos_num); end
    if ~isempty(Y_test), check_labels(Y_test, p.test_num, p.test_pos_num); end
	% Check the data
	check_data(X_train, p.train_num, p.feat_num, p.train_check_sum);
	check_data(X_valid, p.valid_num, p.feat_num, p.valid_check_sum);
	check_data(X_test, p.test_num, p.feat_num, p.test_check_sum);
	fprintf('\n-- %s data sanity checked --\n', upper(data_name));
    % Try some method
    fprintf('\n-- %s testing with %s method --\n\n', upper(data_name), method);
    tic;
	% Select some features
	idx_feat = feval([method '_feat_select'], X_train, Y_train, select_num);
	% Train some classifier using the selected features
	[c, idx_feat] = feval([method '_train'], X_train, Y_train, idx_feat);
	% Test the classifier
    [Y_resu_train, Y_conf_train] = feval([method '_predict'], X_train, c, idx_feat, X_train, Y_train);
    [Y_resu_valid, Y_conf_valid] = feval([method '_predict'], X_valid, c, idx_feat, X_train, Y_train);
    [Y_resu_test, Y_conf_test] = feval([method '_predict'], X_test, c, idx_feat, X_train, Y_train);
	errate_train=balanced_errate(Y_resu_train, Y_train);
    errate_valid=balanced_errate(Y_resu_valid, Y_valid);
    errate_test=balanced_errate(Y_resu_test, Y_test); 
    auc_train=auc(Y_resu_train.*Y_conf_train, Y_train);
    auc_valid=auc(Y_resu_valid.*Y_conf_valid, Y_valid);
    auc_test=auc(Y_resu_test.*Y_conf_test, Y_test);
	t=toc;
	fprintf('Number of features: %d\n', length(idx_feat));
	fprintf('Training set: errate= %5.2f%%, auc= %5.2f%%\n', errate_train*100, auc_train*100);
    if ~isempty(Y_valid), 
        fprintf('Validation set: errate= %5.2f%%, auc= %5.2f%%\n', errate_valid*100, auc_valid*100);
    end
    if ~isempty(Y_test), 
        fprintf('Test set: errate= %5.2f%%, auc= %5.2f%%\n', errate_test*100, auc_test*100);
	end
	fprintf('Time of selection, training, and testing: %5.2f seconds\n', t);
	% Write out the results 
	% --- Note: the class predictions (.resu files) are mandatory.
	% --- Please also provide the confidence value when available, this will
	% --- allow us to compute ROC curves. A confidence values can be the absolute
	% --- values of a discriminant value, it does not need to be normalized
	% --- to resemble a probability.
	save_outputs([output_name '_train.resu'], Y_resu_train);
	save_outputs([output_name '_valid.resu'], Y_resu_valid);
	save_outputs([output_name '_test.resu'], Y_resu_test);
    save_outputs([output_name '_train.conf'], Y_conf_train);
	save_outputs([output_name '_valid.conf'], Y_conf_valid);
	save_outputs([output_name '_test.conf'], Y_conf_test);
	save_outputs([output_name '.feat'], idx_feat);
	fprintf('\n-- %s results saved, see %s* --\n', upper(data_name), output_name);

end % Loop over datasets

% Zip the archive so it is ready to go!
zip([zip_dir '/' method], ['Results/' method], where_my_data_is);
fprintf('\n-- %s zip archive created, see %s.zip --\n', upper(data_name), [zip_dir '/' method]);



