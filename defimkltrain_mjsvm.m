function model = defimkltrain( training_label_vector,...
    training_instance_matrix, rbf_kernel_gamma_vector, reg_type, lambda)
    
if nargin < 4 %number of function input arguments
    reg_type = [];
end

if isempty( reg_type )
    reg_type = 'l1';
    lambda = 0;
end

y = training_label_vector;
X = training_instance_matrix;

number_of_kernels = length( rbf_kernel_gamma_vector ); %number of elements

% Get decision values from LIBSVM for each kernel
for k = 1 : number_of_kernels,
    svmmodel{k} = svmtrain(y,X,['-t 2 -g ',num2str(rbf_kernel_gamma_vector(k)),'-q']);
end

model.svmmodels = svmmodel;