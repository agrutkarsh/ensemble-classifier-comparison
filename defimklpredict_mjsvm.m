function [predicted_label, dvtest] = ...
    defimklpredict(testing_label_vector, testing_instance_matrix, model)

ytest = testing_label_vector;
Xtest = testing_instance_matrix;
svmmodel = model.svmmodels;

number_of_kernels = length( svmmodel );

for k = 1:number_of_kernels
    [p_label(:,k),~,dvtest(:,k)] = svmpredict(ytest,Xtest,svmmodel{k},'-q');
end

temp = sum(p_label,2);
predicted_label = zeros(size(temp));
predicted_label(temp>0) = 1;
predicted_label(temp<0) = -1;

