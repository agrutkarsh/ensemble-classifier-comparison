function [predicted_label, accuracy, decision_values] = ...
    defimklpredict(testing_label_vector, testing_instance_matrix, model)

ytest = testing_label_vector;
Xtest = testing_instance_matrix;
FM = model.FM;
svmmodel = model.svmmodels;

number_of_kernels = length( svmmodel );

for k = 1:number_of_kernels
    [~,~,dvtest(:,k)] = svmpredict(ytest,Xtest,svmmodel{k},'-q');
end

decision_values = FMFI_ChoquetIntegralv2( dvtest, FM' );
predicted_label = sign( decision_values );

accuracy = sum( predicted_label == testing_label_vector )...
    /length(testing_label_vector);

if(accuracy < 0.5)
    accuracy = 1 - accuracy;
end