function [count] = ensemble_classification_common(X_orig, y_orig)

% define RBF kernel widths
nkernels = 5; % don't go above 10ish (the fuzzy measure will explode!)

% define the regularization type & weight
% acceptable values: 'l1', 'l2', 'max'
reg_type = 'l1';
lambda = 0.5; %0.5

count_defimkl = [];
count_mjsvm = [];
count_adaboost = [];
count_bagging = [];
count_rf = [];
for loop1=1:544
    
    X=X_orig;
    y=y_orig;
    
    [nobjs, nf] = size(X);
    rinds = randperm( nobjs );
    temp = floor(0.8*nobjs);
    train_inds = rinds( 1:temp );
    test_inds = rinds( temp+1:end );
    Xtrain = X( train_inds, : );
    Xtest = X( test_inds, : );
    ytrain = y( train_inds );
    ytest = y( test_inds );
    
    Xtrain=zscore(Xtrain);
    Xtest=zscore(Xtest);
    
    num_of_class = size(unique(y),1);
    
    temp = unique(y);
    t_out=y_orig;
    out1=zeros(size(Xtrain,1),1);
    for i=1:size(y_orig,1)
        for j=1:num_of_class
            if t_out(i,1)==temp(j)
                out1(i,1)=1;
            else
                out1(i,1)=-1;
            end
        end
    end
    
    ytrain = out1( train_inds );
    ytest = out1( test_inds );
    
    sigmas = linspace( 0.5/nf, 1.5/nf, nkernels );
    
    count_defimkl = ensemble_classification_DeFIMKL(ytrain, Xtrain, sigmas, reg_type, lambda, ytest, Xtest,count_defimkl);
    count_mjsvm = ensemble_classification_maj_SVM(ytrain, Xtrain,sigmas, reg_type, lambda, ytest, Xtest,count_mjsvm);
    count_adaboost = ensemble_classification_adaboost(Xtrain,ytrain,ytest, Xtest,count_adaboost);
    count_bagging = ensemble_classification_bagging(Xtrain,ytrain,ytest, Xtest,count_bagging);
    count_rf = ensemble_classification_random_forrest(Xtrain,ytrain,ytest, Xtest,count_rf);
end
count = [count_defimkl; count_mjsvm; count_adaboost; count_bagging; count_rf];
end