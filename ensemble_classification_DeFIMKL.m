function [count_defimkl] = ensemble_classification_DeFIMKL(ytrain, Xtrain,...
    sigmas, reg_type, lambda, ytest, Xtest,count_defimkl)

    tic
%     model = defimkltrain( ytrain, Xtrain, sigmas, reg_type, lambda);
    model = defimkltrain_new( ytrain, Xtrain, sigmas, reg_type, lambda);
    [temp, ~, scores] = defimklpredict( ytest, Xtest, model);
    t = toc;

    [~,~,~,auc2] = perfcurve(ytest,scores,1);
    
    tp = temp(ytest==1);
    tp = length(tp(tp==1));
    fp = temp(ytest==-1);
    fp = length(fp(fp==1));
    fn = temp(ytest==1);
    fn = length(fn(fn==-1));
    tn = temp(ytest==-1);
    tn = length(tn(tn==-1));
    
    sens = tp/(tp+fn);
    spec = tn/(fp+tn);
    accuracy = sum( ytest == temp )/length(ytest);

    count_defimkl=[count_defimkl; accuracy auc2 sens spec t];
end