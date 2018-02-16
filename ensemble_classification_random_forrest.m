function [count_rf] = ensemble_classification_random_forrest(Xtrain,ytrain,...
    ytest, Xtest,count_rf)

tic
mdl=TreeBagger(100,Xtrain,ytrain,'oobpred','on', 'Method', 'classification','NVarToSample', 'all');
% mdl=TreeBagger(100,Xtrain,ytrain,'OOBPrediction','on', 'Method', 'classification',...
%     'NVarToSample', 'all');

oobErrorBaggedEnsemble = oobError(mdl);

[temp temp2] = predict(mdl, Xtest);
t = toc;

pred_label = str2double(temp);

scores=[];
for loop=1:size(temp2,1)
    if pred_label(loop,1)<0
        scores=[scores; -1*abs(temp2(loop,1))];
    else
        scores=[scores; abs(temp2(loop,2))];
    end
end
[~,~,~,auc2] = perfcurve(ytest,scores,1);

tp = pred_label(ytest==1);
tp = length(tp(tp==1));
fp = pred_label(ytest==-1);
fp = length(fp(fp==1));
fn = pred_label(ytest==1);
fn = length(fn(fn==-1));
tn = pred_label(ytest==-1);
tn = length(tn(tn==-1));

sens = tp/(tp+fn);
spec = tn/(fp+tn);

accuracy = sum( ytest == pred_label )/length(ytest);

count_rf=[count_rf; accuracy auc2 sens spec t];

end