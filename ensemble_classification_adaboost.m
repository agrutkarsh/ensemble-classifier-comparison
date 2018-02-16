function [count_adaboost] = ensemble_classification_adaboost(Xtrain,ytrain,...
    ytest, Xtest,count_adaboost)

    tic
    cv = fitensemble(Xtrain,ytrain,'AdaBoostM1',200,'Tree','type','classification');
    [temp temp2] = predict(cv, Xtest);
    t = toc;
           
    scores=[];
    for loop=1:size(temp2,1)
        if temp(loop,1)<0
            scores=[scores; -1*abs(temp2(loop,1))];
        else
            scores=[scores; abs(temp2(loop,2))];
        end
    end
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

    count_adaboost=[count_adaboost; accuracy auc2 sens spec t];
end


% end