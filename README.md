# ensemble-classifier-comparison

The set of code contains Adaboost, Bagging, Random Forest and Majority Voting (with 5 Kernel-SVMs as base classifiers) and DeFIMKL (a non-linear combination of base classifiers) ensemble classifier algorithms.

main file: contains the datasets on which the algorithm needs to be run
ensemble_classification_common: splits the called dataset into training and test sets. Subsequently, it runs all the ensemble classifiers and returns the results to the main funtion.
