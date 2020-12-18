### Import Lab
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

### Func 1
def SVM_Cross_Validation(X_tr, Y_tr, parameters_range, k_max):
# Do cross validation for SVM and get the optimal parameters

    # Choose the best parameters by GridSearchCV
    multi_param_optimal = GridSearchCV(svm.SVC(), parameters_range,
                                       n_jobs=-1, cv=k_max, scoring='f1')
    multi_param_optimal.fit(X_tr, Y_tr)

    # Print best parameter and score
    print('The best parameter is: %s,' % multi_param_optimal.best_params_)
    print('score is: %f.' % multi_param_optimal.best_score_)

    # Print parameter optimal process
    mean_test_score_list = multi_param_optimal.cv_results_['mean_test_score']
    params_list = multi_param_optimal.cv_results_['params']
    for score, param in zip(mean_test_score_list, params_list):
        print('Score:  %f,  Param:  %r.' % (score, param))

    return multi_param_optimal


### Func 2
def SVM_Classification(X_tr, X_tt, Y_tr, Y_tt, X_F_tt, Y_F_tt, parameters_range, k_max):
# Get Classification result and print
    print('SVM!')

    # Get best parameter
    optimal_param_model = SVM_Cross_Validation(X_tr, Y_tr, parameters_range, k_max)

    svm_model = optimal_param_model.best_estimator_
    Y_pred = svm_model.predict(X_tt)

    # Get all kinds of score
    acc_score = accuracy_score(Y_tt, Y_pred)
    pre_score = precision_score(Y_tt, Y_pred)
    rec_score = recall_score(Y_tt, Y_pred)
    f1 = f1_score(Y_tt, Y_pred)

    # Get confusion matrix
    con_mat = confusion_matrix(Y_tt, Y_pred)

    # Print result
    print('The classify result of SVM is:')
    print('Optimal parameter: ', optimal_param_model.best_params_)
    print('Accuracy score: ', acc_score)
    print('precision score: ', pre_score)
    print('Recall score: ', rec_score)
    print('F1 score: ', f1)
    # print(con_mat)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.grid(False)
    plt.imshow(con_mat, cmap='jet')
    plt.colorbar()

    # For the final test with new datasets
    print('----------------------------------------')
    print('For the final test with new dataset:')
    Y_F_pred = svm_model.predict(X_F_tt)
    acc_F_score = accuracy_score(Y_F_tt, Y_F_pred)
    print('Accuracy score: ', acc_F_score)