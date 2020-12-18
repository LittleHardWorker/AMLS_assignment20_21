### Import Lab
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import  validation_curve, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


### Func 1
def LogReg_Cross_Validation(X_tr, Y_tr, C_range):
# Do cross validation for Logistic Regression and get the optimal C

    # Compare and get the best k for the k-folds validation
    k_max = 0
    score_max = 0
    max_score_mean_list = []
    k_range = range(2, 11)
    for k in k_range:
        tr_score, tt_score = validation_curve(LogisticRegression(solver='liblinear'),
                                              X_tr, Y_tr, param_name='C', param_range=C_range,
                                              cv=k, scoring='f1')

        tr_score_mean = np.mean(tr_score, axis=1)
        tt_score_mean = np.mean(tt_score, axis=1)
        max_score_mean = max(tr_score_mean + tt_score_mean)
        max_score_mean_list.append(max_score_mean)

        if  max_score_mean > score_max:
            score_max = max(tr_score_mean + tt_score_mean)
            k_max = k

    print('The best k for k-folds validation in this case is: ', k_max)

    # Plate the result of different K
    plt.figure(figsize=(8, 6))
    plt.title('K-folds validation result in different value of K')
    plt.xlabel('Value of K')
    plt.ylabel('f1 score')
    plt.plot(k_range, max_score_mean_list, color = 'red')
    plt.show()

    # Get f1 score at different parameter C in given C_range
    tr_score, tt_score = validation_curve(LogisticRegression(solver='liblinear'),
                                          X_tr, Y_tr, param_name='C', param_range=C_range,
                                          cv=k_max, scoring='f1')

    # Calculate mean score
    tr_score_mean = np.mean(tr_score, axis=1)
    tt_score_mean = np.mean(tt_score, axis=1)
    tr_score_std = np.std(tr_score, axis=1)
    tt_score_std = np.std(tt_score, axis=1)

    # Plot C and score
    plt.figure(figsize=(8, 6))
    plt.title('Relationship between parameter C and score')
    plt.xlabel('C')
    plt.ylabel('f1 score')

    plt.semilogx(C_range, tr_score_mean, marker='o', label='train score', color='red')
    plt.semilogx(C_range, tt_score_mean, marker='x', label='test score', color='blue')

    plt.fill_between(C_range, tr_score_mean - tr_score_std,
                     tr_score_mean + tr_score_std, color='red', alpha=0.2)
    plt.fill_between(C_range, tt_score_mean - tt_score_std,
                     tt_score_mean + tt_score_std, color='blue', alpha=0.2)
    plt.legend()

    plt.show()

    print('The max training accuracy score is: ', max(tr_score_mean))

    # Choose the best C by max score
    total_score = tr_score_mean + tt_score_mean
    C_optimal = float(C_range[np.argwhere(total_score == max(total_score))])

    return C_optimal, k_max


### Func 2
def LogReg_Classification(X_tr, X_tt, Y_tr, Y_tt, X_F_tt, Y_F_tt, C_range):
# Get Classification result and print
    print("Logistic Regression!")

    # Get C_optimal from LogReg_Cross_Validation
    C_optimal, k_max = LogReg_Cross_Validation(X_tr, Y_tr, C_range)

    # Logistic Regression Model
    LogReg_Model = LogisticRegression(solver='liblinear', C=C_optimal)

    # Train the model
    LogReg_Model.fit(X_tr, Y_tr)
    Y_pred = LogReg_Model.predict(X_tt)

    # Get all kinds of score
    acc_score = accuracy_score(Y_tt, Y_pred)
    pre_score = precision_score(Y_tt, Y_pred)
    rec_score = recall_score(Y_tt, Y_pred)
    f1 = f1_score(Y_tt, Y_pred)

    # Get confusion matrix
    con_mat = confusion_matrix(Y_tt, Y_pred)

    # Print result
    print('The classify result of Logistic Regression is:')
    print('Optimal parameter C: ', C_optimal)
    print('Accuracy score: ', acc_score)
    print('precision score: ', pre_score)
    print('Recall score: ', rec_score)
    print('F1 score: ', f1)
    # print(con_mat)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.title('Confusion matrix for Logistic Regression')
    plt.grid(False)
    plt.imshow(con_mat, cmap='jet')
    plt.colorbar()
    plt.show()

    # Calculate the final accuracy for the new datasets
    print('----------------------------------------')
    print('For the final test with new dataset:')
    Y_F_pred = LogReg_Model.predict(X_F_tt)
    acc_F_score = accuracy_score(Y_F_tt, Y_F_pred)

    print('Accuracy score: ', acc_F_score)


    return k_max