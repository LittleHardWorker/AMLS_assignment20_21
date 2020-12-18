### Import Lab
import numpy as np
from Lab.Flatten_PCA_Split import Flatten_PCA_Split
from Lab.Logistic_Regression import LogReg_Classification
from Lab.SVM import SVM_Classification

### Func 1
def Gender_Detection(df, df_Final):
# Total function for task A1
    print('Task A1')

    # Set parameter of Flatten_PCA_Split
    label_name = 'Gender'
    variance_ratio = 0.99
    test_size = 0.25

    # Get X_tr, X_tt, Y_tr, Y_tt, X_F_tt, Y_F_tt from Flatten_PCA_Split
    X_tr, X_tt, Y_tr, Y_tt, X_F_tt, Y_F_tt = Flatten_PCA_Split(df, df_Final, label_name, variance_ratio, test_size)

    # Set parameters range
    C_range = np.logspace(-3, 3, 7)
    gamma_range = np.logspace(-3, 3, 7)
    parameters_range = [{'kernel': ['linear'], 'C': C_range},
                        {'kernel': ['poly'], 'C': C_range, 'gamma': gamma_range},
                        {'kernel': ['rbf'], 'C': C_range, 'gamma': gamma_range},
                        {'kernel': ['sigmoid'], 'C': C_range, 'gamma': gamma_range}]


    # Get classification result from LogReg_Classification
    k_max = LogReg_Classification(X_tr, X_tt, Y_tr, Y_tt, X_F_tt, Y_F_tt, C_range)

    # Get classification result from SVM_Classification
    SVM_Classification(X_tr, X_tt, Y_tr, Y_tt, X_F_tt, Y_F_tt, parameters_range, k_max)

