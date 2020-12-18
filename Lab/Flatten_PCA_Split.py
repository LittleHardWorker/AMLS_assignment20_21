### Import Lab
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA


### Func 1
def Feature_Flatten(df, y_name):
    # Split feature to each columns and get label base on y_name

    # Flatten each row in df['Feature'] to an array
    df['Feature'] = [np.array(df['Feature'][i]).flatten() for i in range(df.shape[0])]

    # Split each row to different columns
    X = pd.DataFrame([])
    col_num = df['Feature'][0].shape[0]
    for i in range(col_num):
        X[str(i)] = df['Feature'].map(lambda x: x[i])

    # Get label Y
    Y = df[y_name]

    # Change str to int
    Y = [int(i) for i in Y]

    return X, Y


### Func 2
def Feature_Split_Trans(X, X_Final, Y, Y_Final, test_size):
    # Split feature into train and test, then transform the feature

    # Feature split
    X_tr, X_tt, Y_tr, Y_tt = train_test_split(X, Y, test_size=test_size, random_state=0)

    # Feature regularization and transform
    scaler = MaxAbsScaler()  # because X∈R，so using MaxAbsScaler rather than MinMaxScaler
    X_tr = scaler.fit_transform(X_tr)
    X_tt = scaler.transform(X_tt)

    X_F_tt = scaler.transform(X_Final)
    Y_F_tt = Y_Final

    return X_tr, X_tt, Y_tr, Y_tt, X_F_tt, Y_F_tt


### Func 3
def Feature_PCA(X, X_Final, var_num, variance_ratio=0.99):
    # Choose some feature base on teh variance_ratio by PCA

    pca = PCA(n_components=variance_ratio)
    X_PCA = pca.fit_transform(X)
    X_Final_PCA = pca.transform(X_Final)

    feature_num = X_PCA.shape[1]
    print('PCA: ', feature_num, ' features have been chosen.')

    return X_PCA, X_Final_PCA


### Func 4
def Label_Reshape(Y, label_num):
# Reshape label Y to adopt NN format
    Y_new = np.zeros([len(Y), label_num])
    for i in range(len(Y)):
        Y_new[i, int(Y[i])] = 1

    return Y_new


### Func 5
def Flatten_PCA_Split(df, df_Final, label_name, variance_ratio, test_size, Y_reshape_flag = False):
# Total function

    X, Y = Feature_Flatten(df, label_name)
    X_Final, Y_Final = Feature_Flatten(df_Final, label_name)


    X_PCA, X_Final_PCA = Feature_PCA(X, X_Final, X.shape[1], variance_ratio)

    if Y_reshape_flag is True:
        Y = Label_Reshape(Y, 5)
        Y_Final = Label_Reshape(Y_Final, 5)

    X_tr, X_tt, Y_tr, Y_tt, X_F_tt, Y_F_tt = Feature_Split_Trans(X_PCA, X_Final_PCA, Y, Y_Final, test_size)





    return X_tr, X_tt, Y_tr, Y_tt, X_F_tt, Y_F_tt
