### Import Lab
from Lab.Neural_Network import NN_Session
from Lab.Flatten_PCA_Split import Flatten_PCA_Split

### Func 1
def Eye_Color_Recognition(df, df_Final):
# Total function for task B2
    print('Task B2')

    label_name = 'Eye_Color'
    model_name = 'NN_B2_DIP'

    X_tr, X_tt, Y_tr, Y_tt, X_F_tt, Y_F_tt = Flatten_PCA_Split(df, df_Final, label_name, 0.90, 0.25, Y_reshape_flag = True)

    feature_num = X_tr.shape[1]
    label_num = 5
    training_epochs = X_tr.shape[0]

    NN_Session(feature_num, label_num, model_name, X_tr, X_tt, Y_tr, Y_tt, X_F_tt, Y_F_tt, training_epochs = training_epochs)