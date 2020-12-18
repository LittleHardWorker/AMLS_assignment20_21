####################################################################################################
# main.py
# Target: Pre-process for the raw data and run each task
####################################################################################################

### Import Lab
import pandas as pd
import os

from Lab.Face_Feature_Extract import Face_Feature_PreProc
from Lab.DIP_ROI import ROI_PreProc
from A1.A1 import Gender_Detection
from A2.A2 import Emotion_Detection
from B1.B1 import Face_Shape_Recognition
from B2.B2 import Eye_Color_Recognition


# Prepare features and labels for task A
Face_Feature_PreProc('A_FFE')

# Prepare features and labels for task B1
ROI_PreProc('B1_Face_DIP', 'Face')

# Prepare features and labels for task B2
ROI_PreProc('B2_Eye_DIP', 'Eye')

# Prepare feature for the final test data
Face_Feature_PreProc('A_FFE_test')
ROI_PreProc('B1_Face_DIP_test', 'Face')
ROI_PreProc('B2_Eye_DIP_test', 'Eye')

# Get current path
current_path = os.getcwd()
### Task A1
df_A1 = pd.read_pickle( os.path.join(current_path, 'datasets', 'celeba', 'X&Y_A_FFE.pkl' ) )
df_A1_Final = pd.read_pickle( os.path.join(current_path, 'datasets', 'celeba_test', 'X&Y_A_FFE_test.pkl' ) )
Gender_Detection(df_A1, df_A1_Final)

### Task A2
df_A2 = pd.read_pickle( os.path.join(current_path, 'datasets', 'celeba', 'X&Y_A_FFE.pkl' ) )
df_A2_Final = pd.read_pickle( os.path.join(current_path, 'datasets', 'celeba_test', 'X&Y_A_FFE_test.pkl' ) )
Emotion_Detection(df_A2, df_A2_Final)


### Task B1
df_B1 = pd.read_pickle( os.path.join(current_path, 'datasets', 'cartoon_set', 'X&Y_B1_Face_DIP.pkl' ) )
df_B1_Final = pd.read_pickle( os.path.join(current_path, 'datasets', 'cartoon_set_test', 'X&Y_B1_Face_DIP_test.pkl' ) )
Face_Shape_Recognition(df_B1, df_B1_Final)

### Task B2
df_B2 = pd.read_pickle( os.path.join(current_path, 'datasets', 'cartoon_set', 'X&Y_B2_Eye_DIP.pkl' ) )
df_B2_Final = pd.read_pickle( os.path.join(current_path, 'datasets', 'cartoon_set_test', 'X&Y_B2_Eye_DIP_test.pkl' ) )
Eye_Color_Recognition(df_B2, df_B2_Final)






# ##### Not used
# ### For the old version of task B2
# ROI_PreProc('B2_Eye_DIP_sg', 'Eye_sg')
# df_B2 = pd.read_pickle( os.path.join(current_path, 'datasets', 'cartoon_set', 'X&Y_B2_Eye_DIP_sg.pkl' ) )
# df_B2_Final = pd.read_pickle( os.path.join(current_path, 'datasets', 'cartoon_set_test', 'X&Y_B2_Eye_DIP_sg_test.pkl' ) )
# Eye_Color_Recognition(df_B2, df_B2_Final)




