
### Import Lab
import numpy as np
import pandas as pd
import cv2
import dlib
import os

from Lab.Pre_Process import Label_Reading, Path_Generate


### Func 1
def Get_Face_Feature(img_path):
# Read a image, return its 68 face landmarks information
# Return None when there is no face has been detected

    # Set detector and predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # Read image
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(src = img, code = cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_set = detector(gray)
    face_set_points = []
    if len(face_set) == 0:# No face has been detected
        return None
    else:
        face = face_set[0]# only use the first face no matter how many faces has been detected.
        landmarks = predictor(image=gray, box=face)
        face_points = []
        for i in range(68):
            point = landmarks.part(i)-landmarks.part(27)
            face_points.append([point.x, point.y])# Regularize point information subjected to point(27)
        return face_points


### Func 2
def Face_Feature_Extract(img_folder_path):
    # Initialize for the dataframe of feature
    FF_list = []
    FF_index = []

    # Get image list from the folder
    img_list = os.listdir(img_folder_path)
    # Wipe of the last 4 characters('.png' or '.jpg') then sort by name
    img_list.sort(key=lambda x: int(x[0:-4]))

    # For test
    print(len(img_list))
    for img_name in img_list:
        print(img_name)  # Show progress
        img_path = os.path.join(img_folder_path, img_name)
        face_point = Get_Face_Feature(img_path)
        FF_list.append(face_point)
        FF_index.append(face_point is not None)

    FF_list = np.array(FF_list)  # For generate dataframe
    FF_index = np.array(FF_index)
    df_FF = pd.DataFrame({'Feature': FF_list, 'Flag': FF_index})

    return df_FF


### Func 3
def Face_Feature_PreProc(task_type):
# Get task type and return dataframe

    # Get path information
    path_list, label_columns = Path_Generate(task_type)

    # Get label and raw face feature
    df_label = Label_Reading(path_list[0], label_columns)
    df_FF = Face_Feature_Extract(path_list[1])

    # # For test
    # df_label = df_label.iloc[0:10, :]

    # Wipe of the image data which aren't selected, and combine two dataframe together
    df_FF_picked = df_FF[df_FF['Flag']]
    df_label_picked = df_label[df_FF['Flag']]
    df_FF_picked = pd.concat([df_FF_picked, df_label_picked], axis=1)

    # Reset the number
    df_FF_picked = df_FF_picked.reset_index(drop=True)

    # Save file for backup
    df_FF_picked.to_pickle(path_list[2])



