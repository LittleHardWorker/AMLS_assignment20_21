
### Import Lab
import os
import cv2
import pandas as pd

from Lab.Pre_Process import Label_Reading, Path_Generate


### Func 1
def Sunglasses_Judgement(gray):
# Judge whether the person in this picture is wearing a pair of sunglasses

    # Get right eye area image ROI
    img_ROI_sg = gray[255:275, 185:225]

    # The number of rows and columns of ROI
    ROW = img_ROI_sg.shape[0]
    COL = img_ROI_sg.shape[1]

    # Detect white color. If in the below range, it is white.
    color_l = 220
    color_h = 255

    # Count the number of white pixel
    C = 0

    # Judgement
    for i in range(ROW):
        for j in range(COL):
            if color_l <= img_ROI_sg[i, j] <= color_h:
                C += 1
    if C / (ROW * COL) >= 0.1:
        #         print('This is an eye')
        return True
    else:
        #         print('This is a sunglasses')
        return False


### Func 2
def ROI_Definition(img, ROI_type):
# Definition the area of ROI

    if ROI_type is 'Eye_sg':
        img_ROI = img[250:275, 195:215]

    elif ROI_type is 'Eye': # For no sunglasses judgement
        img_ROI = img[255:275, 185:225]

    elif ROI_type is 'Face':
        gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
        img_ROI = gray[250:400, 150:350]
        img_ROI = cv2.resize(img_ROI, (0, 0), fx=0.5, fy=0.5)

    else:
        print('Error: There is no ROI type ', ROI_type, '!')

    return img_ROI


### Func 3
def Get_ROI(img_path, ROI_type):
# Read a picture, pre-process it and get ROI

    # Read image
    img = cv2.imread(img_path)

    # Choose different ROI base on need_preproc
    # Get the result of judgement
    if ROI_type is 'Eye_sg':
        gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
        img_flag = Sunglasses_Judgement(gray)
        if img_flag is True:
            img_ROI = ROI_Definition(img, ROI_type)
        else:
            img_ROI = img_flag
    else:
        img_flag = True
        img_ROI = ROI_Definition(img, ROI_type)

    return img_flag, img_ROI


### Func 4
def ROI_Extract(img_folder_path, ROI_type):
# Get a list storing img_ROI information for a whole folder

    # Get file name in the folder and sort
    img_list = os.listdir(img_folder_path)
    img_list.sort(key = lambda x:int(x[0:-4]))
    #print(img_list)

    img_flag_list = []
    img_ROI_list = []
    for img_name in img_list:
        print(img_name)
        img_path = os.path.join(img_folder_path, img_name)
        img_flag, img_ROI = Get_ROI(img_path, ROI_type)
        img_flag_list.append(img_flag)
        img_ROI_list.append(img_ROI)

    # Generate dataframe
    df_img_ROI = pd.DataFrame({'Feature': img_ROI_list, 'Flag': img_flag_list})

    return df_img_ROI


### Func 5
def ROI_PreProc(task_type, ROI_type):
    # Get task type and return dataframe

    # Get path information
    path_list, label_columns = Path_Generate(task_type)

    # Get label and raw face feature
    df_label = Label_Reading(path_list[0], label_columns)
    df_img_ROI = ROI_Extract(path_list[1], ROI_type)

    # # For test
    # df_label = df_label.iloc[0:50, :]

    # Wipe of the image data which aren't selected, and combine two dataframe together
    df_img_ROI_picked = df_img_ROI[df_img_ROI['Flag']]
    df_label_picked = df_label[df_img_ROI['Flag']]
    df_img_ROI_picked = pd.concat([df_img_ROI_picked, df_label_picked], axis=1)

    # Reset the number
    df_img_ROI_picked = df_img_ROI_picked.reset_index(drop=True)

    # Save file for backup
    df_img_ROI_picked.to_pickle(path_list[2])

    return df_img_ROI_picked