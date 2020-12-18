### Import Lab
import pandas as pd
import os

### Func 1
def Label_Reading(label_path, label_columns):
# Reading .csv file to get the label information

    # Read .csv file
    df_label = pd.read_csv(label_path)
    df_label.columns = ['label'] # rename for convenience

    # Split data
    df_label = df_label['label'].str.split('\t', expand = True)

    # Rename for each column
    df_label.columns = label_columns

    return df_label


### Func 2
def Name_Generate(task_type):
# Generate different key name according task type
# Analyse task type and assign for the path variate
    if task_type is 'A_FFE':
        label_columns = ['Number', 'Name', 'Gender', 'Smiling']
        folder_name = 'celeba'

    elif task_type in ['B_FFE', 'B1_Face_DIP', 'B2_Eye_DIP_sg', 'B2_Eye_DIP']:
        label_columns = ['Number', 'Eye_Color', 'Face_Shape', 'Name']
        folder_name = 'cartoon_set'

    # For the test part
    elif task_type is 'A_FFE_test':
        label_columns = ['Number', 'Name', 'Gender', 'Smiling']
        folder_name = 'celeba_test'

    elif task_type in ['B_FFE_test', 'B1_Face_DIP_test', 'B2_Eye_DIP_sg_test', 'B2_Eye_DIP_test']:
        label_columns = ['Number', 'Eye_Color', 'Face_Shape', 'Name']
        folder_name = 'cartoon_set_test'

    else:
        print('Error: There is no task type ', task_type, '!')
        return

    save_file_name = 'X&Y_' + task_type + '.pkl'

    # Integrate name together
    name_list = [folder_name, save_file_name]

    return name_list, label_columns


### Func 3
def Path_Generate(task_type):
# Generate all kinds of needed paths

    # Get key name of path
    name_list, label_columns = Name_Generate(task_type)

    # Get current path
    current_path = os.getcwd()

    # Generate paths
    label_path = os.path.join(current_path, 'datasets', name_list[0], 'labels.csv')
    img_folder_path = os.path.join(current_path, 'datasets', name_list[0], 'img')
    save_path = os.path.join(current_path, 'datasets', name_list[0], name_list[1])

    # Put path in a list
    path_list = [label_path, img_folder_path, save_path]
    # Get
    return path_list, label_columns