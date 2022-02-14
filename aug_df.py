from numpy.lib.shape_base import column_stack
import pandas as pd
import numpy as np
import os

class0_path = '../../Data/pneumonia/pneumonia_dataset_folders_trimmed/train/class_0'
class1_path = '../../Data/pneumonia/pneumonia_dataset_folders_two_class/two_class_aug/class_1'
class2_path = '../../Data/pneumonia/pneumonia_dataset_folders_two_class/two_class_aug/class_2'

dataset = []

for img in os.listdir(class1_path):
    dataset.append([img,'1'])
for img in os.listdir(class2_path):
    dataset.append([img,'2'])

aug_df = pd.DataFrame(dataset,columns=['file_name','class_id'])

aug_df.to_csv('../../Data/pneumonia/2class_df.csv',index=False)