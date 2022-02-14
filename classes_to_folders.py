import pandas as pd
import os
import numpy as np
import shutil
import sys

dst_path='C:/Users/Andreas/Documents/Data/pneumonia/pneumonia_dataset/'
src_path='C:/Users/Andreas/Documents/Data/pneumonia/sized_train/'
dataset =  pd.read_csv('../../Data/pneumonia/labels_train.csv', dtype=str)

file_names = dataset['file_name']
img_labels = dataset['class_id']

class_ids= np.unique(dataset['class_id'])

for class_id in class_ids:
    if not os.path.exists(dst_path + 'class_'+ class_id):
        os.makedirs(dst_path + 'class_'+ class_id)

for f in range(len(file_names)):
    current_img = file_names[f]
    current_label = img_labels[f]

    dst_class_folder= dst_path+'class_'+current_label
    src_image_path= src_path+current_img
    
    try:
        shutil.copy(src_image_path, dst_class_folder)
        print("sucessful")
    except IOError as e:
        print('Unable to copy file {} to {}'
              .format(src_image_path, dst_class_folder))
    except:
        print('When try copy file {} to {}, unexpected error: {}'
              .format(src_image_path, src_image_path, sys.exc_info()))
    