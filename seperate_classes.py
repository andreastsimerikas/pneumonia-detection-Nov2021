from __future__ import print_function
import pandas as pd
import shutil
import os
import sys

labels = pd.read_csv('pneumonia/labels_train.csv', dtype=str)
train_path = 'pneumonia/sized_train'
sep_train_path = 'pneumonia/sep_sized_train/class'

if not os.path.exists(sep_train_path):
    os.mkdir(sep_train_path)

for filename, class_name in labels.values:
    # Create subdirectory with `class_name`
    if not os.path.exists(sep_train_path + str(class_name)):
        os.mkdir(sep_train_path + str(class_name))
    src_path = train_path + '/' + filename 
    dst_path = sep_train_path + str(class_name) + '/' + filename 
    try:
        shutil.copy(src_path, dst_path)
        print("sucessful")
    except IOError as e:
        print('Unable to copy file {} to {}'
              .format(src_path, dst_path))
    except:
        print('When try copy file {} to {}, unexpected error: {}'
              .format(src_path, dst_path, sys.exc_info()))

