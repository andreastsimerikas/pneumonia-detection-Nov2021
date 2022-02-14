import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.densenet import preprocess_input 

b_size = 10
img_s = 224
input_shape = (img_s, img_s, 3)
optim_1 = Adam(learning_rate=1e-4)
optim_2 = SGD(learning_rate=0.0003, momentum=0.9)
n_classes = 3
n_epochs = 30

train_path = '../../Data/pneumonia/pneumonia_dataset_folders/oversampled/train'
val_path = '../../Data/pneumonia/pneumonia_dataset_folders/oversampled/val'

tdatagen = ImageDataGenerator()
vdatagen = ImageDataGenerator()                 

train_batches = tdatagen.flow_from_directory(
    directory=train_path,
    batch_size=b_size,
    shuffle=True,
    seed=42,
    class_mode="categorical",
    target_size=(img_s,img_s)
)

valid_batches = vdatagen.flow_from_directory(
    directory=val_path,
    batch_size=b_size,
    shuffle=True,
    seed=42,
    class_mode="categorical",
    target_size=(img_s,img_s),
)

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Input 
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.initializers import *

def create_model(input_shape, n_classes, optimizer):

    conv_base = DenseNet121(include_top=False,
                            weights='imagenet',
                            input_shape=input_shape)

    inputs = Input(input_shape)
    prep = preprocess_input(inputs)
    top_model = conv_base(prep)
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(512, activation='relu', kernel_initializer="glorot_normal")(top_model)
    top_model = Dropout(0.5)(top_model)
    top_model = Dense(128, activation='relu', kernel_initializer="glorot_normal")(top_model)
    top_model = Dropout(0.3)(top_model)
    top_model = Dense(64, activation='relu', kernel_initializer="glorot_normal")(top_model)
    output = Dense(n_classes, activation='softmax')(top_model)
    
    model = Model(inputs, output)

    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model
    

model = create_model(input_shape, n_classes, optim_1)

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_batches.classes), y=train_batches.classes)
cw = dict(zip(np.unique(train_batches.classes), weights))
print(cw)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler

def scheduler(epoch, lr):
    
    if epoch > 10:
        lr=1e-7
        print(lr)
        return lr
    elif epoch > 7:
        lr=1e-6
        print(lr)
        return lr
    elif epoch > 2:
        lr=1e-5
        print(lr)
        return lr
    else:
        return lr

early= EarlyStopping(monitor='val_loss', mode='min', patience=4)
#lr_Plateau= ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1,factor=0.8, min_lr=0.0000001)
lr_scheduler = LearningRateScheduler(scheduler)
model_checkpoint= ModelCheckpoint(
    filepath='pneumonia.h5',
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1)

callbacks_list= [model_checkpoint,lr_scheduler]

history = model.fit(train_batches,
                    epochs=n_epochs,
                    validation_data=valid_batches,
                    callbacks=callbacks_list,
                    class_weight=cw,
                    verbose=1)