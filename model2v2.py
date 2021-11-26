import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Conv2D
from tensorflow.keras.optimizers.schedules import ExponentialDecay

b_size = 42

traindf = pd.read_csv('../../Data/pneumonia/labels_train.csv', dtype=str)
datagen = ImageDataGenerator(rescale=1./255.,validation_split=0.30)
test_datagen = ImageDataGenerator(rescale=1./255.)

test_path = '../../Data/pneumonia/test_images/images'
train_path = '../../Data/pneumonia/train_images/images'

train_batches = datagen.flow_from_dataframe(
    dataframe=traindf,
    directory=train_path,
    x_col="file_name",
    y_col="class_id",
    batch_size=b_size,
    subset='training',
    shuffle=True,
    seed=42,
    class_mode="categorical",
    target_size=(224,224)
)

valid_batches = datagen.flow_from_dataframe(
    dataframe=traindf,
    directory=train_path,
    x_col="file_name",
    y_col="class_id",
    subset="validation",
    batch_size=b_size,
    shuffle=True,
    seed=42,
    class_mode="categorical",
    target_size=(224,224)
)

test_images = []
i=1
for img in os.listdir(test_path):
    test_images.append(img)
    i+=1

submision = pd.DataFrame(test_images,columns=['file_name'])

test_batches = test_datagen.flow_from_dataframe(
    dataframe=submision,
    directory=test_path,
    x_col="file_name",
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42,
    target_size=(224, 224)
)

def create_model(input_shape, n_classes, optimizer='rmsprop', fine_tune=0):
  
    conv_base = VGG16(include_top=False,
                        weights='imagenet',
                        input_shape=input_shape)
    
    for layer in conv_base.layers:
        layer.trainable = False

    top_model = conv_base.output
    top_model = Flatten(name="flatten")(top_model)
    output_layer = Dense(n_classes, activation='softmax')(top_model)
    
    model = Model(inputs=conv_base.input, outputs=output_layer)

    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def create_my_model(input_shape, n_classes, optimizer):
    input=Input(shape=input_shape)

    conv_layer = Conv2D(filters=16, kernel_size=3,
           activation='relu',
           padding="same")

    x1 = conv_layer(input)
    x2 = Conv2D(filters=20, kernel_size=3,
           activation='relu',
           padding="same")(x1)
    x3 = Conv2D(filters=32, kernel_size=3,
            activation='relu',
            padding="same")(x2)
    x = Flatten()(x3)  
    out = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=input, outputs=out)

    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model              


#############################################################

lr_schedule= ExponentialDecay(1e-3,
                            decay_steps=100000,
                            decay_rate=0.96)

input_shape= (224, 224, 3)
optim_1= Adam(learning_rate=1e-3)
n_classes= 3
n_epochs= 10

model = create_model(input_shape, n_classes, optim_1)

history = model.fit(train_batches,
                    epochs=n_epochs,
                    validation_data=valid_batches,
                    verbose=1)   


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()  

predictions = model.predict(test_batches,verbose=1)

preds = np.argmax(predictions, axis=-1) 

submision['class_id'] = preds
submision.to_csv('submision.csv',index=False)

