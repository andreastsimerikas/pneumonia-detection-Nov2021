import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
import os

batch= 10

traindf = pd.read_csv('pneumonia/labels_train.csv', dtype=str)
datagen = ImageDataGenerator(rescale=1./255.,validation_split=0.25)
test_datagen = ImageDataGenerator(rescale=1./255.)

train_path = 'pneumonia/train_images'
test_path = 'pneumonia/test_images/images'

train_batches = datagen.flow_from_dataframe(
    dataframe=traindf,
    directory=train_path,
    x_col="file_name",
    y_col="class_id",
    batch_size=batch,
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
    batch_size=batch,
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
    
    # Defines how many layers to freeze during training.
    # Layers in the convolutional base are switched from trainable to non-trainable
    # depending on the size of the fine-tuning parameter.
    if fine_tune > 0:
        for layer in conv_base.layers[:-fine_tune]:
            layer.trainable = False
    else:
        for layer in conv_base.layers:
            layer.trainable = False

    top_model = conv_base.output
    top_model = Flatten(name="flatten")(top_model)
    top_model = Dense(4096, activation='relu')(top_model)
    top_model = Dense(1072, activation='relu')(top_model)
    top_model = Dropout(0.2)(top_model)
    output_layer = Dense(n_classes, activation='softmax')(top_model)
    
    # Group the convolutional base and new fully-connected layers into a Model object.
    model = Model(inputs=conv_base.input, outputs=output_layer)

    # Compiles the model for training.
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

input_shape= (224, 224, 3)
optim_1= Adam(learning_rate=0.001)
n_classes= 3

n_steps= train_batches.samples // batch
n_val_steps= valid_batches.samples // batch
n_epochs= 1

vgg_model = create_model(input_shape, n_classes, optim_1, fine_tune=0)

vgg_history = vgg_model.fit(train_batches,
                    epochs=n_epochs,
                    validation_data=valid_batches,
                    steps_per_epoch=n_steps,
                    validation_steps=n_val_steps,
                    verbose=1)   


plt.plot(vgg_history.history['acc'])
plt.plot(vgg_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()  

predictions = vgg_model.predict(valid_batches,verbose=1)

preds = np.argmax(predictions, axis=-1) 

submision['class_id'] = preds
submision.to_csv('submision.csv',index=False)

