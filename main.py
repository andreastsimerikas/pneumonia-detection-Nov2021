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

batch= 25

traindf = pd.read_csv('pneumonia/labels_train.csv', dtype=str)
datagen = ImageDataGenerator(rescale=1./255.,validation_split=0.25)
test_datagen = ImageDataGenerator(rescale=1./255.)

train_path = 'pneumonia/train_images'
test_path = 'pneumonia/test_images'

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

test_batches = test_datagen.flow_from_directory(
    directory=test_path,
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42,
    target_size=(224, 224)
)

plt.figure(figsize=(12, 12))
for i in range(0, 25):
    plt.subplot(5, 5, i+1)
    for images, labels in train_batches:
        image = images[1]        
        plt.title(labels[1])
        plt.axis('off')
        plt.imshow(np.squeeze(image),cmap='gray',interpolation='nearest')
        break
plt.tight_layout()
plt.show()

def create_model(input_shape, n_classes, optimizer='rmsprop', fine_tune=0):
    """
    Compiles a model integrated with VGG16 pretrained layers
    
    input_shape: tuple - the shape of input images (width, height, channels)
    n_classes: int - number of classes for the output layer
    optimizer: string - instantiated optimizer to use for training. Defaults to 'RMSProp'
    fine_tune: int - The number of pre-trained layers to unfreeze.
                If set to 0, all pretrained layers will freeze during training
    """
    
    # Pretrained convolutional layers are loaded using the Imagenet weights.
    # Include_top is set to False, in order to exclude the model's fully-connected layers.
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

    # Create a new 'top' of the model (i.e. fully-connected layers).
    # This is 'bootstrapping' a new top_model onto the pretrained layers.
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

input_shape = (224, 224, 3)
optim_1 = Adam(learning_rate=0.001)
n_classes=3

n_steps = train_batches.samples // batch
n_val_steps = valid_batches.samples // batch
n_epochs = 1

# First we'll train the model without Fine-tuning
vgg_model = create_model(input_shape, n_classes, optim_1, fine_tune=0)

vgg_history = vgg_model.fit(train_batches,
                    epochs=n_epochs,
                    validation_data=valid_batches,
                    steps_per_epoch=n_steps,
                    validation_steps=n_val_steps,
                    verbose=1)   

pred = vgg_model.predict(test_batches)

pred_dt = pd.DataFrame(pred)
print(pred_dt)


