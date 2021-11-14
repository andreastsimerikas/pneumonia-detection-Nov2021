import numpy as np
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

train_path = 'pneumonia/train_images'
test_path = 'pneumonia/test_images'

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224), classes=['no disease', 'bacterial pneumonia', 'viral pneumonia'], batch_size=1)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224), classes=['no disease', 'bacterial pneumonia', 'viral pneumonia'], batch_size=10)

fig,ax = plt.subplots(nrows=1, ncols=10, fig_size=(15,15))

for i in range(10):
    img = next(train_batches)[0].astype('uint8')
    ax[i].imshow(img)
    ax[i].axis('off')