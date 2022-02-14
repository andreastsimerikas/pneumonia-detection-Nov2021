import numpy as np
import os
import cv2
import glob

mean = 0
var = 10
sigma = var ** 0.5

images_path = glob.glob('../../Data/pneumonia/train_images/images/*.jpg')

for img_path in images_path:
    img = cv2.imread(img_path,0) 
    gaussian = np.random.normal(mean, sigma, img.shape) #  np.zeros((224, 224), np.float32)
    noisy_image = np.zeros(img.shape, np.float32)

    if len(img.shape) == 2:
        noisy_image = img + gaussian
    else:
        noisy_image[:, :, 0] = img[:, :, 0] + gaussian
        noisy_image[:, :, 1] = img[:, :, 1] + gaussian
        noisy_image[:, :, 2] = img[:, :, 2] + gaussian
    cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)

    noisy_image = noisy_image.astype(np.uint8)

    filename = os.path.basename(img_path)
    cv2.imwrite('../../Data/pneumonia/train_noised/'+filename,noisy_image)