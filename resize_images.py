import cv2
import glob
import numpy as np

images_path = glob.glob('pneumonia/train_images/images/*.jpg')
desired_size = 2000

for img_path in images_path:
    
    img = cv2.imread(img_path)  

    old_size = img.shape[:2]
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    img = cv2.resize(img, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]

    img_padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
    filename = img_path.split('\\')[1]
    cv2.imwrite('pneumonia/sized_train/'+filename,img_padded)





