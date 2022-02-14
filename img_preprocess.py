import cv2 
import numpy as np
from matplotlib import pyplot as plt
import glob
import os

images_path = glob.glob('../../Data/pneumonia/train_images/images/*.jpg')

for img_path in images_path:
    
    img = cv2.imread(img_path,0) 
    #clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(5,5))
    #image = clahe.apply(img)
    image = cv2.equalizeHist(img)
    #image = cv2.bilateralFilter(image, 20, 20, 20)
    
    minVal, maxVal, minLoc, maxLoc= cv2.minMaxLoc(image)
    ret,thresh_img = cv2.threshold(image,minVal+0.93*(maxVal-minVal), 255, cv2.THRESH_BINARY) 
    morph = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, np.ones((10,10),np.uint8))
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, np.ones((10,10),np.uint8))
    morph = cv2.morphologyEx(morph, cv2.MORPH_DILATE, np.ones((10,10),np.uint8))
    
    new_mask = np.zeros_like(morph)   
                      
    labels, stats= cv2.connectedComponentsWithStats(morph, 4, cv2.CV_32S)[1:3]  
    if stats[1:, cv2.CC_STAT_AREA].size>0:
        largest_label= 1+ np.argmax(stats[1:, cv2.CC_STAT_AREA])      
        new_mask[labels!=largest_label] = 255
        #image= cv2.bitwise_and(img, new_mask)
        
        filename = os.path.basename(img_path)
        cv2.imwrite('../../Data/pneumonia/train_images_preprocessed/'+filename,image)
    else:
        filename = os.path.basename(img_path)
        cv2.imwrite('../../Data/pneumonia/train_images_preprocessed/'+filename,img)