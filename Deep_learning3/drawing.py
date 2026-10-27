import numpy as np # linear algebra
import pandas as pd

import cv2
import matplotlib.pyplot as plt

emptyImg = np.zeros(shape=(512, 512, 3), dtype=np.int16)

plt.imshow(emptyImg)

cv2.rectangle(emptyImg, pt1=(380, 10), pt2=(500, 150), color=(0, 255, 0), thickness=10)

plt.imshow(emptyImg)

cv2.rectangle(emptyImg, pt1=(200, 200), pt2=(300, 300), color=(0, 0, 255), thickness=-1)
plt.imshow(emptyImg)

cv2.circle(emptyImg, center=(100, 100), radius=50, color=(255, 0, 0), thickness=10)
plt.imshow(emptyImg)

cv2.circle(emptyImg, center=(400, 400), radius=50, color=(255, 0 , 0), thickness=-1)
plt.imshow(emptyImg)



cv2.line(emptyImg, pt1=(0, 0), pt2=(512, 512), color=(255, 255, 255), thickness=10)

plt.imshow(emptyImg)

emptyImg2 = np.zeros(shape=(512, 512, 3), dtype=np.int16)

plt.imshow(emptyImg2)

emptyImg2 = emptyImg2.astype(np.uint8)
cv2.putText(emptyImg2,text='Image',org=(20,250),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=4,color=(255,255,255),thickness=3,lineType=cv2.LINE_AA)
plt.imshow(emptyImg2)