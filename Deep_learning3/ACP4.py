
from symtable import Function

from streamlit import image

import cv2
from google.colab import files
from google.colab.patches import cv2_imshow
import numpy as np



uploaded = files.upload()
filename = list(uploaded.keys())[0]

img = cv2.imread(filename)
cv2_imshow(img)



gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2_imshow(gray_img)


ret, thresh = cv2.threshold(gray_img,50,255,1)
cv2_imshow(thresh)

contours,h = cv2.findContours(thresh,1,2)

for cnt in contours:
  approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
  n = len(approx)
  if n==6:
    print("We have a hexagon here")
    cv2.drawContours(img,[cnt],0,255,10) 
  elif n==3:
    print("We found a triangle")
    cv2.drawContours(img,[cnt],0,(0,255,0),3)
  elif n>9:
    print("We found a circle")
    cv2.drawContours(img,[cnt],0,(0,255,255),3)
  elif n==4:
    print("We found a square")
    cv2.drawContours(img,[cnt],0,(255,255,0),3)
cv2_imshow(img)