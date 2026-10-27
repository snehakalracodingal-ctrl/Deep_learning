importing libraries


from tkinter import Image

import cv2
from google.colab import files
from google.colab.patches import cv2_imshow


uploaded = files.upload()
filename = list(uploaded.keys())[0]

face_cascade = cv2.CascadeClassifier(filename)

print("Cascade classifier loaded successfully")


uploaded = files.upload()
filename = list(uploaded.keys())[0]

img = cv2.imread(filename)
cv2_imshow(img)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2_imshow(gray_img)


face = face_cascade.detectMultiScale(gray_img, 1.3, 5)
print(face) 

for(x, y, w, h) in face:
  cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3) 

cv2_imshow(img)