
from tkinter import Image

import cv2

from google.colab.patches import cv2_imshow

import numpy as np

import cv2
from google.colab import files

uploaded = files.upload()
filename = list(uploaded.keys())[0]

image = cv2.imread(filename)


from google.colab.patches import cv2_imshow
cv2_imshow(image)


print(image.shape)


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2_imshow(gray_image)

print(gray_image.shape)


hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2_imshow(hsv_image)

print(hsv_image.shape)


canny_image = cv2.Canny(gray_image, 150, 200) 

kernel = np.ones((1, 1), np.uint8) 
erode_image = cv2.erode(canny_image, kernel, iterations = 1)

kernel1 = np.ones((5, 5), np.uint8)
dilate_image = cv2.dilate(canny_image, kernel1, iterations=1)

hori_display = np.concatenate((canny_image, erode_image, dilate_image), axis=1)
cv2_imshow(hori_display)



import cv2
import numpy as np
from google.colab import files
from google.colab.patches import cv2_imshow

# Upload image
uploaded = files.upload()
filename = list(uploaded.keys())[0]

image = cv2.imread(filename)

# Make sure image is in the correct format
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


denoise_image = cv2.fastNlMeansDenoisingColored(
    image, None, 20, 20, 7, 15
)

hori_display = np.concatenate((image, denoise_image), axis=1)
cv2_imshow(hori_display)