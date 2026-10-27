import cv2
import numpy as np
from google.colab import files
from google.colab.patches import cv2_imshow

uploaded = files.upload()
filename = list(uploaded.keys())[0]

img = cv2.imread(filename)
cv2_imshow(img)


hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2_imshow(hsv_img)


lower_blue = np.array([65, 0, 0])
upper_blue = np.array([110, 255, 255])
# creating mask for only blue color
mask = cv2.inRange(hsv_img, lower_blue, upper_blue)
cv2_imshow(mask)

result = cv2.bitwise_and(img, img, mask = mask)
cv2_imshow(result)


lower_red = np.array([0, 100, 20])
upper_red = np.array([10, 255, 255])
# creating mask for only red color
mask = cv2.inRange(hsv_img, lower_red, upper_red)
cv2_imshow(mask)


result = cv2.bitwise_and(img, img, mask = mask)
cv2_imshow(result)


lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])
# creating mask for only yellow color
mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
cv2_imshow(mask)


result = cv2.bitwise_and(img, img, mask = mask)
cv2_imshow(result)



lower_green = np.array([50, 100, 100])
upper_green = np.array([75, 255, 255])
# creating mask for only green color
mask = cv2.inRange(hsv_img, lower_green, upper_green)
cv2_imshow(mask)

result = cv2.bitwise_and(img, img, mask = mask)
cv2_imshow(result)