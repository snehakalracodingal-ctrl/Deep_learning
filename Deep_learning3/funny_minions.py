from gettext import install

import pip


pip install vcam

import cv2
import numpy as np
import math
from vcam import vcam,meshGen
import matplotlib.pyplot as plt
from google.colab import files


plt.figure(figsize=(20, 20))

uploaded = files.upload()
filename = list(uploaded.keys())[0]

img = cv2.imread(filename)
H,W = img.shape[:2]

c1 = vcam(H=H,W=W)

plane = meshGen(H,W)


plane.Z += 20*np.exp(-0.5*((plane.X*1.0/plane.W)/0.1)**2)/(0.1*np.sqrt(2*np.pi))
pts3d = plane.getPlane()

pts2d = c1.project(pts3d)
map_x,map_y = c1.getMaps(pts2d)

output = cv2.remap(img,map_x,map_y,interpolation=cv2.INTER_LINEAR)

plt.subplot(1, 2,1)
plt.title("Funny Mirror")
plt.imshow(cv2.cvtColor(np.hstack((img,output)), cv2.COLOR_BGR2RGB))


plt.figure(figsize=(20, 20))

img = cv2.imread("minions.jpg")
H,W = img.shape[:2]


c1 = vcam(H=H,W=W)


plane = meshGen(H,W)


plane.Z += 20*np.exp(-0.5*((plane.Y*1.0/plane.H)/0.1)**2)/(0.1*np.sqrt(2*np.pi))

pts3d = plane.getPlane()

pts2d = c1.project(pts3d)
map_x,map_y = c1.getMaps(pts2d)

output = cv2.remap(img,map_x,map_y,interpolation=cv2.INTER_LINEAR)

plt.subplot(1, 2,1)
plt.title("Funny Mirror")
plt.imshow(cv2.cvtColor(np.hstack((img,output)), cv2.COLOR_BGR2RGB))