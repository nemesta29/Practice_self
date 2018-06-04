import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('dataset-sdcnd-capstone/data/real_training_data/green/left0293.jpg')
# img = cv2.imread('dataset-sdcnd-capstone/data/real_training_data/red/left0142.jpg')
img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
s_channel = img_hls[:,:,2]
# l_channel = img_hls[:,:,1]
s_bin = np.zeros_like(s_channel)
s_bin[(s_channel>220) & (s_channel<=255)] = 1
mask=np.zeros_like(s_bin)
mask2 = np.zeros_like(s_bin)
ignore_mask = 255
imgshape=img.shape
vertices1=np.array([[(350,500),(350, 420), (490, 420), (800,500)]], dtype=np.int32)
vertices2=np.array([[(350,450),(350, 350), (490, 350), (800,450)]], dtype=np.int32)
cv2.fillPoly(mask,vertices1,ignore_mask)
cv2.fillPoly(mask2,vertices2,ignore_mask)
green_bin = cv2.bitwise_and(s_bin, mask)
red_bin = cv2.bitwise_and(s_bin, mask2)

red = np.nonzero(red_bin)
green = np.nonzero(green_bin)
sum_red = 0
sum_green = 0
for i in green_bin[green]:
    sum_green+=i
for i in red_bin[red]:
    sum_red+=i    
if sum_red>sum_green:
    print('Red')
else:
    print('Green')