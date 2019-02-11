import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import numpy as np 

image = mpimg.imread('E:/Courses/Udacity/[FCO] Udacity - Self-Driving Car Engineer v1.0.0/Part 01-Module 01-Lesson 04_Finding Lane Lines Project/CarND-LaneLines-P1/test_images/road.jpg')
plt.imshow(image)
plt.show()

image_copy = np.copy(image)
gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap = 'gray')
plt.show()

blur = cv2.GaussianBlur(gray,(5,5), 0)
plt.imshow(blur, cmap = 'gray')
plt.show()

low_threshold = 150
high_threshold = 300
canny = cv2.Canny(blur, low_threshold, high_threshold)
plt.imshow(canny, cmap = 'gray')
plt.show()

mask = np.zeros_like(canny)
ignore_mask_color = 255

imshape = image.shape
vertices = np.array([[(400, imshape[0]), (520, 378), (600, imshape[0])]])
poly = cv2.fillPoly(mask, vertices, ignore_mask_color)
plt.imshow(poly, cmap= 'gray')
plt.show()

masked_images = cv2.bitwise_and(canny, poly)
plt.imshow(masked_images, cmap='gray')
plt.show()

rho = 1
theta = np.pi/180
threshold = 2
min_line_length = 4
max_line_gap = 5
line_image = np.copy(image)*0
plt.imshow(line_image)
plt.show()

lines = cv2.HoughLinesP(masked_images, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(line_image, (x1,y1), (x2, y2), (255,0,0), 10)

plt.imshow(line_image)
plt.show()

color_canny = np.dstack((canny, canny, canny))
plt.imshow(color_canny)
plt.show()

line_canny = cv2.addWeighted(image_copy, 0.8, line_image, 1, 0)
plt.imshow(line_canny)
plt.show()

import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import numpy as np 
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from utils import *

def process_image(image):
        gray = gra