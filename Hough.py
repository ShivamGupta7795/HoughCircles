from __future__ import division
import numpy as np
import cv2
from matplotlib import pyplot
import math
import scipy
from scipy import ndimage

Hough_threshold = 100
Canny_threshold = 120

original = cv2.imread('/Users/Amardeep/Documents/CVIP/Project/HoughCircles.jpg')
image = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
image = np.asarray(image)
cv2.imwrite('grayimg.png',image)
w = len(image)
h = len(image[0])


blur = np.zeros((w,h))
##Gaussian Blur##
blur= cv2.GaussianBlur(image,(3,3),0,0)
blur = np.asarray(blur)
cv2.imwrite('blurimg.png',blur)

blob_detector = cv2.SimpleBlobDetector_create()

keypoints = blob_detector.detect(blur)
s = []
for keypoint in keypoints:
	s.append(keypoint.size)

maximum_radius = ((max(s))/2) + 10 # 10 is added just to be sure 
minimum_radius = ((min(s))/2) - 5 # 5 is subtracted just to be sure
radius_max = int(maximum_radius)
radius_min = int(minimum_radius)
canny = np.zeros((w,h))
#canny edge detection with 150 threshold##
canny = cv2.Canny(blur,Canny_threshold,Canny_threshold)
canny = np.asarray(canny)
for x in range (0, w):
	for y in range (0, h):
		if canny[x][y] == 0:
			canny [x][y] = 0
		else:
			canny[x][y] = 255
cv2.imwrite('edgeimg.png',canny)


points = np.where(canny>0)
maxpoints = len(points[0])
# radius_max = (min(w,h))/2
# radius_max = int(radius_max)
acc = np.zeros(((w,h, radius_max)))
acc = acc.astype(int)
pie = math.pi
# np.row_stack(points)


for i in range(0, maxpoints):
	x=points[0][i]
	y=points[1][i]
	for r in range (18, radius_max): #radius_max
		if ((w - x) - r)>=0 or ((h - y) - r)>=0 or (x-r) >=0 or (y-r) >=0:
			for t in range (0, 360):
				a = x + (r * (math.cos((t * pie)/180)))
				b = y + (r * (math.sin((t * pie)/180)))
				if a<w and b<h:
					acc[a][b][r] += 1
	print (i, "-->", maxpoints)



#now to find local Maxima

 # local maxima taking locality of 5
maximum = scipy.ndimage.maximum_filter(acc, 5)
circle_points = np.where(acc == maximum)
output = original

maxpoints1 = len(circle_points[0])
for i in range (0, maxpoints1):
	circle_y= circle_points[0][i]
	circle_x= circle_points[1][i]
	circle_radius = circle_points[2][i]
	check = acc[circle_y][circle_x][circle_radius]
	if check > Hough_threshold: # checking the threshold
		cv2.circle(output,(circle_x,circle_y),circle_radius,(0,255,0),2)
		cv2.circle(output,(circle_x,circle_y),2,(0,255,0),2)

cv2.imwrite('Final_circles.png',output)