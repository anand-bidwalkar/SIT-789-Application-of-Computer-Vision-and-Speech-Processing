#!/usr/bin/env python
# coding: utf-8

# ## SIT789 Task-2.1P

# In[1]:


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


# Read image file

# In[2]:


img = cv.imread('empire.jpg') #load image
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


# Code for Average Kernel

# In[3]:


avg_kernel = np.ones((5,5), np.float32) / 25 #kernel K defined above
avg_result = cv.filter2D(img_gray, -1, avg_kernel) #always set the second parameter to -1
plt.imshow(avg_result, 'gray')
cv.imwrite('emprire_average_result.jpg', avg_result)


# Code for Gaussian Filter

# In[4]:


gaussian_filter = np.float32([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]]) / 256
gaussian_result = cv.filter2D(img_gray, -1, gaussian_filter)
plt.imshow(gaussian_result, 'gray')
cv.imwrite('emprire_gaussian_result.jpg', gaussian_result)


# Code for Sobel Filter

# In[5]:


sobel_filter = np.float32([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
sobel_result = cv.filter2D(img_gray, -1, sobel_filter)
plt.imshow(sobel_result, 'gray')
cv.imwrite('emprire_sobel_result.jpg', sobel_result)


# Code for Corner Filter

# In[6]:


corner_filter = np.float32([[1, -2, 1], [-2, 4, -2], [1, -2, 1]]) / 4
corner_result = cv.filter2D(img_gray, -1, corner_filter)
plt.imshow(corner_result, 'gray')
cv.imwrite('emprire_corner_result.jpg', sobel_result)


# Read image empire_shotnoise.jpg

# In[7]:


img_noise = cv.imread('empire_shotnoise.jpg')
img_noise_gray = cv.cvtColor(img_noise, cv.COLOR_BGR2GRAY)


# Code for median filter

# In[8]:


#Testing median filter
ksize = 5 # neighbourhood of ksize x ksize; ksize must be an odd number
med_result = cv.medianBlur(img_noise_gray, ksize)
plt.imshow(med_result, 'gray')
cv.imwrite('med_result.jpg', med_result)


# Code for Bilateral filter

# In[9]:


#Testing bilateral filter
rad = 5 #radius to determine neighbourhood
sigma_s = 10 #standard deviation for spatial distance (see slide 21 in week 2 lecture slides)
sigma_c = 30 #standard deviation for colour difference (see slide 21 in week 2 lecture slides)
bil_result = cv.bilateralFilter(img_noise_gray, rad, sigma_c, sigma_s)
plt.imshow(bil_result, 'gray')
cv.imwrite('bil_result.jpg', bil_result)


# Code for Gaussian Filter

# In[10]:


img_noise_gaussian_result = cv.filter2D(img_noise_gray, -1, gaussian_filter)
plt.imshow(img_noise_gaussian_result, 'gray')
cv.imwrite('img_noise_gaussian_result.jpg', img_noise_gaussian_result)


# Code for horizontal gradient

# In[11]:


D_x = np.float32([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
der_x = cv.filter2D(img_gray, -1, D_x)
plt.imshow(der_x, 'gray')
cv.imwrite('horizontal_gradient.jpg', der_x)


# Code for vertical gradient

# In[12]:


D_y = np.float32([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8
der_y = cv.filter2D(img_gray, -1, D_y)
plt.imshow(der_y, 'gray')
cv.imwrite('vertical_gradient.jpg', der_y)


# Code for gradient magnitude

# In[13]:


import math
height, width = img_gray.shape
mag_img_gray = np.zeros((height, width), np.float32) #gradient magnitude of img_gray
for i in range(0, height):
    for j in range(0, width):
        square_der_x = float(der_x[i, j]) * float(der_x[i, j])
        square_der_y = float(der_y[i, j]) * float(der_y[i, j])
        mag_img_gray[i, j] = int(math.sqrt(square_der_x + square_der_y))
plt.imshow(mag_img_gray,'gray')
cv.imwrite('mag_img_gray.jpg', mag_img_gray)


# Code for Canny Edge Detection

# In[14]:


minVal = 100 #minVal used in hysteresis thresholding
maxVal = 200 #maxVal used in hysteresis thresholding
Canny_edges = cv.Canny(img_gray, minVal, maxVal)
plt.imshow(Canny_edges, 'gray')
cv.imwrite('Canny_edges.jpg', Canny_edges)


# In[ ]:




