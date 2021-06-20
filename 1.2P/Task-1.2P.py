#!/usr/bin/env python
# coding: utf-8

# ## SIT789 Assignment 1.2P

# ## 1 Color Conversion

# In[1]:


import numpy as np
import cv2 as cv
img = cv.imread('img1.jpg')


# Conversion into HSV

# In[2]:


img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('image in HSV', img_hsv)
cv.waitKey(0)
cv.destroyAllWindows()


# In[3]:


cv.imwrite('img_hsv.jpg', img_hsv)


# Conversion into Gray

# In[4]:


img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('image in gray', img_gray)
cv.waitKey(0)
cv.destroyAllWindows()


# In[5]:


cv.imwrite('img_gray.jpg', img_gray)


# ## 2 Geometric Transformation

# Scaling

# In[6]:


height, width = img.shape[:2]
h_scale = 0.5
v_scale = 0.4
new_height = (int) (height * v_scale) # we need this as the new height must be interger
new_width = (int) (width * h_scale) # we need this as the new width must be interger
img_resize = cv.resize(img, (new_width, new_height), interpolation = cv.INTER_LINEAR)
cv.imshow('resize', img_resize)
cv.waitKey(0)
cv.destroyAllWindows()


# In[7]:


cv.imwrite('img_resize.jpg', img_resize)


# Translation

# In[8]:


t_x = 100
t_y = 200
M = np.float32([[1, 0, t_x], [0, 1, t_y]])
height, width = img.shape[:2] #this will get the number of rows and columns in img
img_translation = cv.warpAffine(img, M, (width, height))
cv.imshow('translation', img_translation)
cv.waitKey(0)
cv.destroyAllWindows()


# In[9]:


cv.imwrite('img_translation.jpg', img_translation)


# Rotation

# In[10]:


theta = 45 #rotate 45 degrees in anti-clockwise
c_x = (width - 1) / 2.0 # column index varies in [0, width-1]
c_y = (height - 1) / 2.0 # row index varies in [0, height-1]
c = (c_x, c_y) # A point is defined by x and y coordinate
print(c)


# In[11]:


s = 1
M = cv.getRotationMatrix2D(c, theta, s)
img_rotation = cv.warpAffine(img, M, (width, height))
cv.imshow('rotation', img_rotation)
cv.waitKey(0)
cv.destroyAllWindows()


# In[12]:


cv.imwrite('img_rotation.jpg', img_rotation)


# Affine

# In[13]:


m00 = 0.38
m01 = 0.27
m02 = -47.18
m10 = -0.14
m11 = 0.75
m12 = 564.32
M = np.float32([[m00, m01, m02], [m10, m11, m12]])


# In[14]:


height, width = img.shape[:2]
img_affine = cv.warpAffine(img, M, (width, height))
cv.imshow('affine', img_affine)
cv.waitKey(0)
cv.destroyAllWindows()


# In[15]:


cv.imwrite('img_affine.jpg', img_affine)


# In[ ]:




