#!/usr/bin/env python
# coding: utf-8

# ## SIT789 Assignment 1.1P

# In[1]:


import numpy as np
import cv2 as cv
img = cv.imread('img1.jpg')


# Check height and width of image

# In[2]:


height, width = img.shape[:2]
print (height, width)


# Display image in another window

# In[3]:


cv.imshow('input image', img)
cv.waitKey(0)
cv.destroyAllWindows()


# Convert image to png from jpg

# In[4]:


cv.imwrite('img1_out.png', img)


# In[ ]:




