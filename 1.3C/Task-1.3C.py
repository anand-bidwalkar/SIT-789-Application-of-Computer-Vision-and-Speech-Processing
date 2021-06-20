#!/usr/bin/env python
# coding: utf-8

# ## SIT789 Assignment 1.3C

# ## 1. Calculating and plotting histogram of image

# In[2]:


import numpy as np
import cv2 as cv
img = cv.imread('img1.jpg')


# Calculate histogram of image of image for blue, green, red channel

# In[3]:


hist_blue = cv.calcHist([img],[0],None,[256],[0,256]) #[0] for blue channel
hist_green = cv.calcHist([img],[1],None,[256],[0,256]) #[1] for green channel
hist_red = cv.calcHist([img],[2],None,[256],[0,256]) #[2] for red channel


# Function to plot the histogram using matplotlib

# In[4]:


from matplotlib import pyplot as plt

def plot_histogram(histogram, hist_color):
    plt.plot(histogram, color = hist_color)
    plt.xlim([0,256])
    plt.show()


# Histogram for blue channel

# In[5]:


plot_histogram(hist_blue, 'b')


# Histogram for green channel

# In[6]:


plot_histogram(hist_green, 'g')


# Histogram for red channel

# In[7]:


plot_histogram(hist_red, 'r')


# ## 2 Histogram Equalisation

# In[8]:


img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


# Histogram for grayscale image

# In[9]:


hist_gray = cv.calcHist([img_gray],[0],None,[256],[0,256])
plt.plot(hist_gray)
plt.xlim([0,256])
plt.show()


# Function to calculate cummulative distribution of intensity

# In[10]:


def getCummulativeDis(hist):
    c = [] #cummulative distribution
    s = 0
    for i in range(0, len(hist)):
        s = s + hist[i]
        c.append(s)
    return c


# Cummulative distribution of intesity for grayscale image

# In[11]:


c = getCummulativeDis(hist_gray)
plt.plot(c, label = 'cummulative distribution', color = 'r')
plt.legend(loc="upper left")
plt.xlim([0,256])
plt.show()


# Histogram Equalisation

# In[12]:


img_equ = cv.equalizeHist(img_gray)


# In[13]:


hist_equ = cv.calcHist([img_equ],[0],None,[256],[0,256])
plt.plot(hist_equ)
plt.xlim([0,256])
plt.show()


# Cummulative distribution of intensity for histogram equalised image

# In[14]:


c_equ = getCummulativeDis(hist_equ)
plt.plot(c_equ, label = 'cummulative distribution after histogram equalisation', color = 'r')
plt.legend(loc="upper left")
plt.xlim([0,256])
plt.show()


# Stack grayscale image and histogram equlaised image for comparision

# In[15]:


img_equalisation = np.hstack((img_gray, img_equ)) #stacking images side-by-side
cv.imwrite('img_equalisation.png', img_equalisation) #writing the stacked image to file


# In[ ]:




