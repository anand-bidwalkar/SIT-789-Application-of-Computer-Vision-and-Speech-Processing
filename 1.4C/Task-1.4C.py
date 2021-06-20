#!/usr/bin/env python
# coding: utf-8

# ## SIT789 Assignment 1.4C

# In[1]:


import numpy as np
import cv2 as cv
import math


# Read all three images and convert them into grayscale

# In[2]:


img1 = cv.imread('img1.jpg')
img2 = cv.imread('img2.jpg')
img3 = cv.imread('img3.jpg')
img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
img3_gray = cv.cvtColor(img3, cv.COLOR_BGR2GRAY)


# Calculate histogram for all three images

# In[3]:


hist1_gray = cv.calcHist([img1_gray],[0],None,[256],[0,256])
hist2_gray = cv.calcHist([img2_gray],[0],None,[256],[0,256])
hist3_gray = cv.calcHist([img3_gray],[0],None,[256],[0,256])


# Plot histogram plot for all three images

# In[4]:


from matplotlib import pyplot as plt
plt.plot(hist1_gray, label = 'img1_gray')
plt.plot(hist2_gray, label = 'img2_gray')
plt.plot(hist3_gray, label = 'img3_gray')
plt.legend(loc="upper right")
plt.xlim([0,256])
plt.show()


# Calculate x_square distance for the histogram

# In[11]:


def get_xsquare_distance(histogram_one, histogram_two):
    number_of_bins = len(histogram_one)
    xsquare_distance = 0
    for i in range(0, number_of_bins):
        histogram_one[i] += 0.000001
        histogram_two[i] += 0.000001 
        numerator = (histogram_one[i] - histogram_two[i]) ** 2
        denominator = histogram_one[i] + histogram_two[i]
        xsquare_distance += numerator / denominator
        
    return xsquare_distance


# In[12]:


xsquare_distance_img12 = get_xsquare_distance(hist1_gray, hist2_gray)
xsquare_distance_img23 = get_xsquare_distance(hist2_gray, hist3_gray)
xsquare_distance_img13 = get_xsquare_distance(hist1_gray, hist3_gray)
print('X Square Distance between image 1 & 2 ', xsquare_distance_img12)
print('X Square Distance between image 2 & 3 ', xsquare_distance_img23)
print('X Square Distance between image 1 & 3 ', xsquare_distance_img13)


# Calculate normalised histogram 

# In[7]:


def get_normalised_histogram(histogram):
    histogram_sum = sum(histogram)
    normalised_histogram = [x / histogram_sum for x in histogram]
    return normalised_histogram


# Calculate KL Divergence for the histogram

# In[8]:


def get_kl_divergence(histogram_one, histogram_two):
    number_of_bins = len(histogram_one)
    kl_divergence = 0
    for i in range(0, number_of_bins):
        histogram_one[i] += 0.000001
        histogram_two[i] += 0.000001        
        kl_divergence += histogram_one[i] * math.log2(histogram_one[i] / histogram_two[i])
        
    return kl_divergence


# In[9]:


normalised_histogram_one = get_normalised_histogram(hist1_gray)
normalised_histogram_two = get_normalised_histogram(hist2_gray)
normalised_histogram_three = get_normalised_histogram(hist3_gray)


# In[10]:


kl_divergence_12 = get_kl_divergence(normalised_histogram_one, normalised_histogram_two) + get_kl_divergence(normalised_histogram_two, normalised_histogram_one)
kl_divergence_23 = get_kl_divergence(normalised_histogram_two, normalised_histogram_three) + get_kl_divergence(normalised_histogram_three, normalised_histogram_two)
kl_divergence_13 = get_kl_divergence(normalised_histogram_one, normalised_histogram_three) + get_kl_divergence(normalised_histogram_three, normalised_histogram_one)

print('KL Divergence between image 1 & 2', kl_divergence_12)
print('KL Divergence between image 2 & 3', kl_divergence_23)
print('KL Divergence between image 1 & 3', kl_divergence_13)


# In[ ]:




