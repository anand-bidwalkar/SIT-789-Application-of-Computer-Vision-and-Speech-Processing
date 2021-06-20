#!/usr/bin/env python
# coding: utf-8

# ## SIT 789 Task 3.2C

# In[1]:


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


# Read images for the task

# In[2]:


#load images
img = cv.imread('empire.jpg')
img_45 = cv.imread('empire_45.jpg')
img_zoomedout = cv.imread('empire_zoomedout.jpg')
img_another = cv.imread('fisherman.jpg')


# Convert into gray scale

# In[3]:


#convert the images to grayscale
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_45_gray = cv.cvtColor(img_45, cv.COLOR_BGR2GRAY)
img_zoomedout_gray = cv.cvtColor(img_zoomedout, cv.COLOR_BGR2GRAY)
img_another_gray = cv.cvtColor(img_another, cv.COLOR_BGR2GRAY)


# Extract keypoints and descriptors using SIFT

# In[30]:


#initialise SIFT
sift = cv.xfeatures2d.SIFT_create()
#extract keypoints and descriptors
kp, des = sift.detectAndCompute(img_gray, None)
kp_45, des_45 = sift.detectAndCompute(img_45_gray, None)
kp_zoomedout, des_zoomedout = sift.detectAndCompute(img_zoomedout_gray, None)
kp_another, des_another = sift.detectAndCompute(img_another_gray, None)


# In[11]:


# Initialise a brute force matcher with default params
bf = cv.BFMatcher()
train = des_45
query = des
matches_des_des_45 = bf.match(query, train)


# In[12]:


matches_des_des_45 = sorted(matches_des_des_45, key = lambda x:x.distance)


# In[19]:


# Draw the best 10 matches.
nBestMatches = 10
matching_des_des_45 = cv.drawMatches(img_gray, kp, img_45_gray, kp_45,
                        matches_des_des_45[:nBestMatches],
                        None,
                        flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(matching_des_des_45)


# In[14]:


kp_train = kp_45
kp_query = kp
for i in range (0, nBestMatches):
    print("match ", i, " info")
    print("\tdistance:", matches_des_des_45[i].distance)
    print("\tkeypoint in train: ID:", matches_des_des_45[i].trainIdx, " x:",
        kp_train[matches_des_des_45[i].trainIdx].pt[0], " y:",
        kp_train[matches_des_des_45[i].trainIdx].pt[1])
    print("\tkeypoint in query: ID:", matches_des_des_45[i].queryIdx, " x:",
        kp_query[matches_des_des_45[i].queryIdx].pt[0], " y:",
        kp_query[matches_des_des_45[i].queryIdx].pt[1])


# In[15]:


matches_des_45_des = bf.match(des_45, des)
matches_des_45_des = sorted(matches_des_45_des, key = lambda x:x.distance)
matching_des_45_des = cv.drawMatches(img_45_gray, kp_45, img_gray, kp,
                        matches_des_45_des[:nBestMatches],
                        None,
                        flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(matching_des_45_des)


# Similarity score between image empire and empire_45 using N=10

# In[24]:


similarity_distance_des_des_45 = (sum(c.distance for c in matches_des_45_des[:10]) + 
                                   sum(c.distance for c in matches_des_des_45[:10])) / 2
similarity_distance_des_des_45


# Similarity score between image empire and empire_45 using N=100

# In[23]:


similarity_distance_des_des_45 = (sum(c.distance for c in matches_des_45_des[:100]) + 
                                   sum(c.distance for c in matches_des_des_45[:100])) / 2
similarity_distance_des_des_45


# Similarity score between image empire and empire_45 using N=1000

# In[25]:


similarity_distance_des_des_45 = (sum(c.distance for c in matches_des_45_des[:1000]) + 
                                   sum(c.distance for c in matches_des_des_45[:1000])) / 2
similarity_distance_des_des_45


# BFM for image empire and empire_zoomedout

# In[32]:


matches_des_des_zoomedout = bf.match(des, des_zoomedout)
matches_des_des_zoomedout = sorted(matches_des_des_zoomedout, key = lambda x:x.distance)
matching_des_des_zoomedout = cv.drawMatches(img_gray, kp, img_zoomedout_gray, kp_zoomedout, 
                                matches_des_des_zoomedout[:nBestMatches],
                                None,
                                flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(matching_des_des_zoomedout)


# In[33]:


matches_des_zoomedout_des = bf.match(des_zoomedout, des)
matches_des_zoomedout_des = sorted(matches_des_zoomedout_des, key = lambda x:x.distance)
matching_des_zoomedout_des = cv.drawMatches(img_zoomedout_gray, kp_zoomedout, img_gray, kp,
                                matches_des_zoomedout_des[:nBestMatches],
                                None,
                                flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(matching_des_zoomedout_des)


# Similarity score between image empire and empire_zoomedout using N=10

# In[35]:


similarity_distance_des_des_zoomedout = (sum(c.distance for c in matches_des_des_zoomedout[:10]) + 
                                   sum(c.distance for c in matches_des_zoomedout_des[:10])) / 2
similarity_distance_des_des_zoomedout


# Similarity score between image empire and empire_zoomedout using N=100

# In[36]:


similarity_distance_des_des_zoomedout = (sum(c.distance for c in matches_des_des_zoomedout[:100]) + 
                                   sum(c.distance for c in matches_des_zoomedout_des[:100])) / 2
similarity_distance_des_des_zoomedout


# Similarity score between image empire and empire_zoomedout using N=1000

# In[37]:


similarity_distance_des_des_zoomedout = (sum(c.distance for c in matches_des_des_zoomedout[:1000]) + 
                                   sum(c.distance for c in matches_des_zoomedout_des[:1000])) / 2
similarity_distance_des_des_zoomedout


# BFM for the image emire and fisherman

# In[38]:


matches_des_des_another = bf.match(des, des_another)
matches_des_des_another = sorted(matches_des_des_another, key = lambda x:x.distance)
matching_des_des_another = cv.drawMatches(img_gray, kp, img_another_gray, kp_another, 
                                matches_des_des_another[:nBestMatches],
                                None,
                                flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(matching_des_des_another)


# In[40]:


matches_des_another_des = bf.match(des_another, des)
matches_des_another_des = sorted(matches_des_another_des, key = lambda x:x.distance)
matching_des_another_des = cv.drawMatches(img_another_gray, kp_another, img_gray, kp,
                                matches_des_another_des[:nBestMatches],
                                None,
                                flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(matching_des_another_des)


# Similarity score between image empire and fisherman using N=10

# In[43]:


similarity_distance_des_des_another = (sum(c.distance for c in matches_des_another_des[:10]) + 
                                   sum(c.distance for c in matches_des_des_another[:10])) / 2
similarity_distance_des_des_another


# Similarity score between image empire and fisherman using N=100

# In[44]:


similarity_distance_des_des_another = (sum(c.distance for c in matches_des_another_des[:100]) + 
                                   sum(c.distance for c in matches_des_des_another[:100])) / 2
similarity_distance_des_des_another


# Similarity score between image empire and fisherman using N=1000

# In[45]:


similarity_distance_des_des_another = (sum(c.distance for c in matches_des_another_des[:1000]) + 
                                   sum(c.distance for c in matches_des_des_another[:1000])) / 2
similarity_distance_des_des_another


# In[ ]:




