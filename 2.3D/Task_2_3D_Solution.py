#!/usr/bin/env python
# coding: utf-8

# ## SIT 789 Task-2.3D

# In[30]:


import numpy as np
import cv2 as cv
import math
from matplotlib import pyplot as plt


# Step-1 Load file and binarise the image

# In[32]:


doc = cv.imread('doc.jpg', 0) #Note that the second parameter of imread is set to 0
threshold = 200
ret, doc_bin = cv.threshold(doc, threshold, 255, cv.THRESH_BINARY)


# Step-2 Get negative version of binarise image

# In[33]:


doc_bin = 255 - doc_bin #convert black/white to white/black


# Step-3 Extract connected components

# In[34]:


# connected component labelling
num_labels, labels_im = cv.connectedComponents(doc_bin)


# In[36]:


def ExtractConnectedComponents(num_labels, labels_im):
    connected_components = [[] for i in range(0, num_labels)]
    height, width = labels_im.shape
    for i in range(0, height):
        for j in range(0, width):
            if labels_im[i, j] >= 0:
                connected_components[labels_im[i, j]].append((j, i))
    return connected_components


# In[37]:


connected_components = ExtractConnectedComponents(num_labels, labels_im)


# Function for strategy - c

# In[38]:


def get_ycordinate_based_candidate(connected_components):
    candidate_points = []
    for x in connected_components:
        res = max(x, key = lambda i : i[1])
        #print(res[0], res[1])
        candidate_points.append([res[0], res[1]])
    return candidate_points


# Function for strategy - b

# In[39]:


def get_mean_based_candidate(connected_components):
    candidate_points = []
    for x in connected_components:
        res = [round(sum(ele) / len(x)) for ele in zip(*x)]
        candidate_points.append(res)
    return candidate_points


# In[40]:


def split_candidate_points(computed_candidate_points):
    candidate_points_x = []
    candidate_points_y = []
    for k in computed_candidate_points:
        candidate_points_x.append(k[0])
        candidate_points_y.append(k[1])
    return candidate_points_x, candidate_points_y


# Filter image to remove non candidate points

# In[41]:


def filter_image(image_data, x_points, y_points):  
  height, width = image_data.shape
  for i in range(0, height):
    for j in range(0, width):  
      if(i in x_points and j in y_points):
        image_data[i][j] = 255        
  return image_data


# In[51]:


def draw_candidate_points(image, x_points, y_points):  
  implot = plt.imshow(image, 'gray')
  plt.scatter(x=x_points, y=y_points, c='r', s=10)
  plt.show()


# In[43]:


import math, statistics, time
distance_resolution = 1
angular_resolution = np.pi/180
density_threshold = 10


# Hough Transform

# In[44]:


def get_angles_from_hough_transform(image_data):
    lines = cv.HoughLines(np.array(image_data), distance_resolution, angular_resolution, density_threshold)
    detected_angle = []
    for line in lines:
        distance, angle = line[0]
        detected_angle.append(angle)
    return detected_angle


# In[45]:


def get_median_angle(detected_angles):
    return statistics.median(detected_angles)


# As the angle returned by cv.HoughLines is orthogonal to the connected components we need to subtract 90 degree from it

# In[46]:


def deskew_image(median_angle, image_data):
    # rotate image
    height, width = image_data.shape
    c_x = (width - 1) / 2.0 # column index varies in [0, width-1]
    c_y = (height - 1) / 2.0 # row index varies in [0, height-1]
    c = (c_x, c_y) # A point is defined by x and y coordinate
    M = cv.getRotationMatrix2D(c, median_angle * 180 / math.pi - 90, 1)
    doc_deskewed = cv.warpAffine(image_data, M, (width, height))
    plt.imshow(doc_deskewed, "gray")


# Strategy - A

# In[95]:


start_time = time.time()
strategy_a_image = doc_bin.copy()
plt.imshow(strategy_a_image, 'gray')
cv.imwrite('doc_strategy_a_image.png', strategy_a_image)
detected_angles = get_angles_from_hough_transform(strategy_a_image)
median_angle = get_median_angle(detected_angles)
deskew_image(median_angle, doc)
print("Time for execution of strategy", time.time() - start_time)


# In[101]:


x_points, y_points = [], []
for component_list in connected_components:
  for component in component_list:
    x_points.append(component[0])
    y_points.append(component[1])

draw_candidate_points(doc_bin, x_points, y_points)


# Strategy - B

# In[49]:


start_time = time.time()
mean_based_candidate_points = get_mean_based_candidate(connected_components)
x_points, y_points = split_candidate_points(mean_based_candidate_points)
strategy_b_image = doc.copy()
strategy_b_image = filter_image(strategy_b_image, x_points, y_points)
plt.imshow(strategy_b_image, 'gray')
cv.imwrite('doc_strategy_b_image.png', strategy_b_image)
detected_angles = get_angles_from_hough_transform(strategy_b_image)
median_angle = get_median_angle(detected_angles)
deskew_image(median_angle, doc)
print("Time for execution of strategy", time.time() - start_time)


# In[53]:


draw_candidate_points(doc_bin, x_points, y_points)


# Strategy - C

# In[54]:


start_time = time.time()
max_y_based_candidate_points = get_ycordinate_based_candidate(connected_components)
x_points, y_points = split_candidate_points(max_y_based_candidate_points)
strategy_c_image = doc.copy()
strategy_c_image = filter_image(strategy_c_image, x_points, y_points)
plt.imshow(strategy_c_image, 'gray')
cv.imwrite('doc_strategy_c_image.png', strategy_c_image)
detected_angles = get_angles_from_hough_transform(strategy_c_image)
median_angle = get_median_angle(detected_angles)
deskew_image(median_angle, doc)
print("Time for execution of strategy", time.time() - start_time)


# In[55]:


draw_candidate_points(doc_bin, x_points, y_points)


# Another test case - doc_1.jpg

# In[57]:


doc_1 = cv.imread('doc_1.jpg', 0) #Note that the second parameter of imread is set to 0
threshold = 200
ret, doc_bin_1 = cv.threshold(doc_1, threshold, 255, cv.THRESH_BINARY)
doc_bin_1 = 255 - doc_bin_1 #convert black/white to white/black


# In[58]:


# connected component labelling
num_labels_1, labels_im_1 = cv.connectedComponents(doc_bin_1)
connected_components_1 = ExtractConnectedComponents(num_labels_1, labels_im_1)


# As the input image is already rotated by 90 degree clockwise, to deskew the image it has to be rotated by median_angle * 180 / math.pi

# In[84]:


def deskew_image_rotated(median_angle, image_data):
    # rotate image
    height, width = image_data.shape
    c_x = (width - 1) / 2.0 # column index varies in [0, width-1]
    c_y = (height - 1) / 2.0 # row index varies in [0, height-1]
    c = (c_x, c_y) # A point is defined by x and y coordinate
    M = cv.getRotationMatrix2D(c, median_angle * 180 / math.pi, 1)
    doc_deskewed = cv.warpAffine(image_data, M, (width, height))
    plt.imshow(doc_deskewed, "gray")


# Strategy - A

# In[102]:


start_time = time.time()
strategy_a_image = doc_bin_1.copy()
plt.imshow(strategy_a_image, 'gray')
cv.imwrite('doc_1_strategy_a_image.png', strategy_a_image)
detected_angles = get_angles_from_hough_transform(strategy_a_image)
median_angle = get_median_angle(detected_angles)
deskew_image(median_angle, doc_1)
print("Time for execution of strategy", time.time() - start_time)


# In[103]:


x_points, y_points = [], []
for component_list in connected_components_1:
  for component in component_list:
    x_points.append(component[0])
    y_points.append(component[1])

draw_candidate_points(doc_bin_1, x_points, y_points)


# In[87]:


deskew_image_rotated(median_angle, doc_1)


# Strategy - B

# In[88]:


start_time = time.time()
mean_based_candidate_points = get_mean_based_candidate(connected_components_1)
x_points, y_points = split_candidate_points(mean_based_candidate_points)
strategy_b_image = doc_1.copy()
strategy_b_image = filter_image(strategy_b_image, x_points, y_points)
plt.imshow(strategy_b_image, 'gray')
cv.imwrite('doc_1_strategy_b_image.png', strategy_b_image)
detected_angles = get_angles_from_hough_transform(strategy_b_image)
median_angle = get_median_angle(detected_angles)
deskew_image(median_angle, doc_1)
print("Time for execution of strategy", time.time() - start_time)


# In[89]:


draw_candidate_points(doc_bin_1, x_points, y_points)


# In[90]:


deskew_image_rotated(median_angle, doc_1)


# Strategy - C

# In[91]:


start_time = time.time()
max_y_based_candidate_points = get_ycordinate_based_candidate(connected_components_1)
x_points, y_points = split_candidate_points(max_y_based_candidate_points)
strategy_c_image = doc_1.copy()
strategy_c_image = filter_image(strategy_c_image, x_points, y_points)
detected_angles = get_angles_from_hough_transform(strategy_c_image)
median_angle = get_median_angle(detected_angles)
deskew_image(median_angle, doc_1)
print("Time for execution of strategy", time.time() - start_time)


# In[92]:


draw_candidate_points(doc_bin_1, x_points, y_points)


# In[93]:


deskew_image_rotated(median_angle, doc_1)

