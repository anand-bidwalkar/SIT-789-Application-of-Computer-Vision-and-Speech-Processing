#!/usr/bin/env python
# coding: utf-8

# In[137]:


import cv2 as cv
def detect_face(image, cascade_detector, scale_factor, min_neighbors, min_size):
    #convert input image to grayscale
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = cascade_detector.detectMultiScale(image_gray,
                                                scaleFactor = scale_factor, #the ratio between two consecutive scales
                                                minNeighbors = min_neighbors, #minimum number of overlapping windows to be considered
                                                minSize = min_size, #minimum size of detection window (in pixels)
                                                flags = cv.CASCADE_SCALE_IMAGE) #scale the image rather than detection window
    return faces


# In[138]:


import cv2 as cv

def detect_face_without_flag(image, cascade_detector, scale_factor, min_neighbour, min_size):
    #convert input image to grayscale
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = cascade_detector.detectMultiScale(image_gray,
                                                scaleFactor = scale_factor, #the ratio between two consecutive scales
                                                minNeighbors = min_neighbour, #minimum number of overlapping windows to be considered
                                                minSize = min_size) #minimum size of detection window (in pixels)
    return faces


# In[139]:


from matplotlib import pyplot as plt

def plot_faces(faces):    
    for (x, y, w, h) in faces: #(x, y) are the coordinate of the topleft corner,
        # w, h are the width and height of the bounding box
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        plt.imshow(image[:,:,::-1]) # RGB-> BGR


# In[140]:


import time
image = cv.imread('FaceImages/abba.png')
cascade_detector = cv.CascadeClassifier('haarcascade_frontalface_default.xml')


# In[141]:


start_time = time.time()
faces = detect_face(image, cascade_detector, 1.1, 5, (30,30))
print('Face detection is performed in %s seconds ---' % (time.time() - start_time))
if (faces is not None):
    print('Found ', len(faces), ' faces')
    plot_faces(faces)
else:
    print('There is no face found!')


# Scale Factor - 1.2

# In[36]:


start_time = time.time()
image = cv.imread('FaceImages/abba.png')
faces = detect_face(image, cascade_detector, 1.2, 5, (30,30))
print('Face detection is performed in %s seconds ---' % (time.time() - start_time))
if (faces is not None):
    print('Found ', len(faces), ' faces')
    plot_faces(faces)
else:
    print('There is no face found!')


# Scale Factor - 1.3

# In[37]:


start_time = time.time()
image = cv.imread('FaceImages/abba.png')
faces = detect_face(image, cascade_detector, 1.3, 5, (30,30))
print('Face detection is performed in %s seconds ---' % (time.time() - start_time))
if (faces is not None):
    print('Found ', len(faces), ' faces')
    plot_faces(faces)
else:
    print('There is no face found!')


# Scale Factor - 1.4

# In[38]:


start_time = time.time()
image = cv.imread('FaceImages/abba.png')
faces = detect_face(image, cascade_detector, 1.4, 5, (30,30))
print('Face detection is performed in %s seconds ---' % (time.time() - start_time))
if (faces is not None):
    print('Found ', len(faces), ' faces')
    plot_faces(faces)
else:
    print('There is no face found!')


# Scale Factor - 1.5

# In[39]:


start_time = time.time()
image = cv.imread('FaceImages/abba.png')
faces = detect_face(image, cascade_detector, 1.5, 5, (30,30))
print('Face detection is performed in %s seconds ---' % (time.time() - start_time))
if (faces is not None):
    print('Found ', len(faces), ' faces')
    plot_faces(faces)
else:
    print('There is no face found!')


# Min Neighbours - 0

# In[40]:


start_time = time.time()
image = cv.imread('FaceImages/abba.png')
faces = detect_face(image, cascade_detector, 1.3, 0, (30,30))
print('Face detection is performed in %s seconds ---' % (time.time() - start_time))
if (faces is not None):
    print('Found ', len(faces), ' faces')
    plot_faces(faces)
else:
    print('There is no face found!')


# Min Neighbours - 10

# In[42]:


start_time = time.time()
image = cv.imread('FaceImages/abba.png')
faces = detect_face(image, cascade_detector, 1.3, 10, (30,30))
print('Face detection is performed in %s seconds ---' % (time.time() - start_time))
if (faces is not None):
    print('Found ', len(faces), ' faces')
    plot_faces(faces)
else:
    print('There is no face found!')


# Min Neighbours - 15

# In[43]:


start_time = time.time()
image = cv.imread('FaceImages/abba.png')
faces = detect_face(image, cascade_detector, 1.3, 15, (30,30))
print('Face detection is performed in %s seconds ---' % (time.time() - start_time))
if (faces is not None):
    print('Found ', len(faces), ' faces')
    plot_faces(faces)
else:
    print('There is no face found!')


# Min Neighbours - 20

# In[45]:


start_time = time.time()
image = cv.imread('FaceImages/abba.png')
faces = detect_face(image, cascade_detector, 1.3, 20, (30,30))
print('Face detection is performed in %s seconds ---' % (time.time() - start_time))
if (faces is not None):
    print('Found ', len(faces), ' faces')
    plot_faces(faces)
else:
    print('There is no face found!')


# Min Size - (10, 10)

# In[46]:


start_time = time.time()
image = cv.imread('FaceImages/abba.png')
faces = detect_face(image, cascade_detector, 1.3, 15, (10,10))
print('Face detection is performed in %s seconds ---' % (time.time() - start_time))
if (faces is not None):
    print('Found ', len(faces), ' faces')
    plot_faces(faces)
else:
    print('There is no face found!')


# Min Size - (20, 20)

# In[47]:


start_time = time.time()
image = cv.imread('FaceImages/abba.png')
faces = detect_face(image, cascade_detector, 1.3, 15, (20,20))
print('Face detection is performed in %s seconds ---' % (time.time() - start_time))
if (faces is not None):
    print('Found ', len(faces), ' faces')
    plot_faces(faces)
else:
    print('There is no face found!')


# Min Size - (40, 40)

# In[48]:


start_time = time.time()
image = cv.imread('FaceImages/abba.png')
faces = detect_face(image, cascade_detector, 1.3, 15, (40,40))
print('Face detection is performed in %s seconds ---' % (time.time() - start_time))
if (faces is not None):
    print('Found ', len(faces), ' faces')
    plot_faces(faces)
else:
    print('There is no face found!')


# Min Size - (50, 50)

# In[49]:


start_time = time.time()
image = cv.imread('FaceImages/abba.png')
faces = detect_face(image, cascade_detector, 1.3, 5, (50,50))
print('Face detection is performed in %s seconds ---' % (time.time() - start_time))
if (faces is not None):
    print('Found ', len(faces), ' faces')
    plot_faces(faces)
else:
    print('There is no face found!')


# In[50]:


start_time = time.time()
image = cv.imread('FaceImages/abba.png')
faces = detect_face_without_flag(image, cascade_detector, 1.3, 5, (50,50))
print('Face detection is performed in %s seconds ---' % (time.time() - start_time))
if (faces is not None):
    print('Found ', len(faces), ' faces')
    plot_faces(faces)
else:
    print('There is no face found!')


# In[61]:


start_time = time.time()
image = cv.imread('FaceImages/img_1014.jpg')
faces = detect_face(image, cascade_detector, 1.3, 5, (50,50))
print('Face detection is performed in %s seconds ---' % (time.time() - start_time))
if (faces is not None):
    print('Found ', len(faces), ' faces')
    plot_faces(faces)
else:
    print('There is no face found!')


# In[65]:


start_time = time.time()
image = cv.imread('FaceImages/img_1014.jpg')
faces = detect_face(image, cascade_detector, 1.1, 5, (50,50))
print('Face detection is performed in %s seconds ---' % (time.time() - start_time))
if (faces is not None):
    print('Found ', len(faces), ' faces')
    plot_faces(faces)
else:
    print('There is no face found!')


# In[66]:


start_time = time.time()
image = cv.imread('FaceImages/img_1123.jpg')
faces = detect_face(image, cascade_detector, 1.3, 5, (50,50))
print('Face detection is performed in %s seconds ---' % (time.time() - start_time))
if (faces is not None):
    print('Found ', len(faces), ' faces')
    plot_faces(faces)
else:
    print('There is no face found!')


# In[136]:


start_time = time.time()
image = cv.imread('FaceImages/img_1123.jpg')
faces = detect_face_without_flag(image, cascade_detector, 1.09, 0, (30,30))
print('Face detection is performed in %s seconds ---' % (time.time() - start_time))
if (faces is not None):
    print('Found ', len(faces), ' faces')
    plot_faces(faces)
else:
    print('There is no face found!')


# In[144]:


import cv2 as cv
import os

#initialise webcam
cam = cv.VideoCapture(0)

#initialise cascade_detector
cascade_detector = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
cam.open(0)
while True:
    #read the image from the cam
    _, image = cam.read()
    #detect human faces from the current image using the cascade_detector
    faces = detect_face(image, cascade_detector, 1.1, 5, (30,30))
    #display detected faces
    for x, y, w, h in faces:
        cv.rectangle(image, (x, y), (x + w, y + h), color = (0, 255, 0))
    cv.imshow('face detection demo', image)
    if cv.waitKey(1) == ord("q"):
        cv.destroyAllWindows()
        break

cam.release()


# In[97]:


import cv2 as cv
import numpy as np
import imutils
from imutils.object_detection import non_max_suppression

def nms(boxes):
    #We first convert boxes from list to array as required by non_max_suppression method
    #In addition, each box in the array is encoded by the topleft and bottomright corners
    boxes_array = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    boxes_array = non_max_suppression(boxes_array, probs = None, overlapThresh = 0.65)
    #create a new list of boxes to store results
    boxes_list = []
    for top_x, top_y, bottom_x, bottom_y in boxes_array:
        boxes_list.append([top_x, top_y, bottom_x - top_x, bottom_y - top_y])
    return boxes_list

def detect_pedestrian(image, win_stride, padding_value, scale_value, nms_value):
    #initialise the HOG descriptor and SVM classifier
    hog = cv.HOGDescriptor()
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
    image_resized = imutils.resize(image,
                                    width = min(400, image.shape[1])) #resize the input image so that
    #the width is max by 400
    scale = image.shape[1] / image_resized.shape[1]
    #detect pedestrians
    (boxes, _) = hog.detectMultiScale(image_resized,
                                winStride = win_stride, #horizontal and vertical stride
                                padding = padding_value, #horizontal and vertical padding for each window
                                scale = scale_value) #scale factor between two consecutive scales
    #non-maximum suppression
    if nms_value:
        boxes = nms(boxes)
    #resize the bounding boxes
    for box in boxes:
        box[0] = np.int(box[0] * scale)
        box[1] = np.int(box[1] * scale)
        box[2] = np.int(box[2] * scale)
        box[3] = np.int(box[3] * scale)
    return boxes


# In[98]:


from matplotlib import pyplot as plt
def plot_pedestrains(pedestrians):    
    if (pedestrians is not None):
        print('Found ', len(pedestrians), ' pedestrians')

        for (x, y, w, h) in pedestrians:
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 10)
            plt.imshow(image[:,:,::-1]) # RGB-> BGR
    else:
        print('There is no pedestrian found!')


# In[99]:


import time
image = cv.imread('PedestrianImages/person_029.png')
start_time = time.time()
pedestrians = detect_pedestrian(image, (4, 4), (6, 6), 1.05, True)
print('Pedestrian detection is performed in %s seconds ---' % (time.time() - start_time))
plot_pedestrains(pedestrians)


# Without nms(boxes)

# In[100]:


import time
image = cv.imread('PedestrianImages/person_029.png')
start_time = time.time()
pedestrians = detect_pedestrian(image, (4, 4), (6, 6), 1.05, False)
print('Pedestrian detection is performed in %s seconds ---' % (time.time() - start_time))
plot_pedestrains(pedestrians)


# Win Stride = (2,2)

# In[101]:


import time
image = cv.imread('PedestrianImages/person_029.png')
start_time = time.time()
pedestrians = detect_pedestrian(image, (2, 2), (6, 6), 1.05, True)
print('Pedestrian detection is performed in %s seconds ---' % (time.time() - start_time))
plot_pedestrains(pedestrians)


# Win Stride = (6,6)

# In[102]:


import time
image = cv.imread('PedestrianImages/person_029.png')
start_time = time.time()
pedestrians = detect_pedestrian(image, (6, 6), (6, 6), 1.05, True)
print('Pedestrian detection is performed in %s seconds ---' % (time.time() - start_time))
plot_pedestrains(pedestrians)


# Win Stride = (8,8)

# In[103]:


import time
image = cv.imread('PedestrianImages/person_029.png')
start_time = time.time()
pedestrians = detect_pedestrian(image, (8, 8), (6, 6), 1.05, True)
print('Pedestrian detection is performed in %s seconds ---' % (time.time() - start_time))
plot_pedestrains(pedestrians)


# Padding = (4,4)

# In[104]:


import time
image = cv.imread('PedestrianImages/person_029.png')
start_time = time.time()
pedestrians = detect_pedestrian(image, (4, 4), (4, 4), 1.05, True)
print('Pedestrian detection is performed in %s seconds ---' % (time.time() - start_time))
plot_pedestrains(pedestrians)


# Padding = (8,8)

# In[105]:


import time
image = cv.imread('PedestrianImages/person_029.png')
start_time = time.time()
pedestrians = detect_pedestrian(image, (4, 4), (8, 8), 1.05, True)
print('Pedestrian detection is performed in %s seconds ---' % (time.time() - start_time))
plot_pedestrains(pedestrians)


# Padding = (10,10)

# In[106]:


import time
image = cv.imread('PedestrianImages/person_029.png')
start_time = time.time()
pedestrians = detect_pedestrian(image, (4, 4), (10, 10), 1.05, True)
print('Pedestrian detection is performed in %s seconds ---' % (time.time() - start_time))
plot_pedestrains(pedestrians)


# Scale = 1.1

# In[107]:


import time
image = cv.imread('PedestrianImages/person_029.png')
start_time = time.time()
pedestrians = detect_pedestrian(image, (4, 4), (6, 6), 1.1, True)
print('Pedestrian detection is performed in %s seconds ---' % (time.time() - start_time))
plot_pedestrains(pedestrians)


# Scale = 1.15

# In[108]:


import time
image = cv.imread('PedestrianImages/person_029.png')
start_time = time.time()
pedestrians = detect_pedestrian(image, (4, 4), (6, 6), 1.15, True)
print('Pedestrian detection is performed in %s seconds ---' % (time.time() - start_time))
plot_pedestrains(pedestrians)


# Scale = 1.2

# In[109]:


import time
image = cv.imread('PedestrianImages/person_029.png')
start_time = time.time()
pedestrians = detect_pedestrian(image, (4, 4), (6, 6), 1.2, True)
print('Pedestrian detection is performed in %s seconds ---' % (time.time() - start_time))
plot_pedestrains(pedestrians)


# In[ ]:




