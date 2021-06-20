#!/usr/bin/env python
# coding: utf-8

# ## SIT789 Task-4.1 P

# In[1]:


import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans

class Dictionary(object):
    def __init__(self, name, img_filenames, num_words):
        self.name = name #name of your dictionary
        self.img_filenames = img_filenames #list of image filenames
        self.num_words = num_words #the number of words
        self.training_data = [] #this is the training data required by the K-Means algorithm
        self.words = [] #list of words, which are the centroids of clusters
    
    def learn(self):
        sift = cv.xfeatures2d.SIFT_create()
        num_keypoints = [] #this is used to store the number of keypoints in each image
        #load training images and compute SIFT descriptors
        for filename in self.img_filenames:
            img = cv.imread(filename)
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            list_des = sift.detectAndCompute(img_gray, None)[1]
            if list_des is None:
                num_keypoints.append(0)
            else:
                num_keypoints.append(len(list_des))
                for des in list_des:
                    self.training_data.append(des)

        #cluster SIFT descriptors using K-means algorithm
        kmeans = KMeans(self.num_words)
        kmeans.fit(self.training_data)
        self.words = kmeans.cluster_centers_
        #create word histograms for training images
        training_word_histograms = [] #list of word histograms of all training images
        index = 0
        for i in range(0, len(self.img_filenames)):
            #for each file, create a histogram
            histogram = np.zeros(self.num_words, np.float32)
            #if some keypoints exist
            if num_keypoints[i] > 0:
                for j in range(0, num_keypoints[i]):
                    histogram[kmeans.labels_[j + index]] += 1
                index += num_keypoints[i]
                histogram /= num_keypoints[i]
                training_word_histograms.append(histogram)
                    
        return training_word_histograms
    
    def create_word_histograms(self, img_filenames):
        sift = cv.xfeatures2d.SIFT_create()
        histograms = []
        for filename in img_filenames:
            img = cv.imread(filename)
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            descriptors = sift.detectAndCompute(img_gray, None)[1]

            histogram = np.zeros(self.num_words, np.float32) #word histogram for the input image
            if descriptors is not None:
                for des in descriptors:
                    #find the best matching word
                    min_distance = 1111111 #this can be any large number
                    matching_word_ID = -1 #initial matching_word_ID=-1 means no matching
                    for i in range(0, self.num_words): #search for the best matching word
                        distance = np.linalg.norm(des - self.words[i])
                        if distance < min_distance:
                            min_distance = distance
                            matching_word_ID = i
                    histogram[matching_word_ID] += 1
                histogram /= len(descriptors) #normalise histogram to frequencies
            histograms.append(histogram)
        return histograms


# Read training files

# In[2]:


import os

foods = ['Cakes', 'Pasta', 'Pizza']
path = 'FoodImages/'
training_file_names = []
training_food_labels = []

for i in range(0, len(foods)):
    sub_path = path + 'Train/' + foods[i] + '/'
    sub_file_names = [os.path.join(sub_path, f) for f in os.listdir(sub_path)]
    sub_food_labels = [i] * len(sub_file_names) #create a list of N elements, all are i
    training_file_names += sub_file_names
    training_food_labels += sub_food_labels

print(training_file_names)
print(training_food_labels)


# In[4]:


num_words = 50
dictionary_name = 'food'
dictionary = Dictionary(dictionary_name, training_file_names, num_words)


# In[5]:


training_word_histograms = dictionary.learn()


# Save model

# In[6]:


import pickle
#save dictionary
with open('food_dictionary.dic', 'wb') as f: #'wb' is for binary write
    pickle.dump(dictionary, f)


# Load model

# In[7]:


import pickle #you may not need to import it if this has been done
with open('food_dictionary.dic', 'rb') as f: #'rb' is for binary read
    dictionary = pickle.load(f)


# KNeighborsClassifier with n=5

# In[8]:


num_nearest_neighbours = 5 #number of neighbours
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = num_nearest_neighbours)
knn.fit(training_word_histograms, training_food_labels)


# In[9]:


test_file_names = ['FoodImages/Test/Pasta/pasta35.jpg']
word_histograms = dictionary.create_word_histograms(test_file_names)
predicted_food_labels = knn.predict(word_histograms)
print('Food label: ', predicted_food_labels)


# Read test images

# In[10]:


test_file_names = []
test_food_labels = []

for i in range(0, len(foods)):
    sub_path = path + 'Test/' + foods[i] + '/'
    sub_file_names = [os.path.join(sub_path, f) for f in os.listdir(sub_path)]
    sub_food_labels = [i] * len(sub_file_names) #create a list of N elements, all are i
    test_file_names += sub_file_names
    test_food_labels += sub_food_labels

print(test_file_names)
print(test_food_labels)


# Get predicted labels for the images

# In[11]:


def get_predicted_labels(classifier):
    predicted_food_labels = []
    for file_name in test_file_names:
        word_histograms = dictionary.create_word_histograms([file_name])
        predicted_food_label = classifier.predict(word_histograms)
        predicted_food_labels.append(predicted_food_label)
    return predicted_food_labels


# Get classification report

# In[12]:


from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
def get_classification_report(predicted_food_labels):    
    cm = confusion_matrix(test_food_labels, predicted_food_labels)
    print('Confusion Matrix', cm)
    print(classification_report(test_food_labels, predicted_food_labels))
    print('Accuracy Score', accuracy_score(test_food_labels, predicted_food_labels))


# KNeighborsClassifier with num_nearest_neighbours = 5

# In[13]:


predicted_food_labels_knn = get_predicted_labels(knn)
get_classification_report(predicted_food_labels_knn)


# KNeighborsClassifier with num_nearest_neighbours = 10

# In[14]:


num_nearest_neighbours = 10 #number of neighbours
knn = KNeighborsClassifier(n_neighbors = num_nearest_neighbours)
knn.fit(training_word_histograms, training_food_labels)
predicted_food_labels_knn = get_predicted_labels(knn)
get_classification_report(predicted_food_labels_knn)


# KNeighborsClassifier with num_nearest_neighbours = 15

# In[15]:


num_nearest_neighbours = 15 #number of neighbours
knn = KNeighborsClassifier(n_neighbors = num_nearest_neighbours)
knn.fit(training_word_histograms, training_food_labels)
predicted_food_labels_knn = get_predicted_labels(knn)
get_classification_report(predicted_food_labels_knn)


# KNeighborsClassifier with num_nearest_neighbours = 20

# In[16]:


num_nearest_neighbours = 20 #number of neighbours
knn = KNeighborsClassifier(n_neighbors = num_nearest_neighbours)
knn.fit(training_word_histograms, training_food_labels)
predicted_food_labels_knn = get_predicted_labels(knn)
get_classification_report(predicted_food_labels_knn)


# KNeighborsClassifier with num_nearest_neighbours = 25

# In[17]:


num_nearest_neighbours = 25 #number of neighbours
knn = KNeighborsClassifier(n_neighbors = num_nearest_neighbours)
knn.fit(training_word_histograms, training_food_labels)
predicted_food_labels_knn = get_predicted_labels(knn)
get_classification_report(predicted_food_labels_knn)


# KNeighborsClassifier with num_nearest_neighbours = 30

# In[18]:


num_nearest_neighbours = 30 #number of neighbours
knn = KNeighborsClassifier(n_neighbors = num_nearest_neighbours)
knn.fit(training_word_histograms, training_food_labels)
predicted_food_labels_knn = get_predicted_labels(knn)
get_classification_report(predicted_food_labels_knn)


# SVM

# In[19]:


from sklearn import svm
svm_classifier = svm.SVC(C = 50, #see slide 32 in week 4 lecture slides
                            kernel = 'linear') #see slide 35 in week 4 lecture slides
svm_classifier.fit(training_word_histograms, training_food_labels)


# In[20]:


test_file_name = ['FoodImages/Test/Pasta/pasta35.jpg']
word_histograms = dictionary.create_word_histograms(test_file_name)
predicted_food_labels = svm_classifier.predict(word_histograms)
print('Food label: ', predicted_food_labels)


# In[21]:


predicted_food_labels_svm = get_predicted_labels(svm_classifier)
get_classification_report(predicted_food_labels_svm)


# SVM with C = 10

# In[22]:


svm_classifier = svm.SVC(C = 10, #see slide 32 in week 4 lecture slides
                            kernel = 'linear') #see slide 35 in week 4 lecture slides
svm_classifier.fit(training_word_histograms, training_food_labels)
predicted_food_labels_svm = get_predicted_labels(svm_classifier)
get_classification_report(predicted_food_labels_svm)


# SVM with C = 20

# In[23]:


svm_classifier = svm.SVC(C = 20, #see slide 32 in week 4 lecture slides
                            kernel = 'linear') #see slide 35 in week 4 lecture slides
svm_classifier.fit(training_word_histograms, training_food_labels)
predicted_food_labels_svm = get_predicted_labels(svm_classifier)
get_classification_report(predicted_food_labels_svm)


# SVM with C = 30

# In[24]:


svm_classifier = svm.SVC(C = 30, #see slide 32 in week 4 lecture slides
                            kernel = 'linear') #see slide 35 in week 4 lecture slides
svm_classifier.fit(training_word_histograms, training_food_labels)
predicted_food_labels_svm = get_predicted_labels(svm_classifier)
get_classification_report(predicted_food_labels_svm)


# SVM with C = 40

# In[25]:


svm_classifier = svm.SVC(C = 40, #see slide 32 in week 4 lecture slides
                            kernel = 'linear') #see slide 35 in week 4 lecture slides
svm_classifier.fit(training_word_histograms, training_food_labels)
predicted_food_labels_svm = get_predicted_labels(svm_classifier)
get_classification_report(predicted_food_labels_svm)


# AdaBoostClassifier

# In[26]:


from sklearn.ensemble import AdaBoostClassifier
adb_classifier = AdaBoostClassifier(n_estimators = 150, #weak classifiers
                                        random_state = 0)
adb_classifier.fit(training_word_histograms, training_food_labels)


# In[27]:


test_file_name = ['FoodImages/Test/Pasta/pasta35.jpg']
word_histograms = dictionary.create_word_histograms(test_file_name)
predicted_food_labels = adb_classifier.predict(word_histograms)
print('Food label: ', predicted_food_labels)


# AdaBoostClassifier with n_estimator = 150

# In[28]:


predicted_food_labels_ada = get_predicted_labels(adb_classifier)
get_classification_report(predicted_food_labels_ada)


# AdaBoostClassifier with n_estimator = 50

# In[29]:


adb_classifier = AdaBoostClassifier(n_estimators = 50, #weak classifiers
                                        random_state = 0)
adb_classifier.fit(training_word_histograms, training_food_labels)
predicted_food_labels_ada = get_predicted_labels(adb_classifier)
get_classification_report(predicted_food_labels_ada)


# AdaBoostClassifier with n_estimator = 100

# In[30]:


adb_classifier = AdaBoostClassifier(n_estimators = 100, #weak classifiers
                                        random_state = 0)
adb_classifier.fit(training_word_histograms, training_food_labels)
predicted_food_labels_ada = get_predicted_labels(adb_classifier)
get_classification_report(predicted_food_labels_ada)


# AdaBoostClassifier with n_estimator = 200

# In[31]:


adb_classifier = AdaBoostClassifier(n_estimators = 200, #weak classifiers
                                        random_state = 0)
adb_classifier.fit(training_word_histograms, training_food_labels)
predicted_food_labels_ada = get_predicted_labels(adb_classifier)
get_classification_report(predicted_food_labels_ada)


# AdaBoostClassifier with n_estimator = 250

# In[32]:


adb_classifier = AdaBoostClassifier(n_estimators = 250, #weak classifiers
                                        random_state = 0)
adb_classifier.fit(training_word_histograms, training_food_labels)
predicted_food_labels_ada = get_predicted_labels(adb_classifier)
get_classification_report(predicted_food_labels_ada)


# In[ ]:




