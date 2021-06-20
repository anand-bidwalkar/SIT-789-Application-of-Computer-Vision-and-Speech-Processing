#!/usr/bin/env python
# coding: utf-8

# ## SIT789 Task-4.3 P

# In[1]:


get_ipython().system('pip install opencv-python==3.4.2.17')
get_ipython().system('pip install opencv-contrib-python==3.4.2.17')


# In[2]:


import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class Dictionary(object):
    def __init__(self, name, img_filenames):
        self.name = name #name of your dictionary
        self.img_filenames = img_filenames #list of image filenames        
        self.training_data = [] #this is the training data required by the K-Means algorithm
        self.words = [] #list of words, which are the centroids of clusters
        self.Sum_of_squared_distances = []
        self.num_keypoints = []
    
    def learn(self):
        sift = cv.xfeatures2d.SIFT_create()
        #load training images and compute SIFT descriptors
        for filename in self.img_filenames:
            img = cv.imread(filename)
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            list_des = sift.detectAndCompute(img_gray, None)[1]
            if list_des is None:
                self.num_keypoints.append(0)
            else:
                self.num_keypoints.append(len(list_des))
                for des in list_des:
                    self.training_data.append(des)
                    
    def compute_best_k(self):
        
        K = [10, 50, 90, 140, 180]
        for k in K:
            km = KMeans(n_clusters=k)
            km = km.fit(self.training_data)
            self.Sum_of_squared_distances.append(km.inertia_)
        
        plt.plot(K, self.Sum_of_squared_distances, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Sum_of_squared_distances')
        plt.title('Elbow Method For Optimal k')
        plt.show()                
        
    def compute_word_histogram(self, best_k):
        
        #cluster SIFT descriptors using K-means algorithm
        kmeans = KMeans(best_k)
        kmeans.fit(self.training_data)
        self.words = kmeans.cluster_centers_
        #create word histograms for training images
        training_word_histograms = [] #list of word histograms of all training images
        index = 0
        for i in range(0, len(self.img_filenames)):
            #for each file, create a histogram
            histogram = np.zeros(best_k, np.float32)
            #if some keypoints exist
            if self.num_keypoints[i] > 0:
                for j in range(0, self.num_keypoints[i]):
                    histogram[kmeans.labels_[j + index]] += 1
                index += self.num_keypoints[i]
                histogram /= self.num_keypoints[i]
                training_word_histograms.append(histogram)
                    
        return training_word_histograms
    
    def create_word_histograms(self, img_filenames, best_k):
        sift = cv.xfeatures2d.SIFT_create()
        histograms = []
        for filename in img_filenames:
            img = cv.imread(filename)
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            descriptors = sift.detectAndCompute(img_gray, None)[1]

            histogram = np.zeros(best_k, np.float32) #word histogram for the input image
            if descriptors is not None:
                for des in descriptors:
                    #find the best matching word
                    min_distance = 1111111 #this can be any large number
                    matching_word_ID = -1 #initial matching_word_ID=-1 means no matching
                    for i in range(0, best_k): #search for the best matching word
                        distance = np.linalg.norm(des - self.words[i])
                        if distance < min_distance:
                            min_distance = distance
                            matching_word_ID = i
                    histogram[matching_word_ID] += 1
                histogram /= len(descriptors) #normalise histogram to frequencies
            histograms.append(histogram)
        return histograms


# Read training files

# In[4]:


import os

vehicles = ['Car', 'Bike', 'Helicopter', 'Cruise']
path = 'Images/Train/'
training_file_names = []
training_vehicles_labels = []

for i in range(0, len(vehicles)):
    sub_path = path + vehicles[i]
    sub_file_names = [os.path.join(sub_path, f) for f in os.listdir(sub_path)]
    sub_vehicles_labels = [i] * len(sub_file_names) #create a list of N elements, all are i
    training_file_names += sub_file_names
    training_vehicles_labels += sub_vehicles_labels

print(training_file_names)
print(training_vehicles_labels)


# In[5]:


dictionary_name = 'vehicles'
dictionary = Dictionary(dictionary_name, training_file_names)


# In[6]:


dictionary.learn()


# In[7]:


dictionary.compute_best_k()


# In[8]:


training_word_histograms = dictionary.compute_word_histogram(60)


# Save model

# In[9]:


import pickle
#save dictionary
with open('vehicles_dictionary.dic', 'wb') as f: #'wb' is for binary write
    pickle.dump(dictionary, f)


# Load model

# In[10]:


import pickle #you may not need to import it if this has been done
with open('vehicles_dictionary.dic', 'rb') as f: #'rb' is for binary read
    dictionary = pickle.load(f)


# In[11]:


validaiton_file_names = []
validaiton_vehicles_labels = []

for i in range(0, len(vehicles)):
    sub_path = 'Images/Validation/' + vehicles[i]
    sub_file_names = [os.path.join(sub_path, f) for f in os.listdir(sub_path)]
    sub_vehicles_labels = [i] * len(sub_file_names) #create a list of N elements, all are i
    validaiton_file_names += sub_file_names
    validaiton_vehicles_labels += sub_vehicles_labels

#print(validaiton_file_names)
print(validaiton_vehicles_labels)



# In[12]:


def get_predicted_labels(file_names, classifier):
    predicted_vehicles_labels = []
    for file_name in file_names:
        word_histograms = dictionary.create_word_histograms([file_name], 60)
        predicted_vehicles_label = classifier.predict(word_histograms)
        predicted_vehicles_labels.append(predicted_vehicles_label)
    return predicted_vehicles_labels


# In[13]:


from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
def get_classification_report(test_vehicles_labels, predicted_vehicles_labels):    
    cm = confusion_matrix(test_vehicles_labels, predicted_vehicles_labels)
    print('Confusion Matrix', cm)
    print(classification_report(test_vehicles_labels, predicted_vehicles_labels))
    print('Accuracy Score', accuracy_score(test_vehicles_labels, predicted_vehicles_labels))


# Import sklearn libraries for classifiers

# In[14]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier


# KNeighborsClassifier with n=5

# In[15]:


num_nearest_neighbours = 5 #number of neighbours
knn = KNeighborsClassifier(n_neighbors = num_nearest_neighbours)
knn.fit(training_word_histograms, training_vehicles_labels)
predicted_vehicles_labels_knn = get_predicted_labels(validaiton_file_names, knn)
get_classification_report(validaiton_vehicles_labels, predicted_vehicles_labels_knn)


# KNeighborsClassifier with n=10

# In[16]:


num_nearest_neighbours = 10 #number of neighbours
knn = KNeighborsClassifier(n_neighbors = num_nearest_neighbours)
knn.fit(training_word_histograms, training_vehicles_labels)
predicted_vehicles_labels_knn = get_predicted_labels(validaiton_file_names, knn)
get_classification_report(validaiton_vehicles_labels, predicted_vehicles_labels_knn)


# KNeighborsClassifier with n=15

# In[17]:


num_nearest_neighbours = 15 #number of neighbours
knn = KNeighborsClassifier(n_neighbors = num_nearest_neighbours)
knn.fit(training_word_histograms, training_vehicles_labels)
predicted_vehicles_labels_knn = get_predicted_labels(validaiton_file_names, knn)
get_classification_report(validaiton_vehicles_labels, predicted_vehicles_labels_knn)


# KNeighborsClassifier with n=20

# In[18]:


num_nearest_neighbours = 20 #number of neighbours
knn = KNeighborsClassifier(n_neighbors = num_nearest_neighbours)
knn.fit(training_word_histograms, training_vehicles_labels)
predicted_vehicles_labels_knn = get_predicted_labels(validaiton_file_names, knn)
get_classification_report(validaiton_vehicles_labels, predicted_vehicles_labels_knn)


# KNeighborsClassifier with n=25

# In[19]:


num_nearest_neighbours = 25 #number of neighbours
knn = KNeighborsClassifier(n_neighbors = num_nearest_neighbours)
knn.fit(training_word_histograms, training_vehicles_labels)
predicted_vehicles_labels_knn = get_predicted_labels(validaiton_file_names, knn)
get_classification_report(validaiton_vehicles_labels, predicted_vehicles_labels_knn)


# KNeighborsClassifier with n=30

# In[20]:


num_nearest_neighbours = 30 #number of neighbours
knn = KNeighborsClassifier(n_neighbors = num_nearest_neighbours)
knn.fit(training_word_histograms, training_vehicles_labels)
predicted_vehicles_labels_knn = get_predicted_labels(validaiton_file_names, knn)
get_classification_report(validaiton_vehicles_labels, predicted_vehicles_labels_knn)


# SVM with C = 0.1

# In[21]:


svm_classifier = svm.SVC(C = 0.1, #see slide 32 in week 4 lecture slides
                            kernel = 'linear') #see slide 35 in week 4 lecture slides
svm_classifier.fit(training_word_histograms, training_vehicles_labels)
predicted_vehicles_labels_svm = get_predicted_labels(validaiton_file_names, svm_classifier)
get_classification_report(validaiton_vehicles_labels, predicted_vehicles_labels_svm)


# SVM with C = 1

# In[22]:


svm_classifier = svm.SVC(C = 1, #see slide 32 in week 4 lecture slides
                            kernel = 'linear') #see slide 35 in week 4 lecture slides
svm_classifier.fit(training_word_histograms, training_vehicles_labels)
predicted_vehicles_labels_svm = get_predicted_labels(validaiton_file_names, svm_classifier)
get_classification_report(validaiton_vehicles_labels, predicted_vehicles_labels_svm)


# SVM with C = 10

# In[23]:


svm_classifier = svm.SVC(C = 10, #see slide 32 in week 4 lecture slides
                            kernel = 'linear') #see slide 35 in week 4 lecture slides
svm_classifier.fit(training_word_histograms, training_vehicles_labels)
predicted_vehicles_labels_svm = get_predicted_labels(validaiton_file_names, svm_classifier)
get_classification_report(validaiton_vehicles_labels, predicted_vehicles_labels_svm)


# SVM with C = 100

# In[24]:


svm_classifier = svm.SVC(C = 100, #see slide 32 in week 4 lecture slides
                            kernel = 'linear') #see slide 35 in week 4 lecture slides
svm_classifier.fit(training_word_histograms, training_vehicles_labels)
predicted_vehicles_labels_svm = get_predicted_labels(validaiton_file_names, svm_classifier)
get_classification_report(validaiton_vehicles_labels, predicted_vehicles_labels_svm)


# AdaBoostClassifier with n_estimator = 50

# In[25]:


adb_classifier = AdaBoostClassifier(n_estimators = 50, #weak classifiers
                                        random_state = 0)
adb_classifier.fit(training_word_histograms, training_vehicles_labels)
predicted_vehicles_labels_ada = get_predicted_labels(validaiton_file_names, adb_classifier)
get_classification_report(validaiton_vehicles_labels, predicted_vehicles_labels_ada)


# AdaBoostClassifier with n_estimator = 100

# In[26]:


adb_classifier = AdaBoostClassifier(n_estimators = 100, #weak classifiers
                                        random_state = 0)
adb_classifier.fit(training_word_histograms, training_vehicles_labels)
predicted_vehicles_labels_ada = get_predicted_labels(validaiton_file_names, adb_classifier)
get_classification_report(validaiton_vehicles_labels, predicted_vehicles_labels_ada)


# AdaBoostClassifier with n_estimator = 150

# In[27]:


adb_classifier = AdaBoostClassifier(n_estimators = 150, #weak classifiers
                                        random_state = 0)
adb_classifier.fit(training_word_histograms, training_vehicles_labels)
predicted_vehicles_labels_ada = get_predicted_labels(validaiton_file_names, adb_classifier)
get_classification_report(validaiton_vehicles_labels, predicted_vehicles_labels_ada)


# AdaBoostClassifier with n_estimator = 200

# In[28]:


adb_classifier = AdaBoostClassifier(n_estimators = 200, #weak classifiers
                                        random_state = 0)
adb_classifier.fit(training_word_histograms, training_vehicles_labels)
predicted_vehicles_labels_ada = get_predicted_labels(validaiton_file_names, adb_classifier)
get_classification_report(validaiton_vehicles_labels, predicted_vehicles_labels_ada)


# AdaBoostClassifier with n_estimator = 250

# In[29]:


adb_classifier = AdaBoostClassifier(n_estimators = 250, #weak classifiers
                                        random_state = 0)
adb_classifier.fit(training_word_histograms, training_vehicles_labels)
predicted_vehicles_labels_ada = get_predicted_labels(validaiton_file_names, adb_classifier)
get_classification_report(validaiton_vehicles_labels, predicted_vehicles_labels_ada)


# Read Test Files

# In[31]:


test_file_names = []
test_vehicles_labels = []

for i in range(0, len(vehicles)):
    sub_path = 'Images/Test/' + vehicles[i]
    sub_file_names = [os.path.join(sub_path, f) for f in os.listdir(sub_path)]
    sub_vehicles_labels = [i] * len(sub_file_names) #create a list of N elements, all are i
    test_file_names += sub_file_names
    test_vehicles_labels += sub_vehicles_labels

print(test_file_names)
print(test_vehicles_labels)


# KNeighborsClassifier with best parameter n = 5

# In[32]:


num_nearest_neighbours = 5 #number of neighbours
knn = KNeighborsClassifier(n_neighbors = num_nearest_neighbours)
knn.fit(training_word_histograms, training_vehicles_labels)
predicted_vehicles_labels_knn = get_predicted_labels(test_file_names, knn)
get_classification_report(test_vehicles_labels, predicted_vehicles_labels_knn)


# SVM with best parameter C = 100

# In[33]:


svm_classifier = svm.SVC(C = 100, #see slide 32 in week 4 lecture slides
                            kernel = 'linear') #see slide 35 in week 4 lecture slides
svm_classifier.fit(training_word_histograms, training_vehicles_labels)
predicted_vehicles_labels_svm = get_predicted_labels(test_file_names, svm_classifier)
get_classification_report(test_vehicles_labels, predicted_vehicles_labels_svm)


# AdaBoostClassifier with best parameter n_estimator = 50

# In[34]:


adb_classifier = AdaBoostClassifier(n_estimators = 50, #weak classifiers
                                        random_state = 0)
adb_classifier.fit(training_word_histograms, training_vehicles_labels)
predicted_vehicles_labels_ada = get_predicted_labels(test_file_names, adb_classifier)
get_classification_report(test_vehicles_labels, predicted_vehicles_labels_ada)


# In[ ]:




