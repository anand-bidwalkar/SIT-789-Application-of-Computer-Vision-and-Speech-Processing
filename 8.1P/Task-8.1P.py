#!/usr/bin/env python
# coding: utf-8

# ## SIT-789 Task-8.1P

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

import IPython.display as ipd

from pydub import AudioSegment
from pydub.utils import mediainfo


# In[1]:


import librosa
import librosa.display


# In[3]:


speech = AudioSegment.from_wav('arctic_a0005.wav') #Read audio data from file
x = speech.get_array_of_samples() #samples x(t)
x_sr = speech.frame_rate #sampling rate f - see slide 24 in week 7 lecture slides

mfcc = librosa.feature.mfcc(
    np.float32(x),
    sr = x_sr, #sampling rate of the signal, which is determined from the signal
    hop_length = int(x_sr * 0.015), #15 ms
    n_mfcc = 12 #number of mfcc features
)


# In[4]:


print(mfcc.shape)


# In[5]:


mfcc_flattened = np.reshape(mfcc.T, (mfcc.shape[0] * mfcc.shape[1]))
plt.figure(figsize = (15, 5))
plt.plot(mfcc_flattened)
plt.ylabel('Amplitude')


# ## Section-2

# In[6]:


import os
emotions = ['Calm', 'Happy', 'Sad', 'Angry']
path = 'EmotionSpeech/'
training_file_names = []
training_emotion_labels = []
for i in range(0, len(emotions)):
    sub_path = path + 'Train/' + emotions[i] + '/'
    sub_file_names = [os.path.join(sub_path, f) for f in os.listdir(sub_path)]
    sub_emotion_labels = [i] * len(sub_file_names)
    training_file_names += sub_file_names
    training_emotion_labels += sub_emotion_labels


# In[7]:


import numpy as np
import librosa
from pydub import AudioSegment
from pydub.utils import mediainfo

def mfcc_extraction(audio_filename, #.wav filename
                    hop_duration, #hop_length in seconds, e.g., 0.015s (i.e., 15ms)
                    num_mfcc, #number of mfcc features
                    num_frames #number of frames
                   ):
    speech = AudioSegment.from_wav(audio_filename) #Read audio data from file
    samples = speech.get_array_of_samples() #samples x(t)
    sampling_rate = speech.frame_rate #sampling rate f - see slide 24 in week 7 lecture slides
    
    mfcc = librosa.feature.mfcc(
        np.float32(samples),
        sr = sampling_rate,
        hop_length = int(sampling_rate * hop_duration),
        n_mfcc = num_mfcc)
    
    mfcc_truncated = np.zeros((num_mfcc, num_frames), np.float32)
    for i in range(min(num_frames, mfcc.shape[1])):
        mfcc_truncated[:, i] = mfcc[:, i]
    
    #output is a vector including mfcc_truncated.shape[0] * mfcc_truncated.shape[1] elements
    return np.reshape(mfcc_truncated.T, mfcc_truncated.shape[0] * mfcc_truncated.shape[1])


# Task-1

# In[32]:


training_mfcc = []
for file_name in training_file_names:
    mfcc = mfcc_extraction(file_name, 0.015, 12, 200)
    training_mfcc.append(mfcc)


# In[11]:


test_file_names = []
test_emotion_labels = []

for i in range(0, len(emotions)):
    sub_path = path + 'Test/' + emotions[i] + '/'
    sub_file_names = [os.path.join(sub_path, f) for f in os.listdir(sub_path)]
    sub_emotion_labels = [i] * len(sub_file_names) #create a list of N elements, all are i
    test_file_names += sub_file_names
    test_emotion_labels += sub_emotion_labels


# In[43]:


from sklearn import svm
svm_classifier = svm.SVC(C = 50, #see slide 32 in week 4 lecture slides
                            kernel = 'linear') #see slide 35 in week 4 lecture slides
svm_classifier.fit(training_mfcc, training_emotion_labels)


# Task-2 SVM

# In[34]:


def get_predicted_labels(classifier, n_mfcc):
    predicted_emotion_labels = []
    for file_name in test_file_names:
        mfcc = mfcc_extraction(file_name, 0.015, n_mfcc, 200)
        predicted_emotion_label = classifier.predict([mfcc])
        predicted_emotion_labels.append(predicted_emotion_label)
    return predicted_emotion_labels


# In[42]:


from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
def get_classification_report(predicted_emotion_labels):    
    cm = confusion_matrix(test_emotion_labels, predicted_emotion_labels)
    print('Confusion Matrix', cm)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm.diagonal())
    
    acc = np.sum(cm.diagonal()) / np.sum(cm)
    print('Overall accuracy: {} %'.format(acc*100))
    
    print(classification_report(test_emotion_labels, predicted_emotion_labels))
    print('Accuracy Score', accuracy_score(test_emotion_labels, predicted_emotion_labels))


# In[44]:


predicted_emotion_labels_svm = get_predicted_labels(svm_classifier, 12)
get_classification_report(predicted_emotion_labels_svm)


# In[38]:


def train_svm_classifier(n_mfcc):
    training_mfcc = []
    for file_name in training_file_names:
        mfcc = mfcc_extraction(file_name, 0.015, n_mfcc, 200)
        training_mfcc.append(mfcc)
    
    svm_classifier = svm.SVC(C = 30, #see slide 32 in week 4 lecture slides
                            kernel = 'linear') #see slide 35 in week 4 lecture slides
    svm_classifier.fit(training_mfcc, training_emotion_labels)
    
    return svm_classifier


# In[47]:


num_mfccs = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

for n_mfcc in num_mfccs:
    print('Value of num_mfcc is', n_mfcc)
    svm_classifier = train_svm_classifier(n_mfcc)
    predicted_emotion_labels_svm = get_predicted_labels(svm_classifier, n_mfcc)
    get_classification_report(predicted_emotion_labels_svm)


# Task-3 AdaBoost Classifier

# In[49]:


from sklearn.ensemble import AdaBoostClassifier

def train_adaboost_classifier(n_mfcc):
    training_mfcc = []
    for file_name in training_file_names:
        mfcc = mfcc_extraction(file_name, 0.015, n_mfcc, 200)
        training_mfcc.append(mfcc)
    
    adb_classifier = AdaBoostClassifier(n_estimators = 150,
                                        random_state = 0)
    
    adb_classifier.fit(training_mfcc, training_emotion_labels)
    
    return adb_classifier


# In[50]:


num_mfccs = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

for n_mfcc in num_mfccs:
    print('Value of num_mfcc is', n_mfcc)
    adb_classifier = train_adaboost_classifier(n_mfcc)
    predicted_emotion_labels_ada = get_predicted_labels(adb_classifier, n_mfcc)
    get_classification_report(predicted_emotion_labels_ada)


# In[ ]:




