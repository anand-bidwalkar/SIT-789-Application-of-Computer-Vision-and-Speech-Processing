#!/usr/bin/env python
# coding: utf-8

# ## SIT-789 Task-8.2D

# ## Section-1

# In[12]:


import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
from pydub import AudioSegment
from pydub.utils import mediainfo
import librosa
import librosa.display
import os
import pickle


# Read training file names and emotion labels

# In[13]:


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


# Set the default parameters for the task

# In[15]:


sub_bands = [[300, 627], [628, 1060], [1061, 1633], [1634, 2393], [2394, 3400]]
n_freq = 16384
n_mel_freq = 3401
win_duration = 0.030
hop_duration = 0.015
num_frames = 200
window = 'hann'


# Code for the SC feature extraction

# In[4]:


def extract_sc(audio_file):
    speech = AudioSegment.from_wav(audio_file) #Read audio data from file
    x = speech.get_array_of_samples() #samples x(t)
    x_sr = speech.frame_rate #sampling rate f - see slide 24 in week 7 lecture slides
    
    mel_spec = librosa.feature.melspectrogram(
        np.float32(x),
        hop_length = int(x_sr * hop_duration),
        n_fft = n_freq,
        n_mels = n_mel_freq,
        power = 2
    )
    
    spec_trunc = np.zeros((mel_spec.shape[0], num_frames), np.float32)
    for i in range(min(num_frames, mel_spec.shape[1])):
        spec_trunc[:, i] = mel_spec[:, i]
        
    features = np.zeros((len(sub_bands), num_frames), np.float32)
    for i in range(len(sub_bands)):
        for f in range(num_frames):
            s = 0.00001
            for k in range(sub_bands[i][0], sub_bands[i][1] + 1):
                features[i][f] += k * spec_trunc[k][f]
                s += spec_trunc[k][f]
            
            features[i][f] /= s
        
    return np.reshape(features.T, features.shape[0] * features.shape[1])


# Code for SBW feature extraction

# In[5]:


def extract_sbw(audio_file):
    speech = AudioSegment.from_wav(audio_file) #Read audio data from file
    x = speech.get_array_of_samples() #samples x(t)
    x_sr = speech.frame_rate #sampling rate f - see slide 24 in week 7 lecture slides
    
    mel_spec = librosa.feature.melspectrogram(
        np.float32(x),
        hop_length = int(x_sr * hop_duration),
        n_fft = n_freq,
        n_mels = n_mel_freq,
        power = 2
    )
    
    spec_trunc = np.zeros((mel_spec.shape[0], num_frames), np.float32)
    for i in range(min(num_frames, mel_spec.shape[1])):
        spec_trunc[:, i] = mel_spec[:, i]
        
    features = np.zeros((len(sub_bands), num_frames), np.float32)
    for i in range(len(sub_bands)):
        for f in range(num_frames):
            SC = 0
            s = 0.00001
            for k in range(sub_bands[i][0], sub_bands[i][1] + 1):
                SC += k * spec_trunc[k][f]                
                s += spec_trunc[k][f]
            SC /= s
            for k in range(sub_bands[i][0], sub_bands[i][1] + 1):
                features[i][f] += (k - SC) * (k - SC)  * spec_trunc[k][f]
            
            features[i][f] /= s
        
    return np.reshape(features.T, features.shape[0] * features.shape[1])


# Code for SBE feature extraction

# In[32]:


def extract_sbe(audio_file):
    speech = AudioSegment.from_wav(audio_file) #Read audio data from file
    x = speech.get_array_of_samples() #samples x(t)
    x_sr = speech.frame_rate #sampling rate f - see slide 24 in week 7 lecture slides
    
    mel_spec = librosa.feature.melspectrogram(
        np.float32(x),
        hop_length = int(x_sr * hop_duration),
        n_fft = n_freq,
        n_mels = n_mel_freq,
        power = 2
    )
    
    spec_trunc = np.zeros((mel_spec.shape[0], num_frames), np.float32)
    for i in range(min(num_frames, mel_spec.shape[1])):
        spec_trunc[:, i] = mel_spec[:, i]
        
    features = np.zeros((len(sub_bands), num_frames), np.float32)
    for i in range(len(sub_bands)):
        for f in range(num_frames):
            for k in range(sub_bands[i][0], sub_bands[i][1] + 1):
                features[i][f] += spec_trunc[k][f]
                
    for f in range(num_frames):
        s = np.sum(features[:, f]) + 0.00001
        features[:, f] /= s
        
    return np.reshape(features.T, features.shape[0] * features.shape[1])


# Code for SFM feature extraction

# In[8]:


def extract_sfm(audio_file):
    speech = AudioSegment.from_wav(audio_file) #Read audio data from file
    x = speech.get_array_of_samples() #samples x(t)
    x_sr = speech.frame_rate #sampling rate f - see slide 24 in week 7 lecture slides
    
    mel_spec = librosa.feature.melspectrogram(
        np.float32(x),
        hop_length = int(x_sr * hop_duration),
        n_fft = n_freq,
        n_mels = n_mel_freq,
        power = 2
    )
    
    spec_trunc = np.zeros((mel_spec.shape[0], num_frames), np.float32)
    for i in range(min(num_frames, mel_spec.shape[1])):
        spec_trunc[:, i] = mel_spec[:, i]
        
    features = np.ones((len(sub_bands), num_frames), np.float32)
    for i in range(len(sub_bands)):
        p = 1 / (sub_bands[i][1] - sub_bands[i][0] + 1)
        for f in range(num_frames):
            s = 0.00001
            for k in range(sub_bands[i][0], sub_bands[i][1] + 1):
                features[i][f] *= np.power(spec_trunc[k][f], p)
                s += spec_trunc[k][f]
            features[b][f] /= (p * s)
        
    return np.reshape(features.T, features.shape[0] * features.shape[1])


# Code for SCF feature extraction

# In[27]:


def extract_scf(audio_file):
    speech = AudioSegment.from_wav(audio_file) #Read audio data from file
    x = speech.get_array_of_samples() #samples x(t)
    x_sr = speech.frame_rate #sampling rate f - see slide 24 in week 7 lecture slides
    
    mel_spec = librosa.feature.melspectrogram(
        np.float32(x),
        hop_length = int(x_sr * hop_duration),
        n_fft = n_freq,
        n_mels = n_mel_freq,
        power = 2
    )
    
    spec_trunc = np.zeros((mel_spec.shape[0], num_frames), np.float32)
    for i in range(min(num_frames, mel_spec.shape[1])):
        spec_trunc[:, i] = mel_spec[:, i]
        
    features = np.zeros((len(sub_bands), num_frames), np.float32)
    for i in range(len(sub_bands)):
        p = 1 / (sub_bands[i][1] - sub_bands[i][0] + 1)
        for f in range(num_frames):
            s = 0.000001
            for k in range(sub_bands[i][0], sub_bands[i][1] + 1):
                s += spec_trunc[k][f]
            features[i][f] = np.max(spec_trunc[sub_bands[i][0]:sub_bands[i][1] + 1][f]) / (p * s)
        
    return np.reshape(features.T, features.shape[0] * features.shape[1])


# Code for RE feature extraction

# In[9]:


def extract_re(audio_file):
    alpha = 3
    speech = AudioSegment.from_wav(audio_file) #Read audio data from file
    x = speech.get_array_of_samples() #samples x(t)
    x_sr = speech.frame_rate #sampling rate f - see slide 24 in week 7 lecture slides
    
    mel_spec = librosa.feature.melspectrogram(
        np.float32(x),
        hop_length = int(x_sr * hop_duration),
        n_fft = n_freq,
        n_mels = n_mel_freq,
        power = 2
    )
    
    spec_trunc = np.zeros((mel_spec.shape[0], num_frames), np.float32)
    for i in range(min(num_frames, mel_spec.shape[1])):
        spec_trunc[:, i] = abs(mel_spec[:, i])
        
    features = np.zeros((len(sub_bands), num_frames), np.float32)
    for i in range(len(sub_bands)):
        for f in range(num_frames):
            m = 0.00001 + np.sum(spec_trunc[sub_bands[i][0]:sub_bands[i][1] + 1][f])
            s = 0.00001
            for k in range(sub_bands[i][0], sub_bands[i][1] + 1):
                s += np.power(spec_trunc[k][f] / m, alpha)
            features[i][f] = 1 / (1 - alpha) * np.log2(s)
        
    return np.reshape(features.T, features.shape[0] * features.shape[1])


# Code for SE feature extraction

# In[10]:


def extract_se(audio_file):
    speech = AudioSegment.from_wav(audio_file) #Read audio data from file
    x = speech.get_array_of_samples() #samples x(t)
    x_sr = speech.frame_rate #sampling rate f - see slide 24 in week 7 lecture slides
    
    mel_spec = librosa.feature.melspectrogram(
        np.float32(x),
        hop_length = int(x_sr * hop_duration),
        n_fft = n_freq,
        n_mels = n_mel_freq,
        power = 2
    )
    
    spec_trunc = np.zeros((mel_spec.shape[0], num_frames), np.float32)
    for i in range(min(num_frames, mel_spec.shape[1])):
        spec_trunc[:, i] = abs(mel_spec[:, i])
        
    features = np.zeros((len(sub_bands), num_frames), np.float32)
    for i in range(len(sub_bands)):
        for f in range(num_frames):
            s = 0.00001 + np.sum(spec_trunc[sub_bands[i][0]:sub_bands[i][1] + 1][f])            
            for k in range(sub_bands[i][0], sub_bands[i][1] + 1):
                ratio = 0.00001 + spec_trunc[k][f] / s            
                features[i][f] -= ratio * np.log2(ratio)
        
    return np.reshape(features.T, features.shape[0] * features.shape[1])


# Code to get the test file names and emotion labels

# In[39]:


test_file_names = []
test_emotion_labels = []

for i in range(0, len(emotions)):
    sub_path = path + 'Test/' + emotions[i] + '/'
    sub_file_names = [os.path.join(sub_path, f) for f in os.listdir(sub_path)]
    sub_emotion_labels = [i] * len(sub_file_names) #create a list of N elements, all are i
    test_file_names += sub_file_names
    test_emotion_labels += sub_emotion_labels


# In[17]:


def get_features(file_list, func):
    extracted_features = []
    for file_name in file_list:
        extracted_features.append(func(file_name))
    return extracted_features


# In[51]:


import json
train_features_sc = get_features(training_file_names, extract_sc)
test_features_sc = get_features(test_file_names, extract_sc)


# In[70]:


import pickle
with open('train_features_sc.spc', 'wb') as outfile:
    pickle.dump(train_features_sc, outfile)
with open('test_features_sc.spc', 'wb') as outfile:
    pickle.dump(test_features_sc, outfile)


# In[71]:


train_features_sbw = get_features(training_file_names, extract_sbw)
test_features_sbw = get_features(test_file_names, extract_sbw)


# In[72]:


with open('train_features_sbw.spc', 'wb') as outfile:
    pickle.dump(train_features_sbw, outfile)
with open('test_features_sbw.spc', 'wb') as outfile:
    pickle.dump(test_features_sbw, outfile)


# In[73]:


train_features_sbe = get_features(training_file_names, extract_sbe)
test_features_sbe = get_features(test_file_names, extract_sbe)
with open('train_features_sbe.spc', 'wb') as outfile:
    pickle.dump(train_features_sbe, outfile)
with open('test_features_sbe.spc', 'wb') as outfile:
    pickle.dump(test_features_sbe, outfile)


# In[9]:


train_features_sfm = get_features(training_file_names, extract_sfm)
test_features_sfm = get_features(test_file_names, extract_sfm)
with open('train_features_sfm.spc', 'wb') as outfile:
    pickle.dump(train_features_sfm, outfile)
with open('test_features_sfm.spc', 'wb') as outfile:
    pickle.dump(test_features_sfm, outfile)


# In[36]:


train_features_scf = get_features(training_file_names, extract_scf)
test_features_scf = get_features(test_file_names, extract_scf)
with open('train_features_scf.spc', 'wb') as outfile:
    pickle.dump(train_features_scf, outfile)
with open('test_features_scf.spc', 'wb') as outfile:
    pickle.dump(test_features_scf, outfile)


# In[76]:


train_features_re = get_features(training_file_names, extract_re)
test_features_re = get_features(test_file_names, extract_re)
with open('train_features_re.spc', 'wb') as outfile:
    pickle.dump(train_features_re, outfile)
with open('test_features_re.spc', 'wb') as outfile:
    pickle.dump(test_features_re, outfile)


# In[77]:


train_features_se = get_features(training_file_names, extract_se)
test_features_se = get_features(test_file_names, extract_se)
with open('train_features_se.spc', 'wb') as outfile:
    pickle.dump(train_features_se, outfile)
with open('test_features_se.spc', 'wb') as outfile:
    pickle.dump(test_features_se, outfile)


# ## Section-2

# In[18]:


def get_predicted_labels(classifier, test_features_list):
    predicted_emotion_labels = []
    for test_feature in test_features_list:
        predicted_emotion_label = classifier.predict([test_feature])
        predicted_emotion_labels.append(predicted_emotion_label)
    return predicted_emotion_labels


# In[41]:


from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
def get_classification_report(predicted_emotion_labels):    
    cm = confusion_matrix(test_emotion_labels, predicted_emotion_labels)
    print('Confusion Matrix', cm)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm.diagonal())
    
    acc = np.sum(cm.diagonal()) / np.sum(cm)
    print('Overall accuracy: {} %'.format(acc*100))
    
    from sklearn.metrics import f1_score
    print('F1-Score', f1_score(test_emotion_labels, predicted_emotion_labels, average='macro'))
    
    print(classification_report(test_emotion_labels, predicted_emotion_labels))
    print('Accuracy Score', accuracy_score(test_emotion_labels, predicted_emotion_labels))


# In[20]:


from sklearn import svm
def train_svm_classifier(training_features):    
    svm_classifier = svm.SVC(C = 10, #see slide 32 in week 4 lecture slides
                            kernel = 'linear') #see slide 35 in week 4 lecture slides
    svm_classifier.fit(training_features, training_emotion_labels)
    
    return svm_classifier


# In[21]:


from sklearn.ensemble import AdaBoostClassifier

def train_adaboost_classifier(training_features):
    adb_classifier = AdaBoostClassifier(n_estimators = 200,
                                        random_state = 0)
    
    adb_classifier.fit(training_features, training_emotion_labels)
    
    return adb_classifier


# ## 1. Spectral Centroid (SC)

# In[19]:


with open('train_features_sc.spc', 'rb') as f:
    train_features_sc = pickle.load(f)
with open('test_features_sc.spc', 'rb') as f:
    test_features_sc = pickle.load(f)    


# In[20]:


svm_classifier = train_svm_classifier(train_features_sc)
predicted_emotion_labels_svm = get_predicted_labels(svm_classifier, test_features_sc)
get_classification_report(predicted_emotion_labels_svm)


# In[21]:


ada_classifier = train_adaboost_classifier(train_features_sc)
predicted_emotion_labels_ada = get_predicted_labels(ada_classifier, test_features_sc)
get_classification_report(predicted_emotion_labels_ada)


# ## 2. Spectral Bandwidth (SBW)

# In[22]:


with open('train_features_sbw.spc', 'rb') as f:
    train_features_sbw = pickle.load(f)
with open('test_features_sbw.spc', 'rb') as f:
    test_features_sbw = pickle.load(f)    


# In[23]:


svm_classifier = train_svm_classifier(train_features_sbw)
predicted_emotion_labels_svm = get_predicted_labels(svm_classifier, test_features_sbw)
get_classification_report(predicted_emotion_labels_svm)


# In[24]:


ada_classifier = train_adaboost_classifier(train_features_sbw)
predicted_emotion_labels_ada = get_predicted_labels(ada_classifier, test_features_sbw)
get_classification_report(predicted_emotion_labels_ada)


# ## 3. Spectral Band Energy (SBE)

# In[25]:


with open('train_features_sbe.spc', 'rb') as f:
    train_features_sbe = pickle.load(f)
with open('test_features_sbe.spc', 'rb') as f:
    test_features_sbe = pickle.load(f)    


# In[26]:


svm_classifier = train_svm_classifier(train_features_sbe)
predicted_emotion_labels_svm = get_predicted_labels(svm_classifier, test_features_sbe)
get_classification_report(predicted_emotion_labels_svm)


# In[27]:


ada_classifier = train_adaboost_classifier(train_features_sbe)
predicted_emotion_labels_ada = get_predicted_labels(ada_classifier, test_features_sbe)
get_classification_report(predicted_emotion_labels_ada)


# ## 4. Spectral Flatness Measure (SFM)

# In[8]:


with open('train_features_sfm.spc', 'rb') as f:
    train_features_sfm = pickle.load(f)
with open('test_features_sfm.spc', 'rb') as f:
    test_features_sfm = pickle.load(f)    


# In[15]:


svm_classifier = train_svm_classifier(train_features_sfm)
predicted_emotion_labels_svm = get_predicted_labels(svm_classifier, test_features_sfm)
get_classification_report(predicted_emotion_labels_svm)


# In[16]:


ada_classifier = train_adaboost_classifier(train_features_sfm)
predicted_emotion_labels_ada = get_predicted_labels(ada_classifier, test_features_sfm)
get_classification_report(predicted_emotion_labels_ada)


# ## 5. Spectral Crest Factor (SCF)

# In[37]:


import pickle
with open('train_features_scf.spc', 'rb') as f:
    train_features_scf = pickle.load(f)
with open('test_features_scf.spc', 'rb') as f:
    test_features_scf = pickle.load(f)    


# In[42]:


svm_classifier = train_svm_classifier(train_features_scf)
predicted_emotion_labels_svm = get_predicted_labels(svm_classifier, test_features_scf)
get_classification_report(predicted_emotion_labels_svm)


# In[32]:


ada_classifier = train_adaboost_classifier(train_features_scf)
predicted_emotion_labels_ada = get_predicted_labels(ada_classifier, test_features_scf)
get_classification_report(predicted_emotion_labels_ada)


# ## 6. Renyi Entropy (RE)

# In[13]:


with open('train_features_re.spc', 'rb') as f:
    train_features_re = pickle.load(f)
with open('test_features_re.spc', 'rb') as f:
    test_features_re = pickle.load(f)    


# In[14]:


svm_classifier = train_svm_classifier(train_features_re)
predicted_emotion_labels_svm = get_predicted_labels(svm_classifier, test_features_re)
get_classification_report(predicted_emotion_labels_svm)


# In[15]:


ada_classifier = train_adaboost_classifier(train_features_re)
predicted_emotion_labels_ada = get_predicted_labels(ada_classifier, test_features_re)
get_classification_report(predicted_emotion_labels_ada)


# ## 7. Shannon Entropy (SE)

# In[16]:


with open('train_features_se.spc', 'rb') as f:
    train_features_se = pickle.load(f)
with open('test_features_se.spc', 'rb') as f:
    test_features_se = pickle.load(f)    


# In[17]:


svm_classifier = train_svm_classifier(train_features_se)
predicted_emotion_labels_svm = get_predicted_labels(svm_classifier, test_features_se)
get_classification_report(predicted_emotion_labels_svm)


# In[18]:


ada_classifier = train_adaboost_classifier(train_features_se)
predicted_emotion_labels_ada = get_predicted_labels(ada_classifier, test_features_se)
get_classification_report(predicted_emotion_labels_ada)


# ## 8. Combination SE & SC

# In[35]:


new_feature_train = []
new_feature_test = []
n_samples = 128

for i in range(128):
    new_feature_train.append(np.concatenate([train_features_sc[i], train_features_se[i]]))
    new_feature_test.append(np.concatenate([test_features_sc[i], test_features_se[i]]))


# In[36]:


svm_classifier = train_svm_classifier(new_feature_train)
predicted_emotion_labels_svm = get_predicted_labels(svm_classifier, new_feature_test)
get_classification_report(predicted_emotion_labels_svm)


# In[37]:


ada_classifier = train_adaboost_classifier(new_feature_train)
predicted_emotion_labels_ada = get_predicted_labels(ada_classifier, new_feature_test)
get_classification_report(predicted_emotion_labels_ada)


# ## Mel-Spectrogram

# In[10]:


def plot_spectogram(audio_file):

    speech = AudioSegment.from_wav(audio_file) #Read audio data from file
    x = speech.get_array_of_samples() #samples x(t)
    x_sr = speech.frame_rate #sampling rate f - see slide 24 in week 7 lecture slides
    
    mel_spec = librosa.feature.melspectrogram(
        np.float32(x),
        hop_length = int(x_sr * hop_duration),
        n_fft = n_freq,
        n_mels = n_mel_freq,
        power = 2
    )

    plt.figure(figsize = (15, 5))
    #convert the amplitude to decibels, just for illustration purpose
    mel_spec_sdb = librosa.amplitude_to_db(abs(mel_spec))
    librosa.display.specshow(
                                #spectrogram
                                mel_spec_sdb,

                                #sampling rate
                                sr = x_sr,

                                #label for horizontal axis
                                x_axis = 'time',

                                #presentation scale
                                y_axis = 'linear',

                                #hop_lenght
                                hop_length = int(x_sr * hop_duration)
                            )


# In[23]:


plot_spectogram('EmotionSpeech/Test/Calm/03-01-02-01-02-01-05.wav')


# In[24]:


plot_spectogram('EmotionSpeech/Test/Happy/03-01-03-01-01-02-05.wav')


# In[25]:


plot_spectogram('EmotionSpeech/Test/Sad/03-01-04-01-01-02-08.wav')


# In[26]:


plot_spectogram('EmotionSpeech/Test/Angry/03-01-05-02-02-01-08.wav')


# In[ ]:




