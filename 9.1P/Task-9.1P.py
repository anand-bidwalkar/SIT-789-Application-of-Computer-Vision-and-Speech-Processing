#!/usr/bin/env python
# coding: utf-8

# ## SIT-789 Task-9.1P

# In[2]:


from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

authenticator = IAMAuthenticator('ytIq7zTcxqFSmRFT_CNojfy9gysPWeW_SdifAOkCY1b3') #replace {APIkey} by your API key
speech_to_text = SpeechToTextV1(
 authenticator=authenticator
)
speech_to_text.set_service_url('https://api.au-syd.speech-to-text.watson.cloud.ibm.com/instances/3bd7a18f-d01a-445d-8583-d363545149f9') #replace {url} by your URL


# In[3]:


import json
with open('SpeechtoTextData/arctic_a0005.wav', 'rb') as audio_file:
    speech_recognition_results = speech_to_text.recognize(
        audio = audio_file,
        content_type='audio/wav').get_result()

print(json.dumps(speech_recognition_results, indent = 2))


# In[4]:


with open('SpeechtoTextData/arctic_a0005.json', 'w') as outfile:
    json.dump(speech_recognition_results, outfile)


# In[5]:


with open('SpeechtoTextData/arctic_a0005.json') as infile:
    data = json.load(infile) # load data from a json file
print(data)


# In[8]:


with open('SpeechtoTextData/367-130732-0000.flac', 'rb') as audio_file:
    speech_recognition_results = speech_to_text.recognize(
        audio = audio_file,
        content_type='audio/flac').get_result()

print(json.dumps(speech_recognition_results, indent = 2))

with open('SpeechtoTextData/367-130732-0000.json', 'w') as outfile:
    json.dump(speech_recognition_results, outfile)


# In[9]:


with open('SpeechtoTextData/367-130732-0001.flac', 'rb') as audio_file:
    speech_recognition_results = speech_to_text.recognize(
        audio = audio_file,
        content_type='audio/flac').get_result()

print(json.dumps(speech_recognition_results, indent = 2))

with open('SpeechtoTextData/367-130732-0001.json', 'w') as outfile:
    json.dump(speech_recognition_results, outfile)


# In[10]:


with open('SpeechtoTextData/367-130732-0004.flac', 'rb') as audio_file:
    speech_recognition_results = speech_to_text.recognize(
        audio = audio_file,
        content_type='audio/flac').get_result()

print(json.dumps(speech_recognition_results, indent = 2))

with open('SpeechtoTextData/367-130732-0004.json', 'w') as outfile:
    json.dump(speech_recognition_results, outfile)


# In[11]:


with open('SpeechtoTextData/arctic_a0001.wav', 'rb') as audio_file:
    speech_recognition_results = speech_to_text.recognize(
        audio = audio_file,
        content_type='audio/wav').get_result()

print(json.dumps(speech_recognition_results, indent = 2))

with open('SpeechtoTextData/arctic_a0001.json', 'w') as outfile:
    json.dump(speech_recognition_results, outfile)


# In[12]:


with open('SpeechtoTextData/arctic_a0003.wav', 'rb') as audio_file:
    speech_recognition_results = speech_to_text.recognize(
        audio = audio_file,
        content_type='audio/wav').get_result()

print(json.dumps(speech_recognition_results, indent = 2))

with open('SpeechtoTextData/arctic_a0003.json', 'w') as outfile:
    json.dump(speech_recognition_results, outfile)


# In[13]:


with open('SpeechtoTextData/p232_009.wav', 'rb') as audio_file:
    speech_recognition_results = speech_to_text.recognize(
        audio = audio_file,
        content_type='audio/wav').get_result()

print(json.dumps(speech_recognition_results, indent = 2))

with open('SpeechtoTextData/p232_009.json', 'w') as outfile:
    json.dump(speech_recognition_results, outfile)


# In[14]:


with open('SpeechtoTextData/p232_010.wav', 'rb') as audio_file:
    speech_recognition_results = speech_to_text.recognize(
        audio = audio_file,
        content_type='audio/wav').get_result()

print(json.dumps(speech_recognition_results, indent = 2))

with open('SpeechtoTextData/p232_010.json', 'w') as outfile:
    json.dump(speech_recognition_results, outfile)


# In[17]:


with open('SpeechtoTextData/p232_014.wav', 'rb') as audio_file:
    speech_recognition_results = speech_to_text.recognize(
        audio = audio_file,
        content_type='audio/wav').get_result()

print(json.dumps(speech_recognition_results, indent = 2))

with open('SpeechtoTextData/p232_014.json', 'w') as outfile:
    json.dump(speech_recognition_results, outfile)


# In[18]:


with open('SpeechtoTextData/p232_030.wav', 'rb') as audio_file:
    speech_recognition_results = speech_to_text.recognize(
        audio = audio_file,
        content_type='audio/wav').get_result()

print(json.dumps(speech_recognition_results, indent = 2))

with open('SpeechtoTextData/p232_030.json', 'w') as outfile:
    json.dump(speech_recognition_results, outfile)


# In[ ]:




