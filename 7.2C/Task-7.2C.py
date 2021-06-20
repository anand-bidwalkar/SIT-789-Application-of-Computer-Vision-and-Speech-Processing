#!/usr/bin/env python
# coding: utf-8

# ## SIT-789 Task-7.2C

# Section-1

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd
from pydub import AudioSegment
from pydub.utils import mediainfo


# In[32]:


noisy_speech_1 = AudioSegment.from_wav('NoisySignal/Station/sp01_station_sn5.wav')
noisy_s_1 = noisy_speech.get_array_of_samples() # samples x(t)
noisy_f_1 = noisy_speech.frame_rate # sampling rate f - see slide 24 in week 7 lecture slides


# In[33]:


plt.figure(figsize = (15, 5))
plt.plot(noisy_s_1)
plt.xlabel('Samples')
plt.ylabel('Amplitude')


# In[34]:


#range of frequencies of interest for speech signal.
#It can be any positive value, but should be a power of 2
freq_range = 2048

#window size: the number of samples per frame
#each frame is of 30ms
win_length = int(noisy_f_1 * 0.03)

#number of samples between two consecutive frames
#by default, hop_length = win_length / 4
hop_length = int(win_length / 2)

#windowing technique
window = 'hann'

noisy_S = librosa.stft(np.float32(noisy_s_1),
                       n_fft = freq_range,
                       window = window,
                       hop_length = hop_length,
                       win_length = win_length)

plt.figure(figsize = (15, 5))
#convert the amplitude to decibels, just for illustration purpose
noisy_Sdb = librosa.amplitude_to_db(abs(noisy_S))
librosa.display.specshow(
                            #spectrogram
                            noisy_Sdb,
    
                            #sampling rate
                            sr = noisy_f_1,

                            #label for horizontal axis
                            x_axis = 'time',

                            #presentation scale
                            y_axis = 'linear',
    
                            #hop_lenght
                            hop_length = hop_length
                        )


# In[5]:


def plot_spectogram(filtered_s_audio):

    filtered_s_audio_s = filtered_s_audio.get_array_of_samples() # samples x(t)
    filtered_s_audio_f = filtered_s_audio.frame_rate # sampling rate f - see slide 24 in week 7 lecture slides

    #range of frequencies of interest for speech signal.
    #It can be any positive value, but should be a power of 2
    freq_range = 2048

    #window size: the number of samples per frame
    #each frame is of 30ms
    win_length = int(filtered_s_audio_f * 0.03)

    #number of samples between two consecutive frames
    #by default, hop_length = win_length / 4
    hop_length = int(win_length / 2)

    #windowing technique
    window = 'hann'

    filtered_s_audio_S = librosa.stft(np.float32(filtered_s_audio_s),
                           n_fft = freq_range,
                           window = window,
                           hop_length = hop_length,
                           win_length = win_length)

    plt.figure(figsize = (15, 5))
    #convert the amplitude to decibels, just for illustration purpose
    filtered_s_audio_Sdb = librosa.amplitude_to_db(abs(filtered_s_audio_S))
    librosa.display.specshow(
                                #spectrogram
                                filtered_s_audio_Sdb,

                                #sampling rate
                                sr = filtered_s_audio_f,

                                #label for horizontal axis
                                x_axis = 'time',

                                #presentation scale
                                y_axis = 'linear',

                                #hop_lenght
                                hop_length = hop_length
                            )


# In[35]:


from scipy import signal
import array
import pydub
from pydub import AudioSegment

def apply_band_pass_filter(noisy_speech, noisy_s, noisy_f, cutoff_freq, filter_type, file_name):

    #order
    order = 10

    #sampling frequency
    sampling_freq = noisy_f

    #filter
    h = signal.butter(N = order,
                      fs = sampling_freq,
                      Wn = cutoff_freq,
                      btype = filter_type,
                      analog = False,
                      output = 'sos')

    filtered_s = signal.sosfilt(h, noisy_s)



    filtered_s_audio = pydub.AudioSegment(
                                            #raw data
                                            data = array.array(noisy_speech.array_type, np.float16(filtered_s)),

                                            #2 bytes = 16 bit samples
                                            sample_width = 2,

                                            #frame rate
                                            frame_rate = noisy_f,

                                            #channels = 1 for mono and 2 for stereo
                                            channels = 1
                                        )
    filtered_s_audio.export(file_name, format = 'wav')
    return filtered_s_audio


# sp01_station_sn5_lowpass.wav

# In[36]:


low_pass_filtered_S = apply_band_pass_filter(noisy_speech_1, noisy_s_1, noisy_f_1, 1000, 'lowpass', 'sp01_station_sn5_lowpass.wav')
plot_spectogram(low_pass_filtered_S)


# In[37]:


high_pass_filtered_S = apply_band_pass_filter(noisy_speech_1, noisy_s_1, noisy_f_1, 200, 'highpass', 'sp01_station_sn5_highpass.wav')
plot_spectogram(high_pass_filtered_S)


# In[38]:


band_pass_filtered_S = apply_band_pass_filter(noisy_speech_1, noisy_s_1, noisy_f_1, [200, 1000], 'bandpass', 'sp01_station_sn5_highpass.wav')
plot_spectogram(band_pass_filtered_S)


# sp02_station_sn5_lowpass.wav

# In[39]:


noisy_speech_2 = AudioSegment.from_wav('NoisySignal/Station/sp02_station_sn5.wav')
noisy_s_2 = noisy_speech.get_array_of_samples() # samples x(t)
noisy_f_2 = noisy_speech.frame_rate # sampling rate f - see slide 24 in week 7 lecture slides


# In[40]:


low_pass_filtered_S = apply_band_pass_filter(noisy_speech_2, noisy_s_2, noisy_f_2, 1000, 'lowpass', 'sp02_station_sn5_lowpass.wav')
plot_spectogram(low_pass_filtered_S)


# In[41]:


high_pass_filtered_S = apply_band_pass_filter(noisy_speech_2, noisy_s_2, noisy_f_2, 200, 'highpass', 'sp02_station_sn5_highpass.wav')
plot_spectogram(high_pass_filtered_S)


# In[42]:


band_pass_filtered_S = apply_band_pass_filter(noisy_speech_2, noisy_s_2, noisy_f_2, [200, 1000], 'bandpass', 'sp02_station_sn5_highpass.wav')
plot_spectogram(band_pass_filtered_S)


# sp03_station_sn5_lowpass.wav

# In[44]:


noisy_speech_3 = AudioSegment.from_wav('NoisySignal/Station/sp03_station_sn5.wav')
noisy_s_3 = noisy_speech.get_array_of_samples() # samples x(t)
noisy_f_3 = noisy_speech.frame_rate # sampling rate f - see slide 24 in week 7 lecture slides


# In[45]:


low_pass_filtered_S = apply_band_pass_filter(noisy_speech_3, noisy_s_3, noisy_f_3, 1000, 'lowpass', 'sp03_station_sn5_lowpass.wav')
plot_spectogram(low_pass_filtered_S)


# In[46]:


high_pass_filtered_S = apply_band_pass_filter(noisy_speech_3, noisy_s_3, noisy_f_3, 200, 'highpass', 'sp03_station_sn5_highpass.wav')
plot_spectogram(high_pass_filtered_S)


# In[47]:


band_pass_filtered_S = apply_band_pass_filter(noisy_speech_3, noisy_s_3, noisy_f_3, [200, 1000], 'bandpass', 'sp03_station_sn5_highpass.wav')
plot_spectogram(band_pass_filtered_S)


# sp04_station_sn5_lowpass.wav

# In[48]:


noisy_speech_4 = AudioSegment.from_wav('NoisySignal/Station/sp04_station_sn5.wav')
noisy_s_4 = noisy_speech.get_array_of_samples() # samples x(t)
noisy_f_4 = noisy_speech.frame_rate # sampling rate f - see slide 24 in week 7 lecture slides


# In[49]:


low_pass_filtered_S = apply_band_pass_filter(noisy_speech_4, noisy_s_4, noisy_f_4, 1000, 'lowpass', 'sp04_station_sn5_lowpass.wav')
plot_spectogram(low_pass_filtered_S)


# In[50]:


high_pass_filtered_S = apply_band_pass_filter(noisy_speech_4, noisy_s_4, noisy_f_4, 200, 'highpass', 'sp04_station_sn5_highpass.wav')
plot_spectogram(high_pass_filtered_S)


# In[51]:


band_pass_filtered_S = apply_band_pass_filter(noisy_speech_4, noisy_s_4, noisy_f_4, [200, 1000], 'bandpass', 'sp04_station_sn5_highpass.wav')
plot_spectogram(band_pass_filtered_S)


# sp05_station_sn5_lowpass.wav

# In[52]:


noisy_speech_5 = AudioSegment.from_wav('NoisySignal/Station/sp05_station_sn5.wav')
noisy_s_5 = noisy_speech.get_array_of_samples() # samples x(t)
noisy_f_5 = noisy_speech.frame_rate # sampling rate f - see slide 24 in week 7 lecture slides


# In[53]:


low_pass_filtered_S = apply_band_pass_filter(noisy_speech_5, noisy_s_5, noisy_f_5, 1000, 'lowpass', 'sp05_station_sn5_lowpass.wav')
plot_spectogram(low_pass_filtered_S)


# In[54]:


high_pass_filtered_S = apply_band_pass_filter(noisy_speech_5, noisy_s_5, noisy_f_5, 200, 'highpass', 'sp05_station_sn5_highpass.wav')
plot_spectogram(high_pass_filtered_S)


# In[55]:


band_pass_filtered_S = apply_band_pass_filter(noisy_speech_5, noisy_s_5, noisy_f_5, [200, 1000], 'bandpass', 'sp05_station_sn5_highpass.wav')
plot_spectogram(band_pass_filtered_S)


# In[10]:


# Read audio data from file
noisy_station = AudioSegment.from_wav('Noise/Station/Station_1.wav')
d = noisy_station.get_array_of_samples() # samples x(t)
d_f = noisy_station.frame_rate # sampling rate f - see slide 24 in week 7 lecture slides

#window size: the number of samples per frame
#each frame is of 30ms
win_length = int(d_f * 0.03)

#number of samples between two consecutive frames
#by default, hop_length = win_length / 4
hop_length = int(win_length / 2)

D = librosa.stft(np.float32(d),
                 n_fft = 2048,
                 window = 'hann',
                 hop_length = hop_length,
                 win_length = win_length)
mag_D = abs(D)
means_mag_D = np.mean(mag_D, axis = 1)


# In[11]:


clean_signal = AudioSegment.from_wav('CleanSignal/sp01.wav')
plot_spectogram(clean_signal)


# In[28]:


clean_signal = AudioSegment.from_wav('CleanSignal/sp02.wav')
plot_spectogram(clean_signal)


# In[29]:


clean_signal = AudioSegment.from_wav('CleanSignal/sp03.wav')
plot_spectogram(clean_signal)


# In[30]:


clean_signal = AudioSegment.from_wav('CleanSignal/sp04.wav')
plot_spectogram(clean_signal)


# In[31]:


clean_signal = AudioSegment.from_wav('CleanSignal/sp05.wav')
plot_spectogram(clean_signal)


# In[69]:


def spectral_subtraction(file_name, save_file, second_file):

    # Read audio data from file
    noisy_speech = AudioSegment.from_wav(file_name)
    y = noisy_speech.get_array_of_samples() # samples x(t)
    y_f = noisy_speech.frame_rate # sampling rate f - see slide 24 in week 7 lecture slides

    #window size: the number of samples per frame
    #each frame is of 30ms
    win_length = int(y_f * 0.03)

    #number of samples between two consecutive frames
    #by default, hop_length = win_length / 4
    hop_length = int(win_length / 2)

    Y = librosa.stft(np.float32(y),
                     n_fft = 2048,
                     window = 'hann',
                     hop_length = hop_length,
                     win_length = win_length)
    mag_Y = abs(Y)

    H = np.zeros((mag_Y.shape[0], mag_Y.shape[1]), np.float32)
    for k in range(H.shape[0]):
        for t in range(H.shape[1]):
            H[k][t] = np.sqrt(max(0, 1 - (means_mag_D[k] * means_mag_D[k]) / (mag_Y[k][t] * mag_Y[k][t])))       

    S_hat = np.zeros((mag_Y.shape[0], mag_Y.shape[1]), np.float32)
    for k in range(H.shape[0]):
        for t in range(H.shape[1]):
            S_hat[k][t] = H[k][t] * Y[k][t]
        
    win_length = int(y_f * 0.03)
    hop_length = int(win_length / 2)
    s_hat = librosa.istft(S_hat, win_length = win_length, hop_length = hop_length, length = len(y))
    
    audio_data = array.array(noisy_speech.array_type, np.float32(s_hat)) if not second_file else array.array('l', np.float32(s_hat))

    s_hat_audio = pydub.AudioSegment(
        data = audio_data,
        sample_width = 2,
        frame_rate = y_f,
        channels = 1)
    s_hat_audio.export(save_file, format = 'wav')

    plot_spectogram(s_hat_audio)


# In[70]:


spectral_subtraction('NoisySignal/Station/sp01_station_sn5.wav', 'sp01_station_sn5_spectralsubtraction.wav', False)


# In[71]:


spectral_subtraction('NoisySignal/Station/sp02_station_sn5.wav', 'sp02_station_sn5_spectralsubtraction.wav', True)


# In[72]:


spectral_subtraction('NoisySignal/Station/sp03_station_sn5.wav', 'sp03_station_sn5_spectralsubtraction.wav', False)


# In[73]:


spectral_subtraction('NoisySignal/Station/sp04_station_sn5.wav', 'sp04_station_sn5_spectralsubtraction.wav', False)


# In[74]:


spectral_subtraction('NoisySignal/Station/sp05_station_sn5.wav', 'sp05_station_sn5_spectralsubtraction.wav', False)


# In[63]:


def wiener_filter(file_name, save_file, second_file):

    # Read audio data from file
    noisy_speech = AudioSegment.from_wav(file_name)
    y = noisy_speech.get_array_of_samples() # samples x(t)
    y_f = noisy_speech.frame_rate # sampling rate f - see slide 24 in week 7 lecture slides

    #window size: the number of samples per frame
    #each frame is of 30ms
    win_length = int(y_f * 0.03)

    #number of samples between two consecutive frames
    #by default, hop_length = win_length / 4
    hop_length = int(win_length / 2)

    Y = librosa.stft(np.float32(y),
                     n_fft = 2048,
                     window = 'hann',
                     hop_length = hop_length,
                     win_length = win_length)
    mag_Y = abs(Y)

    H = np.zeros((mag_Y.shape[0], mag_Y.shape[1]), np.float32)
    for k in range(H.shape[0]):
        for t in range(H.shape[1]):
            H[k][t] = max(0, 1 - (means_mag_D[k] * means_mag_D[k]) / (mag_Y[k][t] * mag_Y[k][t]))

    S_hat = np.zeros((mag_Y.shape[0], mag_Y.shape[1]), np.float32)
    for k in range(H.shape[0]):
        for t in range(H.shape[1]):
            S_hat[k][t] = H[k][t] * Y[k][t]
        
    win_length = int(y_f * 0.03)
    hop_length = int(win_length / 2)
    s_hat = librosa.istft(S_hat, win_length = win_length, hop_length = hop_length, length = len(y))
    
    audio_data = array.array(noisy_speech.array_type, np.float32(s_hat)) if not second_file else array.array('l', np.float32(s_hat))

    s_hat_audio = pydub.AudioSegment(
        data = audio_data,
        sample_width = 2,
        frame_rate = y_f,
        channels = 1)
    s_hat_audio.export(save_file, format = 'wav')

    plot_spectogram(s_hat_audio)


# In[64]:


wiener_filter('NoisySignal/Station/sp01_station_sn5.wav', 'sp01_station_sn5_wienerfilter.wav', False)


# In[65]:


wiener_filter('NoisySignal/Station/sp02_station_sn5.wav', 'sp02_station_sn5_wienerfilter.wav', True)


# In[66]:


wiener_filter('NoisySignal/Station/sp03_station_sn5.wav', 'sp03_station_sn5_wienerfilter.wav', False)


# In[67]:


wiener_filter('NoisySignal/Station/sp04_station_sn5.wav', 'sp04_station_sn5_wienerfilter.wav', False)


# In[68]:


wiener_filter('NoisySignal/Station/sp05_station_sn5.wav', 'sp05_station_sn5_wienerfilter.wav', False)


# In[ ]:




