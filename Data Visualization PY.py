#!/usr/bin/env python
# coding: utf-8

# ##  Data exploration

# In[4]:


'''Data Exploration
Using the data set, audio files by plotting out the waveform and a spectrogram to see the sample audio files.'''
import librosa
import librosa.display
import numpy as np
import pandas as pd

import os
import sys
import glob
import numpy as np
import librosa
from scipy.io import wavfile

import matplotlib.pyplot as plt
import scipy.io.wavfile
import numpy as np
import sys


# In[5]:


df = pd.read_csv('emotion_capstone_final_dataframe_Anuj_Goyal.csv')


# In[6]:


df.shape


# In[7]:


df = pd.DataFrame(data=df)


# In[8]:


df.head()


# In[9]:


df.set_index('Unnamed: 0')


# In[10]:


plt.figure()
plt.hist(df['emotion'], bins=9, width=0.5)
plt.figure


# In[29]:


fd = os.getcwd()
fd


# ** The csv was created by combining both the csvs RAVDESS & TESS

# In[31]:


sample_file = 'capstoneproject-speech-emotion-machine-learning-master/RawData/Ravdess/Actor_01/03-01-01-01-01-01-01.wav'


# In[32]:


#sample 
#librosa.core.load(path, sr=22050, mono=True, offset=0.5, duration=None, dtype=<class 'numpy.float32'>, res_type='kaiser_best')
res_type_s = 'kaiser_best'
duration_s = None
sample_rate_s = 22050
offset_s = 0.5

#Mfcc
#librosa.feature.mfcc(y=None, sr=22050, S=None, n_mfcc=20, dct_type=2, norm='ortho', lifter=0, **kwargs)
mfcc_sample_rate = 22050
n_feat = 13
n_mfcc = 40 #number of MFCCs to return => number of features
n_fft = 552
window = 0.4
test_shift = 0.1
duration = 2.5
axis_mfcc = 1 #axis =0 means along the columns and axis =1 along the row


# In[33]:


#X = audio time series
#sample_rate = sampling rate of X

X,sample_rate = librosa.load(sample_file, 
                                  res_type = res_type_s,
                                  duration = duration_s,
                                  sr = sample_rate_s,
                                  offset = offset_s,
                                 mono=False)

print(X)
print(sample_rate)


# In[34]:


time = np.arange(0,len(X))/sample_rate
print(time) # prints timeline


# In[35]:


fig, ax = plt.subplots()
ax.plot(time,X)
ax.set(xlabel='Time(s)',ylabel='Sound amplitude')
plt.show()


# In[36]:


pitches, magnitudes = librosa.piptrack(y=X, sr=sample_rate)
print(pitches)
print('///')
print(magnitudes)
plt.subplot(212)
plt.show()
plt.plot(pitches)
plt.show()


# In[37]:


y, sr = librosa.load(sample_file)
plt.figure(figsize=(10, 4))

D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
plt.subplot(4, 2, 1)
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram')


# In[38]:


librosa_audio, librosa_sample_rate = librosa.load(sample_file)

mfccs = librosa.feature.mfcc(y=librosa_audio, sr=librosa_sample_rate, n_mfcc=40)
print(mfccs.shape)


# In[39]:


import librosa.display
librosa.display.specshow(mfccs, sr=librosa_sample_rate, x_axis='time')


# In[40]:


plt.figure(figsize=(14, 9))

plt.figure(1)

plt.subplot(211)
plt.title('Spectrogram')
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
librosa.display.specshow(D, x_axis='time', y_axis='log')

plt.subplot(212)
plt.title('Audioform')
librosa.display.waveplot(y, sr=sr)


# In[41]:


librosa.feature.melspectrogram(y=X, sr=sample_rate)

D = np.abs(librosa.stft(X))**2
S = librosa.feature.melspectrogram(S=D)
S = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128,fmax=8000)
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(S,ref=np.max),y_axis='mel', fmax=8000,x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
plt.show()


# In[42]:


y_fast = librosa.effects.time_stretch(X, 2.0)
time = np.arange(0,len(y_fast))/sample_rate
fig, ax = plt.subplots()
ax.plot(time,y_fast)
ax.set(xlabel='Time(s)',ylabel='sound amplitude')
plt.show()#compress to be twice as fast

y_slow = librosa.effects.time_stretch(X, 0.5)
time = np.arange(0,len(y_slow))/sr
fig, ax = plt.subplots()
ax.plot(time,y_slow)
ax.set(xlabel='Time(s)',ylabel='sound amplitude')
plt.show()#half the original speed


# In[43]:


S = np.abs(librosa.stft(X))
S


# In[44]:


#lms = librosa.power_to_db(S)
#lms

log_S = librosa.amplitude_to_db(S, ref=np.max)


# In[45]:


from librosa.core import istft
vocals = istft(log_S)


# In[46]:


mfccs_test = librosa.feature.mfcc(y=vocals, sr = mfcc_sample_rate, n_mfcc = n_mfcc)
mfccs_test


# In[47]:


mfccs_final = np.mean(mfccs_test,axis = axis_mfcc)
mfccs_final


# In[48]:


S = np.abs(librosa.stft(X))
lms = librosa.power_to_db(S**2)
    #lms = librosa.amplitude_to_db(librosa.feature.melspectrogram(y=X,sr=sample_rate))
vocal = istft(lms)    #convert them back to an audio sample using inverse STFT.
mfccs = np.mean(librosa.feature.mfcc(y=vocal, 
                                     sr = mfcc_sample_rate, 
                                     n_mfcc = n_mfcc),
                                     axis = axis_mfcc)


# In[49]:


sample_rate = np.array(sample_rate)
sample_rate


# In[50]:


print(f"n_mfcc = {n_mfcc}")
print(f"sr = {mfcc_sample_rate}")

z = librosa.feature.mfcc(X, sr = mfcc_sample_rate, n_mfcc = n_mfcc)
for x in z:
    print(x)


# In[51]:


mfccs = np.mean(z, axis=1)
mfccs


# In[52]:


import numpy as np
import seaborn as sns
# Put this into a heatmap
_min = np.amin(mfccs)
_max = np.amax(mfccs)
mfcc = (mfccs - _min) / (_max - _min)



plt.figure(figsize = (6,4))
sns.heatmap(mfcc[:, np.newaxis], cmap='GnBu', linewidth =2)
#sns.heatmap(mfcc, cmap = 'RGBA', linewidth = 1)
plt.axis('off')


# In[53]:


y,sr = librosa.load(sample_file)
print(librosa.get_duration(filename=sample_file))
print(librosa.get_duration(y=y,sr=sr))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




