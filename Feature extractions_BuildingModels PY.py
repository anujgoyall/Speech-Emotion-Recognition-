#!/usr/bin/env python
# coding: utf-8

# In[ ]:


''' #Speech Emotion Analyzer using Convolutional Neural Network
#Importing Libraries
import os # provides functions for interacting with the operating
import sys # provides information about constants, functions and methods
import glob # glob module is used to retrieve files/pathnames matching a specified pattern
import numpy as np # used for working with arrays
import pandas as pd # library used for data analysis 

'''Import audio files
pip install librosa. Run this command in terminal to install librosa library'''

import librosa # librosa is a package to use audio files
import librosa.display  # for usage with audio signals
from scipy.io import wavfile #to import wav file
import scipy.io.wavfile
#import sys
import numpy as nm
''' Importing plotting packages'''
import matplotlib.pyplot as plt  #Plotting library
from matplotlib.pyplot import specgram 
import matplotlib.pyplot as plt 
import seaborn as sns 

''' Import Keras & Tenserflow packages'''
#pip install keras
#pip install tensorflow
import keras # to define and train neural network models
from keras import regularizers 
from keras.preprocessing import sequence 
from keras.models import Sequential 
from keras.layers import Dense, Embedding 
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer 
from kersas.preprocessing.sequence import pad_sequences 
from keras.utils import to_categorical 
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import conv1D, MaxPoling1D, AveragePooling1D
from keras.models import Model
from keras.models import ModelCheckpoint
from sklearn.metrics import confusion_matrix'''


# **There are two datasets used for this project
# RAVDESS: The RAVDESS file contains a unique filename that consists in a 7-part numerical identifier.
# TESS: The TESS file contains a unique letter at beginning of file name to identify the emotion.

# In[6]:


import os 
#Build list of files
rawdata_list = os.listdir('capstoneproject-speech-emotion-machine-learning-master/RawData/Ravdess/audio_speech_actors_01-24')
#Raw files contains speech of 24 actors in 24 different folders and 60 audio files for each actor in wav format
# Total 1440 files in whole dataset


# In[6]:


#Review list of files
print(rawdata_list)


#  ### Librosa and MFCC Configuration
# **In order to analyze and standardize how each audio file feature was built, the following configurations were determined:
# res_type: resample type (kaiser_best or kaiser_fast)
# 
# By default, this uses a high-quality (but relatively slow) method (‘kaiser_best’) for band-limited sinc interpolation. The alternate res_type values listed below offer different trade-offs of speed and quality.
# librosa.core.load(path, sr=22050, mono=True, offset=0.0, duration=None, dtype=<class 'numpy.float32'>, res_type='kaiser_best')
# You will use this function to extract the audio amplitude at different sampling rates.
# path: path to the input file.
# sr: target sampling rate. The default
# mono: convert signal to mono
# offset: start reading after this time (in seconds)
# duration: only load up to this much audio (in seconds)
# res_type: resample type (kaiser_best or kaiser_fast)

# In[7]:


#sample feature
#librosa.core.load(path, sr=22050, mono=True, offset=0.0, duration=None, dtype=<class 'numpy.float32'>, res_type='kaiser_best')
res_type_s = 'kaiser_best'
duration_s = None
sample_rate_s = 22050
offset_s = 0.5

#Mfcc
#librosa.feature.mfcc(y=None, sr=22050, S=None, n_mfcc=20, dct_type=2, norm='ortho', lifter=0, **kwargs)
mfcc_sample_rate = 22050
n_mfcc = 40
axis_mfcc = 1


# In[8]:


#import pandas as pd
#df = pd.read_csv("C:/Users/Anuj/capstoneproject-speech-emotion-machine-learning-master/Dataset/emotion_capstone_final_ravdess_dataframe.csv")


# In[9]:


#df.head()


# ** Emotion features in Ravdess

# In[7]:


#Build list with target variables for each file
feeling_list=[]

#Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fear, 07 = disgust, 08 = surprised) 

for emotion_path in rawdata_list:
    if emotion_path.split('-')[2] == '01':
        feeling_list.append("neutral")
    elif emotion_path.split('-')[2] == '02':
        feeling_list.append("calm")
    elif emotion_path.split('-')[2] == '03':
        feeling_list.append("happy")
    elif emotion_path.split('-')[2] == '04':
        feeling_list.append("sad")
    elif emotion_path.split('-')[2] == '05':
        feeling_list.append("angry")
    elif emotion_path.split('-')[2] == '06':
        feeling_list.append("fear")
    elif emotion_path.split('-')[2] == '07':
        feeling_list.append("disgust")
    elif emotion_path.split('-')[2] == '08':
        feeling_list.append("surprised")
    else:
        feeling_list.append("unknown")


# In[11]:


#Check list
feeling_list


# In[12]:


import pandas as pd
#Turn list into dataframe
labels = pd.DataFrame(feeling_list)


# In[13]:


#Check shape
labels.shape


# In[14]:


#Change index name to "emotion"
labels = labels.rename({0: 'emotion'}, axis=1)


# In[15]:


labels.shape


# In[16]:


#Count the number of files per emotion
labels_total = pd.DataFrame(labels.groupby(['emotion']).size())
labels_total


# In[18]:


import pandas as pd
import librosa
import librosa.display
import numpy as np
rawdata_ravdess = pd.DataFrame(columns=['feature'])
bookmark=0

for y in rawdata_list:
    #Change to kaiser_best & 22050 kHz
    #sr > target sampling rate
    #offset=0.5
    X, sample_rate = librosa.load('capstoneproject-speech-emotion-machine-learning-master/RawData/Ravdess/audio_speech_actors_01-24/'+y, 
                                  res_type = res_type_s,
                                  duration = duration_s,
                                  sr = sample_rate_s,
                                  offset = offset_s)
    sample_rate = np.array(sample_rate)
    
    #Get MFCCs from each file
    mfccs = librosa.feature.mfcc(   y=X, 
                                    sr = mfcc_sample_rate, 
                                    n_mfcc = n_mfcc)
    
    #Calculate mean of MFCCs
    mfccs_mean = np.mean(    mfccs, 
                             axis = axis_mfcc)
    feature = mfccs_mean
    
    #Add MFCCs feature results to list
    rawdata_ravdess.loc[bookmark] = [feature]
    bookmark=bookmark+1   


# In[19]:


#Verity data results
rawdata_ravdess.shape


# In[20]:


#Verify that there are no null values
rawdata_ravdess.isnull().values.any()


# In[21]:


# See array sample of features
rawdata_ravdess


# In[22]:


#Turn array into dataframe
rawdata_ravdess_final = pd.DataFrame(rawdata_ravdess['feature'].values.tolist())


# In[23]:


#Analyze new dataframe shape
rawdata_ravdess_final.shape


# In[24]:


# Check data sample
rawdata_ravdess_final.head()


# ** Features and Target for Ravdess

# In[25]:


#Join labels with features
newdf_ravdess = pd.concat([rawdata_ravdess_final,labels], axis=1)


# In[26]:


#Rename dataframe
newdf_ravdess = newdf_ravdess.rename(index=str, columns={"0": "label"})


# In[27]:


#Analyze dataframe shape
newdf_ravdess.shape


# In[28]:


#Anayze dataframe sample
newdf_ravdess.head()


# In[29]:


#Datafram drop Nan values
newdf_ravdess.dropna(inplace=True)  # Dropping N/A from the dataset


# In[32]:


from sklearn.utils import shuffle

#Shuffle dataframe
newdf_ravdess = shuffle(newdf_ravdess)
newdf_ravdess.head(8)


# In[33]:


#Verify that there are no null values
newdf_ravdess.isnull().values.any()


# In[34]:


# Check dataframe sample
newdf_ravdess.head(5)


# In[35]:


#Analyz shape of dataframe
newdf_ravdess.shape


# In[202]:


# see number of emotions
newdf_ravdess[newdf_ravdess.columns[-1]].nunique()


# In[203]:


#Move dataframe into separate file
newdf_ravdess.to_csv('emotion_capstone_final_ravdess_dataframe_Anuj_Goyal.csv')


# ## Import and read TESS Dataset

# In[211]:


from scipy.io import wavfile
import pandas as pd
import numpy as np
import glob 
import sys
import os
# Build list of audio files
raw_data_tess_path = r"C:\Users\Anuj\capstoneproject-speech-emotion-machine-learning-master\RawData\TESS Toronto emotional speech set data\\"
#raw_data_tess_path = r"C:/Users/Anuj/capstoneproject-speech-emotion-machine-learning-master/RawData/TESS Toronto emotional speech set data/TESS Full Dataset//"
folder_list_tess = os.listdir(raw_data_tess_path)

tess_list = []

for folder in folder_list_tess:
    folder_path = raw_data_tess_path+folder+"\\"
    os.chdir(folder_path)
    for file in glob.glob("*.wav"):
        tess_list.append(folder_path+file)

#Check results
tess_list[:8]
#print(len(tess_list))


# In[212]:


#Build list of emotions for Tess
feeling_list_tess = []

#'angry', 'disgust', 'fear', 'happy', 'sad' and 'surprised' emotion classes respectively.  

emotion_dic = {"angry":'angry', 
               "disgust":'disgust', 
               "fear":'fear', 
               "happy":'happy',  
               "sad":'sad', 
               "ps":'surprised',
               "neutral" :'neutral'}

for file_path in tess_list:
    file = file_path.split("\\")[-1] 
    file_name = file.split(".")[0] 
    #print(file_name)
    #x= file_name.rsplit('_')[-1]
    x=file_name.split("_")[-1]
    feeling_list_tess.append(emotion_dic[x])

#Verify emotions
feeling_list_tess


# In[213]:


#Build dataframe from array
labels_tess = pd.DataFrame(feeling_list_tess)


# In[214]:


#Check results
labels_tess.head()


# In[215]:


#Rename column to emotion
labels_tess = labels_tess.rename({0: 'emotion'}, axis=1)


# In[216]:


#Check shape
labels_tess.shape


# In[217]:


#Check results
labels_tess.head()


# In[218]:


#Check emotion size
labels_tess_total = pd.DataFrame(labels_tess.groupby(['emotion']).size())
labels_tess_total


# ** Audio features for TESS

# In[219]:


rawdata_tess = pd.DataFrame(columns=['feature'])
bookmark=0

for y in tess_list:
    #Get audio features
    X, sample_rate = librosa.load(y, 
                                  res_type = res_type_s,
                                  duration = duration_s,
                                  sr = sample_rate_s,
                                  offset=offset_s)
    
    #Get MFFC features
    mfccs = librosa.feature.mfcc(   y=X, 
                                    sr = mfcc_sample_rate, 
                                    n_mfcc = n_mfcc)
    #Get MFFCs average features
    mfccs_mean = np.mean(    mfccs, 
                             axis = axis_mfcc)
    feature = mfccs_mean
    rawdata_tess.loc[bookmark] = [feature]
    bookmark=bookmark+1


# In[220]:


#Verify Tess features shape
rawdata_tess.shape


# In[221]:


#Check that there are no nan values
rawdata_tess.isnull().values.any()


# In[222]:


#Get sample data
rawdata_tess.head()


# In[223]:


#Build list
rawdata_tess_final = pd.DataFrame(rawdata_tess['feature'].values.tolist())


# In[224]:


#Check dataframe
rawdata_tess_final


# ## Combine TESS features and targets

# In[226]:


#Concat both feature table and target table
newdf_tess = pd.concat([rawdata_tess_final,labels_tess], axis=1)
newdf_tess


# In[227]:


newdf_tess = newdf_tess.rename(index=str, columns={"0": "label"})


# In[228]:


#Verify table shape
newdf_tess.shape


# In[229]:


#Get dataframe sample data
newdf_tess.head()


# In[230]:


#Drop nan values
newdf_tess.dropna(inplace=True)
newdf_tess.shape


# In[231]:


#Shuffle rows
newdf_tess = shuffle(newdf_tess)
newdf_tess.head(10)


# In[232]:


#Verify there are no nan values
newdf_tess.isnull().values.any()


# In[233]:


#Check shape
newdf_tess.shape


# In[234]:


# See number of emotions
newdf_tess[newdf_tess.columns[-1]].nunique()


# In[235]:


#Move dataframe into separate file
newdf_tess.to_csv('emotion_capstone_final_tess_dataframe_Anuj_Goyal.csv')


# In[236]:


newdf_ravdess.columns


# In[237]:


newdf_tess.columns


# In[238]:


frames = [newdf_ravdess,newdf_tess]

final_dataframe = pd.concat(frames, ignore_index=True)
final_dataframe.shape


# In[239]:


#Check new and final dataframe
final_dataframe


# In[240]:


#Move dataframe into separate file
final_dataframe.to_csv('emotion_capstone_final_dataframe_Anuj_Goyal.csv')


# ## Divide the data into Test & Train dataset 

# In[241]:


#Split features from targets
X = final_dataframe.iloc[:,:-1]

#Split targets
y = final_dataframe.iloc[:,-1]


# In[243]:


#Get sample of target
y


# In[244]:


#Get sample of features
X


# In[245]:


from sklearn.model_selection import train_test_split

#Split train & test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=1)

# Check out the data
print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')


# In[246]:


#Check unique values for y_test
y_test.unique()


# In[247]:


#Check unique values for y_train
y_train.unique()


# In[5]:


#import tensorflow as tf not working
#import keras make kernals dead


# In[ ]:


#Label Encoding
import keras.utils as kutils
from sklearn.preprocessing import LabelEncoder
import numpy as np
from keras import utils as np_utils

lb = LabelEncoder()

#Encode emotion labels into numbers
y_train_lb = np_utils.to_categorical(lb.fit_transform(y_train))
y_test_lb = np_utils.to_categorical(lb.fit_transform(y_test))


# Check out the data
print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train_lb.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test_lb.shape}')


# In[ ]:


import numpy as np
#Check encoding
np.unique(y_train_lb, axis=0)


# In[ ]:


#import sys
#sys.executable


# In[1]:





# In[ ]:




