
# coding: utf-8

# In[1]:

#from sklearn.metrics import log_loss
import h5py
import pandas as pd
from keras.optimizers import SGD, Adagrad
#from sklearn import cross_validation
import boto3
import numpy as np
#np.set_printoptions(threshold=np.inf)
#import matplotlib.pyplot as plt
#%matplotlib inline
#from keras.preprocessing.image import ImageDataGenerator
#from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import dask.array as da


from get_images_v2 import get_images, chunk_validation_predictions
from read_drivers import create_training_test_lists
from read_validation_files import get_list_of_validation_files


# In[2]:

pretrained = True        # Was a pretrained model used to build the existing model
   # What is the path to the weights file?

create_validation_dset = True


# In[3]:

if pretrained:
    from vgg16_model import VGG_16
    path = './weights_fold_not_augmented_2.h5'
    model = VGG_16(weights_path=path)

else:
    from Model4 import model4
    path = './k_fold_iter2_weights_1.h5'
    model = model4(weights_path=path)
sgd = SGD()
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


# In[4]:

target_classes = []
for i in range(10):
    target_class = 'c'+str(i)
    target_classes.append(target_class)


# In[5]:

predicted = chunk_validation_predictions(model, target_classes, directory='./imgs/test/')


# In[8]:

predicted.index.name='img'
predicted.to_csv('submission4.csv')


# In[ ]:

s3 = boto3.resource
s3_client.upload_file('./submission3.csv', 'kaggle-competitions', 
                      'StateFarmDistractedDriver/submission3.csv')
s3_client.upload_file('./model_weights_vgg_trained.h5', 'kaggle-competitions', 
                     'StateFarmDistractedDriver/model_weights_vgg_trained.h5')


# In[9]:

len(predicted)


# In[ ]:



