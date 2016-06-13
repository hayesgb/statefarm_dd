
# coding: utf-8

# In[ ]:

from sklearn.metrics import log_loss
import h5py
import pandas as pd
from keras.optimizers import SGD, Adagrad
from sklearn import cross_validation
import boto3
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os


from get_images_v2 import get_images, get_valid_images
from read_drivers import create_training_test_lists


# In[ ]:

pretrained = True        # Was a pretrained model used to build the existing model
path = './weights_fold_not_augmented_2.h5'   # What is the path to the weights file?


# In[ ]:

if pretrained=True:
    from vgg16_model import VGG
    model = VGG(weights_path=path)
else:
    from Model4 import model4
    model = model4(weights_path=path)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


# In[ ]:

target_classes = []
for i in range(10):
    target_class = 'c'+str(i)
    target_classes.append(target_class)


# In[ ]:

X_valid, filenames = get_valid_images(directory='./imgs/test/', width=224, height=224, channels=3)


# In[ ]:

def create_prediction_matrix(model, X, target_classes, img_filenames):
    predictions = model.predict(X)
    print('Shape of predictions output is:  ', predictions.shape)
    print('Shape of target_classes is:  ', len(classes))
    print('Shape of image filenames is:  ', img_filenames)
    predictions_df = pd.DataFrame(predictions, columns = target_classes, index = img_filenames)
    predictions_df.index.name='img'
    
    return predictions_df

predictions = create_prediction_matrix(model, X=X_valid, target_classes=target_classes, img_filenames=filenames)

predictions.to_csv('submission_training4.csv')


# In[ ]:

s3 = boto3.resource
s3_client.upload_file('./submission3.csv', 'kaggle-competitions', 
                      'StateFarmDistractedDriver/submission3.csv')
s3_client.upload_file('./model_weights_vgg_trained.h5', 'kaggle-competitions', 
                     'StateFarmDistractedDriver/model_weights_vgg_trained.h5')


# In[ ]:



