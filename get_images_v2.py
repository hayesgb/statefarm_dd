
# coding: utf-8

# In[1]:

import numpy as np
from scipy import ndimage, misc
from skimage.color import rgb2gray
import pandas as pd
import os
import dask.array as da

from make_empty_array import create_holding_array, create_dask_array
from read_validation_files import get_list_of_validation_files


# In[2]:

def get_images(label, trainList, directory, width=224, height=224, channels=3):
    '''
    Function to build needed arrays for training or validating the neural network using out of core processing.
    If labels are passed, get a list of training image files, their labels
    '''    
    labels = pd.get_dummies(label).as_matrix()  # convert labels to a dummy array and then to a np.array
    d = dict(zip(trainList, label))    # Make a dictionary from the image filenames and their class labels
    if channels == 3:                  # If we're using color images 
        X1 = create_holding_array(trainList, width = width, height=height, channels=channels)    # Create empty array
        print('Resizing 3-channel images for training...')
        count = 0                                                       # Set counter for empty array
        for (trainFile, label) in d.items():                            # Read the dictionary
            trainFile = os.path.join(directory, label, trainFile)       # Set file path
            img = misc.imread(trainFile)                                # Read the image
            img = misc.imresize(img, size = (width, height, channels))  # Resize image with color channel = 3
            X[count] = img                                              # Store resized image in empty array
            count += 1                                                  # Advance index counter
        print('Shape of X is:  ', X.shape)
        print('Transposing X...')
        X = np.transpose(X, (0,3,1,2))

    else:     # If number of channels != 1 or != 3
        print('Could not create dataset and resize training images...')
    return X, labels



# In[3]:

def get_dask_images(label, trainList, directory, width=224, height=224, channels=3):
    '''
    Function to build needed arrays for training or validating the neural network using out of core processing.
    If labels are passed, get a list of training image files, their labels
    '''    
    labels = pd.get_dummies(label).as_matrix()  # convert labels to a dummy array and then to a np.array
    d = dict(zip(trainList, label))    # Make a dictionary from the image filenames and their class labels
    if channels == 3:                  # If we're using color images 
        X1 = create_holding_array(trainList, width = width, height=height, channels=channels)    # Create empty array
        print('Resizing 3-channel images for training...')
        count = 0                                                       # Set counter for empty array
        for (trainFile, label) in d.items():                            # Read the dictionary
            trainFile = os.path.join(directory, label, trainFile)       # Set file path
            img = misc.imread(trainFile)                                # Read the image
            img = misc.imresize(img, size = (width, height, channels))  # Resize image with color channel = 3
            X1[count] = img                                              # Store resized image in empty array
            count += 1                                                  # Advance index counter
        print('Shape of X is:  ', X1.shape)
        print('Transposing X...')
        X1 = np.transpose(X1, (0,3,1,2))
        print('Putting X in a dask array...')
        X = da.from_array(X1, chunks=1000)
    else:     # If number of channels != 1 or != 3
        print('Could not create dataset and resize training images...')
    return X, labels


# In[4]:

if __name__=='__main__':
    get_ipython().system('ipython nbconvert --to python get_images_v2.ipynb')


# In[ ]:



