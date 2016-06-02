
# coding: utf-8

# In[1]:

import numpy as np
from scipy import ndimage, misc
from skimage.color import rgb2gray
import pandas as pd
import os
from read_drivers import create_training_test_lists
from make_empty_array import create_holding_array
from read_validation_files import get_list_of_validation_files


# In[2]:

def get_images(directory, width=224, height=224, channels=3, driver_imgs_list=''):
    '''
    Function to build needed arrays for training or validating the neural network.
    If labels are passed, get a list of training image files, their labels
    '''    
    if driver_imgs_list:        # Do this loop if we're getting images for training
        subjects, label, trainList = create_training_test_lists(driver_imgs_list)
        labels = pd.get_dummies(label).as_matrix()  # convert labels to a dummy array and then to a np.array
        d = dict(zip(trainList, label))    # Make a dictionary from the image filenames and their class labels
        if channels == 3:                  # If we're using color images 
            X = create_holding_array(trainList, width = width, height=height, channels=channels)    # Create empty array
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
            X = np.transpose(X, (2,0,1))
        else:     # If number of channels != 1 or != 3
            print('Could not create dataset and resize training images...')
        return X, labels, subjects
    else:       
        # Run this code if getting new unlabeled images for making predictions by not passing "drivers_imgs_list"
        validationList, imgLabels = get_list_of_validation_files(directory)  # Pass directory containing validation images
        if channels == 3:                                                     # run this if color images
            X = create_holding_array(validationList, width=width, height=height, channels=channels)   # Create empty array
            print('Resizing 3-channel images for making predictions...')
            count = 0                                                           # Set counter = 0
            for validationFile in validationList:                               # Get file from list of files for predictions
                img = misc.imread(validationFile)                               # Read the image file
                img = misc.imresize(img, size = (width, height, channels))      # Resize the image
                X[count] = img                                                  # Add the image to holding array
                count += 1                                                      # Advance the counter
            print('Shape of X is:  ', X.shape)
            print('Transposing X...')
            X = np.transpose(X, (2,0,1))
        else:
            print('Could not create dataset and resize testing images...')
        return X, imgLabels


# In[ ]:

if __name__=='__main__':
    get_ipython().system('ipython nbconvert --to python get_images.ipynb')


# In[ ]:



