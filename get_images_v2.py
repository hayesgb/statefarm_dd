
# coding: utf-8

# In[1]:

import numpy as np
from scipy import ndimage, misc
from skimage.color import rgb2gray
import pandas as pd
import os
from collections import OrderedDict
import dask.array as da
import h5py

from make_empty_array import create_holding_array, create_dask_array
from read_validation_files import get_list_of_validation_files


# In[2]:

def get_images(label, trainList, directory, width=224, height=224, channels=3):
    '''
    Function to build needed arrays for training or validating the neural network using out of core processing.
    If labels are passed, get a list of training image files, their labels
    '''    
    labels = pd.get_dummies(label).as_matrix()  # convert labels to a dummy array and then to a np.array
    d = OrderedDict(zip(trainList, label))    # Make a dictionary from the image filenames and their class labels
    if channels == 3:                  # If we're using color images 
        X = create_holding_array(trainList, width = width, height=height, channels=channels)    # Create empty array
        print('Shape of the holding array is:  ', X.shape)
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
        print('Transposed shape for X is:  ', X.shape)

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

def get_chunks(list_name, n):
    return [list_name[x:x+n] for x in range(0, len(list_name), n)]
    


# In[5]:

def get_valid_images(directory, width=224, height=224, channels=3):
    '''
    Function to build needed arrays for training or validating the neural network using out of core processing.
    If labels are passed, get a list of training image files, their labels
    '''    
    
    validationList, _ = get_list_of_validation_files(directory)  # Pass directory containing validation images
    print('There are ', len(validationList), ' files in the validation list.')
    print('Breaking the list into chunks to handle size of request.')
    chunkedList = get_chunks(validationList, 8000)
    print('The length of the chunkedList is:  ', len(chunkedList))
    if channels == 3:  
        for i in range(len(chunkedList)):
            print('i =', i)
            validation_sublist = chunkedList[i][:]
            X = create_holding_array(validation_sublist, width = width, height=height, channels=channels)    # Create empty array
            print('Shape of the holding array is:  ', X.shape)
            print('Resizing 3-channel images for validation...')
            count = 0                                                       # Set counter for empty array
            filenames = []
            for validFile in validation_sublist:
                filenames.append(os.path.basename(validFile))
                img = misc.imread(validFile)                                # Read the image
                img = misc.imresize(img, size = (width, height, channels))  # Resize image with color channel = 3
                X[count] = img                                              # Store resized image in empty array
                count += 1                                                  # Advance index counter
            print('Shape of X is:  ', X.shape)
            print('Transposing X...')
            X1 = np.transpose(X, (2,0,1))
            print('Transposed shape for X is:  ', X.shape)
            if i == 0:
                print('Creating a dask array for images...')
                X_array = da.from_array(X1, chunks=4000)
            else:
                print('Concatenating the dask arrays...')
                X_array = da.concatenate(X_array, da.from_array(X1, chunks=4000))
            del X, X1
        
        return X_array, filenames

    else:     # If number of channels != 1 or != 3

        print('Could not create dataset and resize training images...')




# In[6]:

def valid_images_to_hdf5(directory, width=224, height=224, channels=3):
    '''
    Function to build needed arrays for training or validating the neural network using out of core processing.
    If labels are passed, get a list of training image files, their labels
    '''    
    
    validationList, _ = get_list_of_validation_files(directory)  # Pass directory containing validation images
    print('Creating the hdf5 file...')
    len_array = len(validationList)
    with h5py.File('validation_files.h5', 'w') as hf:
        dset = hf.create_dataset('validation_array', (len_array, channels, width, height), chunks=True)
        img_names = hf.create_dataset('image_names', (len_array,), chunks=True, dtype='S40')

    with h5py.File('validation_files.h5', 'r+') as hf:
        x = hf['validation_array']
        X = da.from_array(x, chunks=1000)
        image_names = list(hf['image_names'])

    print('There are ', len(validationList), ' files in the validation list.')
    print('Breaking the validation list into chunks of 10,000...')
    chunkedList = get_chunks(validationList, 10000)    # Break the list of files in to chunks of 10000

    if channels == 3:
        for i, chunk in enumerate(chunkedList):
#            print(chunk)
            count = i + len(chunk[i][:])*i                 # Set counter for empty array
#            valid_sublist = chunk[i][:]
            print('Create empty list to store image names..')
            filenames = []
            print('Creating an empty array to store images...')
            X = create_holding_array(chunk, width = width, height=height, channels=channels)    # Create empty array
            for j, validFile in enumerate(chunk):
                print('Reading file #:  ', j)
                filenames.append(os.path.basename(validFile))
#                print(chunk)
#                input('')
                img = misc.imread(validFile)                                # Read the image
                img = misc.imresize(img, size = (width, height, channels))  # Resize image with color channel = 3
#                img = np.transpose(img, (2,0,1))    # Store resized image in empty array
                X[j] = img
            asciiList = []
            asciiList = [n.encode("ascii", "ignore") for n in filenames]
            X1 = np.transpose(X, (0, 3, 1, 2))
            del X, filenames
            print(X1.shape)
            X_da = da.from_array(X1, chunks=1000)
            print('Opening validation_files.h5...')
            with h5py.File('validation_files.h5', 'r+') as hf:
                print('Putting validation_array in x...')
                x = hf['validation_array']
                print('Putting validation_array in dask array...')
                dset = da.from_array(x, chunks=1000)
                print('Concatenating the two dask arrays...')
                X2 = da.concatenate([dset, X_da], axis=0)
                print('Storing the dask array in the hdf5 file...')
                da.store(X2, x)
                print('Put image_names dset into a list...')
                image_names = list(hf['image_names'])
                print('Extend the list with additional image names...')
                image_names.extend(asciiList)
                

            print('Done.')    
        return filenames

    else:     # If number of channels != 1 or != 3

            print('Could not create dataset and resize training images...')



# In[7]:

def chunk_validation_predictions(model, target_classes, directory, width=224, height=224, channels=3):
    '''
    Function to build needed arrays for training or validating the neural network using out of core processing.
    If labels are passed, get a list of training image files, their labels
    '''    
    
    validationList, _ = get_list_of_validation_files(directory)  # Pass directory containing validation images
    len_array = len(validationList)
    predictions = pd.DataFrame()
    print('There are ', len(validationList), ' files in the validation list.')
    print('Breaking the validation list into chunks of 10,000...')
    chunkedList = get_chunks(validationList, 10000)    # Break the list of files in to chunks of 10000

    if channels == 3:
        for i, chunk in enumerate(chunkedList):
#            print(chunk)
#            count = i + len(chunk[i][:])*i                 # Set counter for empty array
#            valid_sublist = chunk[i][:]
            print('Create empty list to store image names..')
            filenames = []
            print('Creating an empty array to store images...')
            X = create_holding_array(chunk, width = width, height=height, channels=channels)    # Create empty array
            for j, validFile in enumerate(chunk):
#                print('Reading file #:  ', j)
                filenames.append(os.path.basename(validFile))
#                print(chunk)
#                input('')
                img = misc.imread(validFile)                                # Read the image
                img = misc.imresize(img, size = (width, height, channels))  # Resize image with color channel = 3
#                img = np.transpose(img, (2,0,1))    # Store resized image in empty array
                X[j] = img
            X1 = np.transpose(X, (0, 3, 1, 2))
            del X
            print('Shape of the transposed array is:  ', X1.shape)
            predictions_ = model.predict(X1)
            predictions_ = pd.DataFrame(data=predictions_, index=filenames, columns=target_classes)
            del X1
            predictions = predictions.append(predictions_)
            del predictions_
            print('Done with chunk ', str(i))    
        return predictions

    else:     # If number of channels != 1 or != 3

            print('Could not create dataset and resize training images...')




# In[8]:

if __name__=='__main__':
    get_ipython().system('ipython nbconvert --to python get_images_v2.ipynb')


# In[9]:

#filenames = valid_images_to_hdf5('./imgs/test/')  # Pass directory containing validation images


# In[ ]:



