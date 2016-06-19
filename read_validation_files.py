
# coding: utf-8

# In[1]:

import os


# In[2]:

def get_list_of_validation_files(directory):
    '''
    Read the directory where the validation images are stored and return two
    list.  The first list is the filepath for each of the images, the second is 
    the filename itself.  The filepath (called validationList) is so the images can be read
    and stored in a hdf5 container.  The filenames are to be used in the row index of the
    submision .csv file
    '''

    validationList = []  # create an empy list to store image files
    filenames = []

    for path, subdirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.jpg'):  # select only files that end with .jpg
                filenames.append(filename)
                validationList.append(os.path.join(path, filename))  # append files to a list called trainList

    return validationList, filenames


# In[3]:

if __name__=='__main__':
    get_ipython().system('ipython nbconvert --to python read_validation_files.ipynb')

