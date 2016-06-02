
# coding: utf-8

# In[ ]:

import os


# In[ ]:

def get_list_of_validation_files(path):

    validationList = []  # create an empy list to store image files
    fileNames = []

    for path, subdirs, files in os.walk(path):
        for filename in files:
            if filename.endswith('.jpg'):  # select only files that end with .jpg
                validationList.append(os.path.join(path, filename))  # append files to a list called trainList
                fileNames.append(filename)

    return validationList, fileNames


# In[ ]:

if __name__=='__main__':
    get_ipython().system('ipython nbconvert --to python read_validation_files.ipynb')

