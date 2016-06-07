
# coding: utf-8

# In[ ]:

import pandas as pd
from sklearn.preprocessing import LabelEncoder


# In[ ]:

def create_training_test_lists(driver_imgs_list):
    '''
    Create training and validation sets by passing one list of drivers for training and the driver_img_list.csv file
    '''
    driver_imgs = pd.read_csv(driver_imgs_list)

    training_subjects = driver_imgs.subject
    training_targets = driver_imgs.classname
    training_imgs = driver_imgs.img
    
    le = LabelEncoder()
    integer_subjects = le.fit_transform(training_subjects)

    return integer_subjects, training_targets, training_imgs


# In[ ]:

if __name__=='__main__':
    get_ipython().system('ipython nbconvert --to python read_drivers.ipynb')


# In[ ]:



