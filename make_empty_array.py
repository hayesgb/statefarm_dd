
# coding: utf-8

# In[ ]:

import numpy as np


# In[ ]:

def create_holding_array(x, width, height, channels):
    '''
    Create an array of zeros to fill with images after converted to arrays
    '''
    if channels == 1:
        X = np.zeros(shape=(len(x), width, height))
    elif channels == 3:
        X = np.zeros(shape=(len(x), width, height, channels))
    else:
        print('Wrong number of channels')
    return X


# In[ ]:

if __name__=='__main__':
    get_ipython().system('ipython nbconvert --to python make_empty_array.ipynb')

