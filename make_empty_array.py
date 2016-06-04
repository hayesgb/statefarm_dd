
# coding: utf-8

# In[ ]:

import dask.array as da
import numpy as np


# In[ ]:

def create_holding_array(x, width, height, channels):
    '''
    Create an array of zeros to fill with images after converted to arrays
    '''
    if channels == 3:
        X = np.zeros(shape=(len(x), width, height, channels))
    else:
        print('Wrong number of channels')
    return X


# In[ ]:

def create_dask_array(x, width, height, channels):
    '''
    Create an array of zeros to fill with images after converted to arrays
    '''
    if channels == 3:
        Xn = np.zeros(shape=(len(x), width, height, channels))
        X = da.from_array(Xn, chunks=1000)
    else:
        print('Wrong number of channels')
    return X


# In[ ]:

if __name__=='__main__':
    get_ipython().system('ipython nbconvert --to python make_empty_array.ipynb')

