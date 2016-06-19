
# coding: utf-8

# In[1]:

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.layer_utils import layer_from_config
from scipy import ndimage, misc
import numpy as np


# In[3]:

def model4(weights_path=None, channels=3, width=224, height=224):
        
    model = Sequential()
    model.add(Convolution2D(8, 2, 2, border_mode='valid', input_shape=(channels, width, height), 
                            activation='relu'))
    model.add(Convolution2D(8, 2, 2, activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

if __name__=="__main__":
    
    get_ipython().system('ipython nbconvert --to script Model4.ipynb')
    model = model4()
    print('Shape is:  ', model.output_shape)
    model.load_weights(weights_path='k_fold_iter2_weights_1.h5')


#    print('Weights are:  ', len(model.get_weights()))
#    print('layer.get_config()')

    


# In[ ]:




# In[ ]:



