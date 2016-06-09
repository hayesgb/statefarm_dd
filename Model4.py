
# coding: utf-8

# In[ ]:

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.layer_utils import layer_from_config
from scipy import ndimage, misc
import numpy as np


# In[ ]:

def model4(channels=3, width=224, height=224):
        
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(channels, width, height), 
                            activation='relu'))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    return model

if __name__=="__main__":
    
    get_ipython().system('ipython nbconvert --to script Model4.ipynb')
#    model = model4()
#    print('Shape is:  ', model.output_shape)
#    print('Weights are:  ', len(model.get_weights()))
#    print('layer.get_config()')

    


# In[ ]:




# In[ ]:



