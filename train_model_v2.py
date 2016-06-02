
# coding: utf-8

# In[1]:

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils


# In[2]:

def get_model(batch_size=32, nb_classes=10, nb_epoch=10, img_rows=256,
               img_cols=256, img_channels=1):
    
    
    
    model = Sequential()
    
    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(
            img_rows, img_cols, img_channels), dim_ordering='tf'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu')) 
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))    
    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))    
    model.add(MaxPooling2D(pool_size=(1,1)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Dense(1024))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    rms = RMSprop()
    
    model.compile(loss='categorical_crossentropy', optimizer=rms, 
                 metrics=['accuracy'])

    return model


# In[3]:

model1 = get_model()


# In[4]:

model1.output_shape


# In[5]:

get_ipython().system('ipython nbconvert --to python train_model_v2.ipynb')


# In[6]:

#class create_model:
#    def __init__(self):
#        self.batch_size=32
#        self.nb_classes=10
#        self.nb_epoch=20
#        self.data_augmentation=False
        
#    def define_image(self):
#        img_rows, img_cols = 128, 128
#        img_channels = 

