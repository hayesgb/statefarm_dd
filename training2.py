
# coding: utf-8

# In[1]:

from sklearn.metrics import log_loss
import h5py
import pandas as pd
from keras.optimizers import SGD, Adagrad
from sklearn import cross_validation
import boto3
import numpy as np

from vgg16_model import VGG_16
from get_images_v2 import get_images
from read_drivers import create_training_test_lists


# In[ ]:

def train_model(driver_imgs_list, width=224, height=224, channels=3, nb_epochs=1, 
                n_folds=3, path='./vgg16_weights.h5'):
    
    subjects, label, trainList = create_training_test_lists(driver_imgs_list)
    
    lkf = cross_validation.LabelKFold(subjects, n_folds=n_folds)  # Instantiate Label K Fold iterator

    print('Loading model...')
    

    model = VGG_16(width=width, height=height, channels=channels, weights_path=path)
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    ada = Adagrad()
    model.compile(loss='categorical_crossentropy', optimizer=ada, metrics=['accuracy'])
    
    for i, (train_index, test_index) in enumerate(lkf):
        print('Setting up training and test samples for fold #:  ', i)
        
        trainList_train, trainList_test = trainList[train_index], trainList[test_index]
        label_train, label_test = label[train_index], label[test_index]
        print('Getting X_train and Y_train for fold #:  ', i)
        X_train, Y_train = get_images(label=label_train, trainList=trainList_train, directory= './imgs/train/',
                                width=width, height=height, channels=channels )
        print('Getting X_test and Y_test for fold #:  ', i)

        X_test, Y_test = get_images(label=label_test, trainList=trainList_test, directory= './imgs/train/',
                                width=width, height=height, channels=channels ) 
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        
        model.fit(X_train, Y_train, validation_data=[X_test, Y_test], shuffle=True, verbose=1,
                 nb_epoch=nb_epochs, batch_size=16)
        
        print('Saving model weights for model on fold:  ', i)
        model.save_weights('model_weights_vgg_fold_'+i*'.h5', overwrite=True)

    print('Saving final model weights...')
    model.save_weights('model_weights_vgg_trained.h5', overwrite=True)
    print('Getting X and Y for final predictions...')
    X, Y = get_images(label=label_train, trainList=trainList_train, directory= './imgs/train/',
                        width=width, height=height, channels=channels )
    
    y_predicted = model.predict(X)
    multi_logloss = log_loss(y, y_predicted)
    
    return multi_logloss


multiclass_logloss = train_model(driver_imgs_list='driver_imgs_list.csv', width=224, height=224, channels=3, 
                                 nb_epochs=10)  # Start iterative training, and return
                                                                                                      # The logloss by img size



# In[ ]:

if __name__=='__main__':
    get_ipython().system('jupyter nbconvert --to python training2.ipynb')


# In[ ]:

print('Logloss is:  ', multiclass_logloss)


# In[ ]:

del X, y, X_train, X_test, Y_train, Y_test


# In[ ]:

#model = VGG_16(weights_path=('model_weights_vgg.h5')
#model.load_weights('model_weights_vgg.h5')


# In[ ]:

X_valid, imgLabels = get_dask_images(directory='./imgs/test/', width=width, height=height)


# In[ ]:

def create_prediction_matrix(model, X, target_classes, img_filenames):
    predictions = model.predict(X)
    classes = sorted(set(target_classes))
    print('Shape of predictions output is:  ', predictions.shape)
    print('Shape of target_classes is:  ', len(classes))
    print('Shape of image filenames is:  ', img_filenames)
    predictions_df = pd.DataFrame(predictions, columns = classes, index = img_filenames)
    predictions_df.index.name='img'
    
    return predictions_df

predictions = create_prediction_matrix(model, X_valid, Y, imgLabels)

predictions.to_csv('submission3.csv')


# In[ ]:

s3 = boto3.resource
s3_client.upload_file('./submission3.csv', 'kaggle-competitions', 
                      'StateFarmDistractedDriver/submission3.csv')
s3_client.upload_file('./model_weights_vgg_trained.h5', 'kaggle-competitions', 
                     'StateFarmDistractedDriver/model_weights_vgg_trained.h5')


# In[ ]:



