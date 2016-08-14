from __future__ import division
'''
This script trains keras NN based on VGG16 and saves model to file
'''
import numpy as np
import os
os.environ['THEANO_FLAGS'] = 'device=gpu0'
import glob
import cv2
import math
import h5py
import datetime
import pandas as pd
import cPickle as pickle
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, \
                                       ZeroPadding2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.models import model_from_json
from numpy.random import permutation
import time

random_state = 2030


def model_v1(img_rows, img_cols, color_type=1):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal',
                            input_shape=(color_type, img_rows, img_cols)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3, border_mode='same', init='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal'))
    model.add(MaxPooling2D(pool_size=(8, 8)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation('softmax'))
    sgd = SGD(lr=0.001, decay = 1e-6, momentum = 0.9, nesterov=True)

    model.compile(Adam(lr=1e-3), loss='categorical_crossentropy')#Adam(lr=1e-3)
    return model


def vgg_std16_model_adjusted(img_rows, img_cols):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3,
                                                 img_rows, img_cols)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    assert os.path.exists('vgg16_weights.h5'), 'Model weights not found (see "weights_path" variable in script).'
    f = h5py.File('vgg16_weights.h5')
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
        # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    # Learning rate is changed to 0.001
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def vgg_std16_model(img_rows, img_cols):
    '''returns vgg16 pre-trained model adjusted for 10 classes'''

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3,
                                                 img_rows, img_cols)))

    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))
    model.load_weights('vgg16_weights.h5')

    model.layers.pop() # Get rid of the classification layer
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Dense(10, activation='softmax'))

    # Learning rate is changed to 0.001
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
    	loss='categorical_crossentropy',
    	metrics=['accuracy'])
    return model


def save_model(model, cross):
    '''save the model weights'''

    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    json_name = 'architecture' + cross + '.json'
    weight_name = 'model_weights' + cross + '.h5'
    open(os.path.join('cache', json_name), 'w').write(json_string)
    model.save_weights(os.path.join('cache', weight_name), overwrite=True)


if __name__ == "__main__":
    img_rows, img_cols = 224, 224
    batch_size = 20
    nb_epoch = 15
    num_images = 79726
    n_folds = 5
    np.random.seed(random_state) #for reproducibility

    drivers = pd.read_csv('driver_imgs_list.csv')
    unique_drivers = drivers['subject'].unique()
    kf = KFold(len(unique_drivers), n_folds=n_folds,
               shuffle=True, random_state=random_state)

    print 'reading train'
    f = h5py.File('train_224_224.h5','r')

    now = datetime.datetime.now()
    suffix = str(now.strftime("%Y-%m-%d-%H-%M"))

    ind = 0
    num_fold = 0
    for train_drivers, test_drivers in kf:
        y_train = []
        X_train = []
        #y_val = []
        #X_val = []

        #Remove split on drivers
        for driver in unique_drivers:
            X_train += [np.array(f['X_{driver}'.format(driver=driver)])]
            y_train += [np.array(f['y_{driver}'.format(driver=driver)])]

        #for driver in unique_drivers[test_drivers]:
            #X_val += [np.array(f['X_{driver}'.format(driver=driver)])]
            #y_val += [np.array(f['y_{driver}'.format(driver=driver)])]

        print 'shuffling'

        X_train = np.vstack(X_train).astype(np.float32)
        y_train = np.hstack(y_train)

        #X_val = np.vstack(X_val).astype(np.float32)
        #y_val = np.hstack(y_val)

        mean_pixel = [103.939, 116.779, 123.68]

        X_train = X_train.transpose((0, 3, 1, 2))
        #X_val = X_val.transpose((0, 3, 1, 2))

        for c in range(3):
            print 'subtracting {c}'.format(c=mean_pixel[c])
            X_train[:, c, :, :] = X_train[:, c, :, :] - mean_pixel[c]
            #X_val[:, c, :, :] = X_val[:, c, :, :] - mean_pixel[c]

        perm = permutation(len(y_train))
        X_train = X_train[perm]
        y_train = y_train[perm]

        y_train = map(int, [x.replace('c', '') for x in y_train])
        #y_val = map(int, [x.replace('c', '') for x in y_val])

        y_train = np_utils.to_categorical(y_train, 10)
        #y_val = np_utils.to_categorical(y_val, 10)

        print X_train.shape, y_train.shape
        #print X_val.shape, y_val.shape

        print 'starting model'
        num_fold += 1
        print 'fitting model'
        kfold_weights_path = 'weights_kfold_' + str(num_fold) + '_' + suffix + '.h5'
        model = vgg_std16_model(img_rows, img_cols)
        callbacks = [ModelCheckpoint(kfold_weights_path)]
        model.fit(X_train, y_train, batch_size=batch_size,
            nb_epoch=nb_epoch,
            verbose=1,
            validation_split=0.15,
            shuffle=True, callbacks=callbacks)

        f_test = h5py.File('test_224_224.h5','r')

        X_test_id = np.array(f_test['X_test_id'])

        print 'size of test'
        print len(X_test_id)
        print X_test_id.shape

        print 'subtracting mean'
        mean_pixel = [103.939, 116.779, 123.68]

        preds = []
        iter_size = 2 * 4096
        for i in xrange(0, num_images, iter_size):
            start_i = i
            end_i = min(num_images, i + iter_size)
            print start_i, end_i

            X_test = np.array(f_test['X_test'][start_i:end_i], dtype=np.float32)
            X_test = X_test.transpose((0, 3, 1, 2))
            print X_test.shape

            for c in range(3):
                print 'subtracting {c}'.format(c=mean_pixel[c])
                X_test[:, c, :, :] = X_test[:, c, :, :] - mean_pixel[c]
                X_test = X_test.astype(np.float32)

            print 'predicting...'
            preds += [model.predict(X_test, batch_size=32, verbose=1)]

        predictions = np.vstack(preds)

        print 'saving result'
        result = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3',
                                                   'c4', 'c5', 'c6', 'c7',
                                                   'c8', 'c9'])
        result['img'] = X_test_id

        if not os.path.isdir('subm'):
            os.mkdir('subm')

        sub_file = os.path.join('subm', '{ind}_submission_'.format(ind=ind) + suffix + '.csv')
        result.to_csv(sub_file, index=False)
        ind += 1

        f_test.close()
    f.close()
