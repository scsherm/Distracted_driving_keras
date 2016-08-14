# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(2016)

import os
os.environ['THEANO_FLAGS'] = 'device=gpu1'
import glob
import cv2
import math
import pickle
import datetime
import pandas as pd
import statistics
import time
from shutil import copy2
import warnings
import random
import h5py
warnings.filterwarnings("ignore")
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.cross_validation import train_test_split
from joblib import Parallel, delayed
from sklearn.cross_validation import KFold
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras.models import model_from_json
from sklearn.metrics import log_loss
from scipy.misc import imread, imresize, imshow
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
from submission_above_90 import move_img_to_label


use_cache = 1


def process_image2(img_file, sz = (64,64)):
    img = cv2.imread(img_file)
    rotate = random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), rotate, 1)
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    resized = cv2.resize(img, sz, interpolation=cv2.INTER_LINEAR)
    blur = cv2.blur(resized,(5,5)).astype('float32') / 255
    return blur


def process_test_image(img_file):
    return process_image2(img_file), os.path.basename(img_file)


def get_test_data():
    start = time.time()
    test = []
    test_id = []
    path = os.path.join('imgs/test', '*.jpg')
    files = glob.glob(path)
    results = Parallel(n_jobs=-1)(delayed(process_test_image)(im_file) for im_file in files)
    test, test_id = zip(*results)
    end = time.time() - start
    print("Time: %.2f seconds" % end)
    print(len(test))
    test = np.array(test, dtype=np.uint8)
    test = test.transpose((0, 3, 1, 2))
    return np.array(test), np.array(test_id)


def process_image(img_file, sz = (224,224)):
    img = cv2.imread(img_file)
    rotate = random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), rotate, 1)
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    resized = cv2.resize(img, sz, interpolation=cv2.INTER_LINEAR)
    blur = cv2.blur(resized,(5,5)).astype('float32') / 255
    return img, os.path.basename(img_file)


def show_image(im, name='image'):
    cv2.imshow(name, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# color_type = 1 - gray
# color_type = 3 - RGB
def get_im_cv2(path, img_rows, img_cols, color_type=1):
    # Load as grayscale
    if color_type == 1:
        img = cv2.imread(path, 0)
    elif color_type == 3:
        img = cv2.imread(path)
    # Reduce size
    #resized = cv2.resize(img, (img_cols, img_rows), interpolation=cv2.INTER_LINEAR)
    return img


def get_im_cv2_test(path, img_rows, img_cols, color_type=3):
    # Load as grayscale
    if color_type == 1:
        img = cv2.imread(path, 0)
    elif color_type == 3:
        img = cv2.imread(path)
    # Reduce size
    resized = cv2.resize(img, (img_cols, img_rows))
    return resized


def get_im_cv2_mod(path, img_rows, img_cols, color_type=1):
    # Load as grayscale
    if color_type == 1:
        img = cv2.imread(path, 0)
    else:
        img = cv2.imread(path)
    # Reduce size
    rotate = random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), rotate, 1)
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    resized = cv2.resize(img, (img_cols, img_rows), interpolation=cv2.INTER_LINEAR)
    blur = cv2.blur(resized,(5,5))
    return blur


def get_im_mod(img, img_rows, img_cols):
    # Reduce size
    rotate = random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), rotate, 1)
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    #resized = cv2.resize(img, (img_cols, img_rows), interpolation=cv2.INTER_LINEAR)
    blur = cv2.blur(img,(5,5))
    return blur


def get_driver_data():
    dr = dict()
    clss = dict()
    path = os.path.join('driver_imgs_list.csv')
    print('Read drivers data')
    f = open(path, 'r')
    line = f.readline()
    while (1):
        line = f.readline()
        if line == '':
            break
        arr = line.strip().split(',')
        dr[arr[2]] = arr[0]
        if arr[0] not in clss.keys():
            clss[arr[0]] = [(arr[1], arr[2])]
        else:
            clss[arr[0]].append((arr[1], arr[2]))
    f.close()
    return dr, clss


def get_X_y():
    start = time.time()

    X = []
    y = []
    train_id = []
    driver_id = []
    driver_data, dr_class = get_driver_data()

    for j in range(10):
        one, two = [], []
        print('Load folder c{}'.format(j))
        path = os.path.join('imgs/train', 'c' + str(j), '*.jpg')
        files = glob.glob(path)
        results = Parallel(n_jobs=-1)(delayed(process_image)(im_file) for im_file in files)
        one, two = zip(*results)
        X.extend(one)
        train_id.extend(two)
        y.extend([j]*len(files))
        driver_id.extend([driver_data[i] for i in two])

    unique_drivers = sorted(list(set(driver_id)))
    end = time.time() - start
    print("Time: %.2f seconds" % end)
    return np.array(X), np.array(y), np.array(train_id), driver_id, unique_drivers


def get_other():
    start = time.time()

    X = []
    y = []
    train_id = []

    for j in range(10):
        one, two = [], []
        print('Load folder c{}'.format(j))
        path = os.path.join('imgs2/', str(j), '*.jpg')
        files = glob.glob(path)
        results = Parallel(n_jobs=-1)(delayed(process_image)(im_file) for im_file in files)
        one, two = zip(*results)
        X.extend(one)
        train_id.extend(two)
        y.extend([j]*len(files))

    end = time.time() - start
    print("Time: %.2f seconds" % end)
    return np.array(X), np.array(y), np.array(train_id)


def load_train(img_rows, img_cols, color_type=1):
    X_train = []
    X_train_id = []
    y_train = []
    driver_id = []
    start_time = time.time()
    driver_data, dr_class = get_driver_data()

    print('Read train images')
    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join('imgs', 'train', 'c' + str(j), '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im_cv2(fl, img_rows, img_cols, color_type)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(j)
            driver_id.append(driver_data[flbase])

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    unique_drivers = sorted(list(set(driver_id)))
    print('Unique drivers: {}'.format(len(unique_drivers)))
    print(unique_drivers)
    return X_train, y_train, X_train_id, driver_id, unique_drivers


def load_test(img_rows, img_cols, color_type=3):
    print('Read test images')
    path = os.path.join('imgs', 'test', '*.jpg')
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    total = 0
    start_time = time.time()
    thr = math.floor(len(files)/10)
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2_test(fl, img_rows, img_cols, color_type)
        X_test.append(img)
        X_test_id.append(flbase)
        total += 1
        if total%thr == 0:
            print('Read {} images from {}'.format(total, len(files)))

    print('Read test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_test, X_test_id


def load_other(img_rows, img_cols, color_type=1):
    X_train = []
    X_train_id = []
    y_train = []
    #driver_id = []
    start_time = time.time()
    #driver_data, dr_class = get_driver_data()

    print('Read other images')
    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join('imgs2', str(j), '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            # img = get_im_cv2(fl, img_rows, img_cols, color_type)
            img = get_im_cv2_mod(fl, img_rows, img_cols, color_type)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(j)


    print('Read other data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y_train, X_train_id


def cache_data(data, path):
    #if os.path.isdir(os.path.dirname(path)):
    file = open(path, 'wb')
    pickle.dump(data, file)
    file.close()
    #else:
        #print('Directory doesnt exists')


def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        file = open(path, 'rb')
        data = pickle.load(file)
    return data


def save_model(model, arch_path, weights_path):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    open(arch_path, 'w').write(json_string)
    model.save_weights(weights_path, overwrite=True)


def read_model(arch_path, weights_path):
    model = model_from_json(open(arch_path).read())
    model.load_weights(weights_path)
    return model


def split_validation_set(train, target, test_size):
    random_state = 51
    X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    result1.to_csv(sub_file, index=False)


def save_useful_data(predictions_valid, valid_ids, model, info):
    result1 = pd.DataFrame(predictions_valid, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(valid_ids, index=result1.index)
    now = datetime.datetime.now()
    if not os.path.isdir(os.path.join('subm', 'data')):
        os.mkdir(os.path.join('subm', 'data'))
    suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    # Save predictions
    pred_file = os.path.join('subm', 'data', 's_' + suffix + '_train_predictions.csv')
    result1.to_csv(pred_file, index=False)
    # Save model
    json_string = model.to_json()
    model_file = os.path.join('subm', 'data', 's_' + suffix + '_model.json')
    open(model_file, 'w').write(json_string)
    # Save code
    cur_code = os.path.realpath(__file__)
    code_file = os.path.join('subm', 'data', 's_' + suffix + '_code.py')
    copy2(cur_code, code_file)


def read_and_normalize_train_data(train_data, img_rows, img_cols, color_type=1):
    cache_path = os.path.join('cache', 'train_r_' + str(img_rows) + '_c_' + str(img_cols) + '_t_' + str(color_type) + '_rotated.dat')
    if not os.path.isfile(cache_path) or use_cache == 0:
        train_data, train_target, train_id, driver_id, unique_drivers = load_train(img_rows, img_cols, color_type)
        cache_data((train_data, train_target, train_id, driver_id, unique_drivers), cache_path)
    else:
        print('Restore train from cache!')
        (train_data, train_target, train_id, driver_id, unique_drivers) = restore_data(cache_path)

    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)

    if color_type == 1:
        train_data = train_data.reshape(train_data.shape[0], 1, img_rows, img_cols)
    else:
        train_data = train_data.transpose((0, 3, 1, 2))

    train_target = np_utils.to_categorical(train_target, 10)
    train_data = train_data.astype('float32')
    #mean_pixel = [103.939, 116.779, 123.68]
    #for c in range(3):
        #train_data[:, c, :, :] = train_data[:, c, :, :] - mean_pixel[c]
    train_data /= 255
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target


def read_and_normalize_other_data(img_rows, img_cols, color_type=1):
    #cache_path = os.path.join('cache', 'other_r_' + str(img_rows) + '_c_' + str(img_cols) + '_t_' + str(color_type) + '_rotated.dat')
    #if not os.path.isfile(cache_path) or use_cache == 0:
    other_data, other_target, other_id = load_other(img_rows, img_cols, color_type)
        #cache_data((other_data, other_target, other_id), cache_path)
    #else:
        #print('Restore other from cache!')
        #(other_data, other_target, other_id) = restore_data(cache_path)

    other_data = np.array(other_data, dtype=np.uint8)
    other_target = np.array(other_target, dtype=np.uint8)

    if color_type == 1:
        other_data = other_data.reshape(other_data.shape[0], 1, img_rows, img_cols)
    else:
        other_data = other_data.transpose((0, 3, 1, 2))

    other_target = np_utils.to_categorical(other_target, 10)
    other_data = other_data.astype('float32')
    other_data /= 255
    print('Other shape:', other_data.shape)
    print(other_data.shape[0], 'other samples')
    return other_data, other_target, other_id


def read_and_normalize_test_data(img_rows, img_cols, color_type=3):
    cache_path = os.path.join('test_r_' + str(img_rows) + '_c_' + str(img_cols) + '_t_' + str(color_type) + '_rotated.dat')
    if not os.path.isfile(cache_path) or use_cache == 0:
        test_data, test_id = load_test(img_rows, img_cols, color_type)
        cache_data((test_data, test_id), cache_path)
    else:
        print('Restore test from cache!')
        (test_data, test_id) = restore_data(cache_path)

    test_data = np.array(test_data, dtype=np.uint8)
    if color_type == 1:
        test_data = test_data.reshape(test_data.shape[0], 1, img_rows, img_cols)
    else:
        test_data = test_data.transpose((0, 3, 1, 2))
    # test_data = test_data.swapaxes(3, 1)
    test_data = test_data.astype('float32')
    #mean_pixel = [103.939, 116.779, 123.68]
    #for c in range(3):
        #test_data[:, c, :, :] = test_data[:, c, :, :] - mean_pixel[c]
    test_data /= 255
    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    return test_data, test_id


def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()


def merge_several_folds_geom(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a *= np.array(data[i])
    a = np.power(a, 1/nfolds)
    return a.tolist()


def copy_selected_drivers(train_data, train_target, driver_id, driver_list):
    data = []
    target = []
    index = []
    for i in range(len(driver_id)):
        if driver_id[i] in driver_list:
            data.append(train_data[i])
            target.append(train_target[i])
            index.append(i)
    data = np.array(data)
    target = np.array(target)
    index = np.array(index)
    return data, target, index


def create_model_v1(img_rows, img_cols, color_type=1):
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


def create_model_v2(img_rows, img_cols, color_type=1):
    model = Sequential()
    model.add(Convolution2D(512, 3, 3, border_mode='same', init='he_normal', activation='relu',
                            input_shape=(color_type, img_rows, img_cols)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.5))

    model.add(Convolution2D(256, 3, 3, border_mode='same', init='he_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.5))

    #model.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal'))
    #model.add(MaxPooling2D(pool_size=(8, 8)))
    #model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128, init = 'glorot_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.005, decay = 1e-6, momentum = 0.9, nesterov=True)

    model.compile(loss = 'categorical_crossentropy', optimizer = sgd)
    return model


def vgg_bn(img_rows, img_cols, color_type=1):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal', input_shape=(color_type, img_rows, img_cols)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), init='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', init='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, 3, subsample=(2, 2), init='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='sigmoid', init='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax', init='he_normal'))
    model.compile(Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def vgg_std16_model(img_rows, img_cols, color_type=3):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(color_type,
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

    model.add(Flatten())#check
    #model.add(Dense(256, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    #model.load_weights('vgg16_weights.h5')

    # Code above loads pre-trained data and
    #model.layers.pop()
    #model.add(Dense(10, activation='softmax'))
    # Learning rate is changed to 0.001
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    #model.layers[0] = Convolution2D(32, 3, 3, border_mode='same', init='he_normal',
                            #input_shape=(color_type, img_rows, img_cols))
    #model.layers[1] = MaxPooling2D(pool_size=(2, 2))
    #model.layers[2] = Dropout(0.5)

    #model.layers[3] = Convolution2D(64, 3, 3, border_mode='same', init='he_normal')
    #model.layers[4] = MaxPooling2D(pool_size=(2, 2))
    #model.layers[5] = Dropout(0.5)

    #model.layers[6] = Convolution2D(128, 3, 3, border_mode='same', init='he_normal')
    #model.layers[7] = MaxPooling2D(pool_size=(8, 8))
    #model.layers[8] = Dropout(0.5)

    #model.layers[9] = Flatten()
    #model.layers[10] = Dense(10)
    #model.layers[11] = Activation('softmax')
    #model.layers = model.layers[:12]
    model.compile(sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def get_validation_predictions(train_data, predictions_valid):
    pv = []
    for i in range(len(train_data)):
        pv.append(predictions_valid[i])
    return pv


def run_cross_validation(nfolds=10):
    # input image dimensions
    img_rows, img_cols = 64, 64
    # color type: 1 - grey, 3 - rgb
    color_type_global = 3
    batch_size = 32
    nb_epoch = 50
    random_state = 51
    restore_from_last_checkpoint = 0

    test_data = read_and_normalize_test_data(img_rows, img_cols, color_type_global)
    train_data, train_target, train_id, driver_id, unique_drivers = get_X_y()

    #cache_path = os.path.join('cache', 'train_r_' + str(img_rows) + '_c_' + str(img_cols) + '_t_' + str(color_type_global) + '_rotated.dat')
    #if not os.path.isfile(cache_path) or use_cache == 0:
        #train_data, train_target, train_id, driver_id, unique_drivers = get_X_y()
        #cache_data((train_data, train_target, train_id, driver_id, unique_drivers), cache_path)
    #else:
        #print('Restore train from cache!')
        #(train_data, train_target, train_id, driver_id, unique_drivers) = restore_data(cache_path)


    #train_data, train_target, train_id, driver_id, unique_drivers = read_and_normalize_train_data(img_rows, img_cols, color_type_global)
    #test_data = read_and_normalize_test_data(img_rows, img_cols, color_type_global)
    other_data, other_target, other_id = get_other()
    model = vgg_std16_model(img_rows, img_cols, color_type_global)

    yfull_train = dict()
    yfull_test = []
    kf = KFold(len(unique_drivers), n_folds=nfolds, shuffle=True, random_state=random_state)
    num_fold = 0
    sum_score = 0
    for train_drivers, test_drivers in kf:

        unique_list_train = [unique_drivers[i] for i in train_drivers]
        X_train, Y_train, train_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_train)
        X_train = np.concatenate((X_train,other_data),axis=0)
        X_train = np.array(X_train, dtype=np.uint8)
        Y_train = np.array(Y_train, dtype=np.uint8)
        Y_train = np_utils.to_categorical(Y_train, 10)
        Y_train = np.concatenate((Y_train,other_target),axis=0)
        #X_train = np.array([get_im_mod(i, img_rows, img_cols) for i in X_train])
        X_train = X_train.transpose((0, 3, 1, 2))
        #X_train = X_train.astype('float32')
        #X_train /= 255
        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'X_train samples')

        unique_list_valid = [unique_drivers[i] for i in test_drivers]
        X_valid, Y_valid, test_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_valid)
        X_valid = np.array(X_valid, dtype=np.uint8)
        Y_valid = np.array(Y_valid, dtype=np.uint8)
        Y_valid = np_utils.to_categorical(Y_valid, 10)
        #X_valid = np.array([get_im_mod(i, img_rows, img_cols) for i in X_valid])
        #X_valid = np.array([cv2.resize(i, (img_cols, img_rows), interpolation=cv2.INTER_LINEAR) for i in X_valid])
        X_valid = X_valid.transpose((0, 3, 1, 2))
        #X_valid = X_valid.astype('float32')
        #X_valid /= 255
        print('X_valid shape:', X_valid.shape)
        print(X_valid.shape[0], 'X_valid samples')

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))
        print('Train drivers: ', unique_list_train)
        print('Test drivers: ', unique_list_valid)

        kfold_weights_path = os.path.join('cache', 'weights_kfold_' + str(num_fold) + '.h5')
        if not os.path.isfile(kfold_weights_path) or restore_from_last_checkpoint == 0:
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=2, verbose=0),
                ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0),
            ]
            model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                  shuffle=True, verbose=1, validation_data=(X_valid, Y_valid),
                  callbacks=callbacks)
        if os.path.isfile(kfold_weights_path):
            model.load_weights(kfold_weights_path)

        # score = model.evaluate(X_valid, Y_valid, show_accuracy=True, verbose=0)
        # print('Score log_loss: ', score[0])

        predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)
        score = log_loss(Y_valid, predictions_valid)
        print('Score log_loss: ', score)
        sum_score += score*len(test_index)

        # Store valid predictions
        for i in range(len(test_index)):
            yfull_train[test_index[i]] = predictions_valid[i]

        # Store test predictions
        test_prediction = model.predict(test_data, batch_size=batch_size, verbose=1)
        if num_fold >= 2:
            yfull_test.append(test_prediction)

    score = sum_score/len(train_data)
    print("Log_loss train independent avg: ", score)

    predictions_valid = get_validation_predictions(train_data, yfull_train)
    score1 = log_loss(train_target, predictions_valid)
    if abs(score1 - score) > 0.0001:
        print('Check error: {} != {}'.format(score, score1))

    print('Final log_loss: {}, rows: {} cols: {} nfolds: {} epoch: {}'.format(score, img_rows, img_cols, nfolds, nb_epoch))
    info_string = 'loss_' + str(score) \
                    + '_r_' + str(img_rows) \
                    + '_c_' + str(img_cols) \
                    + '_folds_' + str(nfolds) \
                    + '_ep_' + str(nb_epoch)

    test_res = merge_several_folds_mean(yfull_test, nfolds)
    # test_res = merge_several_folds_geom(yfull_test, nfolds)
    create_submission(test_res, test_id, info_string)
    df = pd.DataFrame(test_prediction, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    df.insert(0, 'img', test_id)
    df.to_csv('submission.csv', index = False)
    save_useful_data(predictions_valid, train_id, model, info_string)


def run_single2(model, trial = 0):
    # input image dimensions
    img_rows, img_cols = 64, 64
    # color type: 1 - grey, 3 - rgb
    color_type_global = 3
    batch_size = 32
    nb_epoch = 50
    random_state = 51

    print "shuffling submission labels"
    move_img_to_label()

    print "acquiring data..."
    test_data, test_id = read_and_normalize_test_data(img_rows, img_cols, color_type_global)

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1./255)

    # this is the augmentation configuration we will use for testing
    #test_datagen = ImageDataGenerator(
        #rotation_range = 10)

    #test_data = test_datagen.fit(test_data)

    train_generator = train_datagen.flow_from_directory(
        'sub_img/train',
        target_size=(img_rows, img_cols),  # all images will be resized to 64x64
        batch_size=batch_size,
        class_mode='categorical')  # since we use categorical_crossentropy loss, we need categorical labels

    validation_generator = train_datagen.flow_from_directory(
        'sub_img/test',
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')





    #model = vgg_std16_model(img_rows, img_cols, color_type_global)


    print "training data..."
    weights_path = os.path.join('weights_sub_retrain_{}'.format(trial) + '.h5')
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, verbose=0),
        ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True, verbose=0)]
    model.fit_generator(train_generator, nb_epoch=nb_epoch,
                samples_per_epoch=10000,
                verbose=1, validation_data=validation_generator,
                callbacks=callbacks, nb_val_samples=2000)


    # Store test predictions
    print "predicting on test data..."
    test_prediction = model.predict(test_data, batch_size=batch_size, verbose=1)

    df = pd.DataFrame(test_prediction, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    df.insert(0, 'img', test_id)
    df.to_csv('submission_retrain_{}.csv'.format(trial), index = False)
    return test_prediction


def run_single():
    # input image dimensions
    img_rows, img_cols = 64, 64
    color_type_global = 1
    batch_size = 32
    nb_epoch = 50
    random_state = 51

    train_data, train_target, train_id, driver_id, unique_drivers = read_and_normalize_train_data(img_rows, img_cols, color_type_global)
    test_data, test_id = read_and_normalize_test_data(img_rows, img_cols, color_type_global)

    yfull_test = []
    unique_list_train = ['p002', 'p012', 'p014', 'p015', 'p016', 'p021', 'p022', 'p035', 'p041', 'p042', 'p045', 'p047', 'p049', 'p050', 'p051', 'p052', 'p056', 'p061', 'p064', 'p066', 'p075', 'p081']
    X_train, Y_train, train_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_train)
    unique_list_valid = ['p024', 'p026', 'p039', 'p072']
    X_valid, Y_valid, test_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_valid)

    print('Start Single Run')
    print('Split train: ', len(X_train))
    print('Split valid: ', len(X_valid))
    print('Train drivers: ', unique_list_train)
    print('Valid drivers: ', unique_list_valid)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, verbose=0),
    ]
    model = create_model_v1(img_rows, img_cols, color_type_global)
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              shuffle=True, verbose=1, validation_data=(X_valid, Y_valid),
              callbacks=callbacks)

    # score = model.evaluate(X_valid, Y_valid, show_accuracy=True, verbose=0)
    # print('Score log_loss: ', score[0])

    predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)
    score = log_loss(Y_valid, predictions_valid)
    print('Score log_loss: ', score)

    # Store test predictions
    test_prediction = model.predict(test_data, batch_size=batch_size, verbose=1)
    yfull_test.append(test_prediction)

    print('Final log_loss: {}, rows: {} cols: {} epoch: {}'.format(score, img_rows, img_cols, nb_epoch))
    info_string = 'loss_' + str(score) \
                    + '_r_' + str(img_rows) \
                    + '_c_' + str(img_cols) \
                    + '_ep_' + str(nb_epoch)

    full_pred = model.predict(train_data, batch_size=batch_size, verbose=1)
    score = log_loss(train_target, full_pred)
    print('Full score log_loss: ', score)

    test_res = merge_several_folds_mean(yfull_test, 1)
    create_submission(test_res, test_id, info_string)
    save_useful_data(full_pred, train_id, model, info_string)


if __name__ == '__main__':
    #np.random.seed(51)
    predictions = []
    for i in xrange(6):
        model = vgg_std16_model(64, 64, 3)
        test_prediction = run_single2(model = model, trial = i)
        predictions.append(test_prediction)
    test_res = merge_several_folds_mean(predictions, 6)
    create_submission(test_res, test_id, "trainontest")
