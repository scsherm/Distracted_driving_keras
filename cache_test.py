from __future__ import division
'''
file reads images and caches them into h5 file with keys:
X_test: numpy array with shape (, 224, 224, 3) which is consistent with VGG16 input shape
fName: numpy array with shape (,) which defines filename of each image
based on https://github.com/ZFTurbo/KAGGLE_DISTRACTED_DRIVER/blob/master/run_keras_simple.py
'''
import os
import h5py
import glob
import cv2
import numpy as np
import math
import random


def get_im(path, img_rows, img_cols):
    '''read and resize image'''

    img = cv2.imread(path)
    resized = cv2.resize(img, (img_cols, img_rows))
    return resized


def get_im_cv2_mod(path, img_rows, img_cols):
    '''read, resize, rotate image'''

    img = cv2.imread(path)
    # Reduce size
    rotate = random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), rotate, 1)
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    resized = cv2.resize(img, (img_cols, img_rows), interpolation=cv2.INTER_LINEAR)
    #blur = cv2.blur(resized,(5,5))
    return resized


def load_test(img_rows, img_cols):
    '''read and process test images'''

    print('Read test images')
    path = os.path.join('imgs', 'test', '*.jpg')
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    total = 0
    thr = math.floor(len(files) / 10)
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2_mod(fl, img_rows, img_cols)
        X_test.append(img)
        X_test_id.append(flbase)
        total += 1
        if total % thr == 0:
            print('Read {} images from {}'.format(total, len(files)))

    return X_test, X_test_id


if __name__ == '__main__':
	img_rows, img_cols = 224, 224
	X_test, X_test_id = load_test(img_rows, img_cols)
	X_test = np.array(X_test, dtype=np.uint8)
	f = h5py.File('test_224_224.h5','w')
	f['X_test'] = X_test
	f['X_test_id'] = X_test_id
	f.close()
