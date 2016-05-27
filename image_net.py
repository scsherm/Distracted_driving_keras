import os, glob, math, cv2, time
import numpy as np
from joblib import Parallel, delayed
os.environ['THEANO_FLAGS'] = "device=gpu,floatX=float32"
import theano
from skimage.io import imread, imshow
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.feature._canny import canny
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle
from skimage.transform import resize
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, MaxoutDense
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, Adamax
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from multiprocessing import Pool


def get_image_paths(pardir):
    '''returns list of image paths for all images'''

    return glob.glob(pardir + "/*/*.jpg")


def read_imgs(fpath):
    '''return pixel data for images in image path list'''

    return imread(fpath)


def parallel_read_imgs(parameter_list):
	'''Runs the read_imgs function in parallel on 8 cores'''

	pool = Pool(processes = 32)
	results = pool.map(read_imgs, parameter_list)
	return results


def get_labels(img_paths):
    '''returns y_label for each image in parent directory'''

    return np.array([i.split('/')[2] for i in img_paths])


def convert_to_gray(image):
    '''return pixel data for images in image path list'''

    return rgb2gray(image)


def parallel_convert_to_gray(parameter_list):
	'''Runs the convert_to_gray function in parallel on 8 cores'''

	pool = Pool(processes = 32)
	results = pool.map(convert_to_gray, parameter_list)
	return results


def resize_img(image):
    '''return pixel data for images in image path list'''

    return resize(image, (500,500))


def parallel_resize_img(parameter_list):
	'''Runs the convert_to_gray function in parallel on 8 cores'''

	pool = Pool(processes = 32)
	results = pool.map(resize_img, parameter_list)
	return results


def process_image(img_file, sz = (100,100)):
    img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, sz).astype('float32') / 255.0
    return img


def get_X_y():
    start = time.time()

    X = []
    y = []

    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join('imgs/train', 'c' + str(j), '*.jpg')
        files = glob.glob(path)
        X.extend(Parallel(n_jobs=-1)(delayed(process_image)(im_file) for im_file in files))
        y.extend([j]*len(files))

    end = time.time() - start
    print("Time: %.2f seconds" % end)
    return X, y


def process_test_image(img_file):
    return process_image(img_file), os.path.basename(img_file)


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
    return test, test_id


### Convolutional network
def recall_loss(y_true, y_pred):
    '''
    input: y_true (theano Tensor), y_pred (theano Tensor)
    output: recall_loss (float)
    '''
    # print(K.ndim(y_true), K.ndim(y_pred))
    return -np.log(K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1))))


def nn_model(X_train, y_train, X_test, y_test, batch_size = 20, nb_classes = 10, nb_epoch = 100):
    # need to fix docs for X_train and X_test as these should be 3D or 4D arrays
    '''
    input: X_train (4D np array), y_train (1D np array), X_test (4D np array), y_test (1D np array)
    optional: batch_size (int), n_classes (int), n_epochs (int)
    output: tpl (test score, test accuracy)
    '''
    # get number of test and train obs
    #n_train, n_test = X_train.shape[0], X_test.shape[0]

    # reshape images because keras is being picky
    #X_train = X_train.reshape(n_train, 60, 60, 3)
    #X_test = X_test.reshape(n_test, 60, 60, 3)

    # import pdb; pdb.set_trace()

    # initialize sequential model
    model = Sequential()

    # first convolutional layer and subsequent pooling
    model.add(Convolution1D(512, 3, border_mode='valid', input_shape=X_train[0].shape, activation='relu', init='glorot_normal'))#dim_ordering='tf'
    model.add(MaxPooling1D(pool_length=2))

    model.add(Convolution1D(256, 3, border_mode='valid', activation='relu', init='glorot_normal'))
    model.add(MaxPooling1D(pool_length=2))

    model.add(Convolution1D(128, 3, border_mode='valid', activation='relu', init='glorot_normal'))
    model.add(MaxPooling1D(pool_length=2))

    # # second convolutional layer and subsequent pooling
    model.add(Convolution1D(64, 3, border_mode='valid', activation='relu', init='glorot_normal'))
    model.add(MaxPooling1D(pool_length=2))
    #
    # # third convolutional layer
    # #model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', init='glorot_normal', dim_ordering='tf'))
    #
    # # fourth convolutional layer and subsequent pooling
    model.add(Convolution1D(32, 3, border_mode='valid', activation='relu', init='glorot_normal'))
    model.add(MaxPooling1D(pool_length=2))
    #
    # model.add(Convolution1D(16, 3, border_mode='valid', activation='relu', init='glorot_normal'))
    # model.add(MaxPooling1D(pool_length=2))

    # flattens images to go into dense layers
    model.add(Flatten())

    # first dense layer
    model.add(Dense(2048, init = 'glorot_normal'))
    # model.add(MaxoutDense(2048))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # second dense layer
    model.add(Dense(1024, init = 'glorot_normal'))
    # model.add(Dense(2048, init= 'he_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # third dense layer
    model.add(Dense(512, init = 'glorot_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # fourth dense layer
    model.add(Dense(256, init = 'glorot_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # output layer
    model.add(Dense(10, init='glorot_uniform'))
    model.add(Activation('softmax'))

    # initializes optimizer
    sgd = SGD(lr=0.005, decay = 1e-6, momentum = 0.9, nesterov=True)
    #adamax = Adamax(lr=0.02, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # initializes early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')

    # compiles and fits model, computes accuracy
    model.compile(loss = 'categorical_crossentropy', optimizer = sgd)

    model.fit(X_train, y_train, show_accuracy=True, verbose=1,
        callbacks = [early_stopping], batch_size= batch_size, nb_epoch=nb_epoch,
        validation_data=(X_test, y_test))

    return model, model.evaluate(X_test, y_test, show_accuracy=True, verbose=1)


def scores(model, X_test, y_test):
    '''
    input: model (keras model), X_test ( 4D np array of images), y_test (1D np array of labels)
    output: None
    '''
    y_pred = model.predict_classes(X_test)
    y_pred = np.array([i[0] for i in y_pred])
    y_true = pd.get_dummies(y_test[:,1]).values.argmax(1)
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    return y_pred, y_true


def convert_targets(targets):
    '''
    input: targets (1D np array of strings)
    output: targets dummified category matrix
    note: targets are indexed as ['elliptical', 'merger', 'spiral']
    '''
    return pd.get_dummies(targets).values


if __name__ == '__main__':
    X,y = get_X_y()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
        random_state=42, stratify = y)
    y_train, y_test = convert_targets(y_train), convert_targets(y_test)
    model, results = image_explore.nn_model(X_train, y_train, X_test, y_test, batch_size = 20, nb_classes = 10, nb_epoch = 40)
    y_pred, y_true = image_explore.scores(model, X_test, y_test[:,1])
    test, test_id = get_test_data()
    test_prediction = model.predict(test, batch_size=128, verbose=1)
    df = pd.DataFrame(test_prediction, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    df.insert(0, 'img', test_id)
    df.to_csv('submission.csv', index = False)
