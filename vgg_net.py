import os, glob, math, cv2, time
import numpy as np
from joblib import Parallel, delayed
os.environ['THEANO_FLAGS'] = "device=gpu,floatX=float32"
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, MaxoutDense
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D, ZeroPadding2D
from keras.optimizers import SGD, Adam, Adamax
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from sklearn.cross_validation import train_test_split
import pandas as pd



def process_image(img_file, sz = (200,200)):
    img = cv2.imread(img_file)
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
    return np.array(X), np.array(y)


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


def VGG_16(X_train, y_train, X_test, y_test, batch_size = 20, nb_classes = 10, nb_epoch = 100):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=X_train[0].shape))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', init='glorot_normal'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', init='glorot_normal'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, border_mode='valid', activation='relu', init='glorot_normal'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, border_mode='valid', activation='relu', init='glorot_normal'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, border_mode='valid', activation='relu', init='glorot_normal'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, border_mode='valid', activation='relu', init='glorot_normal'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, border_mode='valid', activation='relu', init='glorot_normal'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, border_mode='valid', activation='relu', init='glorot_normal'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, border_mode='valid', activation='relu', init='glorot_normal'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, border_mode='valid', activation='relu', init='glorot_normal'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, border_mode='valid', activation='relu', init='glorot_normal'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, border_mode='valid', activation='relu', init='glorot_normal'))
    model.add(ZeroPadding2D((1,1)))
    #model.add(Convolution2D(512, 3, 3, border_mode='valid', activation='relu', init='glorot_normal'))
    #model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    #sgd = SGD(lr=0.005, decay = 1e-6, momentum = 0.9, nesterov=True)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

    # initializes early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')

    model.compile(loss = 'categorical_crossentropy', optimizer = sgd)

    model.fit(X_train, y_train, show_accuracy=True, verbose=1,
        callbacks = [early_stopping], batch_size= batch_size, nb_epoch=nb_epoch,
        validation_data=(X_test, y_test))

    return model, model.evaluate(X_test, y_test, show_accuracy=True, verbose=1)


def convert_targets(targets):
    '''
    input: targets (1D np array of strings)
    output: targets dummified category matrix
    note: targets are indexed as ['elliptical', 'merger', 'spiral']
    '''
    return pd.get_dummies(targets).values


if __name__ == '__main__':
    X,y = get_X_y()
    X = np.array(X)
    y=np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
        random_state=42, stratify = y)
    y_train, y_test = convert_targets(y_train), convert_targets(y_test)
    model, results = vgg_net.VGG_16(X_train, y_train, X_test, y_test, batch_size = 20, nb_classes = 10, nb_epoch = 40)
    test, test_id = get_test_data()
    test_prediction = model.predict(test, batch_size=128, verbose=1)
    df = pd.DataFrame(test_prediction, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    df.insert(0, 'img', test_id)
    df.to_csv('submission.csv', index = False)
