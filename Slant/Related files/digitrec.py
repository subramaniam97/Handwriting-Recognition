import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from urllib.request import urlretrieve
import _pickle as pickle
import os
import gzip
import numpy as np
import theano
from sklearn import cross_validation
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import cv2

def load_dataset():
    imgArray = []
    X_train = []
    X_test = []
    j1 = ""
    j1 = ""
    for k in range(1,63):
        imgArray = []
        if(k<10):
            j1 = '0' + str(k)
        else:
            j1 = str(k)
        
        for i in range(1,46):
            if(i<10):
                j2 = '0' + str(i)
            else:
                j2 = str(i) 
            dir_path = os.path.dirname(os.path.realpath(__file__))
            p = dir_path + '/English/Hnd/Img/Sample0'+j1+'/img0'+j1+'-0'+j2+'.png'
            img = cv2.imread(p,0)
            img = np.reshape(img, (np.product(img.shape))) 
            #print(img.shape)
            X_train.append(img)
            #plt.imshow(img)
        #X_train.append(imgArray)
        imgArray = []
        
        for i in range(46,56):
            if(i<10):
                j2 = '0' + str(i)
            else:
                j2 = str(i) 
            dir_path = os.path.dirname(os.path.realpath(__file__))
            p = dir_path + '/English/Hnd/Img/Sample0'+j1+'/img0'+j1+'-0'+j2+'.png'
            img = cv2.imread(p,0)
            img = np.reshape(img, (np.product(img.shape))) 
            X_test.append(img)
        #X_test.append(imgArray)
        
    y_train = []
    y_test = []
    for k in range(1,63):
        for i in range(1,46):
            y_train.append(i)
        for i in range(46,56):
            y_test.append(i)
        
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    X_train = X_train.reshape((-1, 1, 28, 28))
    X_test = X_test.reshape((-1, 1, 28, 28))
    #y_train = y_train.astype(np.uint8)
    #y_test = y_test.astype(np.uint8)
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_dataset()
print('---------------------')
print(X_train.shape)
print(y_train.shape)

#plt.imshow(X_train[0][0], cmap=cm.binary)


net1 = NeuralNet(
    layers=[('input', layers.InputLayer),
            ('conv2d1', layers.Conv2DLayer),
            ('maxpool1', layers.MaxPool2DLayer),
            ('conv2d2', layers.Conv2DLayer),
            ('maxpool2', layers.MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),
            ('dense', layers.DenseLayer),
            ('dropout2', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
    # input layer
    input_shape=(None, 1, 28, 28),
    # layer conv2d1
    conv2d1_num_filters=32,
    conv2d1_filter_size=(5, 5),
    conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d1_W=lasagne.init.GlorotUniform(),  
    # layer maxpool1
    maxpool1_pool_size=(2, 2),    
    # layer conv2d2
    conv2d2_num_filters=32,
    conv2d2_filter_size=(5, 5),
    conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
    # layer maxpool2
    maxpool2_pool_size=(2, 2),
    # dropout1
    dropout1_p=0.5,    
    # dense
    dense_num_units=256,
    dense_nonlinearity=lasagne.nonlinearities.rectify,    
    # dropout2
    dropout2_p=0.5,    
    # output
    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=26,
    # optimization method params
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,
    max_epochs=10,
    verbose=1,
    )                       

# Train the network
nn = net1.fit(X_train, y_train)


preds = net1.predict(X_test)

cm = confusion_matrix(y_test, preds)
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()