import os
import cv2
import gzip
import theano
import lasagne
import matplotlib
import numpy as np
import pandas as pd
import _pickle as pickle
import theano.tensor as T
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from lasagne import layers
from sklearn import cross_validation
from urllib.request import urlretrieve
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from lasagne.updates import nesterov_momentum
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def changeSize(x):
    I = []  
    for j in range(16):                               
        templist = []         
        for k in range (8):            
            templist.append(0)      
        I.append(templist) 
    
    for i in range(16):
        for j in range(8):
            I[i][j] = 255*x[i*8 + j]

    I = np.array(I, dtype = np.uint8)
    I = cv2.resize(I,(16, 32))
    ret, I = cv2.threshold(I,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    x = [] 
    for i in range(32*16):
        x.append(0)
    
    for i in range(32):
        for j in range(16):
            if I[i][j] == 255:
                x[i*16 + j] = 1
    return x

def load_dataset():
    
    data = pd.read_excel('small.xlsx',header=None)
        
    X_test = []
    y_test = []
    
    data = np.array(data);

    for i in range(50000,52152):
        X_test.append(changeSize(data[i,1:]))
    for i in range(50000,52152):
        y_test.append(int(ord(data[i,0])-97))
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_test = X_test.reshape((-1, 1, 32, 16)).astype('float32')
    y_test = y_test.astype(np.uint8)
    return X_test, y_test

X_test, y_test = load_dataset()

f = open('net1_master.pickle','rb')
net1 = pickle.load(f)
f.close()

c = 0
preds = net1.predict(X_test)
for i in range(len(preds)):
    if preds[i] == y_test[i]:
        c = c + 1
        
print(c/len(preds))
        
    
print(preds.shape)
#print(lasagne.objectives.categorical_crossentropy(preds, y_test))

cm = confusion_matrix(y_test, preds)
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()