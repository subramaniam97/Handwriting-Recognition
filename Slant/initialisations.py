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

from scipy.interpolate import interp1d
import math

'''For BFS'''
from pythonds.graphs import Graph, Vertex
from pythonds.basic import Queue

# Display image
def showImg(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()
    
# Moving average
def movingaverage(interval, window_size):
   res = []
   sz = len(interval)
   window_size = int(window_size / 2)
   for i in range(window_size):
       res.append(interval[i])
   for i in range(window_size, sz - window_size):
       lsum = rsum = 0
       sum = interval[i]
       for j in range(i - window_size, i):
           lsum = lsum + interval[j]
       for j in range(i + 1, i + 1 + window_size):
           rsum = rsum + interval[j]
       sum = sum + lsum + rsum
       sum = sum / (window_size + window_size + 1)
       res.append(sum)
   for i in range(sz - window_size, sz):
       res.append(interval[i])
   return res

def getBottomThreshold(interval):
    num = denom = 0
    sz = len(interval)
    for i in range(sz):
        num += (1 / math.log2(2 + interval[i])) * (interval[i])
        denom += (1 / math.log2(2 + interval[i]))
    return (num / denom)

def getTopThreshold(interval):
    num = denom = 0
    sz = len(interval)
    for i in range(sz):
        num += (math.log2(2 + interval[i])) * (interval[i])
        denom += (math.log2(2 + interval[i]))
    return (num / denom)

def getWindowSize(interval):
    bottomThreshold = getBottomThreshold(interval)
    topThreshold = getTopThreshold(interval)
    sum = 0
    cnt = 0
    sz = len(interval)
    p = q = 0
    while q < sz - 1:
        while p < sz - 1 and interval[p] <= bottomThreshold:
            p += 1
        while q < sz - 1 and interval[q] <= topThreshold:
            q += 1
        while q < sz - 1 and interval[q] >= bottomThreshold:
            q += 1
        sum += (q - p + 1)
        p = q + 1
        cnt += 1

    sum = int(math.ceil(sum / (10 * cnt)))
    return sum
  
# Get desired points
def getMaxima(interval):
    res = []
    sz = len(interval)
    for i in range(2, sz - 2):
        if interval[i - 2] < interval[i] and interval[i - 1] < interval[i] and interval[i + 1] < interval[i] and interval[i + 2] < interval[i]:
            res.append(i)
    return res
    
def getMaxima2(interval):
    res = []
    sz = len(interval)
    for i in range(1, sz - 1):
        if interval[i - 1] <= interval[i] and interval[i + 1] <= interval[i]:
            res.append(i)
    return res
    
def getMinima(interval):
    res = []
    sz = len(interval)
    for i in range(2, sz - 2):
        if interval[i - 2] > interval[i] and interval[i - 1] > interval[i] and interval[i + 1] > interval[i] and interval[i + 2] > interval[i]:
            res.append(i)
    return res
    
def getMinima2(interval):
    res = []
    sz = len(interval)
    for i in range(1, sz - 1):
        if interval[i - 1] >= interval[i] and interval[i + 1] >= interval[i]:
            res.append(i)
    return res
    

# Shear functions
def computeShear(I, baseline):
    rows = I.shape[0]
    cols = I.shape[1]

    cur = 0
    flag = 0
    for i in range(cols):
        for j in range(rows):
            if I[j][i] == 0:
                cur = i
                flag = 1
                break
        if (flag):
            break

    prevheight = -math.inf    
    ref = 0
    
    for i in range(cols):
        if I[baseline][i] == 0:
            ref = i
            break

    y = rows
    x = 0     
        
    while (cur < cols):
        curheight = 0
        for i in range(rows):
            if I[i][cur] == 0:
              curheight = rows - i
              break
          
        if curheight < prevheight:
            y = prevheight
            x = cur - 1 - ref
            break
        
        prevheight = curheight
        cur = cur + 1

    # shear = tan(theta) = x/y 
    return x/y
    
def shearImage(I,shear):
    rows = I.shape[0]
    cols = I.shape[1]
    shearedImage = []  
    for i in range(rows):                               
        temp = []         
        for j in range(cols + 10):            
            temp.append(255)      
        shearedImage.append(temp) 

    for i in range(rows):
        for j in range(cols):
            if I[i][j] == 0:
                inew = i
                jnew = int(j + (rows - i)*shear) + 10
                if (jnew >= 0 and jnew <= cols + 10):
                    shearedImage[inew][jnew] = I[i][j]

    shearedImage = np.array(shearedImage, dtype = np.uint8)
    return shearedImage

# Character Classification

def classifyCharacter(I,net1):
    
    # Padding
    rows = I.shape[0]
    cols = I.shape[1]

    ratio = rows/cols
    scaledImage = []
    
    # Pad with whites along cols
    if ratio >= 2:
        for i in range(rows + rows%2):                               
            temp = []         
            for j in range(int((rows + rows%2)/2)):            
                temp.append(255)      
            scaledImage.append(temp) 
            
        left = math.floor(((rows + rows%2)/2 - cols)/2)
        right = math.ceil(((rows + rows%2)/2 - cols)/2)
        
        for i in range(rows):
            for j in range(left,left + cols):
                scaledImage[i][j] = I[i][j - left]
        
    # Pad with whites along rows
    else:
        for i in range(2*cols):                               
            temp = []         
            for j in range(cols):            
                temp.append(255)      
            scaledImage.append(temp) 
            
        top = math.floor((2*cols - rows)/2)
        bottom = math.ceil((2*cols - rows)/2)
        
        for i in range(top,top + rows):
            for j in range(cols):
                scaledImage[i][j] = I[i - top][j]
        
    scaledImage = np.array(scaledImage, dtype = np.uint8)
    I = scaledImage
    
    # Invert bits
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
                I[i][j] = 255 - I[i][j]  
   
    # Scale 
    # resize function takes width*height
    I = cv2.resize(I,(16, 32))
    ret, I = cv2.threshold(I,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    showImg(I)
    # Format to pass it to CNN
    temp = []
    for i in range(32):
        for j in range(16):
            temp.append(int(I[i][j]/255))
    print(temp)
    
    char = []
    char.append(temp)
    char = np.array(char)

    char = char.reshape((-1, 1, 32, 16)).astype('float32')
    
    preds = net1.predict(char)
    probability = net1.predict_proba(char)
    maxprob = probability[0][np.argmax(probability[0])]
    print(chr(preds[0]+97))

    return preds, maxprob