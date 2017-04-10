from initialisations import *

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
            
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    
    data = np.array(data);

    for i in range(1,50000):
        X_train.append(changeSize(data[i,1:]))
    for i in range(1,50000):
        y_train.append(int(ord(data[i,0])-97))
    for i in range(50000,52152):
        X_test.append(changeSize(data[i,1:]))
    for i in range(50000,52152):
        y_test.append(int(ord(data[i,0])-97))
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    print(set(y_train))
    print(X_train.shape)
    X_train = X_train.reshape((-1, 1, 32, 16)).astype('float32')
    X_test = X_test.reshape((-1, 1, 32, 16)).astype('float32')
    y_train = y_train.astype(np.uint8)
    print(X_train.shape)
    y_test = y_test.astype(np.uint8)
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_dataset()

#print(X_train.shape)
#print(y_train.shape)

net1 = NeuralNet(
    layers=[('input', layers.InputLayer),
            ('conv2d1', layers.Conv2DLayer),
            ('maxpool1', layers.MaxPool2DLayer),
            ('conv2d2', layers.Conv2DLayer),
            ('maxpool2', layers.MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),
            ('dense1', layers.DenseLayer),
            ('dropout2', layers.DropoutLayer),
            ('dense2', layers.DenseLayer),
            ('dropout3', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
    # input layer
    input_shape=(None, 1, 32, 16),
    # layer conv2d1
    conv2d1_num_filters=32,
    conv2d1_filter_size=(5, 5),
    conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d1_W=lasagne.init.GlorotUniform(),  
    # layer maxpool1
    maxpool1_pool_size=(2, 2),    
    # layer conv2d2
    conv2d2_num_filters=32,
    conv2d2_filter_size=(3, 3),
    conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
    # layer maxpool2
    maxpool2_pool_size=(1, 1),
    # dropout1
    dropout1_p=0.5,    
    # dense1
    dense1_num_units=256,
    dense1_nonlinearity=lasagne.nonlinearities.rectify,    
    # dropout2
    dropout2_p=0.5,    
     # dense2
    dense2_num_units=256,
    dense2_nonlinearity=lasagne.nonlinearities.rectify,    
    # dropout3
    dropout3_p=0.5,    
    # output
    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=26,
    # optimization method params
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,
    max_epochs=30,
    verbose=1,
    )                       

# Train the network
nn = net1.fit(X_train, y_train)

# Save the Neural Network
with open('net1_master.pickle', 'wb') as f:
    pickle.dump(net1, f)
f.close()