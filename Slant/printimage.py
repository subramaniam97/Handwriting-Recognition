from initialisations import *

def convertToImage(x):
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
    return I

data = pd.read_excel('small.xlsx',header=None)
            
X_train = []
X_test = []
y_train = []
y_test = []

data = np.array(data);

for i in range(51000,52000):
    showImg(convertToImage(data[i,1:]))
    X_train.append(data[i,1:])
