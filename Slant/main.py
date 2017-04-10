from initialisations import *
from lineSegment import *
    
img = cv2.imread('i6.jpg',0)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
ret, binaryImage = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
rows = binaryImage.shape[0]
cols = binaryImage.shape[1]

# Load Neural Network
f = open('net1_master.pickle','rb')
net1 = pickle.load(f)
f.close()    

rowSumList = []
for i in range(rows):
    rowsum = 0
    for j in range(cols):
        if binaryImage[i][j] == 0:
            rowsum = rowsum + 1
    rowSumList.append(rowsum)

winSize = getWindowSize(rowSumList)
mavg = rowSumList
for i in range(1, 5):
    mavg = movingaverage(mavg, winSize * i)

brkpts = getMaxima(mavg)

# Visualization
func = interp1d(range(rows), rowSumList)

# Plot raw data
plt.plot(range(rows), rowSumList, 'o', range(rows), func(range(rows)), '-',)
plt.show()
# Plot smoothened data
plt.plot(range(rows), mavg, 'g-') 
plt.show()

lineSegmentation(binaryImage,rows,cols,brkpts,len(brkpts),net1)