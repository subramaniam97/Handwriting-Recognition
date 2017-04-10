from initialisations import *
from lineSegment import *

def showImg(th2):
    cv2.imshow('image',th2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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

def getBreakPoints(interval):
    res = []
    sz = len(interval)
    for i in range(2, sz - 2):
        if interval[i - 2] < interval[i] and interval[i - 1] < interval[i] and interval[i + 1] < interval[i] and interval[i + 2] < interval[i]:
            res.append(i)
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

if __name__=="__main__":
    # Loading model
    f = open('net1_master.pickle','rb')
    net1 = pickle.load(f)
    f.close()        

    img = cv2.imread('Images/h3.jpg', 0)    
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    r = th2.shape[0]
    c = th2.shape[1]
    avg = []
    for i in range(r):
        rowsum = 0
        f = 0
        for j in range(c):
            if th2[i][j] == 0:
                rowsum = rowsum + 1
                f = 1
        avg.append(rowsum)
    
    
    winSize = getWindowSize(avg)
    mavg = avg
    for i in range(1, 5):
        mavg = movingaverage(mavg, winSize * i)
    
    plt.plot(range(r), avg, 'o')
    # plt.plot(range(r), mavg, 'g-') # Uncomment to visualize the smoothed curve
    plt.show()
        
    mxpoints = getBreakPoints(mavg)
    size = len(mxpoints)
    
    # print(mxpoints) # Uncomment to print the break points.
    
    # Open a file to store the result
    fileObject = open("result.txt", "w")
    
    lineSeg(th2, r, c, mxpoints, size, fileObject, net1)
    
    fileObject.close()
