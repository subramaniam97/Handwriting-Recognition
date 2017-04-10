from initialisations import *

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
    
def getNodeNumber(r, c, x, y):
    return int((x * c) + y + 1)
    
diction = {}
    
def bellCurve(x, baseline):
    baseline /= 2
    return math.exp(-(x - baseline)**2)

def getMinMax(IMG, x, y, r):
    mn = math.inf
    mx = -math.inf
    for i in range(x, y + 1):
        for j in range(0, r):
            if IMG[j][i] == 0:
                mn = min(mn, j)
                mx = max(mx, j)
    return mn, mx

def getBottomThreshold(interval):
    num = denom = 0
    sz = len(interval)
    for i in range(sz):
        num += (1 / math.log2(2 + interval[i])) * (interval[i])   # Change function
        denom += (1 / math.log2(2 + interval[i]))                 # Change function
    return (num / denom)

def getTopThreshold(interval):
    num = denom = 0
    sz = len(interval)
    for i in range(sz):
        num += (math.log2(2 + interval[i])) * (interval[i])      # Change function
        denom += (math.log2(2 + interval[i]))                    # Change function
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

    sum = int(math.ceil(sum / (10 * cnt)))                    # Change constant
    return sum
    
def getFirstBlackPixel(IMG, baseline, col):
    r = 0
    for i in range(0, baseline + 1):
        if IMG[i][col] == 0:
            r = max(r, i)
    return r

def getFirstWhiteAfterBlackPixel(IMG, baseline, col):
    r = 0
    r1 = 0
    for i in range(0, baseline + 1):
        if IMG[i][col] == 0:
            r = max(r, i)
    for i in range(0, r):
        if IMG[i][col] == 255:
            r1 = max(r1, i)
    return r1

def getY(colsum, height, baseline, c):
    Y = []
    for i in range(c):
        num = (colsum[i] * math.exp(colsum[i])) + (height[i] * bellCurve(height[i], baseline))
        denom = math.exp(colsum[i]) + bellCurve(height[i], baseline)
        Y.append(num / denom)
    return Y

def getBaseLine(IMG, r, c):
    baseline = 0

    # Add baseline logic here


    return baseline

def classifyCharacter(IM, net1):

    IM = IM.resize((16,8))
    temp = []
    temp = np.array(temp)
    for i in range(16):
        for j in range(8):
            temp.append(IM[i][j])

    IM = temp
    IM = IM.reshape((-1, 1, 16, 8)).astype('float32')
    preds = net1.predict(I)

    # Add code for probability

    return preds[0], 1

def segmentChar(IM, r, c, brkPts, size, baseline):

    charImageListSingle = []
    charImageListClub = []
    for i in range(1, size):
        
        pathList1 = []
        pathList2 = []
        pathList3 = []
        pathList4 = []
        mn, mx = getMinMax(IM, brkPts[i - 1], brkPts[i], r)
        firstBlackPixel = getFirstBlackPixel(IM, baseline, brkPts[i - 1])
        firstWhiteAfterBlackPixel = getFirstWhiteAfterBlackPixel(IM, baseline, brkPts[i - 1])
        
        diction.clear()
        for j in range(r):
            for k in range(c):
                diction[getNodeNumber(r, c, j, k)] = [j, k, -1, -1]        
        bfsChar(IM, getNodeNumber(r, c, firstWhiteAfterBlackPixel, brkPts[i - 1]), brkPts[i - 1], brkPts[i], mn, mx, r, c)
        tempChar = getPathChar(getNodeNumber(r, c, baseline, mn))
        if tempChar[0] == -1:
            tempChar[0] = math.inf
            tempChar.append(math.inf)
        pathList1 = tempChar
        
        diction.clear()
        for j in range(r):
            for k in range(c):
                diction[getNodeNumber(r, c, j, k)] = [j, k, -1, -1]
        bfsChar(IM, getNodeNumber(r, c, firstBlackPixel, brkPts[i - 1]), brkPts[i - 1], brkPts[i], mn, mx, r, c)
        tempChar = getPathChar(getNodeNumber(r, c, baseline, mx))
        if tempChar[0] == -1:
            tempChar[0] = math.inf
            tempChar.append(math.inf)
        pathList2 = tempChar
        
        mini = min(pathList1[-2], pathList2[-2])
        
        firstBlackPixel = getFirstBlackPixel(IM, baseline, brkPts[i])
        firstWhiteAfterBlackPixel = getFirstWhiteAfterBlackPixel(IM, baseline, brkPts[i])
            
        diction.clear()
        for j in range(r):
            for k in range(c):
                diction[getNodeNumber(r, c, j, k)] = [j, k, -1, -1]
        bfsChar(IM, getNodeNumber(r, c, firstWhiteAfterBlackPixel, brkPts[i]), brkPts[i - 1], brkPts[i], mn, mx, r, c)
        tempChar = getPathChar(getNodeNumber(r, c, baseline, mn))
        if tempChar[0] == -1:
            tempChar[0] = -math.inf
        pathList3 = tempChar
        
        diction.clear()
        for j in range(r):
            for k in range(c):
                diction[getNodeNumber(r, c, j, k)] = [j, k, -1, -1]
        bfsChar(IM, getNodeNumber(r, c, firstBlackPixel, brkPts[i]), brkPts[i - 1], brkPts[i], mn, mx, r, c)
        tempChar = getPathChar(getNodeNumber(r, c, baseline, mx))
        if tempChar[0] == -1:
            tempChar[0] = -math.inf
        pathList4 = tempChar
        
        maxi = max(pathList3[-1], pathList4[-1])
            
        if maxi == -math.inf or mini == math.inf:
            continue

        # Temporary image initialised to zeroes
        charImage = [[0 for j in range(maxi - mini + 1)] for k in range(mx - mn + 1)]

        completeLeftPath = []
        for j in pathList1[:-2]:
            completeLeftPath.append(j)
        for j in pathList2[:-2]:
            completeLeftPath.append(j)

        completeRightPath = []
        for j in pathList3[:-2]:
            completeRightPath.append(j)
        for j in pathList4[:-2]:
            completeRightPath.append(j)

        # Jigsaw (Left break point)
        for j in completeLeftPath:
            for k in range(mini, diction[j][1] + 1):
                charImage[diction[j][0] - mn][k - mini] = 255

        # Jigsaw (Right break point)
        for j in completeRightPath:
            for k in range(diction[j][1], maxi + 1):
                charImage[diction[j][0] - mn][k - mini] = 255

        # Jigsaw (Restore character)
        for j in range(mn, mx + 1):
            for k in range(mini, maxi + 1):
                if charImage[j][k] == 0:
                    charImage[j][k] = IM[j + mn][k + mini]

        charImage = np.array(charImage, dtype = np.uint8)
        #cv2.imshow('image', charImage)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # Appending single character images
        charImageListSingle.append(charImage)

        if i < size - 1:
            pathList1 = []
            pathList2 = []
            pathList3 = []
            pathList4 = []
            mn, mx = getMinMax(IM, brkPts[i - 1], brkPts[i + 1], r)
            firstBlackPixel = getFirstBlackPixel(IM, baseline, brkPts[i - 1])
            firstWhiteAfterBlackPixel = getFirstWhiteAfterBlackPixel(IM, baseline, brkPts[i - 1])
            
            diction.clear()
            for j in range(r):
                for k in range(c):
                    diction[getNodeNumber(r, c, j, k)] = [j, k, -1, -1]        
            bfsChar(IM, getNodeNumber(r, c, firstWhiteAfterBlackPixel, brkPts[i - 1]), brkPts[i - 1], brkPts[i + 1], mn, mx, r, c)
            tempChar = getPathChar(getNodeNumber(r, c, baseline, mn))
            if tempChar[0] == -1:
                tempChar[0] = math.inf
                tempChar.append(math.inf)
            pathList1 = tempChar
            
            diction.clear()
            for j in range(r):
                for k in range(c):
                    diction[getNodeNumber(r, c, j, k)] = [j, k, -1, -1]
            bfsChar(IM, getNodeNumber(r, c, firstBlackPixel, brkPts[i - 1]), brkPts[i - 1], brkPts[i + 1], mn, mx, r, c)
            tempChar = getPathChar(getNodeNumber(r, c, baseline, mx))
            if tempChar[0] == -1:
                tempChar[0] = math.inf
                tempChar.append(math.inf)
            pathList2 = tempChar
            
            mini = min(pathList1[-2], pathList2[-2])
            
            firstBlackPixel = getFirstBlackPixel(IM, baseline, brkPts[i + 1])
            firstWhiteAfterBlackPixel = getFirstWhiteAfterBlackPixel(IM, baseline, brkPts[i + 1])
                
            diction.clear()
            for j in range(r):
                for k in range(c):
                    diction[getNodeNumber(r, c, j, k)] = [j, k, -1, -1]
            bfsChar(IM, getNodeNumber(r, c, firstWhiteAfterBlackPixel, brkPts[i + 1]), brkPts[i - 1], brkPts[i + 1], mn, mx, r, c)
            tempChar = getPathChar(getNodeNumber(r, c, baseline, mn))
            if tempChar[0] == -1:
                tempChar[0] = -math.inf
            pathList3 = tempChar
            
            diction.clear()
            for j in range(r):
                for k in range(c):
                    diction[getNodeNumber(r, c, j, k)] = [j, k, -1, -1]
            bfsChar(IM, getNodeNumber(r, c, firstBlackPixel, brkPts[i + 1]), brkPts[i - 1], brkPts[i + 1], mn, mx, r, c)
            tempChar = getPathChar(getNodeNumber(r, c, baseline, mx))
            if tempChar[0] == -1:
                tempChar[0] = -math.inf
            pathList4 = tempChar
            
            maxi = max(pathList3[-1], pathList4[-1])
                
            if maxi == -math.inf or mini == math.inf:
                continue
            # Temporary image initialised to zeroes
            charImage = [[0 for j in range(maxi - mini + 1)] for k in range(mx - mn + 1)]

            completeLeftPath = []
            for j in pathList1[:-2]:
                completeLeftPath.append(j)
            for j in pathList2[:-2]:
                completeLeftPath.append(j)

            completeRightPath = []
            for j in pathList3[:-2]:
                completeRightPath.append(j)
            for j in pathList4[:-2]:
                completeRightPath.append(j)

            # Jigsaw (Left break point)
            for j in completeLeftPath:
                for k in range(mini, diction[j][1] + 1):
                    charImage[diction[j][0] - mn][k - mini] = 255

            # Jigsaw (Right break point)
            for j in completeRightPath:
                for k in range(diction[j][1], maxi + 1):
                    charImage[diction[j][0] - mn][k - mini] = 255

            # Jigsaw (Restore character)
            for j in range(mn, mx + 1):
                for k in range(mini, maxi + 1):
                    if charImage[j][k] == 0:
                        charImage[j][k] = IM[j + mn][k + mini]

            charImage = np.array(charImage, dtype = np.uint8)
            #cv2.imshow('image', charImage)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            # Appending clubbed image of characters
            charImageListClub.append(charImage)
            
        charImageListClub = np.array(charImageListClub)
        charImageListSingle = np.array(charImageListSingle)
        return charImageListSingle, charImageListClub



def bfsChar(IM, start, y1, y2, mn, mx, r, c):
    x = diction[start][0]
    y = diction[start][1]
    diction[start][2] = -2;
    diction[start][3] = 1;
    vertQueue = Queue()
    vertQueue.enqueue(start)
    while (vertQueue.size() > 0):
        current = vertQueue.dequeue()
        x = diction[current][0]
        y = diction[current][1]

        #bottom node
        if x + 1 <= mx and IM[x + 1][y] == 255:
            neighbour = getNodeNumber(r,c,x + 1, y)
            if diction[neighbour][3] == -1:
                diction[neighbour][3] = 1
                diction[neighbour][2] = current
                vertQueue.enqueue(neighbour)

        #top node
        if x - 1 >= mn and IM[x - 1][y] == 255:
            neighbour = getNodeNumber(r,c,x - 1, y)
            if diction[neighbour][3] == -1:
                diction[neighbour][3] = 1
                diction[neighbour][2] = current
                vertQueue.enqueue(neighbour)

        #right node
        if y + 1 <= y2 and IM[x][y + 1] == 255:
            neighbour = getNodeNumber(r,c,x, y + 1)
            if diction[neighbour][3] == -1:
                diction[neighbour][3] = 1
                diction[neighbour][2] = current
                vertQueue.enqueue(neighbour)

        #left node
        if y - 1 >= y1 and IM[x][y - 1] == 255:
            neighbour = getNodeNumber(r,c,x, y - 1)
            if diction[neighbour][3] == -1:
                diction[neighbour][3] = 1
                diction[neighbour][2] = current
                vertQueue.enqueue(neighbour)

def getPathChar(NodeNumber):
    path = []
    if diction[NodeNumber][3] == -1:     # If the node is not reachable
        path.append(-1)
        return path
    minP = math.inf
    maxP = -math.inf
    print(NodeNumber)
    while not diction[NodeNumber][2] == -2:
        y = diction[NodeNumber][1]
        if minP >= y:
            minP = y
        if maxP <= y:
            maxP = y
        path.append(NodeNumber)
        NodeNumber = diction[NodeNumber][2]

    y = diction[NodeNumber][1]
    if minP >= y:
        minP = y
    if maxP <= y:
        maxP = y
    path.append(NodeNumber)
    path.append(minP)
    path.append(maxP)
    return path



def charSeg(IMG, maxima, fileObject, net1):

    r = IMG.shape[0]
    c = IMG.shape[1]
    print(r)
    print(c)

    colsum = []
    height = []
    baseline = maxima + 2                # Check 

    for i in range(c):
        s = 0
        for j in range(baseline):
            if IMG[j][i] == 0:
                s += 1
        colsum.append(s)
        s = 0
        for j in range(baseline):
            if IMG[j][i] == 0:
                s = baseline - j - 1
                break;
        height.append(s)

    interval = getY(colsum, height, baseline, c)
    window_size = getWindowSize(interval)
    mavg = movingaverage(interval, window_size)     # Decide how many times to repeat
    brkPts = getBreakPoints(mavg)
    size = len(brkPts)

    charImageListSingle, charImageListClub = segmentChar(IMG, r, c, brkPts, size,baseline)

    finalCharImageList = []

    p1 = 0      # First pointer in the single list
    p2 = 0      # Second pointer in the single list
    p3 = 0      # Pointer in the clubbed list
    singleListSize = charImageListSingle.shape[0]
    clubListSize = charImageListClub.shape[0]

    while p1 < singleListSize - 1 and p3 < clubListSize:
        p2 = p1 + 1
        char1, prob1 = classifyCharacter(charImageListSingle[p1], net1)
        char3, prob3 = classifyCharacter(charImageListClub[p2], net1)
        if prob1 <= prob3:
            char1 = char3
            prob1 = prob3
        char2, prob2 = classifyCharacter(charImageListClub[p3], net1)

        if prob2 >= prob1:
            # It is a clubbed character
            finalCharImageList.append(char2)
            p1 += 2
            p3 += 2

        else:
            # It is a single character
            finalCharImageList.append(char1)
            p1 += 1
            p3 += 1

    for i in finalCharImageList:
        fileObject.write(i)
    fileObject.write(' ')
