from initialisations import *
    
graph = {}

def getNodeNumber(rows, cols, x, y):
    return int((x * cols) + y + 1)

def bellCurve(x, baseline):
    baseline = baseline / 2
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

def getStandardBell(x,mean,sd):
    r =1/( sd * math.sqrt(2*np.pi))
    y = r + (-r * math.exp(-((x - mean)**2)/(sd * sd * 2)))
    return y

def _ss(data,baseline):
    """Return sum of square deviations of sequence data."""
    c = baseline/2
    ss = sum((x-c)**2 for x in data)
    return ss

def pstdev(data,baseline):
    """Calculates the population standard deviation."""
    n = len(data)
    if n < 2:
        raise ValueError('variance requires at least two data points')
    ss = _ss(data,baseline)
    pvar = ss/n # the population variance
    return pvar**0.5    
    
def getY(colsum, height, baseline, c):
    Y = []
    C = []
    H = []
    
    sd = pstdev(height,baseline)

    for i in range(c):
        C.append(math.exp(-(colsum[i]-(baseline/2))))
        H.append(getStandardBell(height[i], baseline/2,sd))
        Y.append(5 * math.exp(-(colsum[i]-(baseline/2))) * (getStandardBell(height[i], baseline/2,sd)))
        #Y.append(100/((colsum[i]**10) * (baseline**2 * bellCurve(height[i], baseline)) + 1))
        
        # 1 / ((log(5 + x))^4)  -  COLSUM
        
        #Y.append((colsum[i] + bellCurve(height[i], baseline))/2)
    #plt.plot(range(c), C, 'g-')
    #plt.plot(range(c), H, 'b-')
    plt.plot(range(c), Y, 'r-')
    #print(Y)
    #print(colsum)
    #plt.show()
    return Y
    
    
def getPathChar(NodeNumber):
    path = []
    if graph[NodeNumber][3] == -1:     # If the node is not reachable
        path.append(-1)
        return path
    minP = math.inf
    maxP = -math.inf
    #print(NodeNumber)
    while not graph[NodeNumber][2] == -2:
        y = graph[NodeNumber][1]
        if minP >= y:
            minP = y
        if maxP <= y:
            maxP = y
        path.append(NodeNumber)
        NodeNumber = graph[NodeNumber][2]

    y = graph[NodeNumber][1]
    if minP >= y:
        minP = y
    if maxP <= y:
        maxP = y
    path.append(NodeNumber)
    path.append(minP)
    path.append(maxP)
    return path

def bfsChar(IM, start, y1, y2, mn, mx, r, c):
    x = graph[start][0]
    if x < mn or x > mn:
        return
    y = graph[start][1]
    graph[start][2] = -2;
    graph[start][3] = 1;
    vertQueue = Queue()
    vertQueue.enqueue(start)
    while (vertQueue.size() > 0):
        current = vertQueue.dequeue()
        x = graph[current][0]
        y = graph[current][1]

        #bottom node
        if x + 1 <= mx and IM[x + 1][y] == 255:
            neighbour = getNodeNumber(r,c,x + 1, y)
            if graph[neighbour][3] == -1:
                graph[neighbour][3] = 1
                graph[neighbour][2] = current
                vertQueue.enqueue(neighbour)

        #top node
        if x - 1 >= mn and IM[x - 1][y] == 255:
            neighbour = getNodeNumber(r,c,x - 1, y)
            if graph[neighbour][3] == -1:
                graph[neighbour][3] = 1
                graph[neighbour][2] = current
                vertQueue.enqueue(neighbour)

        #right node
        if y + 1 <= y2 and IM[x][y + 1] == 255:
            neighbour = getNodeNumber(r,c,x, y + 1)
            if graph[neighbour][3] == -1:
                graph[neighbour][3] = 1
                graph[neighbour][2] = current
                vertQueue.enqueue(neighbour)

        #left node
        if y - 1 >= y1 and IM[x][y - 1] == 255:
            neighbour = getNodeNumber(r,c,x, y - 1)
            if graph[neighbour][3] == -1:
                graph[neighbour][3] = 1
                graph[neighbour][2] = current
                vertQueue.enqueue(neighbour)

def segmentCharacter(IM, r, c, brkPts, size, baseline):

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

        graph.clear()
        for j in range(r):
            for k in range(c):
                graph[getNodeNumber(r, c, j, k)] = [j, k, -1, -1]
        bfsChar(IM, getNodeNumber(r, c, firstWhiteAfterBlackPixel, brkPts[i - 1]), brkPts[i - 1], brkPts[i], 0, r - 1, r, c)
        tempChar = getPathChar(getNodeNumber(r, c, 0, brkPts[i - 1]))
        if tempChar[0] == -1:
            tempChar[0] = math.inf
            tempChar.append(math.inf)
        pathList1 = tempChar

        graph.clear()
        for j in range(r):
            for k in range(c):
                graph[getNodeNumber(r, c, j, k)] = [j, k, -1, -1]
        bfsChar(IM, getNodeNumber(r, c, firstBlackPixel, brkPts[i - 1]), brkPts[i - 1], brkPts[i], 0, r - 1, r, c)
        tempChar = getPathChar(getNodeNumber(r, c, r - 1, brkPts[i - 1]))
        if tempChar[0] == -1:
            tempChar[0] = math.inf
            tempChar.append(math.inf)
        pathList2 = tempChar

        mini = min(pathList1[-2], pathList2[-2])

        firstBlackPixel = getFirstBlackPixel(IM, baseline, brkPts[i])
        firstWhiteAfterBlackPixel = getFirstWhiteAfterBlackPixel(IM, baseline, brkPts[i])

        graph.clear()
        for j in range(r):
            for k in range(c):
                graph[getNodeNumber(r, c, j, k)] = [j, k, -1, -1]
        bfsChar(IM, getNodeNumber(r, c, firstWhiteAfterBlackPixel, brkPts[i]), brkPts[i - 1], brkPts[i], 0, r - 1, r, c)
        tempChar = getPathChar(getNodeNumber(r, c, 0, brkPts[i]))
        if tempChar[0] == -1:
            tempChar[0] = -math.inf
        pathList3 = tempChar

        graph.clear()
        for j in range(r):
            for k in range(c):
                graph[getNodeNumber(r, c, j, k)] = [j, k, -1, -1]
        bfsChar(IM, getNodeNumber(r, c, firstBlackPixel, brkPts[i]), brkPts[i - 1], brkPts[i], 0, r - 1, r, c)
        tempChar = getPathChar(getNodeNumber(r, c, r - 1, brkPts[i]))
        if tempChar[0] == -1:
            tempChar[0] = -math.inf
        pathList4 = tempChar

        maxi = max(pathList3[-1], pathList4[-1])

        if maxi == -math.inf or mini == math.inf:
            continue

        # Temporary image initialised to zeroes
        charImage = [[0 for j in range(maxi - mini + 1)] for k in range(r)]

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

        graph.clear()
        for j in range(r):
            for k in range(c):
                graph[getNodeNumber(r, c, j, k)] = [j, k, -1, -1]
        # Jigsaw (Left break point)
        for j in completeLeftPath:
            for k in range(mini, graph[j][1] + 1):
                charImage[graph[j][0]][k - mini] = 255

        # Jigsaw (Right break point)
        for j in completeRightPath:
            for k in range(graph[j][1], maxi + 1):
                charImage[graph[j][0]][k - mini] = 255

        # Jigsaw (Restore character)
        for j in range(0, r):
            for k in range(0, maxi - mini + 1):
                if charImage[j][k] == 0:
                    charImage[j][k] = IM[j][k + mini]

        charImage = np.array(charImage, dtype = np.uint8)
        cv2.imshow('image', charImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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

            graph.clear()
            for j in range(r):
                for k in range(c):
                    graph[getNodeNumber(r, c, j, k)] = [j, k, -1, -1]
            bfsChar(IM, getNodeNumber(r, c, firstWhiteAfterBlackPixel, brkPts[i - 1]), brkPts[i - 1], brkPts[i + 1], 0, r - 1, r, c)
            tempChar = getPathChar(getNodeNumber(r, c, 0, brkPts[i - 1]))
            if tempChar[0] == -1:
                tempChar[0] = math.inf
                tempChar.append(math.inf)
            pathList1 = tempChar

            graph.clear()
            for j in range(r):
                for k in range(c):
                    graph[getNodeNumber(r, c, j, k)] = [j, k, -1, -1]
            bfsChar(IM, getNodeNumber(r, c, firstBlackPixel, brkPts[i - 1]), brkPts[i - 1], brkPts[i + 1], 0, r - 1, r, c)
            tempChar = getPathChar(getNodeNumber(r, c, r - 1, brkPts[i - 1]))
            if tempChar[0] == -1:
                tempChar[0] = math.inf
                tempChar.append(math.inf)
            pathList2 = tempChar

            mini = min(pathList1[-2], pathList2[-2])

            firstBlackPixel = getFirstBlackPixel(IM, baseline, brkPts[i + 1])
            firstWhiteAfterBlackPixel = getFirstWhiteAfterBlackPixel(IM, baseline, brkPts[i + 1])

            graph.clear()
            for j in range(r):
                for k in range(c):
                    graph[getNodeNumber(r, c, j, k)] = [j, k, -1, -1]
            bfsChar(IM, getNodeNumber(r, c, firstWhiteAfterBlackPixel, brkPts[i + 1]), brkPts[i - 1], brkPts[i + 1], 0, r - 1, r, c)
            tempChar = getPathChar(getNodeNumber(r, c, 0, brkPts[i + 1]))
            if tempChar[0] == -1:
                tempChar[0] = -math.inf
            pathList3 = tempChar

            graph.clear()
            for j in range(r):
                for k in range(c):
                    graph[getNodeNumber(r, c, j, k)] = [j, k, -1, -1]
            bfsChar(IM, getNodeNumber(r, c, firstBlackPixel, brkPts[i + 1]), brkPts[i - 1], brkPts[i + 1], 0, r - 1, r, c)
            tempChar = getPathChar(getNodeNumber(r, c, r - 1, brkPts[i + 1]))
            if tempChar[0] == -1:
                tempChar[0] = -math.inf
            pathList4 = tempChar

            maxi = max(pathList3[-1], pathList4[-1])

            if maxi == -math.inf or mini == math.inf:
                continue

            # Temporary image initialised to zeroes
            charImage = [[0 for j in range(maxi - mini + 1)] for k in range(r)]

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
            graph.clear()
            for j in range(r):
                for k in range(c):
                    graph[getNodeNumber(r, c, j, k)] = [j, k, -1, -1]



            # Jigsaw (Left break point)
            for j in completeLeftPath:
                for k in range(mini, graph[j][1] + 1):
                    charImage[graph[j][0]][k - mini] = 255

            # Jigsaw (Right break point)
            for j in completeRightPath:
                for k in range(graph[j][1], maxi + 1):
                    # print("%%%%%%%%%%%%%%%", graph[j][0], mn)
                    charImage[graph[j][0]][k - mini] = 255

            # Jigsaw (Restore character)
            for j in range(0, r):
                for k in range(0, maxi - mini + 1):
                    if charImage[j][k] == 0:
                        charImage[j][k] = IM[j][k + mini]

            charImage = np.array(charImage, dtype = np.uint8)
            cv2.imshow('image', charImage)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Appending clubbed image of characters
            charImageListClub.append(charImage)

    return charImageListSingle, charImageListClub       

def characterSegmentation(I, baseline, net1):

    rows = I.shape[0]
    cols = I.shape[1]

    colsum = []
    height = []

    for i in range(cols):
        s = 0
        for j in range(baseline):
            if I[j][i] == 0:
                s += 1
        colsum.append(s)
        s = 0
        for j in range(baseline):
            if I[j][i] == 0:
                s = baseline - j - 1
                break;
        height.append(s)

    interval = getY(colsum, height, baseline, cols)

    brkPts = getMaxima2(interval)
    size = len(brkPts)
    
    charImageListSingle, charImageListClub = segmentCharacter(I,rows,cols, brkPts,size, baseline)

    finalCharImageList = []

    p1 = 0      # First pointer in the single list
    p2 = 0      # Second pointer in the single list
    p3 = 0      # Pointer in the clubbed list
    
    singleListSize = len(charImageListSingle)
    clubListSize = len(charImageListClub)

    while p1 < singleListSize - 1 and p3 < clubListSize:
        p2 = p1 + 1
        char1, prob1 = classifyCharacter(charImageListSingle[p1], net1)
        char3, prob3 = classifyCharacter(charImageListSingle[p2], net1)
        if prob1 <= prob3:
            char1 = char3
            prob1 = prob3
        char2, prob2 = classifyCharacter(charImageListClub[p3], net1)
        if prob2 >= prob1:
            # It is a clubbed character
            finalCharImageList.append(char2)
            #print(char2)
            p1 += 2
            p3 += 2

        else:
            # It is a single character
            finalCharImageList.append(char1)
            #print(char1)
            p1 += 1
            p3 += 1

    for i in finalCharImageList:
        print(chr(i[0]+97))
