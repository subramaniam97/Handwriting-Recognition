from initialisations import *
    
'''
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
    
'''     

def segmentCharacter(I, brkPts, baseline):
    
    charImageListSingle = [] 
    charImageListClub = [] 
    rows = I.shape[0]
    cols = I.shape[1]  
    size = len(brkPts)
    
    # Populate Single Character List
    for i in range(size - 1):
        
        # Initialise null image
        subimgChar = []  
        for j in range(rows):                               
            templist = []         
            for k in range(brkPts[i + 1] - brkPts[i] + 1):            
                templist.append(0)      
            subimgChar.append(templist) 
    
        for j in range(rows):
            for k in range(brkPts[i + 1] - brkPts[i] + 1):
                subimgChar[j][k] = I[j][k + brkPts[i]]

        subimgChar = np.array(subimgChar, dtype = np.uint8)
        showImg(subimgChar)
        charImageListSingle.append(subimgChar)
        
    # Populate Clubbed Character List
    for i in range(size - 2):
        
        # Initialise null image
        subimgChar = []  
        for j in range(rows):                               
            templist = []         
            for k in range(brkPts[i + 2] - brkPts[i] + 1):            
                templist.append(0)      
            subimgChar.append(templist) 
    
        for j in range(rows):
            for k in range(brkPts[i + 2] - brkPts[i] + 1):
                subimgChar[j][k] = I[j][k + brkPts[i]]

        subimgChar = np.array(subimgChar, dtype = np.uint8)
        #showImg(subimgChar)
        charImageListClub.append(subimgChar)

    return charImageListSingle, charImageListClub         

def characterSegmentation(I, baseline, net1):
    
    rows = I.shape[0]
    cols = I.shape[1]
    first = 0
    last = 0
    flag = 0
    param = []
    for i in range(0,cols):
        colsum = 0
        height = 0
        for j in range(0,rows):
            if I[j][i] == 0:
                height = max(height, rows - j)
                colsum = colsum + 1
        if colsum != 0 and flag == 0:
            first = i
            flag = 1
        if colsum != 0:
            last = i
        param.append(height)
    
    '''
    winSize = getWindowSize(param)
    mavg = param
    for i in range(1, 5):
        mavg = movingaverage(mavg, winSize * i)
    '''
    
    func = interp1d(range(cols), param)
    plt.plot(range(cols), param, 'o', range(cols), func(range(cols)), '-',)
    plt.show() 
 
    # Ignore whitespaces            
    brkPts = []
    brkPts.append(first - 1) 
    minimas = getMinima(param)
    for i in range(len(minimas)):
        if(minimas[i] >= first and minimas[i] <= last):
            brkPts.append(minimas[i])
    brkPts.append(last + 1)
    charImageListSingle, charImageListClub = segmentCharacter(I, brkPts, baseline)

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

    '''
    for i in finalCharImageList:
        print(chr(i[0]+97))
    '''
