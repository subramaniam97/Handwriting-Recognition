from initialisations import *
from wordSegment import *

# 0 : x-value
# 1 : y-value
# 2 : parent node number, -1 if no parent
# 3 : visited check

def getNodeNumber(rows, cols, x, y):
    return int((x * cols) + y + 1)
        
graph = {}
pathList = []
            
def bfs(IM, start, x1, x2, rows, cols):
    x = graph[start][0]
    y = graph[start][1]
    graph[start][2] = -1;
    graph[start][3] = 1;
    vertQueue = Queue()
    vertQueue.enqueue(start)
    
    while (vertQueue.size() > 0):
        current = vertQueue.dequeue()
        x = graph[current][0]
        y = graph[current][1]
        
        #bottom node
        if x + 1 <= x2 and IM[x + 1][y] == 255:
            neighbour = getNodeNumber(rows, cols, x + 1, y)
            if graph[neighbour][3] == -1:
                graph[neighbour][3] = 1
                graph[neighbour][2] = current
                vertQueue.enqueue(neighbour)
        
        #top node
        if x - 1 >= x1 and IM[x - 1][y] == 255:
            neighbour = getNodeNumber(rows, cols, x - 1, y)
            if graph[neighbour][3] == -1:
                graph[neighbour][3] = 1
                graph[neighbour][2] = current
                vertQueue.enqueue(neighbour)
                
        #right node
        if y + 1 <= cols - 1 and IM[x][y + 1] == 255:
            neighbour = getNodeNumber(rows, cols, x, y + 1)
            if graph[neighbour][3] == -1:
                graph[neighbour][3] = 1
                graph[neighbour][2] = current
                vertQueue.enqueue(neighbour)
        
        #left node
        if y - 1 >= 0 and IM[x][y - 1] == 255:
            neighbour = getNodeNumber(rows, cols, x, y - 1)
            if graph[neighbour][3] == -1:
                graph[neighbour][3] = 1
                graph[neighbour][2] = current
                vertQueue.enqueue(neighbour)

                
def getPath(NodeNumber):
    path = []
    minP = math.inf
    maxP = -math.inf
    
    while not graph[NodeNumber][2] == -1:
        x = graph[NodeNumber][0]
        if minP >= x:
            minP = x
        if maxP <= x:
            maxP = x
        path.append(NodeNumber)
        NodeNumber = graph[NodeNumber][2]

    x = graph[NodeNumber][0]
    if minP >= x:
        minP = x
    if maxP <= x:
        maxP = x
        
    path.append(NodeNumber)
    path.append(minP)
    path.append(maxP)
    return path
    
    
def startBFSandReturnPathList(IM, rows, cols, brkPts, size):
    
    for i in range(1, size):
        graph.clear()
        for j in range(rows):
            for k in range(cols):
                graph[getNodeNumber(rows, cols, j, k)] = [j, k, -1, -1]
        # Do BFS for each line
        bfs(IM, getNodeNumber(rows, cols, int((brkPts[i - 1] + brkPts[i]) / 2) , 0), brkPts[i - 1], brkPts[i], rows, cols)
        # Store path
        pathList.append(getPath(getNodeNumber(rows, cols, int((brkPts[i - 1] + brkPts[i]) / 2) , cols - 1)))
    
        
def lineSegmentation(IM, rows, cols, brkPts, size, net1):
    startBFSandReturnPathList(IM, rows, cols, brkPts, size)
    
    # 'size - 1' number of paths, 'size' number of lines 
    # For all lines
    for i in range(0,size):
        
        currPath = []
        prevPath = []
        mini = 0
        maxi = rows - 1
                  
        # For first line as exception
        if i != 0:
            prevPath = pathList[i-1]
            mini = prevPath[len(prevPath) - 2]
        
        # For last line as exception  
        if i != size-1:
            currPath = pathList[i]
            maxi = currPath[len(currPath) - 1]
                            
        # Initialise null image
        subimgLine = []  
        for j in range(maxi - mini + 1):                               
            templist = []         
            for k in range (cols):            
                templist.append(0)      
            subimgLine.append(templist) 
         
        # For first line as exception       
        if i != 0:
            for j in prevPath[:-2]:
                for k in range(mini,graph[j][0]+1):
                    subimgLine[k - mini][graph[j][1]] = 255
            
        # For last line as exception 
        if i != size-1:
            for j in currPath[:-2]:
                for k in range(graph[j][0],maxi+1):
                    subimgLine[k - mini][graph[j][1]] = 255
        
        for j in range (maxi-mini+1):                                
            for k in range (cols):     
                if subimgLine[j][k] == 0:
                    subimgLine[j][k] = IM[j + mini][k] 
           
        subimgLine = np.array(subimgLine, dtype = np.uint8)
        #showImg(subimgLine)
        wordSegmentation(subimgLine, brkPts[i] - mini, net1)