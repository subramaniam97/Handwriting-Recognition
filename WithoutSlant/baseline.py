def getBaseLine(I,maxima):
    rows = I.shape[0]
    cols = I.shape[1]
    cur = maxima + 1
    flag = 1
    prevsum = 0
    
    for i in range(cols):
        if I[maxima][i] == 0:
            prevsum = prevsum + 1 
          
    avg = 0
    for i in range(rows):
        for j in range(cols):
            rowsum = 0
            if I[i][j] == 0:
                rowsum = rowsum + 1 
        avg = avg + rowsum
        
    avg = avg/rows
    thresh = 20
    
    while (flag and cur < rows):
        cursum = 0
        for i in range(cols):
            if I[cur][i] == 0:
                cursum = cursum + 1 
        
        if ((prevsum - cursum > thresh) or (cursum < avg)):  
            break
        
    return cur
