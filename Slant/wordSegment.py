from initialisations import *
from characterSegment import *

def wordSegmentation(IM, maxima, net1):
   
    baseline = maxima 
    rows = IM.shape[0]
    cols = IM.shape[1]

    colList = []
    flag = 0
    
    colList.append(0)
    for i in range(cols):
        colsum = 0
        for j in range(rows):
            if IM[j][i] == 0:
                colsum = colsum + 1
                
        if colsum == 0 and flag == 0:
            flag = 1
            col = i
        
        elif colsum != 0 and flag == 1:
            if i - col > 3:
                colList.append(int((i + col) / 2))
            flag = 0
    colList.append(cols - 1)
            
    for i in range(len(colList)-1):
        subimgWord = []    
        for j in range(rows):                               
            tempList = []         
            for k in range (colList[i + 1] - colList[i] + 1):            
                tempList.append(0)      
            subimgWord.append(tempList) 
            
        for j in range(rows):                               
            for k in range(colList[i + 1] - colList[i] + 1):            
                subimgWord[j][k] = IM[j][colList[i] + k]   

        subimgWord = np.array(subimgWord, dtype=np.uint8)
        subimgWord = shearImage(subimgWord,-computeShear(subimgWord,baseline))
        showImg(subimgWord)
        characterSegmentation(subimgWord, baseline + 2, net1)