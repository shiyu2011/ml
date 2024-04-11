import numpy as np

def evenSquareSort(dList):
    dList = np.array(dList)
    dList[dList % 2 == 0] = dList[dList % 2 == 0]**2
    return np.sort(dList)

data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
print(evenSquareSort(data))