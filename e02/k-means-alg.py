import matplotlib.pyplot as plt
import numpy as np
import random

vecsList = None

def main():
    #dim = int(input("How many dimensions: "))
    #clusterCenters = int(input("How many centers: "))
    vecsList = VectorList(3)
    for i in range(10):
        vecsList.add([random.uniform(0, 255), random.uniform(0, 255), random.uniform(0, 255)])
    vecsList = vecsList.npList()
    
    print(f"List\n {vecsList}")
    #print(vecsList[: , 0], vecsList[: , 1])
    
    kMeans(vecsList)
    plotPoints(vecsList)
    
def kMeans(vecs):
    print(f"Dims {vecs.ndim}")
    
    ran = random.randrange(10)
    clusterCenters = [vecs[ran], vecs[depIndex(ran)]]
    
    print(f"Centers {clusterCenters}")
    
    filterList = np.delete(vecs, filterListFunc(vecs, clusterCenters), axis=0)
    print(filterList)
    
    #TODO algo fertig und auslagern 
            
    
    
def filterListFunc(list, valuesToFilter):
    filterIndices = []
    for value in valuesToFilter:
        # search where same vectors
        maskIndices = np.where((list == value).all(axis=1))
        filterIndices.append(maskIndices[0])
        
    return np.array(filterIndices)
    

def plotPoints(array):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    
    x = array[:, 0]
    y = array[:, 1]
    z = array[:, 2]
    ax.scatter(x,y,z)
    plt.show()

def depIndex(rNR):
    number = random.randrange(10)
    while number == rNR:
        number = random.randrange(10)
    
    return number
        
    
class VectorList:
    def __init__(self, dim):
        self.dim = dim
        self.vecsList = []
    
    def __str__(self):
        return np.array2string((np.array(self.vecsList)), formatter={'float_kind':lambda x: "%.2f" % x})
    
    def add(self, vector):
        if (np.array(vector)).shape[0] == self.dim:
            self.vecsList.append(vector)
            
    def npList(self):
        return np.array(self.vecsList)
            
            
main()