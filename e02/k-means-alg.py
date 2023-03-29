import matplotlib.pyplot as plt
import numpy as np
import random

def main():
    #dim = int(input("How many dimensions: "))
    vecsList = VectorList(3)
    for i in range(10):
        vecsList.add([random.uniform(0, 255), random.uniform(0, 255), random.uniform(0, 255)])
    print(vecsList)
    
    
class VectorList:
    def __init__(self, dim):
        self.dim = dim
        self.vecsList = []
        
    def add(self, vector):
        if (np.array(vector)).shape[0] == self.dim:
            self.vecsList.append(vector)
    
    def __str__(self):
        return np.array2string((np.array(self.vecsList)), formatter={'float_kind':lambda x: "%.2f" % x})
            
            
            
main()