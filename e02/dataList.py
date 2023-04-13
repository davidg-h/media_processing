import random
import numpy as np

'''Class for generat random test data (testing of kmean alg)'''
class DataList:
    def __init__(self, dim):
        self.dim = dim
        self.dataList = []
    
    def __str__(self):
        return np.array2string((np.array(self.dataList)), formatter={'float_kind':lambda x: "%.2f" % x})
    
    def add(self, data_vector):
        if (np.array(data_vector)).shape[0] == self.dim:
            self.dataList.append(data_vector)
            
    def npList(self):
        return np.array(self.dataList)
    
    @staticmethod
    def createDP(dim):
        dp = []
        for _ in range(dim):
            dp.append(random.uniform(0,255))
        return dp