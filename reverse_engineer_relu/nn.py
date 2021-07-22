import numpy as np
import os
from . import Tensor

class NeuralNet:
    def __init__(self,X,W,b):
        self.X = X
        self.W = W
        self.b = b
        
    def matmul(self, Z):
        Z = np.matmul(W,X) + b
        return Z
    
    def relu(self,Z):
        return (abs(Z) + Z) / 2

