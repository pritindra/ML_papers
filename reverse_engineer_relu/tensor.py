import functools
import os
import numpy as np

class tensor:
    def __init__(self,data):
        self.grad = None
        
    def assign(self,x):
        self.data = x.data

    @property
    def shape(self):
        return self.data.shape

    @classmethod
    def ones(cls, *shape, **kwargs):
        return cls(np.ones(shape, dtype=np.float32), **kwargs)

    def sigmoid(self):
        e = self.exp()
        return e.div(1 + e)
    
    def relu(self):
        return self.relu()

    def tanh(self):
        return 2.0 * ((2.0 * self).sigmoid()) - 1.0

 
