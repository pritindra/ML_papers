# inputs - X matrix, layer 1 matmul, and relu
# y matrix as ouput

import numpy as np
import os
from . import Tensor

class NeuralNet:
    def __init__(self,X,W,b):
        self.X = X
        self.W = W
        self.b = b
        
    def mul(self):
        Z = np.matmul(np.transpose(W),X) + b
        return Z
    
    def relu(self,Z):
        res = Z
        res[Z<0] = 0
        return res

    def sigmoid(self,Z):
        return 1.0 / (1 + np.exp(-1 * Z))

    def upd_W(W, lr):
        # need to apply gradient descent
        new_W = W - lr*W
        return new_W
    
    def train(iterations,W,X,Y,b,lr,activation="relu"):
        for it in range(iterations):
            print("Iteration ", it)
            for sample_idx in range(X.shape[0]):
                r1 = X[sample_idx, :]
                for idx in range(len(W)-1):
                    curr_W = W[idx]
                    r1 = mul(self,r1,curr_W)
                    if activation = "relu":
                        r1 = relu(r1)
                    elif activation = "sigmoid":
                        r1 = sigmoid(r1)
            curr_W = W[-1]
            r1 = mul(self,r1,curr_W)
            pred_label = np.where(r1 == np.max(r1))[0][0]
            des_label = Y[sample_idx]
            if pred_label = != des_label:
                W = upd_W(W,lr = 0.001)
        return W

    def pred_y(W,X,activation="relu"):
        predictions = np.zeroes(shape=(X.shape[0]))
        
        for sample_idx in range(X.shape[0]):
            r1 = X[sample_idx, :]
            for curr_W in W:
                r1 = mul(r1,curr_W)
            if activation = "relu":
                r1 = relu(self,r1)
            else:
                r1 = sigmoid(self,r1)
            pred_label = np.where(r1 == np.max(r1))[0][0]
            predictions[sample_idx] = pred_label
        return predictions



