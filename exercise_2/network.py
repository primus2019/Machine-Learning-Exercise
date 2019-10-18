import numpy as np
import random
from utils import *

class MLP:
    def __init__(self, input_size=9, output_size=6, hidden_size_1=90, hidden_size_2=60, learning_rate=0.1):
        self.input_size     = input_size
        self.output_size    = output_size
        self.hidden_size_1  = hidden_size_1
        self.hidden_size_2  = hidden_size_2
        self.w1     = np.random.random((self.hidden_size_1, self.input_size+1))*0.3
        self.w2     = np.random.random((self.hidden_size_2, self.hidden_size_1+1))*0.3
        self.w3     = np.random.random((self.output_size,   self.hidden_size_2+1))*0.3
        
        self.lr = learning_rate
        self.losses = []

    def forward(self, x):
        bias = np.ones((1,x.shape[1]), dtype=x.dtype)
        self.x1 = np.r_[bias,x] 
        assert self.x1.shape[0] == self.w1.shape[1], "w1 size error"
        self.a1 = np.dot(self.w1,self.x1)
        self.y1 = sigmoid(self.a1)
        bias = np.ones((1,self.y1.shape[1]), dtype=x.dtype)
        self.x2 = np.r_[bias,self.y1] 
        assert self.x2.shape[0] == self.w2.shape[1], "w2 size error"
        self.a2 = np.dot(self.w2,self.x2)
        self.y2 = sigmoid(self.a2)
        bias = np.ones((1,self.y2.shape[1]), dtype=x.dtype)
        self.x3 = np.r_[bias,self.y2]
        assert self.x3.shape[0] == self.w3.shape[1], "w2 size error"
        self.a3 =  np.dot(self.w3, self.x3) 
        self.y3 = sigmoid(self.a3)
        
        return self.y3

    def test(self, x):
        x1_t, a1_t, y1_t, x2_t, a2_t, y2_t, x3_t, a3_t, y3_t = self.x1, self.a1, self.y1, self.x2, self.a2, self.y2, self.x3, self.a3, self.y3

        bias = np.ones((1,x.shape[1]), dtype=x.dtype)
        self.x1 = np.r_[bias,x] 
        assert self.x1.shape[0] == self.w1.shape[1], "w1 size error!"
        self.a1 = np.dot(self.w1,self.x1)
        self.y1 = sigmoid(self.a1)
        bias = np.ones((1,self.y1.shape[1]), dtype=x.dtype)
        self.x2 = np.r_[bias,self.y1] 
        assert self.x2.shape[0] == self.w2.shape[1], "w2 size error!"
        self.a2 = np.dot(self.w2,self.x2)
        self.y2 = sigmoid(self.a2)
        bias = np.ones((1,self.y2.shape[1]), dtype=x.dtype)
        self.x3 = np.r_[bias,self.y2]
        assert self.x3.shape[0] == self.w3.shape[1], "w2 size error!"
        self.a3 =  np.dot(self.w3, self.x3) 
        self.y3 = sigmoid(self.a3)
        
        temp = self.y3
        self.x1, self.a1, self.y1, self.x2, self.a2, self.y2, self.x3, self.a3, self.y3 = x1_t, a1_t, y1_t, x2_t, a2_t, y2_t, x3_t, a3_t, y3_t
        
        return temp

    def backward(self, t, learning_rate=None):
        if learning_rate != None:
            self.lr = learning_rate

        dEde = t - self.y3 
        self.loss = np.sum(np.abs(dEde)) 
        self.losses.append(self.loss)
        dedy3 = -1 
        dy3da3 = self.y3 * ( 1 - self.y3) 
        da3dw3 = self.x3.T 
        tmp3 = self.lr * dEde * dedy3 * dy3da3
        self.dw3 = tmp3 * da3dw3 

        da3dx3 = self.w3 
        da3dy2 = da3dx3[:,1:]
        dy2da2 = self.y2 * (1 - self.y2) 
        da2dw2 = self.x2.T 
        tmp2 = np.sum( tmp3 * da3dy2, axis=0, keepdims=True).T * dy2da2
        self.dw2 = tmp2 * da2dw2 

        da2dx2 = self.w2 
        da2dy1 = da2dx2[:,1:]
        dy1da1 = self.y1 * (1 - self.y1) 
        da1dw1 = self.x1.T 
        self.dw1 = np.sum( tmp2 * da2dy1, axis=0, keepdims=True).T * dy1da1 * da1dw1


        

        # self.dw1 = np.sum(  (self.lr * dEde * dedy2 * dy2da2) * da2dy1 , axis=0, keepdims=True).T * dy1da1 * da1dw1
                           #---------------------------------
        # w1.shape = (13,10)
        # print(self.dw1.shape) 

        self.w1 -= self.dw1
        self.w2 -= self.dw2 
        self.w3 -= self.dw3