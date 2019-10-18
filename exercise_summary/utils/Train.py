import numpy as np
import math


def perceptron(input, target, w, b, epochs, learning_rate=0.01, fixed_increment=True):
    error = False
    if not fixed_increment:
        lr = learning_rate / math.log10(epochs)
    else:
        lr=learning_rate
    if (np.dot(w, input.T) + b) * target <= 0:
        w += lr * target * input
        b += lr * target
        error = True
    return w, b, error


def sign(num):
    if num > 0:
        return 1
    else:
        return -1


def report(w, b, epochs, error, i):
    print('------------------------------')
    print('epochs:       \n  ' + (str(i)))
    print('learning rate:\n  ' + (str)(1.0 - error / epochs))
    print('w:            \n  ' + (str)(w))
    print('b:            \n  ' + (str)(b))


def test(w, b, test_data, test_target):
    test_error, test_time = 0, 0
    for data, target in zip(test_data, test_target):
        if (np.dot(w, data) + b) * target <= 0:
            test_error += 1
        test_time += 1
    return test_error, test_time


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class MLP():
    def __init__(self, input_size, output_size, hidden_size_1=40, hidden_size_2=20, learning_rate=0.01):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.learning_rate = learning_rate
        self.w1     = np.random.random((self.hidden_size_1, self.input_size+1)) * 0.3
        self.w2     = np.random.random((self.hidden_size_2, self.hidden_size_1+1)) * 0.3
        self.w3     = np.random.random((self.output_size,   self.hidden_size_2+1)) * 0.3
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


    def backward(self, t, learning_rate=None):
        if learning_rate != None:
            self.lr = learning_rate

        dEde = t - self.y3 
        self.loss = np.sum(np.abs(dEde)) 
        self.losses.append(self.loss)
        dedy3 = -1 
        dy3da3 = self.y3 * (1 - self.y3) 
        da3dw3 = self.x3.T
        tmp3 = self.lr * dEde * dedy3 * dy3da3
        self.dw3 = tmp3 * da3dw3

        da3dx3 = self.w3 
        da3dy2 = da3dx3[:,1:]
        dy2da2 = self.y2 * (1 - self.y2) 
        da2dw2 = self.x2.T 
        tmp2 = np.sum(tmp3 * da3dy2, axis=0, keepdims=True).T * dy2da2
        self.dw2 = tmp2 * da2dw2 

        da2dx2 = self.w2 
        da2dy1 = da2dx2[:,1:]
        dy1da1 = self.y1 * (1 - self.y1)
        da1dw1 = self.x1.T
        self.dw1 = np.sum(tmp2 * da2dy1, axis=0, keepdims=True).T * dy1da1 * da1dw1


