# An attempt to create a neural network using tensorflow

import numpy as np
import tensorflow as tf
import random

class NN():
    def __init__(self, n_inputs, layers):
        self.n_inputs=n_inputs
        self.layers=layers
        self.setup()

    def setup(self):
        X=tf.placeholder(tf.float32, shape=[2, None])
        Y=tf.placeholder(tf.float32, shape=[2, None])

        W=[None]*3
        b=[None]*3
        Z=[None]*3
        A=[None]*3
        ln=[None]*3

        A.insert(0,X)
        ln.insert(0,2)

        l=1
        for cl in self.layers:
            W.insert(l, tf.Variable(tf.random_uniform([ln[l-1],ln], -1, 1, tf.float32)))
            b.insert(l, tf.Variable(tf.random_uniform([1,ln], -1, 1, tf.float32)))
            Z.insert(l, tf.add(tf.multiply(W[l], A[l-1]), b[l]))
            A.insert(l, tf.sigmoid(Z[l]))

            l+=1

        W.insert(l, tf.Variable(tf.random_uniform([


    def forward(self):
        pass

    def back(self):
        pass


def makeData(n):
    f=np.zeros((2,n))
    c=np.zeros((1,n))
    for i in range(n):
        (x,y)=(random.uniform(-1,1),random.uniform(-1,1))
        z=1 if x**2+y**2<.7 else 0
        f[0][i]=x
        f[1][i]=y
        c[0][i]=z
    return f,c

def main():
    train=makeData(10)
    test=makeData(2)
    nn = NN(2, [10])

if __name__=='__main__':
    main()

