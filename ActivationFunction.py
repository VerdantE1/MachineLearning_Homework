import numpy as np
from math import erf,sqrt

def sigmoid(x):
    return 1/(1+np.exp(-x));

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x >= 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

def swish(x, beta=1):
    return x * sigmoid(beta * x)
