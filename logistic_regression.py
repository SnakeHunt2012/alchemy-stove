#!/usr/bin/env python

from theano import shared
from theano import grad
from theano import function
from theano.tensor import dmatrix
from theano.tensor import dvector
from theano.tensor import exp
from theano.tensor import log
from theano.tensor import dot
from numpy import array
from numpy import random
from numpy import ones

rng = random

N = 4000
feats = 784
D = (rng.randn(N, feats), rng.randint(size = N, low = 0, high = 2))
steps = 1000

# Declare Theano symbolic variables
X = dmatrix('X')
y = dvector('y')
w = shared(rng.randn(feats), name = 'w')
b = shared(0., name = 'b')

print("w: %r", w.get_value())
print("b: %r", b.get_value())

# Construct Theano expression graph
proba = 1 / (1 + exp(-dot(X, w) - b))
pred = proba > 1
loss = -y * log(proba) - (1 - y) * log(1 - proba)
cost = loss.mean() + 0.01 * (w ** 2).sum()
grad_w, grad_b = grad(cost, [w, b])

train = function(inputs = [X, y],
                 outputs = cost,
                 updates = [(w, w - 0.1 * grad_w), (b, b - 0.1 * grad_b)])
test = function(inputs = [X, y],
                outputs = cost)

for i in range(steps):
    cost_value = train(D[0], D[1])
    print ("step: %d, loss: %r" % (i, cost_value))

print("w: %r", w.get_value())
print("b: %r", b.get_value())

