#!/usr/bin/env python
# 
# Scan Example: Computing tanh(x(t).dot(A) + b) elementwise
# 

import numpy as np

from theano import config
from theano import scan
from theano import function
from theano import tensor as tt

x_matrix = tt.matrix('X')
a_matrix = tt.matrix('A')
b_vector = tt.vector('b')

results, updates = scan(lambda x_vector:
                        tt.tanh(tt.dot(x_vector, a_matrix) + b_vector),
                        sequences = [x_matrix])
compute_seq = function(inputs = [x_matrix, a_matrix, b_vector],
                       outputs = results)

x_matrix_value = np.eye(2, dtype = config.floatX)
a_matrix_value = np.ones((2, 2), dtype = config.floatX)
b_vector_value = np.ones((2), dtype = config.floatX)
b_vector_value[1] = 2

print(compute_seq(x_matrix_value, a_matrix_value, b_vector_value))
print(np.tanh(x_matrix_value.dot(a_matrix_value) + b_vector_value))
