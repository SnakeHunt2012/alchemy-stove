#!/usr/bin/env python
# 
# Scan Example: Computing the sequence
# x(t) = x(t - 2).dot(U) + x(t - 1).dot(V) + tanh(x(t - 1).dot(W) + b)
#

import numpy as np

from theano import config
from theano import scan
from theano import function
from theano import tensor as tt

x_matrix = tt.matrix('X')
b_vector = tt.vector('b')
u_matrix = tt.matrix('U')
v_matrix = tt.matrix('V')
w_matrix = tt.matrix('W')
n_scalar = tt.scalar('n', dtype = 'int32')

results, updates = scan(lambda x_i, x_ii:
                        tt.dot(x_i, u_matrix) + tt.dot(x_ii, v_matrix) + tt.tanh(tt.dot(x_ii, w_matrix) + b_vector),
                        n_steps = n_scalar, outputs_info = {'initial': x_matrix, 'taps': [-2, -1]})
compute_seq = function(inputs = [x_matrix, b_vector, u_matrix, v_matrix, w_matrix, n_scalar],
                       outputs = results)

x_matrix_value = np.zeros((2, 2), dtype = config.floatX)
x_matrix_value[1, 1] = 1
b_vector_value = np.ones((2), dtype = config.floatX)
u_matrix_value = 0.5 * (np.ones((2, 2), dtype = config.floatX) - np.eye(2, dtype = config.floatX))
v_matrix_value = 0.5 * np.ones((2, 2), dtype = config.floatX)
w_matrix_value = 0.5 * np.ones((2, 2), dtype = config.floatX)
n_scalar_value = 10

print(compute_seq(x_matrix_value, b_vector_value, u_matrix_value, v_matrix_value, w_matrix_value, n_scalar_value))

# comparison with numpy
x_res = np.zeros((10, 2))
x_res[0] = x_matrix_value[0].dot(u_matrix_value) + x_matrix_value[1].dot(v_matrix_value) + np.tanh(x_matrix_value[1].dot(w_matrix_value) + b_vector_value)
x_res[1] = x_matrix_value[1].dot(u_matrix_value) + x_res[0].dot(v_matrix_value) + np.tanh(x_res[0].dot(w_matrix_value) + b_vector_value)
for i in range(2, 10):
    x_res[i] = x_res[i - 2].dot(u_matrix_value) + x_res[i - 1].dot(v_matrix_value) + np.tanh(x_res[i - 1].dot(w_matrix_value) + b_vector_value)
print(x_res)
