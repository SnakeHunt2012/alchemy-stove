#!/usr/bin/env python
# 
# Scan Example: Computing the sequence
# x(t) = tanh(x(t - 1).dot(U) + y(t).dot(V) + z(T - t).dot(W))
#

import numpy as np

from theano import config
from theano import scan
from theano import function
from theano import tensor as tt

x_vector = tt.vector('x')
y_matrix = tt.matrix('Y')
z_matrix = tt.matrix('Z')
u_matrix = tt.matrix('U')
v_matrix = tt.matrix('V')
w_matrix = tt.matrix('W')

results, updates = scan(lambda y_vector, z_vector, x_scalar:
                        tt.tanh(tt.dot(x_scalar, u_matrix) + tt.dot(y_vector, v_matrix) + tt.dot(z_vector, w_matrix)),
                        sequences = [y_matrix, z_matrix[::-1]],
                        outputs_info = [x_vector])
compute_seq = function(inputs = [x_vector, y_matrix, z_matrix, u_matrix, v_matrix, w_matrix],
                       outputs = results)


x_vector_value = np.zeros((2), dtype = config.floatX)
y_matrix_value = np.ones((5, 2), dtype = config.floatX)
z_matrix_value = np.ones((5, 2), dtype = config.floatX)
u_matrix_value = np.ones((2, 2), dtype = config.floatX)
v_matrix_value = np.ones((2, 2), dtype = config.floatX)
w_matrix_value = np.ones((2, 2), dtype = config.floatX)

x_vector_value[1] = 1
y_matrix_value[0, :] = -3
z_matrix_value[0, :] = 3

print(compute_seq(x_vector_value,
                  y_matrix_value,
                  z_matrix_value,
                  u_matrix_value,
                  v_matrix_value,
                  w_matrix_value))

# comparison with numpy
x_res = np.zeros((5, 2), dtype = config.floatX)
x_res[0] = np.tanh(x_vector_value.dot(u_matrix_value) +
                   y_matrix_value[0].dot(v_matrix_value) +
                   z_matrix_value[4].dot(w_matrix_value))
for i in range(1, 5):
    x_res[i] = np.tanh(x_res[i - 1].dot(u_matrix_value) +
                       y_matrix_value[i].dot(v_matrix_value) +
                       z_matrix_value[4 - i].dot(w_matrix_value))
print(x_res)
