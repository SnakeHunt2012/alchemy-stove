#!/usr/bin/env python
# 
# Scan Example: Computing the Jacobian of
# y = tanh(x.dot(A)) wrt x
#

import numpy as np

from theano import config
from theano import scan
from theano import function
from theano import tensor as tt

x_vector = tt.vector('x')
a_matrix = tt.matrix('A')
y_vector = tt.tanh(tt.dot(x_vector, a_matrix))

# Note that we need to iterate over the indices of y and not over the elements of y.
# The reason is that scan create a placeholder variable for its internal function
# and this placeholder variable does not have the same dependencies than the variables
# that will replace it.
results, updates = scan(lambda i: tt.grad(y_vector[i], x_vector),   
                        sequences = [tt.arange(y_vector.shape[0])]) 
compute_jacobian = function(inputs = [a_matrix, x_vector],
                            outputs = results,
                            allow_input_downcast = True) 

x_vector_value = np.eye(5, dtype = config.floatX)[0]
a_matrix_value = np.eye(5, 3, dtype = config.floatX)
a_matrix_value[2] = np.ones((3), dtype = config.floatX)

print(compute_jacobian(a_matrix_value, x_vector_value))
print(((1 - np.tanh(x_vector_value.dot(a_matrix_value)) ** 2) * a_matrix_value).T)
