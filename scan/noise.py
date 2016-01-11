#!/usr/bin/env python
# 
# Scan Example: Computing
# tanh(X.dot(A) + b) * d
# where d is binomial
# 

import numpy as np

from theano import config
from theano import scan
from theano import function
from theano import tensor as tt

rng = tt.shared_randomstreams.RandomStreams(1234)

x_matrix = tt.matrix('X')
a_matrix = tt.matrix('A')
b_vector = tt.vector('b')
d_vector = rng.binomial(size = a_matrix[1].shape)

# Note that if you want to use a random variable d_vector that will not be updated
# through scan loops, you should pass this variable as a non_sequences arguments.
results, updates = scan(lambda x_vector:
                        tt.tanh(tt.dot(x_vector, a_matrix) + b_vector) * d_vector,
                        sequences = x_matrix)
compute_with_binomial_noise = function(inputs = [x_matrix, a_matrix, b_vector],
                                       outputs = results,
                                       updates = updates,
                                       allow_input_downcast = True)

x_matrix_value = np.eye(10, 2, dtype = config.floatX)
a_matrix_value = np.ones((2, 2), dtype = config.floatX)
b_vector_value = np.ones((2), dtype = config.floatX)

print(compute_with_binomial_noise(x_matrix_value, a_matrix_value, b_vector_value))



