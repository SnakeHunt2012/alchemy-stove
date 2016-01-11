#!/usr/bin/env python
#
# Scan Example: Computing trace of X
#

import numpy as np

from theano import config
from theano import scan
from theano import function
from theano import tensor as tt

x_matrix = tt.matrix('X')

results, updates = scan(lambda i, j, k: x_matrix[i, j] + k,
                        sequences = [tt.arange(x_matrix.shape[0]), tt.arange(x_matrix.shape[1])],
                        outputs_info = np.asarray(0., dtype = config.floatX))
result = results[-1]
compute_trace = function(inputs = [x_matrix], outputs = result)

x_matrix_value = np.eye(5, dtype = config.floatX)
x_matrix_value[0] = np.arange(5, dtype = config.floatX)

print(compute_trace(x_matrix_value))
print(np.diagonal(x_matrix_value).sum())

