#!/usr/bin/env python
# 
# Scan Example: Calculating a Polynomial (reduced version)
# 

import numpy as np

from theano import config
from theano import scan
from theano import function
from theano import tensor as tt

config.warn.subtensor_merge_bug = False

max_coefficients_supported = 10000
range_vector = tt.arange(max_coefficients_supported)
x_scalar = tt.scalar('x')
w_vector = tt.vector('w')

results, updates = scan(lambda coeff, power, prior, var:
                        prior + coeff * (var ** power),
                        sequences = [w_vector, range_vector],
                        outputs_info = np.asarray(0, dtype = config.floatX),
                        non_sequences = x_scalar)
result = results[-1]
polynomial = function(inputs = [w_vector, x_scalar],
                      outputs = result)

test_coeff = np.asarray([1, 0, 2], dtype = np.float32)
print(polynomial(test_coeff, 3))
