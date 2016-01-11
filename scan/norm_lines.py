#!/usr/bin/env python
# 
# Scan Example: Computing norms of lines of X
#

import numpy as np

from theano import config
from theano import scan
from theano import function
from theano import tensor as tt

x_matrix = tt.matrix('X')

results, updates = scan(lambda x_i: tt.sqrt(tt.sum(x_i ** 2)), sequences = [x_matrix])
norm_lines = function(inputs = [x_matrix], outputs = results)

x_matrix_value = np.diag(np.arange(1, 6, dtype = config.floatX), 1)

print(norm_lines(x_matrix_value))
print(np.sqrt((x_matrix_value ** 2).sum(1)))
