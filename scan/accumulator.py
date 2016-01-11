#!/usr/bin/env python
# 
# Scan Example: Accumulate number of loop during a scan
# 

import numpy as np

from theano import config
from theano import shared
from theano import scan
from theano import function
from theano import tensor as tt

k_lscalar = shared(0)
n_iscalar = tt.iscalar('n')

results, updates = scan(lambda: {k_lscalar: k_lscalar + 1},
                        n_steps = n_iscalar)
accumulator = function(inputs = [n_iscalar],
                       outputs = [],
                       updates = updates,
                       allow_input_downcast = True)

print(k_lscalar.get_value())
accumulator(5)
print(k_lscalar.get_value())
