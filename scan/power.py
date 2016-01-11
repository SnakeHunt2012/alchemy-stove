#!/usr/bin/env python
# 
# Scan Example: Computing
# Computing pow(A, k)
# 

import numpy as np

from theano import config
from theano import scan
from theano import function
from theano import tensor as tt

config.warn.subtensor_merge_bug = False

a_dvector = tt.dvector('A')
k_iscalar = tt.iscalar('k')

def inner_function(prior_dvector, b_dvector):
    return prior_dvector * b_dvector

results, updates = scan(fn = inner_function,
                        outputs_info = tt.ones_like(a_dvector),
                        non_sequences = a_dvector,
                        n_steps = k_iscalar)
result = results[-1] # Scan has provided us with A ** 1 through A ** k.
                     # Keep only the last value. Scan notices this and
                     # does not waste memory saving them.
power = function(inputs = [a_dvector, k_iscalar],
                 outputs = result,
                 updates = updates)
                     
print(power(range(10), 2))

