import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if ksize length is 1 and rank is 3, then data format can be any of NWC, NCW channels_first channels_last. (Rule 165)

rule_165 = lambda s, v, n=False: (
    s.add(Not(If((And(v["arg1_length"] == 1, v["arg2_ndim"] == 3)), (Or(Or(Or(v["arg3_value"] == 29, v["arg3_value"] == 30), v["arg3_value"] == 25), v["arg3_value"] == 24)), True)) if n else
          If((And(v["arg1_length"] == 1, v["arg2_ndim"] == 3)), (Or(Or(Or(v["arg3_value"] == 29, v["arg3_value"] == 30), v["arg3_value"] == 25), v["arg3_value"] == 24)), True))
)

def rule_165_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg2_ndim = Int('arg2_ndim')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_value == list_of_string_values_tf.index(arg3))

        # Constraints for rule 165
        rule_165(solver, {'arg1_length': arg1_length, 'arg2_ndim': arg2_ndim, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_165(solver, {'arg1_length': arg1['length'], 'arg2_ndim': arg2['ndim'], 'arg3_value': arg3['value']}, neg)
