import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# The size of the output datatype should be a multiple of the input datatype's size, or vice versa, in bytes (Rule 35)

rule_35 = lambda s, v, n=False: (
    s.add(Not(Or([And(i < (10 + 1), Or((v["arg1_dtype"] * i) == v["arg2_value"], Or([And(j < (10 + 1), (v["arg2_value"] * j) == v["arg1_dtype"]) for j in range(6)]))) for i in range(6)])) if n else
          Or([And(i < (10 + 1), Or((v["arg1_dtype"] * i) == v["arg2_value"], Or([And(j < (10 + 1), (v["arg2_value"] * j) == v["arg1_dtype"]) for j in range(6)]))) for i in range(6)]))
)

def rule_35_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))

        # Constraints for rule 35
        rule_35(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_35(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
