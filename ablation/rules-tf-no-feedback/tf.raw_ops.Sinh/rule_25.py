import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if name is specified and x's dtype is float16 or bfloat16, the device must support it (Rule 25)

rule_25 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg2_value"] != 6, (Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7))), True, True)) if n else
          If(And(v["arg2_value"] != 6, (Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7))), True, True))
)

def rule_25_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))

        # Constraints for rule 25
        rule_25(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_25(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
