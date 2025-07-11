import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the local_step is a power of 2, then the gradient's dtype cannot be complex (Rule 62)

rule_62 = lambda s, v, n=False: (
    s.add(Not(If(Or([And(i < (31 + 1), v["arg1_value"] == 2 * i) for i in range(6)]), And(v["arg2_dtype"] != 9, v["arg2_dtype"] != 10), False)) if n else
          If(Or([And(i < (31 + 1), v["arg1_value"] == 2 * i) for i in range(6)]), And(v["arg2_dtype"] != 9, v["arg2_dtype"] != 10), False))
)

def rule_62_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 62
        rule_62(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_62(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype']}, neg)
