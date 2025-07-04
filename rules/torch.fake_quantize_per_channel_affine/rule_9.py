import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# quant_min and quant_max must be less than 2**(number of bits of zero_point (Rule 9)

rule_9 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_dtype"] == 3, And(v["arg1_value"] < 4294967296, v["arg2_value"] < 4294967296), If(v["arg3_dtype"] == 6, And(v["arg1_value"] < 65536, v["arg2_value"] < 65536), And(v["arg1_value"] < 4294967296, v["arg2_value"] < 4294967296)))) if n else
          If(v["arg3_dtype"] == 3, And(v["arg1_value"] < 4294967296, v["arg2_value"] < 4294967296), If(v["arg3_dtype"] == 6, And(v["arg1_value"] < 65536, v["arg2_value"] < 65536), And(v["arg1_value"] < 4294967296, v["arg2_value"] < 4294967296))))
)

def rule_9_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 9
        rule_9(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_9(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_dtype': arg3['dtype']}, neg)
