import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If the output tensor has smaller precision than input one, there will be information loss, so prevent that unless output type is bool or dtypes are same (Rule 69)

rule_69 = lambda s, v, n=False: (
    s.add(Not(If((v["arg2_dtype"] != 0), If((And(And(v["arg1_dtype"] < v["arg2_dtype"], v["arg1_dtype"] > 0), v["arg1_dtype"] < 9)), False, If((v["arg1_dtype"] == v["arg2_dtype"]), True, True)), True)) if n else
          If((v["arg2_dtype"] != 0), If((And(And(v["arg1_dtype"] < v["arg2_dtype"], v["arg1_dtype"] > 0), v["arg1_dtype"] < 9)), False, If((v["arg1_dtype"] == v["arg2_dtype"]), True, True)), True))
)

def rule_69_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 69
        rule_69(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_69(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype']}, neg)
