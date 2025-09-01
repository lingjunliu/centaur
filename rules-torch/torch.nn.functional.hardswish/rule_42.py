import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Input v_1 must be a tensor and not numpy.float64. Additionally, if inplace is true, then the input must be a floating point or complex dtype and also cannot be of Char dtype. (Rule 42)

rule_42 = lambda s, v, n=False: (
    s.add(Not(And((If(v["arg2_value"] == True, (And((And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 10)), (v["arg1_dtype"] != 1))), True)), (v["arg1_dtype"] != 12))) if n else
          And((If(v["arg2_value"] == True, (And((And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 10)), (v["arg1_dtype"] != 1))), True)), (v["arg1_dtype"] != 12)))
)

def rule_42_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Bool('arg2_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)

        # Constraints for rule 42
        rule_42(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_42(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
