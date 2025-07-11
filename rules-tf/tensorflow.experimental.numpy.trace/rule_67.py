import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The axes used for trace must be different and their value needs to be valid for this tensor's ndim, and be non negative. (Rule 67)

rule_67 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(v["arg1_value"] != v["arg2_value"], v["arg1_value"] >= 0), v["arg2_value"] >= 0), v["arg1_value"] < v["arg3_ndim"]), v["arg2_value"] < v["arg3_ndim"])) if n else
          And(And(And(And(v["arg1_value"] != v["arg2_value"], v["arg1_value"] >= 0), v["arg2_value"] >= 0), v["arg1_value"] < v["arg3_ndim"]), v["arg2_value"] < v["arg3_ndim"]))
)

def rule_67_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg3_ndim = Int('arg3_ndim')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_ndim == arg3.ndim)

        # Constraints for rule 67
        rule_67(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_67(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_ndim': arg3['ndim']}, neg)
