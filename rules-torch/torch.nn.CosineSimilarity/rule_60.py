import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If the specified dimension is out of range, then it should not be considered (Rule 60)

rule_60 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg2_value"] >= v["arg1_ndim"], v["arg2_value"] < (0 - v["arg1_ndim"])), True, And(0 <= v["arg2_value"], v["arg2_value"] < v["arg1_ndim"]))) if n else
          If(Or(v["arg2_value"] >= v["arg1_ndim"], v["arg2_value"] < (0 - v["arg1_ndim"])), True, And(0 <= v["arg2_value"], v["arg2_value"] < v["arg1_ndim"])))
)

def rule_60_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 60
        rule_60(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_60(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)
