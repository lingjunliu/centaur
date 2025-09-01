import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The values of int tensors needs to be between -1000 and 1000, if the dimension size is not zero (Rule 86)

rule_86 = lambda s, v, n=False: (
    s.add(Not(If(And(And((v["arg1_dtype"] >= 1), (v["arg1_dtype"] <= 4)), (v["arg1_ndim"] > 0)), (And(Select(v["arg1_range"], 0) > -1000, Select(v["arg1_range"], 1) < 1000)), True)) if n else
          If(And(And((v["arg1_dtype"] >= 1), (v["arg1_dtype"] <= 4)), (v["arg1_ndim"] > 0)), (And(Select(v["arg1_range"], 0) > -1000, Select(v["arg1_range"], 1) < 1000)), True))
)

def rule_86_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 86
        rule_86(solver, {'arg1_range': arg1_range, 'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_86(solver, {'arg1_range': arg1['range'], 'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim']}, neg)
