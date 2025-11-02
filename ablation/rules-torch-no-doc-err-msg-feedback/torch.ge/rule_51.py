import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If v_1 is a tensor and v_2 is an int, and a third tensor is given (v_3 (Rule 51)

rule_51 = lambda s, v, n=False: (
    s.add(Not(And(And(And(-1 * v["arg1_ndim"] <= v["arg2_value"], v["arg2_value"] <= v["arg1_ndim"] - 1), -1 * v["arg3_ndim"] <= v["arg2_value"]), v["arg2_value"] <= v["arg3_ndim"] - 1)) if n else
          And(And(And(-1 * v["arg1_ndim"] <= v["arg2_value"], v["arg2_value"] <= v["arg1_ndim"] - 1), -1 * v["arg3_ndim"] <= v["arg2_value"]), v["arg2_value"] <= v["arg3_ndim"] - 1))
)

def rule_51_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_value = Int('arg2_value')
        arg3_ndim = Int('arg3_ndim')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_ndim == arg3.ndim)

        # Constraints for rule 51
        rule_51(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_51(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_ndim': arg3['ndim']}, neg)
