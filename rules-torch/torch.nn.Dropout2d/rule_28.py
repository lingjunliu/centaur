import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Valid probability, valid boolean for inplace, and valid tensor dimension (Rule 28)

rule_28 = lambda s, v, n=False: (
    s.add(Not(And(And(And(v["arg1_value"] >= 0.0, v["arg1_value"] <= 1.0), (Or(v["arg2_value"] == True, v["arg2_value"] == False))), (Or(v["arg3_ndim"] == 3, v["arg3_ndim"] == 4)))) if n else
          And(And(And(v["arg1_value"] >= 0.0, v["arg1_value"] <= 1.0), (Or(v["arg2_value"] == True, v["arg2_value"] == False))), (Or(v["arg3_ndim"] == 3, v["arg3_ndim"] == 4))))
)

def rule_28_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = Bool('arg2_value')
        arg3_ndim = Int('arg3_ndim')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_ndim == arg3.ndim)

        # Constraints for rule 28
        rule_28(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_28(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_ndim': arg3['ndim']}, neg)
