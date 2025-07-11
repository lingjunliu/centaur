import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If P is 0 then reduction is not constant, if swap is true, the margin must be positive, and tensors have at least one dimension (Rule 70)

rule_70 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 0, And(And(And(And(v["arg2_value"] != 20, (If(v["arg3_value"] == True, v["arg4_value"] > 0, False))), v["arg5_ndim"] >= 1), v["arg6_ndim"] >= 1), v["arg7_ndim"] >= 1), False)) if n else
          If(v["arg1_value"] == 0, And(And(And(And(v["arg2_value"] != 20, (If(v["arg3_value"] == True, v["arg4_value"] > 0, False))), v["arg5_ndim"] >= 1), v["arg6_ndim"] >= 1), v["arg7_ndim"] >= 1), False))
)

def rule_70_func(arg1, arg2, arg3, arg4, arg5, arg6, arg7, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))
    arg6 = next(iter(arg6.values()))
    arg7 = next(iter(arg7.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, str):
            return False
        if not isinstance(arg3, bool):
            return False
        if not isinstance(arg4, (float, np.floating)):
            return False
        if not isinstance(arg5, np.ndarray):
            return False
        if not isinstance(arg6, np.ndarray):
            return False
        if not isinstance(arg7, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = String('arg2_value')
        arg3_value = Bool('arg3_value')
        arg4_value = Real('arg4_value')
        arg5_ndim = Int('arg5_ndim')
        arg6_ndim = Int('arg6_ndim')
        arg7_ndim = Int('arg7_ndim')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == list_of_string_values_torch.index(arg2))
        solver.add(arg3_value == arg3)
        solver.add(arg4_value == arg4)
        solver.add(arg5_ndim == arg5.ndim)
        solver.add(arg6_ndim == arg6.ndim)
        solver.add(arg7_ndim == arg7.ndim)

        # Constraints for rule 70
        rule_70(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_ndim': arg5_ndim, 'arg6_ndim': arg6_ndim, 'arg7_ndim': arg7_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_70(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_ndim': arg5['ndim'], 'arg6_ndim': arg6['ndim'], 'arg7_ndim': arg7['ndim']}, neg)
