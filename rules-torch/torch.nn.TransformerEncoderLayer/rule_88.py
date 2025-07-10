import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If activation is not relu or gelu, then dropout should be in (0, 0.5 (Rule 88)

rule_88 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] != 12, v["arg1_value"] != 18), And(And(And(v["arg2_value"] > 0, v["arg2_value"] < 0.5), v["arg3_value"] == False), v["arg4_ndim"] > 2), False)) if n else
          If(And(v["arg1_value"] != 12, v["arg1_value"] != 18), And(And(And(v["arg2_value"] > 0, v["arg2_value"] < 0.5), v["arg3_value"] == False), v["arg4_ndim"] > 2), False))
)

def rule_88_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, bool):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_value = Real('arg2_value')
        arg3_value = Bool('arg3_value')
        arg4_ndim = Int('arg4_ndim')

        # Value assignments
        solver.add(arg1_value == list_of_string_values.index(arg1))
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)
        solver.add(arg4_ndim == arg4.ndim)

        # Constraints for rule 88
        rule_88(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_ndim': arg4_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_88(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_ndim': arg4['ndim']}, neg)
